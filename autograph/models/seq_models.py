import os
import os.path as osp
import torch
import hydra
from torch import nn
import pytorch_lightning as pl
import transformers
from transformers import LogitsProcessor, LogitsProcessorList
from ..evaluation.metrics import get_dataset_metric
from ..evaluation.visualization import plot_gridspec_graphs, plot_smiles
from timeit import default_timer as timer

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")


class SequenceModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.tokenizer = None
        self.instantiate_datamodule()
        self.instantiate_model()
        self.instantiate_loss()
        self.instantiate_metrics()
        self.save_hyperparameters()

    def update_cfg(self, cfg):
        self.cfg = cfg

        self.instantiate_datamodule()
        self.instantiate_loss()
        self.instantiate_metrics()
        self.save_hyperparameters()

    def save_pretrained(self, save_directory, **kwargs):
        self.model.model.save_pretrained(save_directory, **kwargs)

    def instantiate_datamodule(self):
        self._datamodule = hydra.utils.instantiate(self.cfg.datamodule, tokenizer=self.tokenizer)
        self._datamodule.prepare_data()
        self._datamodule.setup()
        if self.tokenizer is None:
            self.tokenizer = self._datamodule.tokenizer
        # self.vocab_size = len(self.tokenizer)

    def instantiate_model(self):
        self.model = hydra.utils.instantiate(self.cfg.model, tokenizer=self.tokenizer)

    def instantiate_loss(self):
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad)

    def instantiate_metrics(self):
        dataset = hydra.utils.instantiate(
            self.cfg.datamodule, init_tokenizer=False
        )
        dataset.setup()
        dataset.setup('test')
        self.sampling_metric = get_dataset_metric(
            self.cfg.datamodule.dataset_names,
            dataset,
            num_ref_graphs=self.cfg.sampling.num_samples
        )

    def shared_step(self, batch, batch_idx, phase='train'):
        if isinstance(batch, (tuple, list)):
            batch, attn_mask = batch
        x, y = batch[:, :-1], batch[:, 1:]
        y_pred = self.model(x)
        y_pred = y_pred.view(-1, y_pred.shape[-1])
        y = y.reshape(-1)
        loss = self.loss_fn(y_pred, y)
        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase='train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase='val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase='test')
        return loss

    def on_validation_epoch_end(self):
        self.evaluate_sampling(phase='val')

    def on_test_epoch_end(self):
        self.evaluate_sampling(phase='test')

    def evaluate_sampling(self, phase='val'):
        if self.sampling_metric is None:
            raise RuntimeError("Please run self.init_metric first!")
        num_samples = self.cfg.sampling.num_samples
        if num_samples <= 0:
            num_samples = (
                self.sampling_metric.num_graphs_test if phase == 'test' else self.sampling_metric.num_graphs_val
            )
        graphs, total_time = self.generate(num_samples=num_samples)
        self.log(f"{phase}/time(s)", total_time)
        metric_log = self.sampling_metric(graphs, split=phase)
        for key in metric_log:
            if key in self.sampling_metric.to_log_metrics:
                self.log(f"{phase}/{key}", metric_log[key])
        if self.cfg.datamodule.dataset_names in ['QM9', 'MOSES', 'Guacamol']:
            if phase == "test":
                with open(osp.join(self.cfg.logs.path, "generated.smiles"), 'w') as f:
                    for smiles in metric_log['smiles']:
                        f.write("%s\n" % smiles)
                    print('All smiles saved')
                smiles_path = f"{self.cfg.logs.path}/smiles"
                os.makedirs(smiles_path, exist_ok=True)
                plot_smiles(smiles_path, metric_log['smiles'][:12])
        else:
            if self.cfg.wandb or phase == "test":
                fig_graphs = plot_gridspec_graphs(graphs[:9])
                plt.savefig(osp.join(self.cfg.logs.path, "generated_graphs.pdf"))
                if self.cfg.wandb:
                    import wandb
                    self.logger.experiment.log({"sampling/graphs": [wandb.Image(fig_graphs)]})
                plt.close(fig_graphs)

    def generate(self, num_samples=None, input_ids=None, return_sent=False):
        num_samples = self.cfg.sampling.num_samples if num_samples is None else num_samples
        if input_ids is not None and isinstance(input_ids, torch.Tensor):
            num_samples = input_ids.shape[0]
        graphs = []
        total_time = 0
        for i in range(0, num_samples, self.cfg.sampling.batch_size):
            batch_size = min(self.cfg.sampling.batch_size, num_samples - i)
            if input_ids is not None and isinstance(input_ids, torch.Tensor):
                init_walk_idx = input_ids[i:i + batch_size]
                init_walk_idx = init_walk_idx.to(self.device)
            else:
                init_walk_idx = torch.full(
                    (batch_size, 1), self.tokenizer.sos, dtype=torch.long, device=self.device
                )
            tic = timer()
            graph = self.model.generate(
                init_walk_idx,
                top_k=self.cfg.sampling.top_k,
                temperature=self.cfg.sampling.temperature,
                max_length=self.cfg.sampling.max_length,
                return_sent=return_sent,
            )
            toc = timer()
            graphs.extend(graph)
            total_time += toc - tic
        total_time /= num_samples
        assert len(graphs) == num_samples, "mismatched length"
        return graphs, total_time

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.cfg.train.optimizer, filter(lambda x: x.requires_grad, self.model.parameters())
        )
        lr_scheduler = hydra.utils.call(self.cfg.train.lr_scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }


class HFSequenceModel(nn.Module):
    model_sizes = {
        'xs': {
            'hidden_size': 384,
            'num_hidden_layers': 6,
            'num_attention_heads': 12,
            'intermediate_size': 4 * 384,
            'n_embd': 384,
            'n_head': 12,
            'n_layer': 6,
        },
        's': {
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 4 * 768,
            'n_embd': 768,
            'n_head': 12,
            'n_layer': 12,
        },
        'm': {
            'hidden_size': 1024,
            'num_hidden_layers': 24,
            'num_attention_heads': 16,
            'intermediate_size': 4 * 1024,
            'n_embd': 1024,
            'n_head': 16,
            'n_layer': 24,
        },
    }

    model_params = {
        'mamba': {
            'initializer_range': 0.02,
            'rescale_prenorm_residual': True,
        },
        'llama2': {
            "rms_norm_eps": 1e-05,
            "max_position_embeddings": 4096,
        },
        'llama3': {
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            },
            "rope_theta": 500000.0,
        }
    }

    model_classes = {
        'gpt2': (transformers.GPT2Config, transformers.GPT2LMHeadModel),
        'llama': (transformers.LlamaConfig, transformers.LlamaForCausalLM),
        'llama2': (transformers.LlamaConfig, transformers.LlamaForCausalLM),
        'llama3': (transformers.LlamaConfig, transformers.LlamaForCausalLM),
        'gpt_neox': (transformers.GPTNeoXConfig, transformers.GPTNeoXForCausalLM),
        'mamba': (transformers.MambaConfig, transformers.MambaForCausalLM),
    }

    def __init__(self, tokenizer, model_name, **kwargs):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = tokenizer.truncation_length
        if self.max_length is not None:
            self.max_length = max(2048, self.max_length)
        self.model_name = model_name

        vocab_params = {
            'vocab_size': len(tokenizer),
            'bos_token_id': tokenizer.sos,
            'eos_token_id': tokenizer.eos,
            'pad_token_id': tokenizer.pad,
        }

        model_name_splits = model_name.split('-')
        model_name = model_name_splits[0]
        model_size = model_name_splits[1]

        model_size_params = self.model_sizes[model_size]
        if model_name == "mamba":
            num_layers = model_size_params['num_hidden_layers'] * 2
            model_size_params = {}
            model_size_params['num_hidden_layers'] = num_layers
        model_params = self.model_params.get(model_name, {})

        model_config, model_cls = self.model_classes[model_name]

        self.model = model_cls(model_config(
            **vocab_params, **model_size_params, **model_params, **kwargs
        ))

    def forward(self, input_ids):
        return self.model(input_ids=input_ids).logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        top_k=10,
        temperature=1.0,
        max_length=None,
        return_sent=False,
    ):
        batch_size = input_ids.shape[0]
        max_length = self.max_length if max_length is None else max_length

        logits_processor = None
        if self.tokenizer.labeled_graph:
            logits_processor = LogitsProcessorList([
                LabeledGraph(self.tokenizer, batch_size, input_ids.device),
            ])
        walk_idx = self.model.generate(
            input_ids,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            max_length=max_length,
            logits_processor=logits_processor,
        )
        edge_index_list = []
        for i in range(batch_size):
            if return_sent:
                edge_index = walk_idx[i]
            else:
                edge_index = self.tokenizer.decode(walk_idx[i])
            edge_index_list.append(edge_index)
        return edge_index_list


class LabeledGraph(LogitsProcessor):
    def __init__(self, tokenizer, batch_size, device='cpu'):
        self.start_bracket = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=device
        )
        self.tokenizer = tokenizer
        idx = torch.arange(len(tokenizer), device=device)
        self.special_idx = idx < tokenizer.idx_offset
        self.idx = (idx >= tokenizer.idx_offset) & (idx < tokenizer.node_idx_offset)
        self.node_idx = (idx >= tokenizer.node_idx_offset) & (idx < tokenizer.edge_idx_offset)
        self.edge_idx = idx >= tokenizer.edge_idx_offset
        self.reset_idx = idx == self.tokenizer.reset
        self.ladj_idx = idx == self.tokenizer.ladj
        self.radj_idx = idx == self.tokenizer.radj
        self.eos_idx = idx == self.tokenizer.eos

    def modify_scores(self, scores, sampled_idx):
        scores[:, self.tokenizer.sos] = float('-inf')
        self.start_bracket = (
            self.start_bracket | (sampled_idx == self.tokenizer.ladj)
        ) & (sampled_idx != self.tokenizer.radj)

        scores_idx = torch.where(self.idx, scores, float('-inf'))
        scores_node_idx = torch.where(self.node_idx, scores, float('-inf'))
        scores_edge_idx = torch.where(self.edge_idx, scores, float('-inf'))

        scores = torch.where(self.idx[sampled_idx] & (~self.start_bracket), scores_node_idx, scores)
        scores = torch.where(
            self.idx[sampled_idx] & self.start_bracket,
            torch.where(self.edge_idx | self.radj_idx, scores, float('-inf')),
            scores
        )
        scores = torch.where(
            self.node_idx[sampled_idx],
            torch.where(self.edge_idx | self.reset_idx | self.ladj_idx | self.eos_idx, scores, float('-inf')),
            scores
        )
        scores = torch.where(self.edge_idx[sampled_idx], scores_idx, scores)
        scores = torch.where(sampled_idx == self.tokenizer.ladj, scores_edge_idx, scores)
        scores = torch.where(
            sampled_idx == self.tokenizer.radj,
            torch.where(self.edge_idx | self.reset_idx | self.eos_idx, scores, float('-inf')),
            scores
        )
        scores = torch.where(sampled_idx == self.tokenizer.reset, scores_idx, scores)
        return scores

    def __call__(self, input_ids, scores):
        return self.modify_scores(scores, input_ids[:, -1:])
