import os
import logging
import hydra
from pyprojroot import here
import numpy as np

import graph_tool
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from autograph.models.seq_models import SequenceModel
from autograph.utils.mol import smiles2graph
from autograph.datamodules.mol_dataset import GuacamolDataset
from autograph.evaluation.molsets import compute_molecular_metrics
from autograph.evaluation.visualization import plot_smiles


torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True

OmegaConf.register_new_resolver('eval', eval)

log = logging.getLogger(__name__)

@hydra.main(
    version_base="1.3", config_path=str(here() / "configs"), config_name="test"
)
def main(cfg):
    log.info(f"Configs:\n{OmegaConf.to_yaml(cfg)}")
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Loading model from {cfg.model.pretrained_path}...")
    model = SequenceModel.load_from_checkpoint(cfg.model.pretrained_path)
    model.update_cfg(cfg)

    datamodule = model._datamodule


    device = device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    model.eval()
    model = model.to(device)

    motif = "C1C=CNC2=CC=CC=C21"
    graph = smiles2graph(motif)
    if cfg.sampling.motif_num > 1:
        from torch_geometric.data import Batch, Data
        graph = Batch.from_data_list([graph] * cfg.sampling.motif_num)
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
    print(graph)

    num_samples = max(100, cfg.sampling.num_samples)
    graph_list = []
    model.tokenizer.append_eos = False

    for i in range(0, num_samples, cfg.sampling.batch_size):
        batch_size = min(cfg.sampling.batch_size, num_samples - i)
        input_ids = model.tokenizer(graph)
        input_ids = input_ids.repeat((batch_size, 1))
        input_ids = input_ids.to(device)
        graphs, time = model.generate(input_ids=input_ids)
        graph_list.extend(graphs)

    print(len(graph_list))
    train_smiles = open(datamodule.train_dataset.datasets[0].smiles_path).read().splitlines()
    logs = compute_molecular_metrics(graph_list, GuacamolDataset.atom_decoder, train_smiles=train_smiles)
    smiles = logs.pop('smiles')
    print(logs)

    smiles_path = f"{cfg.logs.path}/smiles"
    os.makedirs(smiles_path, exist_ok=True)
    plot_smiles(smiles_path, smiles[:50])


if __name__ == "__main__":
    main()
