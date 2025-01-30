import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import numpy as np
from .spectre_utils import (
    degree_stats,
    spectral_stats,
    clustering_stats,
    motif_stats,
    orbit_stats_all,
    eval_acc_sbm_graph,
    eval_acc_planar_graph,
    eval_fraction_isomorphic,
    eval_fraction_unique_non_isomorphic_valid,
    is_planar_graph,
    is_sbm_graph,
)
from .molsets import compute_molecular_metrics


class SamplingMetrics(object):
    def __init__(self, datamodule, compute_emd, metrics_list, num_ref_graphs=None, need_train_graphs=True):
        super().__init__()

        self.to_log_metrics = ['degree', 'spectre', 'clustering', 'orbit', 'sbm_acc', 'planar_acc', 'vun', 'ratio']

        if need_train_graphs:
            self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        else:
            self.train_graphs = None
        self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        try:
            self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())
        except:
            self.test_graphs = []
        if num_ref_graphs is not None and num_ref_graphs > 0:
            num_graphs = min([num_ref_graphs, len(self.val_graphs)])
            if self.train_graphs is not None:
                rng = np.random.RandomState(0)
                indices = rng.permutation(len(self.train_graphs))[:num_graphs]
                self.train_graphs = [self.train_graphs[i] for i in indices]

            rng = np.random.RandomState(0)
            indices = rng.permutation(len(self.val_graphs))[:num_graphs]
            self.val_graphs = [self.val_graphs[i] for i in indices]

        self.num_graphs_val = len(self.val_graphs)
        self.num_graphs_test = len(self.test_graphs)
        self.compute_emd = compute_emd
        self.metrics_list = metrics_list

        self.compute_ratio = "ratio" in self.metrics_list
        if self.compute_ratio:
            self.train_metric_cache = {}

    def loader_to_nx(self, loader):
        networkx_graphs = []
        for i, batch in enumerate(loader):
            data_list = batch.to_data_list()
            for j, data in enumerate(data_list):
                networkx_graphs.append(
                    to_networkx(
                        data,
                        node_attrs=None,
                        edge_attrs=None,
                        to_undirected=True,
                        remove_self_loops=True,
                    )
                )
        return networkx_graphs

    def __call__(
        self,
        generated_graphs: list,
        split='val',
        train_mode=False
    ):
        reference_graphs = self.test_graphs if split == 'test' else self.val_graphs
        networkx_graphs = []
        for graph in generated_graphs:
            if isinstance(graph, Data):
                graph = to_networkx(graph, to_undirected=True)
            networkx_graphs.append(graph)

        to_log = {}

        if "degree" in self.metrics_list:
            degree = degree_stats(
                reference_graphs,
                networkx_graphs,
                is_parallel=False,
                compute_emd=self.compute_emd,
            )

            to_log["degree"] = degree

        if "spectre" in self.metrics_list:
            spectre = spectral_stats(
                reference_graphs,
                networkx_graphs,
                is_parallel=False,
                n_eigvals=-1,
                compute_emd=self.compute_emd,
            )

            to_log["spectre"] = spectre

        if "clustering" in self.metrics_list:
            clustering = clustering_stats(
                reference_graphs,
                networkx_graphs,
                bins=100,
                is_parallel=False,
                compute_emd=self.compute_emd,
            )
            to_log["clustering"] = clustering

        if "motif" in self.metrics_list:
            motif = motif_stats(
                reference_graphs,
                networkx_graphs,
                motif_type="4cycle",
                ground_truth_match=None,
                bins=100,
                compute_emd=self.compute_emd,
            )
            to_log["motif"] = motif

        if "orbit" in self.metrics_list:
            orbit = orbit_stats_all(
                reference_graphs, networkx_graphs, compute_emd=self.compute_emd
            )
            to_log["orbit"] = orbit

        if "sbm" in self.metrics_list:
            acc = eval_acc_sbm_graph(networkx_graphs, refinement_steps=100, strict=True)
            to_log["sbm_acc"] = acc

        if "planar" in self.metrics_list:
            planar_acc = eval_acc_planar_graph(networkx_graphs)
            to_log["planar_acc"] = planar_acc

        if "sbm" in self.metrics_list or "planar" in self.metrics_list:
            (
                frac_unique,
                frac_unique_non_isomorphic,
                fraction_unique_non_isomorphic_valid,
            ) = eval_fraction_unique_non_isomorphic_valid(
                networkx_graphs,
                self.train_graphs,
                is_sbm_graph if "sbm" in self.metrics_list else is_planar_graph,
            )
            frac_non_isomorphic = 1.0 - eval_fraction_isomorphic(
                networkx_graphs, self.train_graphs
            )
            to_log.update(
                {
                    "unique": frac_unique,
                    "un": frac_unique_non_isomorphic,
                    "vun": fraction_unique_non_isomorphic_valid,
                    "novel": frac_non_isomorphic,
                }
            )

        if self.compute_ratio and not train_mode:
            if split in self.train_metric_cache.keys():
                train_log = self.train_metric_cache[split]
            else:
                train_log = self(self.train_graphs, split=split, train_mode=True)
                self.train_metric_cache[split] = train_log
            predicted_matrics = []
            train_metrics = []
            for metric in ['degree', 'spectre', 'clustering', 'orbit']:
                if metric in self.metrics_list:
                    predicted_matrics.append(to_log[metric])
                    train_metrics.append(train_log[metric])
            avg_ratio = np.array(predicted_matrics) / np.clip(np.array(train_metrics), a_min=0.0, a_max=None)
            avg_ratio = np.mean(avg_ratio[np.isfinite(avg_ratio)])
            to_log['ratio'] = avg_ratio

        return to_log

    def reset(self):
        pass


class MoleculeSamplingMetrics(object):
    def __init__(self, datamodule, **kwargs):
        self.datamodule = datamodule
        self.atom_decoder = datamodule.atom_decoder
        self.train_smiles = datamodule.train_smiles
        self.to_log_metrics = ['validity', 'novelty', 'uniqueness', 'mol_stable', 'atm_stable', 'relaxed_validity']
        self.num_graphs_test = len(datamodule.test_dataset)

    def __call__(self, generated_graphs, **kwargs):
        return compute_molecular_metrics(generated_graphs, self.atom_decoder, self.train_smiles)


def get_dataset_metric(dataset_name, datamodule, **kwargs):
    if dataset_name == 'planar':
        return SamplingMetrics(
            datamodule, compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre", "planar", "ratio"],
        )
    elif dataset_name == 'sbm':
        return SamplingMetrics(
            datamodule, compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre", "sbm", "ratio"],
        )
    elif dataset_name == 'community':
        return SamplingMetrics(
            datamodule, compute_emd=True,
            metrics_list=["degree", "clustering", "orbit", "spectre", "ratio"],
        )
    elif dataset_name in ['protein', 'point_cloud']:
        return SamplingMetrics(
            datamodule, compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre", "ratio"],
            **kwargs
        )
    elif dataset_name in ['networkx', 'networkx_all', 'networkx_big']:
        return SamplingMetrics(
            datamodule, compute_emd=False,
            metrics_list=["degree", "clustering", "spectre"],
            need_train_graphs=False,
            **kwargs
        )
    elif dataset_name in ['QM9', 'MOSES', 'Guacamol']:
        return MoleculeSamplingMetrics(datamodule, **kwargs)
    else:
        return SamplingMetrics(
            datamodule, compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre"],
            **kwargs,
        )
