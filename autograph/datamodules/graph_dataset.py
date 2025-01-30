import torch
from functools import partial
import pytorch_lightning as pl
from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    IterableDataset,
    RandomSampler
)
from torch_geometric import datasets
from torch_geometric.loader import DataLoader as DataLoaderPyG
from .spectre_dataset import SpectreGraphDataset
from .synthetic_dataset import SyntheticDataset
from .networkx_dataset import NetworkXDataset
from .protein_dataset import ProteinDataset
from .point_cloud_dataset import PointCloudDataset
from .mol_dataset import QM9Dataset, MOSESDataset, GuacamolDataset
from .data.tokenizer import Graph2TrailTokenizer


def add_dataset_name(data, dataset_name):
    data.dataset_name = dataset_name
    return data


class GraphDataset(pl.LightningDataModule):
    datasets_map = {
        # spectre datasets
        'planar': partial(SpectreGraphDataset, dataset_name='planar'),
        'sbm': partial(SpectreGraphDataset, dataset_name='sbm'),
        'comm20': partial(SpectreGraphDataset, dataset_name='comm20'),
        # Synthetic datasets from GraphGT
        'ER': partial(SyntheticDataset, dataset_name='ER'),
        'BA': partial(SyntheticDataset, dataset_name='BA'),
        'waxman': partial(SyntheticDataset, dataset_name='waxman'),
        'random_geo': partial(SyntheticDataset, dataset_name='random_geo'),
        # Networkx synthetic datasets
        'classic': partial(NetworkXDataset, dataset_name='classic'),
        'lattice': partial(NetworkXDataset, dataset_name='lattice'),
        'small': partial(NetworkXDataset, dataset_name='small'),
        'random': partial(NetworkXDataset, dataset_name='random'),
        'geometric': partial(NetworkXDataset, dataset_name='geometric'),
        'tree': partial(NetworkXDataset, dataset_name='tree'),
        'community': partial(NetworkXDataset, dataset_name='community'),
        'social': partial(NetworkXDataset, dataset_name='social'),
        # Networkx big synthetic datasets
        'classic_big': partial(NetworkXDataset, dataset_name='classic', big=True),
        'lattice_big': partial(NetworkXDataset, dataset_name='lattice', big=True),
        'small_big': partial(NetworkXDataset, dataset_name='small', big=True),
        'random_big': partial(NetworkXDataset, dataset_name='random', big=True),
        'geometric_big': partial(NetworkXDataset, dataset_name='geometric', big=True),
        'tree_big': partial(NetworkXDataset, dataset_name='tree', big=True),
        'community_big': partial(NetworkXDataset, dataset_name='community', big=True),
        'social_big': partial(NetworkXDataset, dataset_name='social', big=True),
        # protein 
        'DD': partial(ProteinDataset, dataset_name='DD'),
        # point cloud
        'FIRSTMM_DB': partial(PointCloudDataset, dataset_name='FIRSTMM_DB'),
        # molecules
        'QM9': QM9Dataset,
        'MOSES': MOSESDataset,
        'Guacamol': GuacamolDataset,
    }

    def __init__(
        self,
        root,
        dataset_names='all',
        tokenizer=None,
        init_tokenizer=True,
        max_length=-1,
        truncation_length=None,
        labeled_graph=False,
        undirected=True,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.dataset_names = dataset_names
        self.set_val_metric(dataset_names)
        if dataset_names == 'all':
            self.dataset_names = list(self.datasets_map.keys())
        elif dataset_names == 'spectre':
            self.dataset_names = ['planar', 'sbm', 'comm20']
        elif dataset_names == 'synthetic':
            self.dataset_names = ['ER', 'BA', 'waxman', 'random_geo']
        elif dataset_names == 'networkx':
            self.dataset_names = ['classic', 'lattice', 'small', 'random', 'geometric']
        elif dataset_names == 'networkx_all':
            self.dataset_names = ['classic', 'lattice', 'small', 'random', 'geometric', 'tree', 'community']
        elif dataset_names == 'networkx_big':
            self.dataset_names = ['classic_big', 'lattice_big', 'small_big', 'random_big', 'geometric_big', 'tree_big', 'community_big']
        elif dataset_names == 'protein':
            self.dataset_names = ['DD']
        elif dataset_names == 'point_cloud':
            self.dataset_names = ['FIRSTMM_DB']
        else:
            if not isinstance(dataset_names, list):
                self.dataset_names = [dataset_names]
            for dataset_name in self.dataset_names:
                assert dataset_name in self.datasets_map.keys(), \
                "Not included in the database!"

        self.labeled_graph = labeled_graph
        self.tokenizer = tokenizer
        if tokenizer is None and init_tokenizer:
            self.tokenizer = Graph2TrailTokenizer(
                # dataset_names=self.dataset_names,
                dataset_names=[],
                max_length=max_length,
                truncation_length=truncation_length,
                labeled_graph=labeled_graph,
                undirected=undirected,
            )

        self.kwargs = kwargs
        self.collate_fn = self.tokenizer.batch_converter() if self.tokenizer is not None else None

    def set_val_metric(self, dataset_names):
        if dataset_names in ['planar', 'sbm']:
            self.val_metric = ('vun', 'max')
        else:
            self.val_metric = ('loss', 'min')

    @property
    def atom_decoder(self):
        if self.labeled_graph:
            # TODO: suppport multiple molecule datasets
            return self.train_dataset.datasets[0].atom_decoder
        return None

    @property
    def train_smiles(self):
        if self.labeled_graph:
            return self.train_dataset.datasets[0].smiles
        return None

    def prepare_data(self):
        max_num_nodes = {}
        for dataset_name in self.dataset_names:
            train_dataset = self.datasets_map[dataset_name](
                root=self.root, split='train', pre_transform=partial(add_dataset_name, dataset_name=dataset_name)
            )
            self.datasets_map[dataset_name](
                root=self.root, split='val', pre_transform=partial(add_dataset_name, dataset_name=dataset_name)
            )
            try:
                self.datasets_map[dataset_name](
                    root=self.root, split='test', pre_transform=partial(add_dataset_name, dataset_name=dataset_name)
                )
            except:
                pass
            if self.tokenizer is not None:
                max_num_nodes[dataset_name] = max([g.num_nodes for g in train_dataset])
        if self.tokenizer is not None:
            self.max_num_nodes = max_num_nodes
            self.tokenizer.set_num_nodes(max(max_num_nodes.values()))
            print(self.max_num_nodes)
            if self.labeled_graph:
                # TODO: suppport multiple molecule datasets
                self.tokenizer.set_num_node_and_edge_types(
                    num_node_types=train_dataset.num_node_types,
                    num_edge_types=train_dataset.num_edge_types,
                )

    def setup(self, stage='fit'):
        if stage == 'fit':
            train_dataset = [self.datasets_map[dataset_name](
                root=self.root,
                split='train',
                transform=self.tokenizer,
                pre_transform=partial(add_dataset_name, dataset_name=dataset_name),
            ) for dataset_name in self.dataset_names]
            self.train_dataset = ConcatDataset(train_dataset)
            val_dataset = [self.datasets_map[dataset_name](
                root=self.root,
                split='val',
                transform=self.tokenizer,
                pre_transform=partial(add_dataset_name, dataset_name=dataset_name),
            ) for dataset_name in self.dataset_names]
            self.val_dataset = ConcatDataset(val_dataset)

        if stage == 'test':
            try:
                test_dataset = [self.datasets_map[dataset_name](
                    root=self.root,
                    split='test',
                    transform=self.tokenizer,
                    pre_transform=partial(add_dataset_name, dataset_name=dataset_name),
                ) for dataset_name in self.dataset_names]
                self.test_dataset = ConcatDataset(test_dataset)
            except:
                pass

    def dataloader(self, dataset, **kwargs):
        if self.tokenizer is None:
            return DataLoaderPyG(dataset, **kwargs)
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            **self.kwargs
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return self.dataloader(
            self.val_dataset,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.kwargs
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return self.dataloader(
            self.test_dataset,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.kwargs
        )

    def predict_dataloader(self) -> DataLoader:
        assert self.pred_dataset is not None
        return self.dataloader(
            self.pred_dataset,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.kwargs
        )


class PackedDataset(IterableDataset):
    def __init__(self, dataset, seq_length=2048, shuffle=False, infinite=False):
        self.dataset = dataset
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.infinite = infinite
        self.current_size = 0
        if shuffle:
            self.iterator = RandomSampler(dataset)
        else:
            self.iterator = range(len(dataset))

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        while True:
            cur_example = []
            cur_seqlens = []
            cur_len = 0
            for index in self.iterator:
                cur_seq = self.dataset[index]
                if cur_len + len(cur_seq) <= self.seq_length:
                    cur_example.append(cur_seq)
                    cur_seqlens.append(len(cur_seq))
                    cur_len += len(cur_seq)
                else:
                    cur_example = torch.cat(cur_example)
                    cur_seqlens = torch.tensor(cur_seq)
                    self.current_size += 1
                    yield cur_example
                    cur_example = []
                    cur_len = 0
            if not self.infinite:
                break
