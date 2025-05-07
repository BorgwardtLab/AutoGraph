import os
import os.path as osp
from pathlib import Path

import torch
from functools import partial
import numpy as np
from torch_geometric import utils
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import TUDataset


class ProteinDataset(InMemoryDataset):
    meta = {
        'DD': partial(TUDataset, name='DD', use_node_attr=False, use_edge_attr=False)
    }
    def __init__(self, root, dataset_name='DD', split='train', transform=None, pre_transform=None, pre_filter=None):
        assert dataset_name in self.meta.keys(), f"Unknown dataset name: {dataset_name}!"
        self.dataset_name = dataset_name
        self.split = split
        root = f'{root}/protein/{dataset_name}'
        min_size = 100
        max_size = 500
        pre_filter = lambda data: (data.num_nodes >= min_size) and (data.num_nodes <= max_size)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return [self.split + '.pt']

    def download(self):
        base_dataset = self.meta[self.dataset_name](
            self.root,
            pre_filter=self.pre_filter
        )
        num_graphs = len(base_dataset)

        test_len = int(round(num_graphs * 0.2))
        train_len = int(round((num_graphs - test_len) * 0.8))
        val_len = num_graphs - train_len - test_len

        train, val, test = torch.utils.data.random_split(
            base_dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(1234),
        )

        torch.save(train.indices, self.raw_paths[0])
        torch.save(val.indices, self.raw_paths[1])
        torch.save(test.indices, self.raw_paths[2])

    def process(self):
        base_dataset = self.meta[self.dataset_name](
            self.root,
            pre_filter=self.pre_filter
        )
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        indices = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for index, i in enumerate(indices):
            data = base_dataset[i]
            data = Data(edge_index=utils.remove_self_loops(data.edge_index)[0], num_nodes=data.num_nodes, index=index)
            data = data.coalesce()

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    dataset = ProteinDataset('../../datasets', split='train')
    print(dataset)
    graph = dataset[0]
    print(graph)
    print(([g.edge_index.shape[1] for g in dataset]))
    print(sum([g.num_nodes for g in dataset]) / len(dataset))
