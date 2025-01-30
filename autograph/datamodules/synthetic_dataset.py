import os
import os.path as osp
from pathlib import Path

import torch
import numpy as np
from torch_geometric import utils
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_tar


class SyntheticDataset(InMemoryDataset):
    meta = {
        'ER': {
            'unzip_folder': 'ER_pre',
            'download_names': ['ER_20_pre', 'ER_40_pre', 'ER_60_pre'],
            'filename': 'input_adj.npy',
            'url': 'https://graphgt.mathcs.emory.edu/datasets/Synthetic/ER_pre.zip'
        },
        'BA': {
            'unzip_folder': '',
            'download_names': ['BA_20_pre', 'BA_40_pre', 'BA_60_pre'],
            'filename': 'input_adj.npy',
            'url': 'https://graphgt.mathcs.emory.edu/datasets/Synthetic/BA_pre.zip'
        },
        'waxman': {
            'unzip_folder': '',
            'download_names': ['waxman_pre'],
            'filename': 'adj.npy',
            'url': 'https://graphgt.mathcs.emory.edu/datasets/Synthetic/waxman_pre.zip',
        },
        'random_geo': {
            'unzip_folder': '',
            'download_names': ['random_geo_pre'],
            'filename': 'adj.npy',
            'url': 'https://graphgt.mathcs.emory.edu/datasets/Synthetic/random_geo_pre.zip',
        },
    }
    def __init__(self, root, dataset_name='ER', split='train', transform=None, pre_transform=None, pre_filter=None):
        assert dataset_name in self.meta.keys(), f"Unknown dataset name: {dataset_name}!"
        assert split in ['train', 'val'], f"Unknown split: {split}!"
        self.dataset_name = dataset_name
        self.split = split
        root = f'{root}/synthetic/{dataset_name}'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt']

    @property
    def processed_file_names(self):
        return [self.split + '.pt']

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        raw_url = self.meta[self.dataset_name]['url']
        unzip_folder = self.meta[self.dataset_name]['unzip_folder']
        download_names = self.meta[self.dataset_name]['download_names']
        filename = self.meta[self.dataset_name]['filename']
        path = download_url(raw_url, self.raw_dir)
        try:
            extract_tar(path, self.raw_dir, mode='r')
        except:
            pass

        train_data = []
        val_data = []

        for download_name in download_names:
            adjs = np.load(osp.join(self.raw_dir, unzip_folder, f'{download_name}/{filename}'))

            num_graphs = adjs.shape[0]
            rng = np.random.RandomState(0)

            train_len = num_graphs - 50 # only 50 graphs for validation
            indices = rng.permutation(num_graphs)

            train_indices = indices[:train_len]
            val_indices = indices[train_len:]

            for i in train_indices:
                train_data.append(adjs[i])
            for i in val_indices:
                val_data.append(adjs[i])

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])

    def process(self):
        file_idx = {'train': 0, 'val': 1}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for adj in raw_dataset:
            num_nodes = adj.shape[0]
            edge_index, _ = utils.dense_to_sparse(torch.from_numpy(adj).float())
            data = Data(edge_index=edge_index, num_nodes=num_nodes)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])
