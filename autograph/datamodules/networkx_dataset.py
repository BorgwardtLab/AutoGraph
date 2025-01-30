import os
import os.path as osp
from pathlib import Path

import torch
import numpy as np
import networkx as nx
from torch_geometric import utils
from torch_geometric.data import Data, InMemoryDataset


class NetworkXDataset(InMemoryDataset):
    meta = {
        "classic": [
            {
                'func': 'nx.balanced_tree(2, np.random.randint(2, 9))',
                'num': 10,
            },
            {
                'func': 'nx.barbell_graph(np.random.randint(3, 21), np.random.randint(41))',
                'num': 100,
            },
            {
                'func': 'nx.binomial_tree(np.random.randint(2, 9))',
                'num': 10,
            },
            {
                'func': 'nx.complete_graph(np.random.randint(3, 21))',
                'num': 10,
            },
            {
                'func': 'nx.circular_ladder_graph(np.random.randint(10, 151))',
                'num': 100,
            },
            {
                'func': 'nx.cycle_graph(np.random.randint(10, 301))',
                'num': 100,
            },
            {
                'func': 'nx.dorogovtsev_goltsev_mendes_graph(np.random.randint(2, 6))',
                'num': 5,
            },
            {
                'func': 'nx.ladder_graph(np.random.randint(10, 201))',
                'num': 100,
            },
            {
                'func': 'nx.lollipop_graph(np.random.randint(3, 11), np.random.randint(10, 31))',
                'num': 100,
            },
            {
                'func': 'nx.star_graph(np.random.randint(10, 201))',
                'num': 100,
            },
            {
                'func': 'nx.turan_graph(np.random.randint(10, 41), 2)',
                'num': 100,
            },
            {
                'func': 'nx.wheel_graph(np.random.randint(10, 201))',
                'num': 100,
            }
        ],
        "lattice": [
            {
                'func': 'nx.grid_2d_graph(np.random.randint(5, 21), np.random.randint(5, 21))',
                'num': 200,
            },
            {
                'func': 'nx.triangular_lattice_graph(np.random.randint(5, 21), np.random.randint(5, 21))',
                'num': 200,
            },
        ],
        "small": [{'func': f'nx.{method}()', 'num': 1} for method in nx.generators.small.__all__ if method != 'LCF_graph'],
        "random": [
            {
                'func': 'nx.erdos_renyi_graph(np.random.randint(20, 71), 0.2)',
                'num': 2000,
            },
            {
                'func': 'nx.random_regular_graph(np.random.randint(3, 11), np.random.choice([20, 30, 40, 50, 60, 70, 80, 90, 100]))',
                'num': 2000,
            },
            {
                'func': 'nx.barabasi_albert_graph(np.random.randint(20, 101), np.random.randint(2, 6))',
                'num': 2000,
            },
            {
                'func': 'nx.random_lobster(80, 0.7, 0.7)',
                'num': 2000,
                'min_node': 20,
                'max_node': 200,
            },
        ],
        "geometric": [
            {
                'func': 'nx.random_geometric_graph(np.random.choice([20, 30, 40, 50]), 0.3)',
                'num': 2000,
            },
            {
                'func': 'nx.waxman_graph(np.random.choice([50, 100, 150]))',
                'num': 2000,
            },
        ],
        "tree": [
            {
                'func': 'nx.random_unlabeled_tree(np.random.randint(20, 301))',
                'num': 1000,
            },
        ],
        "community": [
            {
                'func': 'nx.connected_caveman_graph(np.random.randint(10, 81), np.random.randint(2, 5))',
                'num': 200,
            },
            {
                'func': 'nx.windmill_graph(np.random.randint(10, 81), np.random.randint(2, 5))',
                'num': 200,
            },
        ],
        "social": [{'func': f'nx.{method}()', 'num': 1} for method in nx.generators.social.__all__],
    }

    meta_big = {
        "classic": [
            {
                'func': 'nx.balanced_tree(2, np.random.randint(4, 10))',
                'num': 10,
            },
            {
                'func': 'nx.barbell_graph(np.random.randint(3, 31), np.random.randint(41))',
                'num': 100,
            },
            {
                'func': 'nx.binomial_tree(np.random.randint(2, 9))',
                'num': 10,
            },
            {
                'func': 'nx.complete_graph(np.random.randint(3, 31))',
                'num': 10,
            },
            {
                'func': 'nx.circular_ladder_graph(np.random.randint(10, 501))',
                'num': 300,
            },
            {
                'func': 'nx.cycle_graph(np.random.randint(10, 6001))',
                'num': 2000,
            },
            {
                'func': 'nx.dorogovtsev_goltsev_mendes_graph(np.random.randint(2, 7))',
                'num': 5,
            },
            {
                'func': 'nx.ladder_graph(np.random.randint(10, 1001))',
                'num': 500,
            },
            {
                'func': 'nx.lollipop_graph(np.random.randint(3, 21), np.random.randint(10, 51))',
                'num': 200,
            },
            {
                'func': 'nx.star_graph(np.random.randint(10, 501))',
                'num': 200,
            },
            {
                'func': 'nx.turan_graph(np.random.randint(10, 41), 2)',
                'num': 100,
            },
            {
                'func': 'nx.wheel_graph(np.random.randint(10, 201))',
                'num': 100,
            }
        ],
        "lattice": [
            {
                'func': 'nx.grid_2d_graph(np.random.randint(5, 31), np.random.randint(5, 31))',
                'num': 400,
            },
            {
                'func': 'nx.triangular_lattice_graph(np.random.randint(5, 41), np.random.randint(5, 41))',
                'num': 400,
            },
        ],
        "small": [{'func': f'nx.{method}()', 'num': 1} for method in nx.generators.small.__all__ if method != 'LCF_graph'],
        "random": [
            {
                'func': 'nx.erdos_renyi_graph(np.random.randint(20, 101), 0.2)',
                'num': 4000,
            },
            {
                'func': 'nx.random_regular_graph(np.random.randint(3, 11), np.random.choice([20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]))',
                'num': 2000,
            },
            {
                'func': 'nx.barabasi_albert_graph(np.random.randint(20, 501), np.random.randint(2, 6))',
                'num': 4000,
            },
            {
                'func': 'nx.random_lobster(80, 0.7, 0.7)',
                'num': 4000,
                'min_node': 20,
                'max_node': 1000,
            },
        ],
        "geometric": [
            {
                'func': 'nx.random_geometric_graph(np.random.choice([20, 30, 40, 50, 60, 70, 80, 90, 100]), 0.3)',
                'num': 3000,
            },
            {
                'func': 'nx.waxman_graph(np.random.choice([50, 100, 150, 200, 250, 300]))',
                'num': 2000,
            },
        ],
        "tree": [
            {
                'func': 'nx.random_unlabeled_tree(np.random.randint(20, 501))',
                'num': 1000,
            },
        ],
        "community": [
            {
                'func': 'nx.connected_caveman_graph(np.random.randint(10, 101), np.random.randint(2, 5))',
                'num': 300,
            },
            {
                'func': 'nx.windmill_graph(np.random.randint(10, 101), np.random.randint(2, 5))',
                'num': 300,
            },
        ],
        "social": [{'func': f'nx.{method}()', 'num': 1} for method in nx.generators.social.__all__],
    }
    def __init__(self, root, dataset_name='classic', split='train', big=False, transform=None, pre_transform=None, pre_filter=None):
        self.meta_used = self.meta_big if big else self.meta
        assert dataset_name in self.meta_used.keys(), f"Unknown dataset name: {dataset_name}!"
        assert split in ['train', 'val'], f"Unknown split: {split}!"
        self.dataset_name = dataset_name
        self.split = split
        folder_name = 'networkx_big' if big else 'networkx'
        root = f'{root}/{folder_name}/{dataset_name}'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.split + '.pt']

    def process(self):
        data_list = []
        func_list = self.meta_used[self.dataset_name]
        ratio = 1 if self.split == 'train' else 0.1
        np.random.seed(0)
        for func in func_list:
            for _ in range(max(int(func['num'] * ratio), 1)):
                min_node = func.get('min_node', 0)
                max_node = func.get('max_node', float('inf'))
                to_check = (min_node > 0) or (max_node < float('inf'))
                data = eval(func['func'])
                while to_check:
                    if len(data.nodes()) >= min_node and len(data.nodes()) <= max_node:
                        break
                    data = eval(func['func'])
                data = utils.from_networkx(data)
                data = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == '__main__':
    dataset = NetworkXDataset('../../datasets', dataset_name='random')
    print(dataset)
    graph = dataset[0]
    print(graph)
    print(max([g.edge_index.shape[1] for g in dataset]))
