import torch
import numpy as np
from torch_geometric import utils
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from .sent_utils import (
    sample_sent,
    sample_labeled_sent,
    reconstruct_graph_from_sent,
    reconstruct_graph_from_labeled_sent
)


def sample_sent_from_graph(
    edge_index,
    num_nodes=None,
    max_length=-1,
    idx_offset=0,
    reset=-1,
    ladj=-2,
    radj=-3,
    undirected=True,
    rng=None,
):
    if rng is None:
        rng = np.random.mtrand._rand
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)
    csr_matrix = utils.to_scipy_sparse_matrix(
        edge_index, num_nodes=num_nodes
    ).astype(np.int32).tocsr()
    return sample_sent(
        csr_matrix, max_length, idx_offset, reset, ladj, radj, undirected, rng
    )


def get_graph_from_sent(walk_index, idx_offset, reset, ladj, radj, undirected=True):
    device = walk_index.device
    walk_index = walk_index.cpu().numpy()
    edge_index = reconstruct_graph_from_sent(walk_index, reset, ladj, radj)
    edge_index = torch.from_numpy(edge_index)
    if undirected:
        edge_index_sym = torch.cat([edge_index[[1]], edge_index[[0]]])
        edge_index = torch.cat([edge_index, edge_index_sym], dim=1)
    edge_index,_ = utils.remove_self_loops(edge_index)
    edge_index, _, _ = utils.remove_isolated_nodes(edge_index)
    edge_index = utils.coalesce(edge_index)
    return edge_index


def sample_labeled_sent_from_graph(
    edge_index,
    node_labels,
    edge_labels,
    node_idx_offset=0,
    edge_idx_offset=0,
    num_nodes=None,
    max_length=-1,
    idx_offset=0,
    reset=-1,
    ladj=-2,
    radj=-3,
    undirected=True,
    rng=None,
):
    if rng is None:
        rng = np.random.mtrand._rand
    if isinstance(rng, int):
        rng = np.random.RandomState(rng)
    csr_matrix = utils.to_scipy_sparse_matrix(
        edge_index, num_nodes=num_nodes
    ).astype(np.int32).tocsr()
    if isinstance(node_labels, torch.Tensor):
        node_labels = node_labels.numpy()
    if isinstance(edge_labels, torch.Tensor):
        edge_labels = edge_labels.numpy()
    return sample_labeled_sent(
        csr_matrix, node_labels, edge_labels, node_idx_offset, edge_idx_offset,
        max_length, idx_offset, reset, ladj, radj, undirected, rng
    )

def get_graph_from_labeled_sent(
    walk_index, idx_offset, node_idx_offset, edge_idx_offset,
    num_node_types, num_edge_types,
    reset, ladj, radj, undirected=True
):
    device = walk_index.device
    walk_index = walk_index.cpu().numpy()
    edge_index, node_labels, edge_labels = reconstruct_graph_from_labeled_sent(
        walk_index, reset, ladj, radj, idx_offset,
    )
    max_node_idx = edge_index.flatten().max() + 1
    node_labels = node_labels[idx_offset:max_node_idx]
    edge_index = torch.from_numpy(edge_index)
    node_labels = torch.from_numpy(node_labels)
    edge_labels = torch.from_numpy(edge_labels)
    edge_index -= idx_offset
    node_labels -= node_idx_offset
    edge_labels -= edge_idx_offset
    node_labels[(node_labels < 0) | (node_labels >= num_node_types)] = 0
    edge_labels[(edge_labels < 0) | (edge_labels >= num_edge_types)] = 0
    if undirected:
        edge_index_sym = torch.cat([edge_index[[1]], edge_index[[0]]])
        edge_index = torch.cat([edge_index, edge_index_sym], dim=1)
        edge_labels = torch.cat([edge_labels, edge_labels])
    edge_index, edge_labels = utils.remove_self_loops(edge_index, edge_labels)
    edge_index, edge_labels = utils.coalesce(edge_index, edge_labels, reduce='min')
    return edge_index, node_labels, edge_labels


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    from torch_geometric import datasets
    dataset = datasets.ZINC('../../../datasets/ZINC', subset=True, split='train')
    graph = dataset[100]
    graph = graph.coalesce()

    print(graph.edge_index)
    print(graph.x.flatten())
    print(graph.edge_attr)

    walk_index, walk_node_index, node_index_map = sample_labeled_sent_from_graph(
        graph.edge_index,
        graph.x.flatten().numpy(),
        graph.edge_attr.flatten().numpy(),
        node_idx_offset=100,
        edge_idx_offset=200,
        num_nodes=graph.num_nodes
    )
    print(walk_index)
    print(walk_node_index)
    print(node_index_map)

    node_index_map = torch.from_numpy(node_index_map)
    edge_index, edge_attr = utils.coalesce(node_index_map[graph.edge_index], graph.edge_attr)
    print(edge_attr)

    walk_index = torch.from_numpy(walk_index)

    edge_index_recon, node_labels, edge_labels = get_graph_from_labeled_sent(
        walk_index, idx_offset=0, node_idx_offset=100, edge_idx_offset=200,
        reset=-1,
        ladj=-2,
        radj=-3
    )

    print((edge_index_recon == edge_index).all())
    print((node_labels[node_index_map] == graph.x.flatten()).all())
    print((edge_labels == edge_attr).all())
