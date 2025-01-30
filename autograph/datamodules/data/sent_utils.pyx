# encoding: utf-8
# cython: linetrace=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# distutils: language=c++

cimport numpy as np
import numpy as np
np.import_array()

from libcpp.algorithm cimport sort

ctypedef np.int64_t int64_t
ctypedef np.uint32_t uint32_t
ctypedef np.uint8_t uint8_t


cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

cdef inline uint32_t our_rand_r(uint32_t* seed) nogil:
    seed[0] ^= <uint32_t>(seed[0] << 13)
    seed[0] ^= <uint32_t>(seed[0] >> 17)
    seed[0] ^= <uint32_t>(seed[0] << 5)

    return seed[0] % (<uint32_t>RAND_R_MAX + 1)

cdef inline uint32_t rand_int(uint32_t end, uint32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end

cdef void sort_array(int64_t[::1] arr, int size) nogil:
    sort(&arr[0], (&arr[0]) + size)


def sample_sent(
    csr_matrix,
    int seq_length=-1,
    int idx_offset=0,
    int reset_idx=-1,
    int left_bracket=-2,
    int right_bracket=-3,
    bint undirected=True,
    object rng=None
):
    if rng is None:
        rng = np.random.RandomState(0)

    cdef:
        int[:] indices = csr_matrix.indices
        int[:] indptr = csr_matrix.indptr
        int num_nodes = csr_matrix.shape[0]
        int num_edges = indices.shape[0]

    if seq_length < 0:
        seq_length = max(num_edges, 10) * 2

    cdef:
        uint32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
        uint32_t* rand_r_state = &rand_r_state_seed

        int num_unvisited = num_nodes
        uint8_t[:] unvisited = np.ones(num_nodes, dtype=np.uint8)

        int64_t[:] sent_seq = np.full(seq_length, reset_idx, dtype=np.int64)
        int64_t[:] node_index_map = np.full(num_nodes, -1, dtype=np.int64)

        int sent_seq_idx = 0
        int64_t node_index = 0

        int current_node, sample_idx, prev_node, i

        int[:] neighbors = np.empty(num_nodes, dtype=np.int32)
        int neighbor, k
        int neighbors_num

        int64_t[::1] neighborhood_set = np.empty(num_nodes, dtype=np.int64)
        int neighborhood_set_size

    with nogil:
        current_node = rand_int(<uint32_t> num_nodes, rand_r_state)
        node_index_map[current_node] = node_index
        sent_seq[sent_seq_idx] = node_index + idx_offset
        node_index += 1
        sent_seq_idx += 1
        unvisited[current_node] = False
        num_unvisited -= 1
        while num_unvisited > 0:
            prev_node = current_node
            # Get Neighbors(current_node)
            neighbors_num = 0
            for k in range(indptr[current_node], indptr[current_node + 1]):
                neighbor = indices[k]
                if unvisited[neighbor]:
                    neighbors[neighbors_num] = neighbor
                    neighbors_num += 1
            if neighbors_num == 0: # start a new trail
                sent_seq_idx += 1
                sample_idx = rand_int(<uint32_t> num_unvisited, rand_r_state)
                current_node = -1
                for i in range(num_nodes):
                    if unvisited[i]:
                        current_node += 1
                    if current_node == sample_idx:
                        break
                current_node = i
            else: # sample the next node in the trail
                sample_idx = rand_int(<uint32_t> neighbors_num, rand_r_state)
                current_node = neighbors[sample_idx]

            node_index_map[current_node] = node_index
            sent_seq[sent_seq_idx] = node_index + idx_offset
            node_index += 1
            sent_seq_idx += 1
            unvisited[current_node] = False
            num_unvisited -= 1

            # add neighborhood information
            neighborhood_set_size = 0
            for k in range(indptr[current_node], indptr[current_node + 1]):
                neighbor = indices[k]
                if not unvisited[neighbor] and neighbor != prev_node:
                    neighborhood_set[neighborhood_set_size] = node_index_map[neighbor]
                    neighborhood_set_size += 1
            if neighborhood_set_size > 0:
                sort_array(neighborhood_set, neighborhood_set_size)
                sent_seq[sent_seq_idx] = left_bracket
                sent_seq_idx += 1
                for i in range(neighborhood_set_size):
                    sent_seq[sent_seq_idx] = neighborhood_set[i] + idx_offset
                    sent_seq_idx += 1
                sent_seq[sent_seq_idx] = right_bracket
                sent_seq_idx += 1

    return np.asarray(sent_seq[:sent_seq_idx]), np.asarray(node_index_map)


def sample_labeled_sent(
    csr_matrix,
    int64_t[:] node_labels,
    int64_t[:] edge_labels,
    int node_idx_offset,
    int edge_idx_offset,
    int seq_length=-1,
    int idx_offset=0,
    int reset_idx=-1,
    int left_bracket=-2,
    int right_bracket=-3,
    bint undirected=True,
    object rng=None
):
    if rng is None:
        rng = np.random.RandomState(0)

    cdef:
        int[:] indices = csr_matrix.indices
        int[:] indptr = csr_matrix.indptr
        int num_nodes = csr_matrix.shape[0]
        int num_edges = indices.shape[0]

    if seq_length < 0:
        seq_length = max(num_edges, 10) * 4

    cdef:
        uint32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
        uint32_t* rand_r_state = &rand_r_state_seed

        int num_unvisited = num_nodes
        uint8_t[:] unvisited = np.ones(num_nodes, dtype=np.uint8)

        int64_t[:] sent_seq = np.full(seq_length, reset_idx, dtype=np.int64)
        int64_t[:] node_index_map = np.full(num_nodes, -1, dtype=np.int64)

        int sent_seq_idx = 0
        int64_t node_index = 0

        int current_node, sample_idx, prev_node, i

        int[:] neighbors = np.empty(num_nodes, dtype=np.int32)
        int neighbor, k
        int neighbors_num

        int64_t[::1] neighborhood_set = np.empty(num_nodes, dtype=np.int64)
        int neighborhood_set_size

        int[:] current_edge_indices = np.zeros(num_nodes, dtype=np.int32)
        int64_t edge_index

    with nogil:
        current_node = rand_int(<uint32_t> num_nodes, rand_r_state)
        node_index_map[current_node] = node_index
        sent_seq[sent_seq_idx] = node_index + idx_offset
        node_index += 1
        sent_seq_idx += 1
        unvisited[current_node] = False
        num_unvisited -= 1
        # node label
        sent_seq[sent_seq_idx] = node_labels[current_node] + node_idx_offset
        sent_seq_idx += 1
        while num_unvisited > 0:
            prev_node = current_node
            # Get Neighbors(current_node)
            neighbors_num = 0
            for k in range(indptr[current_node], indptr[current_node + 1]):
                neighbor = indices[k]
                if unvisited[neighbor]:
                    current_edge_indices[neighbors_num] = k
                    neighbors[neighbors_num] = neighbor
                    neighbors_num += 1
            if neighbors_num == 0: # start a new trail
                sent_seq_idx += 1
                sample_idx = rand_int(<uint32_t> num_unvisited, rand_r_state)
                current_node = -1
                for i in range(num_nodes):
                    if unvisited[i]:
                        current_node += 1
                    if current_node == sample_idx:
                        break
                current_node = i
            else: # sample the next node in the trail
                sample_idx = rand_int(<uint32_t> neighbors_num, rand_r_state)
                current_node = neighbors[sample_idx]

                # edge label
                edge_index = current_edge_indices[sample_idx]
                sent_seq[sent_seq_idx] = edge_labels[edge_index] + edge_idx_offset
                sent_seq_idx += 1

            node_index_map[current_node] = node_index
            sent_seq[sent_seq_idx] = node_index + idx_offset
            node_index += 1
            sent_seq_idx += 1
            unvisited[current_node] = False
            num_unvisited -= 1
            # node label
            sent_seq[sent_seq_idx] = node_labels[current_node] + node_idx_offset
            sent_seq_idx += 1

            # add neighborhood information
            neighborhood_set_size = 0
            for k in range(indptr[current_node], indptr[current_node + 1]):
                neighbor = indices[k]
                if not unvisited[neighbor] and neighbor != prev_node:
                    neighborhood_set[neighborhood_set_size] = node_index_map[neighbor]
                    neighborhood_set_size += 1
                    current_edge_indices[node_index_map[neighbor]] = k
            if neighborhood_set_size > 0:
                sort_array(neighborhood_set, neighborhood_set_size)
                sent_seq[sent_seq_idx] = left_bracket
                sent_seq_idx += 1
                for i in range(neighborhood_set_size):
                    # edge label
                    sent_seq[sent_seq_idx] = edge_labels[current_edge_indices[neighborhood_set[i]]] + edge_idx_offset
                    sent_seq_idx += 1

                    sent_seq[sent_seq_idx] = neighborhood_set[i] + idx_offset
                    sent_seq_idx += 1
                sent_seq[sent_seq_idx] = right_bracket
                sent_seq_idx += 1

    return np.asarray(sent_seq[:sent_seq_idx]), np.asarray(node_index_map)


def reconstruct_graph_from_sent(
    int64_t[:] sent_seq,
    int reset_idx,
    int left_bracket,
    int right_bracket,
):
    cdef:
        int i
        int walk_length = sent_seq.shape[0]

        int64_t[:, ::1] edge_index = np.zeros((2, walk_length * 2), dtype=np.int64)
        int idx = 0

        int64_t bracket_idx = 0
        bint start_bracket = False

    with nogil:
        for i in range(walk_length - 1):
            if sent_seq[i] == reset_idx or sent_seq[i + 1] == reset_idx or sent_seq[i + 1] == left_bracket:
                start_bracket = False
                continue
            if sent_seq[i] == left_bracket:
                start_bracket = True
                bracket_idx = sent_seq[i - 1]
            elif sent_seq[i] == right_bracket and start_bracket:
                edge_index[0, idx] = bracket_idx
                edge_index[1, idx] = sent_seq[i + 1]
                idx += 1
                start_bracket = False
            elif start_bracket:
                edge_index[0, idx] = bracket_idx
                edge_index[1, idx] = sent_seq[i]
                idx += 1
            else:
                edge_index[0, idx] = sent_seq[i]
                edge_index[1, idx] = sent_seq[i + 1]
                idx += 1

    return np.asarray(edge_index[:, :idx])


def reconstruct_graph_from_labeled_sent(
    int64_t[:] sent_seq,
    int reset_idx,
    int left_bracket,
    int right_bracket,
    int idx_offset=0,
):
    cdef:
        int i = 0
        int walk_length = sent_seq.shape[0]

        int64_t[:, ::1] edge_index = np.zeros((2, walk_length * 2), dtype=np.int64)
        int64_t[:] node_labels = np.full(walk_length + idx_offset, -1, dtype=np.int64)
        int64_t[:] edge_labels = np.full(walk_length * 2, -1, dtype=np.int64)
        int idx = 0
        int64_t current_node_idx = 0

        int64_t bracket_idx = 0
        bint start_bracket = False

    with nogil:
        while i < walk_length - 1:
            if sent_seq[i] == reset_idx or sent_seq[i + 1] == reset_idx:
                start_bracket = False
                i += 1
                continue
            if sent_seq[i] == left_bracket:
                start_bracket = True
                bracket_idx = sent_seq[i - 2]
                i += 1
            elif sent_seq[i] == right_bracket and start_bracket:
                edge_index[0, idx] = bracket_idx
                edge_index[1, idx] = sent_seq[i + 2]
                if edge_labels[idx] == -1:
                    edge_labels[idx] = sent_seq[i + 1]
                idx += 1
                start_bracket = False
                i += 2
            elif start_bracket:
                edge_index[0, idx] = bracket_idx
                edge_index[1, idx] = sent_seq[i + 1]
                if edge_labels[idx] == -1:
                    edge_labels[idx] = sent_seq[i]
                idx += 1
                i += 2
            elif sent_seq[i + 2] == reset_idx:
                current_node_idx = sent_seq[i]
                if node_labels[current_node_idx] == -1:
                    node_labels[current_node_idx] = sent_seq[i + 1]
                i += 2
            elif sent_seq[i + 2] == left_bracket:
                current_node_idx = sent_seq[i]
                if node_labels[current_node_idx] == -1:
                    node_labels[current_node_idx] = sent_seq[i + 1]
                i += 2
            else:
                current_node_idx = sent_seq[i]
                if i + 3 <= walk_length - 1:
                    edge_index[0, idx] = current_node_idx
                    edge_index[1, idx] = sent_seq[i + 3]
                    if edge_labels[i] == -1:
                        edge_labels[idx] = sent_seq[i + 2]
                    idx += 1
                if node_labels[current_node_idx] == -1:
                    node_labels[current_node_idx] = sent_seq[i + 1]
                i += 3

    return np.asarray(edge_index[:, :idx]), np.asarray(node_labels), np.asarray(edge_labels[:idx])
