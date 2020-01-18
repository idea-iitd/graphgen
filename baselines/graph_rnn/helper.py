import networkx as nx
import torch

from datasets.preprocess import get_random_bfs_seq

MAX_WORKERS = 48


def get_attributes_len_for_graph_rnn(
    len_node_map, len_edge_map, max_prev_node=None, max_head_and_tail=None
):
    """
    Returns (len_node_vec, len_edge_vec, feature_len)
    len_node_vec : Length of vector to represent a node attribute
    len_edge_vec : Length of vector to represent an edge attribute
    num_nodes_to_consider: Number of previous nodes to consider for edges for a given node
    """

    # Last two bits for START node and END node token
    len_node_vec = len_node_map + 2
    # Last three bits in order are NO edge, START egde, END edge token
    len_edge_vec = len_edge_map + 3

    if max_prev_node is not None:
        num_nodes_to_consider = max_prev_node
    elif max_head_and_tail is not None:
        num_nodes_to_consider = max_head_and_tail[0] + max_head_and_tail[1]

    return len_node_vec, len_edge_vec, num_nodes_to_consider


def graph_to_matrix(
    graph, node_map, edge_map, max_prev_node=None, max_head_and_tail=None, random_bfs=False
):
    """
    Method for converting graph to a 2d feature matrix
    :param graph: Networkx graph object
    :param node_map: Node label to integer mapping
    :param edge_map: Edge label to integer mapping
    :param max_prev_node: Number of previous nodes to consider for edge prediction
    :param max_head_and_tail: Head and tail of adjacency vector to consider for edge prediction
    :random_bfs: Whether or not to do random_bfs
    """
    n = len(graph.nodes())
    len_node_vec, _, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
        len(node_map), len(edge_map), max_prev_node, max_head_and_tail)

    if random_bfs:
        bfs_seq = get_random_bfs_seq(graph)
        bfs_order_map = {bfs_seq[i]: i for i in range(n)}
        graph = nx.relabel_nodes(graph, bfs_order_map)

    # 3D adjacecny matrix in case of edge_features (each A[i, j] is a len_edge_vec size vector)
    adj_mat_2d = torch.ones((n, num_nodes_to_consider))
    adj_mat_2d.tril_(diagonal=-1)
    adj_mat_3d = torch.zeros((n, num_nodes_to_consider, len(edge_map)))

    node_mat = torch.zeros((n, len_node_vec))

    for v, data in graph.nodes.data():
        ind = node_map[data['label']]
        node_mat[v, ind] = 1

    for u, v, data in graph.edges.data():
        if max_prev_node is not None:
            if abs(u - v) <= max_prev_node:
                adj_mat_3d[max(u, v), max(u, v) - min(u, v) -
                           1, edge_map[data['label']]] = 1
                adj_mat_2d[max(u, v), max(u, v) - min(u, v) - 1] = 0

        elif max_head_and_tail is not None:
            if abs(u - v) <= max_head_and_tail[1]:
                adj_mat_3d[max(u, v), max(u, v) - min(u, v) -
                           1, edge_map[data['label']]] = 1
                adj_mat_2d[max(u, v), max(u, v) - min(u, v) - 1] = 0
            elif min(u, v) < max_head_and_tail[0]:
                adj_mat_3d[max(u, v), max_head_and_tail[1] +
                           min(u, v), edge_map[data['label']]] = 1
                adj_mat_2d[max(u, v), max_head_and_tail[1] + min(u, v)] = 0

    adj_mat = torch.cat((adj_mat_3d, adj_mat_2d.reshape(adj_mat_2d.size(
        0), adj_mat_2d.size(1), 1), torch.zeros((n, num_nodes_to_consider, 2))), dim=2)
    adj_mat = adj_mat.reshape((adj_mat.size(0), -1))

    return torch.cat((node_mat, adj_mat), dim=1)
