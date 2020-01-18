import pickle
import os
from multiprocessing import Pool
from functools import partial
import networkx as nx
import torch
from tqdm.auto import tqdm

from dfscode.dfs_wrapper import get_min_dfscode

MAX_WORKERS = 48


def mapping(path, dest):
    """
    :param path: path to folder which contains pickled networkx graphs
    :param dest: place where final dictionary pickle file is stored
    :return: dictionary of 4 dictionary which contains forward 
    and backwards mappings of vertices and labels, max_nodes and max_edges
    """

    node_forward, node_backward = {}, {}
    edge_forward, edge_backward = {}, {}
    node_count, edge_count = 0, 0
    max_nodes, max_edges, max_degree = 0, 0, 0
    min_nodes, min_edges = float('inf'), float('inf')

    for filename in tqdm(os.listdir(path)):
        if filename.endswith(".dat"):
            f = open(path + filename, 'rb')
            G = pickle.load(f)
            f.close()

            max_nodes = max(max_nodes, len(G.nodes()))
            min_nodes = min(min_nodes, len(G.nodes()))
            for _, data in G.nodes.data():
                if data['label'] not in node_forward:
                    node_forward[data['label']] = node_count
                    node_backward[node_count] = data['label']
                    node_count += 1

            max_edges = max(max_edges, len(G.edges()))
            min_edges = min(min_edges, len(G.edges()))
            for _, _, data in G.edges.data():
                if data['label'] not in edge_forward:
                    edge_forward[data['label']] = edge_count
                    edge_backward[edge_count] = data['label']
                    edge_count += 1

            max_degree = max(max_degree, max([d for n, d in G.degree()]))

    feature_map = {
        'node_forward': node_forward,
        'node_backward': node_backward,
        'edge_forward': edge_forward,
        'edge_backward': edge_backward,
        'max_nodes': max_nodes,
        'min_nodes': min_nodes,
        'max_edges': max_edges,
        'min_edges': min_edges,
        'max_degree': max_degree
    }

    f = open(dest, 'wb')
    pickle.dump(feature_map, f)
    f.close()

    print('Successfully done node count', node_count)
    print('Successfully done edge count', edge_count)

    return feature_map


def get_bfs_seq(G, start_id):
    """
    Get a bfs node sequence
    :param G: graph
    :param start_id: starting node
    :return: List of bfs node sequence
    """
    successors_dict = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        succ = []
        for current in start:
            if current in successors_dict:
                succ = succ + successors_dict[current]

        output = output + succ
        start = succ
    return output


def get_random_bfs_seq(graph):
    n = len(graph.nodes())
    # Create a random permutaion of graph nodes
    perm = torch.randperm(n)
    adj = nx.to_numpy_matrix(graph, nodelist=perm.numpy(), dtype=int)
    G = nx.from_numpy_matrix(adj)

    # Construct bfs ordering starting from a random node
    start_id = torch.randint(0, n, ()).item()
    bfs_seq = get_bfs_seq(G, start_id)

    return [perm[bfs_seq[i]] for i in range(n)]


def graph_to_min_dfscode(graph_file, graphs_path, min_dfscodes_path, temp_path):
    with open(graphs_path + graph_file, 'rb') as f:
        G = pickle.load(f)
        min_dfscode = get_min_dfscode(G, temp_path)

        if len(G.edges()) == len(min_dfscode):
            with open(min_dfscodes_path + graph_file, 'wb') as f:
                pickle.dump(min_dfscode, f)
        else:
            print('Error in min dfscode for filename', graph_file)
            exit()


def graphs_to_min_dfscodes(graphs_path, min_dfscodes_path, temp_path):
    """
    :param graphs_path: Path to directory of graphs in networkx format
    :param min_dfscodes_path: Path to directory to store the min dfscodes
    :param temp_path: path for temporary files
    :return: length of dataset
    """
    graphs = []
    for filename in os.listdir(graphs_path):
        if filename.endswith(".dat"):
            graphs.append(filename)

    with Pool(processes=MAX_WORKERS) as pool:
        for i, _ in tqdm(enumerate(pool.imap_unordered(
            partial(graph_to_min_dfscode, graphs_path=graphs_path, min_dfscodes_path=min_dfscodes_path,
                    temp_path=temp_path), graphs, chunksize=16), 1)):
            pass
            # if i % 50000 == 0:
            #     print('Processed', i, 'graphs')

    print('Done creating min dfscodes')


def dfscode_to_tensor(dfscode, feature_map):
    max_nodes, max_edges = feature_map['max_nodes'], feature_map['max_edges']
    node_forward_dict, edge_forward_dict = feature_map['node_forward'], feature_map['edge_forward']
    num_nodes_feat, num_edges_feat = len(
        feature_map['node_forward']), len(feature_map['edge_forward'])

    # max_nodes, num_nodes_feat and num_edges_feat are end token labels
    # So ignore tokens are one higher
    dfscode_tensors = {
        't1': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        't2': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'v1': (num_nodes_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'e': (num_edges_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'v2': (num_nodes_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'len': len(dfscode)
    }

    for i, code in enumerate(dfscode):
        dfscode_tensors['t1'][i] = int(code[0])
        dfscode_tensors['t2'][i] = int(code[1])
        dfscode_tensors['v1'][i] = int(node_forward_dict[code[2]])
        dfscode_tensors['e'][i] = int(edge_forward_dict[code[3]])
        dfscode_tensors['v2'][i] = int(node_forward_dict[code[4]])

    # Add end token
    dfscode_tensors['t1'][len(dfscode)], dfscode_tensors['t2'][len(
        dfscode)] = max_nodes, max_nodes
    dfscode_tensors['v1'][len(dfscode)], dfscode_tensors['v2'][len(
        dfscode)] = num_nodes_feat, num_nodes_feat
    dfscode_tensors['e'][len(dfscode)] = num_edges_feat

    return dfscode_tensors


def dfscode_from_file_to_tensor_to_file(
    min_dfscode_file, min_dfscodes_path, min_dfscode_tensors_path, feature_map
):
    with open(min_dfscodes_path + min_dfscode_file, 'rb') as f:
        min_dfscode = pickle.load(f)

    dfscode_tensors = dfscode_to_tensor(min_dfscode, feature_map)

    with open(min_dfscode_tensors_path + min_dfscode_file, 'wb') as f:
        pickle.dump(dfscode_tensors, f)


def min_dfscodes_to_tensors(min_dfscodes_path, min_dfscode_tensors_path, feature_map):
    """
    :param min_dfscodes_path: Path to directory of pickled min dfscodes
    :param min_dfscode_tensors_path: Path to directory to store the min dfscode tensors
    :param feature_map:
    :return: length of dataset
    """
    min_dfscodes = []
    for filename in os.listdir(min_dfscodes_path):
        if filename.endswith(".dat"):
            min_dfscodes.append(filename)

    with Pool(processes=MAX_WORKERS) as pool:
        for i, _ in tqdm(enumerate(pool.imap_unordered(
                partial(dfscode_from_file_to_tensor_to_file, min_dfscodes_path=min_dfscodes_path,
                        min_dfscode_tensors_path=min_dfscode_tensors_path, feature_map=feature_map),
                min_dfscodes, chunksize=16), 1)):
            pass

            # if i % 10000 == 0:
            #     print('Processed', i, 'graphs')


def calc_max_prev_node_helper(idx, graphs_path):
    with open(graphs_path + 'graph' + str(idx) + '.dat', 'rb') as f:
        G = pickle.load(f)

    max_prev_node = []
    for _ in range(100):
        bfs_seq = get_random_bfs_seq(G)
        bfs_order_map = {bfs_seq[i]: i for i in range(len(G.nodes()))}
        G = nx.relabel_nodes(G, bfs_order_map)

        max_prev_node_iter = 0
        for u, v in G.edges():
            max_prev_node_iter = max(max_prev_node_iter, max(u, v) - min(u, v))

        max_prev_node.append(max_prev_node_iter)

    return max_prev_node


def calc_max_prev_node(graphs_path):
    """
    Approximate max_prev_node from simulating bfs sequences 
    """
    max_prev_node = []
    count = len([name for name in os.listdir(
        graphs_path) if name.endswith(".dat")])

    max_prev_node = []
    with Pool(processes=MAX_WORKERS) as pool:
        for max_prev_node_g in tqdm(pool.imap_unordered(
                partial(calc_max_prev_node_helper, graphs_path=graphs_path), list(range(count)))):
            max_prev_node.extend(max_prev_node_g)

    max_prev_node = sorted(max_prev_node)[-1 * int(0.001 * len(max_prev_node))]
    return max_prev_node


def dfscodes_weights(dataset_path, graph_list, feature_map, device):
    freq = {
        't1_freq': torch.ones(feature_map['max_nodes'] + 1, device=device),
        't2_freq': torch.ones(feature_map['max_nodes'] + 1, device=device),
        'v1_freq': torch.ones(len(feature_map['node_forward']) + 1, device=device),
        'e_freq': torch.ones(len(feature_map['edge_forward']) + 1, device=device),
        'v2_freq': torch.ones(len(feature_map['node_forward']) + 1, device=device)
    }

    for idx in graph_list:
        with open(dataset_path + 'graph' + str(idx) + '.dat', 'rb') as f:
            min_dfscode = pickle.load(f)
            for code in min_dfscode:
                freq['t1_freq'][int(code[0])] += 1
                freq['t2_freq'][int(code[1])] += 1
                freq['v1_freq'][feature_map['node_forward'][code[2]]] += 1
                freq['e_freq'][feature_map['edge_forward'][code[3]]] += 1
                freq['v2_freq'][feature_map['node_forward'][code[4]]] += 1

    freq['t1_freq'][-1] = len(graph_list)
    freq['t2_freq'][-1] = len(graph_list)
    freq['v1_freq'][-1] = len(graph_list)
    freq['e_freq'][-1] = len(graph_list)
    freq['v2_freq'][-1] = len(graph_list)

    print('Weights computed')

    return {
        't1_weight': torch.pow(torch.torch.max(freq['t1_freq']), 0.3) / torch.pow(freq['t1_freq'], 0.3),
        't2_weight': torch.pow(torch.max(freq['t2_freq']), 0.3) / torch.pow(freq['t2_freq'], 0.3),
        'v1_weight': torch.pow(torch.max(freq['v1_freq']), 0.3) / torch.pow(freq['v1_freq'], 0.3),
        'e_weight': torch.pow(torch.max(freq['e_freq']), 0.3) / torch.pow(freq['e_freq'], 0.3),
        'v2_weight': torch.pow(torch.max(freq['v2_freq']), 0.3) / torch.pow(freq['v2_freq'], 0.3)
    }


def random_walk_with_restart_sampling(
    G, start_node, iterations, fly_back_prob=0.15,
    max_nodes=None, max_edges=None
):
    sampled_graph = nx.Graph()
    sampled_graph.add_node(start_node, label=G.nodes[start_node]['label'])

    curr_node = start_node

    for _ in range(iterations):
        choice = torch.rand(()).item()

        if choice < fly_back_prob:
            curr_node = start_node
        else:
            neigh = list(G.neighbors(curr_node))
            chosen_node_id = torch.randint(
                0, len(neigh), ()).item()
            chosen_node = neigh[chosen_node_id]

            sampled_graph.add_node(
                chosen_node, label=G.nodes[chosen_node]['label'])
            sampled_graph.add_edge(
                curr_node, chosen_node, label=G.edges[curr_node, chosen_node]['label'])

            curr_node = chosen_node

        if max_nodes is not None and sampled_graph.number_of_nodes() >= max_nodes:
            break

        if max_edges is not None and sampled_graph.number_of_edges() >= max_edges:
            break

    # sampled_graph = G.subgraph(sampled_node_set)

    return sampled_graph
