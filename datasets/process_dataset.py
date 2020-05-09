import os
import random
import time
import math
import pickle
from functools import partial
from multiprocessing import Pool
import bisect
import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from utils import mkdir
from datasets.preprocess import (
    mapping, graphs_to_min_dfscodes,
    min_dfscodes_to_tensors, random_walk_with_restart_sampling
)


def check_graph_size(
    graph, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):

    if min_num_nodes and graph.number_of_nodes() < min_num_nodes:
        return False
    if max_num_nodes and graph.number_of_nodes() > max_num_nodes:
        return False

    if min_num_edges and graph.number_of_edges() < min_num_edges:
        return False
    if max_num_edges and graph.number_of_edges() > max_num_edges:
        return False

    return True


def produce_graphs_from_raw_format(
    inputfile, output_path, num_graphs=None, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    """
    :param inputfile: Path to file containing graphs in raw format
    :param output_path: Path to store networkx graphs
    :param num_graphs: Upper bound on number of graphs to be taken
    :param min_num_nodes: Lower bound on number of nodes in graphs if provided
    :param max_num_nodes: Upper bound on number of nodes in graphs if provided
    :param min_num_edges: Lower bound on number of edges in graphs if provided
    :param max_num_edges: Upper bound on number of edges in graphs if provided
    :return: number of graphs produced
    """

    lines = []
    with open(inputfile, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            lines.append(line)

    index = 0
    count = 0
    graphs_ids = set()
    while index < len(lines):
        if lines[index][0][1:] not in graphs_ids:
            graph_id = lines[index][0][1:]
            G = nx.Graph(id=graph_id)

            index += 1
            vert = int(lines[index][0])
            index += 1
            for i in range(vert):
                G.add_node(i, label=lines[index][0])
                index += 1

            edges = int(lines[index][0])
            index += 1
            for i in range(edges):
                G.add_edge(int(lines[index][0]), int(
                    lines[index][1]), label=lines[index][2])
                index += 1

            index += 1

            if not check_graph_size(
                G, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
            ):
                continue

            if nx.is_connected(G):
                with open(os.path.join(
                        output_path, 'graph{}.dat'.format(count)), 'wb') as f:
                    pickle.dump(G, f)

                graphs_ids.add(graph_id)
                count += 1

                if num_graphs and count >= num_graphs:
                    break

        else:
            vert = int(lines[index + 1][0])
            edges = int(lines[index + 2 + vert][0])
            index += vert + edges + 4

    return count


# For Enzymes dataset
def produce_graphs_from_graphrnn_format(
    input_path, dataset_name, output_path, num_graphs=None,
    node_invariants=[], min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):
    node_attributes = False
    graph_labels = False

    G = nx.Graph()
    # load data
    path = input_path
    data_adj = np.loadtxt(os.path.join(path, dataset_name + '_A.txt'),
                          delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(
            os.path.join(path, dataset_name + '_node_attributes.txt'),
            delimiter=',')

    data_node_label = np.loadtxt(
        os.path.join(path, dataset_name + '_node_labels.txt'),
        delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(
        os.path.join(path, dataset_name + '_graph_indicator.txt'),
        delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(
            os.path.join(path, dataset_name + '_graph_labels.txt'),
            delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)

    # add node labels
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i + 1, feature=data_node_att[i])
        G.add_node(i + 1, label=str(data_node_label[i]))

    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1

    count = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['id'] = data_graph_labels[i]

        if not check_graph_size(
            G_sub, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
        ):
            continue

        if nx.is_connected(G_sub):
            G_sub = nx.convert_node_labels_to_integers(G_sub)
            G_sub.remove_edges_from(nx.selfloop_edges(G_sub))

            if 'CC' in node_invariants:
                clustering_coeff = nx.clustering(G_sub)
                cc_bins = [0, 0.2, 0.4, 0.6, 0.8]

            for node in G_sub.nodes():
                node_label = str(G_sub.nodes[node]['label'])

                if 'Degree' in node_invariants:
                    node_label += '-' + str(G_sub.degree[node])

                if 'CC' in node_invariants:
                    node_label += '-' + str(
                        bisect.bisect(cc_bins, clustering_coeff[node]))

                G_sub.nodes[node]['label'] = node_label

            nx.set_edge_attributes(G_sub, 'DEFAULT_LABEL', 'label')

            with open(os.path.join(
                    output_path, 'graph{}.dat'.format(count)), 'wb') as f:
                pickle.dump(G_sub, f)

            count += 1

            if num_graphs and count >= num_graphs:
                break

    return count


def sample_subgraphs(
    idx, G, output_path, iterations, num_factor, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    count = 0
    deg = G.degree[idx]
    for _ in range(num_factor * int(math.sqrt(deg))):
        G_rw = random_walk_with_restart_sampling(
            G, idx, iterations=iterations, max_nodes=max_num_nodes,
            max_edges=max_num_edges)
        G_rw = nx.convert_node_labels_to_integers(G_rw)
        G_rw.remove_edges_from(nx.selfloop_edges(G_rw))

        if not check_graph_size(
            G_rw, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
        ):
            continue

        if nx.is_connected(G_rw):
            with open(os.path.join(
                    output_path,
                    'graph{}-{}.dat'.format(idx, count)), 'wb') as f:
                pickle.dump(G_rw, f)
                count += 1


def produce_random_walk_sampled_graphs(
    input_path, dataset_name, output_path, iterations, num_factor,
    num_graphs=None, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):
    print('Producing random_walk graphs - num_factor - {}'.format(num_factor))
    G = nx.Graph()

    d = {}
    count = 0
    with open(os.path.join(input_path, dataset_name + '.content'), 'r') as f:
        for line in f.readlines():
            spp = line.strip().split('\t')
            G.add_node(count, label=spp[-1])
            d[spp[0]] = count
            count += 1

    count = 0
    with open(os.path.join(input_path, dataset_name + '.cites'), 'r') as f:
        for line in f.readlines():
            spp = line.strip().split('\t')
            if spp[0] in d and spp[1] in d:
                G.add_edge(d[spp[0]], d[spp[1]], label='DEFAULT_LABEL')
            else:
                count += 1

    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G)

    with Pool(processes=48) as pool:
        for _ in tqdm(pool.imap_unordered(partial(
                sample_subgraphs, G=G, output_path=output_path,
                iterations=iterations, num_factor=num_factor,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges),
                list(range(G.number_of_nodes())))):
            pass

    filenames = []
    for name in os.listdir(output_path):
        if name.endswith('.dat'):
            filenames.append(name)

    random.shuffle(filenames)

    if not num_graphs:
        num_graphs = len(filenames)

    count = 0
    for i, name in enumerate(filenames[:num_graphs]):
        os.rename(
            os.path.join(output_path, name),
            os.path.join(output_path, 'graph{}.dat'.format(i))
        )
        count += 1

    for name in filenames[num_graphs:]:
        os.remove(os.path.join(output_path, name))

    return count


# Routine to create datasets
def create_graphs(args):
    # Different datasets
    if 'Lung' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Lung/')
        input_path = base_path + 'lung.txt'
        min_num_nodes, max_num_nodes = None, 50
        min_num_edges, max_num_edges = None, None

    elif 'Breast' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Breast/')
        input_path = base_path + 'breast.txt'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'Leukemia' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Leukemia/')
        input_path = base_path + 'leukemia.txt'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'Yeast' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Yeast/')
        input_path = base_path + 'yeast.txt'
        min_num_nodes, max_num_nodes = None, 50
        min_num_edges, max_num_edges = None, None

    elif 'All' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'All/')
        input_path = base_path + 'all.txt'
        # No limit on number of nodes and edges
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'ENZYMES' in args.graph_type:
        base_path = os.path.join(args.dataset_path, 'ENZYMES/')
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = ['Degree']
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'citeseer' in args.graph_type:
        base_path = os.path.join(args.dataset_path, 'citeseer/')
        random_walk_iterations = 150  # Controls size of graph
        num_factor = 5  # Controls size of dataset

        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = 20, None

    elif 'cora' in args.graph_type:
        base_path = os.path.join(args.dataset_path, 'cora/')
        random_walk_iterations = 150  # Controls size of graph
        num_factor = 5  # Controls size of dataset

        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = 20, None

    else:
        print('Dataset - {} is not valid'.format(args.graph_type))
        exit()

    args.current_dataset_path = os.path.join(base_path, 'graphs/')
    args.min_dfscode_path = os.path.join(base_path, 'min_dfscodes/')
    min_dfscode_tensor_path = os.path.join(base_path, 'min_dfscode_tensors/')

    if args.note == 'GraphRNN' or args.note == 'DGMG':
        args.current_processed_dataset_path = args.current_dataset_path
    elif args.note == 'DFScodeRNN':
        args.current_processed_dataset_path = min_dfscode_tensor_path

    if args.produce_graphs:
        mkdir(args.current_dataset_path)

        if args.graph_type in ['Lung', 'Breast', 'Leukemia', 'Yeast', 'All']:
            count = produce_graphs_from_raw_format(
                input_path, args.current_dataset_path, args.num_graphs,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        elif args.graph_type in ['ENZYMES']:
            count = produce_graphs_from_graphrnn_format(
                base_path, args.graph_type, args.current_dataset_path,
                num_graphs=args.num_graphs, node_invariants=node_invariants,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        elif args.graph_type in ['cora', 'citeseer']:
            count = produce_random_walk_sampled_graphs(
                base_path, args.graph_type, args.current_dataset_path,
                num_graphs=args.num_graphs, iterations=random_walk_iterations,
                num_factor=num_factor, min_num_nodes=min_num_nodes,
                max_num_nodes=max_num_nodes, min_num_edges=min_num_edges,
                max_num_edges=max_num_edges)

        print('Graphs produced', count)
    else:
        count = len([name for name in os.listdir(
            args.current_dataset_path) if name.endswith(".dat")])
        print('Graphs counted', count)

    # Produce feature map
    feature_map = mapping(args.current_dataset_path,
                          args.current_dataset_path + 'map.dict')
    print(feature_map)

    if args.note == 'DFScodeRNN' and args.produce_min_dfscodes:
        # Empty the directory
        mkdir(args.min_dfscode_path)

        start = time.time()
        graphs_to_min_dfscodes(args.current_dataset_path,
                               args.min_dfscode_path, args.current_temp_path)

        end = time.time()
        print('Time taken to make dfscodes = {:.3f}s'.format(end - start))

    if args.note == 'DFScodeRNN' and args.produce_min_dfscode_tensors:
        # Empty the directory
        mkdir(min_dfscode_tensor_path)

        start = time.time()
        min_dfscodes_to_tensors(args.min_dfscode_path,
                                min_dfscode_tensor_path, feature_map)

        end = time.time()
        print('Time taken to make dfscode tensors= {:.3f}s'.format(
            end - start))

    graphs = [i for i in range(count)]
    return graphs
