import os
import time
import shutil
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


def produce_graphs_from_raw_format(inputfile, output_path, include_file=None, produce_complementary=False):
    """
    :param inputfile: Path to file containing graphs in raw format
    :param output_path: Path to store networkx graphs
    :param include_file: Path to file containing list of graph ids to include
    :param produce_complementary: Whether to create graphs not in include_file
    :return: number of graphs produced
    """

    include = []
    if include_file is not None:
        with open(include_file, 'r') as f:
            for line in f:
                include.append(line.strip()[1:])
        f.close()

    include = frozenset(include)

    lines = []
    with open(inputfile, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            lines.append(line)

    index, l = 0, len(lines)
    count = 0
    graphs_ids = set()
    while index < l:
        if ((lines[index][0][1:] in include) ^ produce_complementary) and lines[index][0][1:] not in graphs_ids:
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

            if nx.is_connected(G):
                with open(output_path + 'graph' + str(count) + '.dat', 'wb') as f:
                    pickle.dump(G, f)

                graphs_ids.add(graph_id)
                count += 1

            index += 1
        else:
            vert = int(lines[index + 1][0])
            edges = int(lines[index + 2 + vert][0])
            index += vert + edges + 4

    return count


def produce_graphs_from_graphrnn_format(
    input_path, dataset_name, output_path, min_num_nodes=20, max_num_nodes=1000
):
    node_attributes = False
    graph_labels = False

    G = nx.Graph()
    # load data
    path = input_path
    data_adj = np.loadtxt(path + dataset_name + '_A.txt',
                          delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(
            path + dataset_name + '_node_attributes.txt', delimiter=',')

    data_node_label = np.loadtxt(
        path + dataset_name + '_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(
        path + dataset_name + '_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(
            path + dataset_name + '_graph_labels.txt', delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)

    # add node attributes
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

        if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
            if nx.is_connected(G_sub):
                G_sub = nx.convert_node_labels_to_integers(G_sub)
                G_sub.remove_edges_from(G_sub.selfloop_edges())

                # clustering_coeff = nx.clustering(G_sub)
                # bins = [0, 0.2, 0.4, 0.6, 0.8]

                # for node in G_sub.nodes():
                #     G_sub.nodes[node]['label'] = str(G_sub.nodes[node]['label']) + '-' + \
                #         str(bisect.bisect(bins, clustering_coeff[node]))

                # for node in G_sub.nodes():
                #     G_sub.nodes[node]['label'] = str(G_sub.nodes[node]['label']) + '-' + \
                #         str(G_sub.degree[node]) + '-' + str(bisect.bisect(bins, clustering_coeff[node]))

                # for u, v in G_sub.edges():
                #     G_sub.edges[u, v]['label'] = str(min(G_sub.nodes[u]['label'], G_sub.nodes[v]['label'])) + \
                #         '-' + str(max(G_sub.nodes[u]['label'], G_sub.nodes[v]['label']))

                # for node in G_sub.nodes():
                #     G_sub.nodes[node]['label'] = str(G_sub.nodes[node]['label']) + '-' + str(G_sub.degree[node])

                for node in G_sub.nodes():
                    G_sub.nodes[node]['label'] = str(
                        G_sub.nodes[node]['label'])

                nx.set_edge_attributes(G_sub, 'DEFAULT_LABEL', 'label')

                with open(output_path + 'graph' + str(count) + '.dat', 'wb') as f:
                    pickle.dump(G_sub, f)

                count += 1

    return count


def sample_subgraphs(idx, G, output_path, iterations, num_factor):
    count = 0
    deg = G.degree[idx]
    for _ in range(num_factor * int(math.sqrt(deg))):
        G_rw = random_walk_with_restart_sampling(G, idx, iterations=iterations)
        G_rw = nx.convert_node_labels_to_integers(G_rw)
        G_rw.remove_edges_from(G_rw.selfloop_edges())

        if nx.is_connected(G_rw):
            with open(output_path + 'graph' + str(idx) + '-' + str(count) + '.dat', 'wb') as f:
                pickle.dump(G_rw, f)
                count += 1


def produce_random_walk_sampled_graphs(
    input_path, dataset_name, output_path, iterations, num_factor
):
    print('Producing random_walk graphs - num_factor - {}'.format(num_factor))
    G = nx.Graph()

    d = {}
    count = 0
    with open(input_path + dataset_name + '.content', 'r') as f:
        for line in f.readlines():
            spp = line.strip().split('\t')
            G.add_node(count, label=spp[-1])
            d[spp[0]] = count
            count += 1

    count = 0
    with open(input_path + dataset_name + '.cites', 'r') as f:
        for line in f.readlines():
            spp = line.strip().split('\t')
            if spp[0] in d and spp[1] in d:
                G.add_edge(d[spp[0]], d[spp[1]], label='DEFAULT_LABEL')
            else:
                count += 1

    G.remove_edges_from(G.selfloop_edges())
    G = nx.convert_node_labels_to_integers(G)

    print(G.number_of_nodes(), G.number_of_edges())

    with Pool(processes=48) as pool:
        for _ in tqdm(pool.imap_unordered(
            partial(sample_subgraphs, G=G, output_path=output_path, iterations=iterations,
                    num_factor=num_factor), list(range(G.number_of_nodes())))):
            pass

    filenames = []
    for name in os.listdir(output_path):
        if name.endswith('.dat'):
            filenames.append(name)

    print(len(filenames))

    for i, name in enumerate(filenames):
        os.rename(output_path + name, output_path + 'graph' + str(i) + '.dat')

    return len(filenames)


# Routine to create datasets
def create_graphs(args):
    if 'Lung' in args.graph_type:
        base_path = args.dataset_path + 'Lung/'
        input_path = base_path + 'lung.txt'
    elif 'Breast' in args.graph_type:
        base_path = args.dataset_path + 'Breast/'
        input_path = base_path + 'breast.txt'
    elif 'Leukemia' in args.graph_type:
        base_path = args.dataset_path + 'Leukemia/'
        input_path = base_path + 'leukemia.txt'
    elif 'Yeast' in args.graph_type:
        base_path = args.dataset_path + 'Yeast/'
        input_path = base_path + 'yeast.txt'
    elif 'All' in args.graph_type:
        base_path = args.dataset_path + 'All/'
        input_path = base_path + 'all.txt'
    elif 'ENZYMES' in args.graph_type:
        base_path = args.dataset_path + 'ENZYMES/'
        min_num_nodes = 0
        max_num_nodes = 10000
    elif 'citeseer' in args.graph_type:
        base_path = args.dataset_path + 'citeseer/'
        iterations = 150
    elif 'cora' in args.graph_type:
        base_path = args.dataset_path + 'cora/'
        iterations = 150

    if 'inactive' in args.graph_type:
        args.current_dataset_path = base_path + 'inactive_graphs/'
        include_file = base_path + 'actives.txt'
        produce_complementary = True
        args.min_dfscode_path = base_path + 'inactive_min_dfscodes/'
        min_dfscode_tensor_path = base_path + 'inactive_min_dfscode_tensors/'

    elif 'active' in args.graph_type:
        args.current_dataset_path = base_path + 'active_graphs/'
        include_file = base_path + 'actives.txt'
        produce_complementary = False
        args.min_dfscode_path = base_path + 'active_min_dfscodes/'
        min_dfscode_tensor_path = base_path + 'active_min_dfscode_tensors/'

    else:
        args.current_dataset_path = base_path + 'graphs/'
        include_file = None
        produce_complementary = True
        args.min_dfscode_path = base_path + 'min_dfscodes/'
        min_dfscode_tensor_path = base_path + 'min_dfscode_tensors/'

    if args.note == 'GraphRNN' or args.note == 'DGMG':
        args.current_processed_dataset_path = args.current_dataset_path
    elif args.note == 'DFScodeRNN':
        args.current_processed_dataset_path = min_dfscode_tensor_path

    if args.produce_graphs:
        # Empty the directory
        if os.path.isdir(args.current_dataset_path):
            is_del = input('Delete ' + args.current_dataset_path + ' Y/N:')
            if is_del.strip().lower() == 'y':
                shutil.rmtree(args.current_dataset_path)
            else:
                exit()

        os.makedirs(args.current_dataset_path)

        if any(graph_type in args.graph_type for graph_type in ['Lung', 'Breast', 'Leukemia', 'Yeast', 'All']):
            count = produce_graphs_from_raw_format(
                input_path, args.current_dataset_path, include_file, produce_complementary=produce_complementary)

        elif any(graph_type in args.graph_type for graph_type in ['ENZYMES']):
            count = produce_graphs_from_graphrnn_format(
                base_path, args.graph_type, args.current_dataset_path,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes)

        elif any(graph_type in args.graph_type for graph_type in ['cora', 'citeseer']):
            grow_size = 5

            count = produce_random_walk_sampled_graphs(
                base_path, args.graph_type, args.current_dataset_path,
                iterations=iterations, num_factor=grow_size)

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
