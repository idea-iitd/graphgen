"""
Code adapted from https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgmg
"""

import pickle
import torch
from torch.utils.data import Dataset
import networkx as nx


class DGMG_Dataset_from_file(Dataset):
    """
    Deep GMG Dataset for target values
    """

    def __init__(self, args, graphs_indices, feature_map):
        # Path to folder containing dataset
        self.dataset_path = args.current_processed_dataset_path
        self.graphs_indices = graphs_indices
        self.feature_map = feature_map

    def __len__(self):
        return len(self.graphs_indices)

    def __getitem__(self, idx):
        with open(self.dataset_path + 'graph' + str(self.graphs_indices[idx]) + '.dat', 'rb') as f:
            graph = pickle.load(f)
        f.close()

        node_map, edge_map = self.feature_map['node_forward'], self.feature_map['edge_forward']

        perm = torch.randperm(len(graph.nodes())).numpy()
        perm_map = {i: perm[i] for i in range(len(perm))}
        graph = nx.relabel_nodes(graph, perm_map)

        actions = []
        for v in range(len(graph.nodes())):
            actions.append(1 + node_map[graph.nodes[v]['label']])  # Add node

            for u, val in graph[v].items():
                if u < v:
                    # Add edge
                    actions.append(1)
                    actions.append(
                        int(u * len(edge_map) + edge_map[val['label']]))

            actions.append(0)  # Stop Edge

        actions.append(0)  # Stop Node

        return actions

    def collate_batch(self, batch):
        return batch
