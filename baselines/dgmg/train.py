"""
Code adapted from https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgmg
"""

import networkx as nx

from baselines.dgmg.model import create_model
from utils import load_model, get_model_attribute


def evaluate_loss(model, data):
    batch_size = len(data)

    log_prob = model['generator'](
        batch_size=batch_size, actions=data, training=True)
    loss = -log_prob / batch_size

    return loss


def predict_graphs(eval_args):
    train_args = eval_args.train_args
    feature_map = get_model_attribute(
        'feature_map', eval_args.model_path, eval_args.device)
    train_args.device = eval_args.device

    model = create_model(train_args, feature_map)
    load_model(eval_args.model_path, eval_args.device, model)

    for _, net in model.items():
        net.eval()

    graphs = []

    for _ in range(eval_args.count // eval_args.batch_size):
        sampled_graphs = model['generator'](
            eval_args.batch_size, training=False)

        nb = feature_map['node_backward']
        eb = feature_map['edge_backward']
        for sampled_graph in sampled_graphs:
            graph = sampled_graph.to_networkx(
                node_attrs=['label'], edge_attrs=['label']).to_undirected()

            labeled_graph = nx.Graph()

            for v in graph.nodes():
                labeled_graph.add_node(
                    v, label=nb[graph.nodes[v]['label'].item() - 1])

            for u, v in graph.edges():
                labeled_graph.add_edge(
                    u, v, label=eb[graph.edges[u, v]['label'].item()])

            # Take maximum connected component
            if len(labeled_graph.nodes()) > 0:
                max_comp = max(nx.connected_components(labeled_graph), key=len)
                labeled_graph = labeled_graph.subgraph(max_comp)

            graphs.append(labeled_graph)

    return graphs
