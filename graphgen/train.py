import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions import Categorical
import networkx as nx

from graphgen.model import create_model
from utils import load_model, get_model_attribute
from dfscode.dfs_wrapper import graph_from_dfscode


def evaluate_loss(args, model, data, feature_map):
    x_len_unsorted = data['len'].to(args.device)
    x_len_max = max(x_len_unsorted)
    batch_size = x_len_unsorted.size(0)

    # sort input for packing variable length sequences
    x_len, sort_indices = torch.sort(x_len_unsorted, dim=0, descending=True)

    max_nodes = feature_map['max_nodes']
    len_node_vec, len_edge_vec = len(
        feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1
    feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_edge_vec

    # Prepare targets with end_tokens already there
    t1 = torch.index_select(
        data['t1'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    t2 = torch.index_select(
        data['t2'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    v1 = torch.index_select(
        data['v1'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    e = torch.index_select(
        data['e'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    v2 = torch.index_select(
        data['v2'][:, :x_len_max + 1].to(args.device), 0, sort_indices)

    x_t1, x_t2 = F.one_hot(t1, num_classes=max_nodes +
                           2)[:, :, :-1], F.one_hot(t2, num_classes=max_nodes + 2)[:, :, :-1]
    x_v1, x_v2 = F.one_hot(v1, num_classes=len_node_vec +
                           1)[:, :, :-1], F.one_hot(v2, num_classes=len_node_vec + 1)[:, :, :-1]
    x_e = F.one_hot(e, num_classes=len_edge_vec + 1)[:, :, :-1]

    x_target = torch.cat((x_t1, x_t2, x_v1, x_e, x_v2), dim=2).float()

    # initialize dfs_code_rnn hidden according to batch size
    model['dfs_code_rnn'].hidden = model['dfs_code_rnn'].init_hidden(
        batch_size=batch_size)

    # Teacher forcing: Feed the target as the next input
    # Start token is all zeros
    dfscode_rnn_input = torch.cat(
        (torch.zeros(batch_size, 1, feature_len, device=args.device), x_target[:, :-1, :]), dim=1)

    # Forward propogation
    dfscode_rnn_output = model['dfs_code_rnn'](
        dfscode_rnn_input, input_len=x_len + 1)

    # Evaluating dfscode tuple
    timestamp1 = model['output_timestamp1'](dfscode_rnn_output)
    timestamp2 = model['output_timestamp2'](dfscode_rnn_output)
    vertex1 = model['output_vertex1'](dfscode_rnn_output)
    edge = model['output_edge'](dfscode_rnn_output)
    vertex2 = model['output_vertex2'](dfscode_rnn_output)

    if args.loss_type == 'BCE':
        x_pred = torch.cat(
            (timestamp1, timestamp2, vertex1, edge, vertex2), dim=2)

        # Cleaning the padding i.e setting it to zero
        x_pred = pack_padded_sequence(x_pred, x_len + 1, batch_first=True)
        x_pred, _ = pad_packed_sequence(x_pred, batch_first=True)

        if args.weights:
            # Weights for BCE
            weight = torch.cat((feature_map['t1_weight'].to(args.device), feature_map['t2_weight'].to(args.device),
                                feature_map['v1_weight'].to(
                                    args.device), feature_map['e_weight'].to(args.device),
                                feature_map['v2_weight'].to(args.device)))

            weight = weight.expand(batch_size, x_len_max + 1, -1)
        else:
            weight = None

        loss_sum = F.binary_cross_entropy(
            x_pred, x_target, reduction='none', weight=weight)
        loss = torch.mean(
            torch.sum(loss_sum, dim=[1, 2]) / (x_len.float() + 1))

    elif args.loss_type == 'NLL':
        timestamp1 = timestamp1.transpose(dim0=1, dim1=2)
        timestamp2 = timestamp2.transpose(dim0=1, dim1=2)
        vertex1 = vertex1.transpose(dim0=1, dim1=2)
        edge = edge.transpose(dim0=1, dim1=2)
        vertex2 = vertex2.transpose(dim0=1, dim1=2)

        loss_t1 = F.nll_loss(
            timestamp1, t1, ignore_index=max_nodes + 1, weight=feature_map.get('t1_weight'))
        loss_t2 = F.nll_loss(
            timestamp2, t2, ignore_index=max_nodes + 1, weight=feature_map.get('t2_weight'))
        loss_v1 = F.nll_loss(vertex1, v1, ignore_index=len(
            feature_map['node_forward']) + 1, weight=feature_map.get('v1_weight'))
        loss_e = F.nll_loss(edge, e, ignore_index=len(
            feature_map['edge_forward']) + 1, weight=feature_map.get('e_weight'))
        loss_v2 = F.nll_loss(vertex2, v2, ignore_index=len(
            feature_map['node_forward']) + 1, weight=feature_map.get('v2_weight'))

        loss = loss_t1 + loss_t2 + loss_v1 + loss_e + loss_v2

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

    max_nodes = feature_map['max_nodes']
    len_node_vec, len_edge_vec = len(
        feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1
    feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_edge_vec

    graphs = []

    for _ in range(eval_args.count // eval_args.batch_size):
        # initialize dfs_code_rnn hidden according to batch size
        model['dfs_code_rnn'].hidden = model['dfs_code_rnn'].init_hidden(
            batch_size=eval_args.batch_size)

        rnn_input = torch.zeros(
            (eval_args.batch_size, 1, feature_len), device=eval_args.device)
        pred = torch.zeros(
            (eval_args.batch_size, eval_args.max_num_edges, 5), device=eval_args.device)

        for i in range(eval_args.max_num_edges):
            rnn_output = model['dfs_code_rnn'](rnn_input)

            # Evaluating dfscode tuple
            timestamp1 = model['output_timestamp1'](
                rnn_output).reshape(eval_args.batch_size, -1)
            timestamp2 = model['output_timestamp2'](
                rnn_output).reshape(eval_args.batch_size, -1)
            vertex1 = model['output_vertex1'](
                rnn_output).reshape(eval_args.batch_size, -1)
            edge = model['output_edge'](rnn_output).reshape(
                eval_args.batch_size, -1)
            vertex2 = model['output_vertex2'](
                rnn_output).reshape(eval_args.batch_size, -1)

            if train_args.loss_type == 'BCE':
                timestamp1 = Categorical(timestamp1).sample()
                timestamp2 = Categorical(timestamp2).sample()
                vertex1 = Categorical(vertex1).sample()
                edge = Categorical(edge).sample()
                vertex2 = Categorical(vertex2).sample()

            elif train_args.loss_type == 'NLL':
                timestamp1 = Categorical(logits=timestamp1).sample()
                timestamp2 = Categorical(logits=timestamp2).sample()
                vertex1 = Categorical(logits=vertex1).sample()
                edge = Categorical(logits=edge).sample()
                vertex2 = Categorical(logits=vertex2).sample()

            rnn_input = torch.zeros(
                (eval_args.batch_size, 1, feature_len), device=eval_args.device)

            rnn_input[torch.arange(eval_args.batch_size), 0, timestamp1] = 1
            rnn_input[torch.arange(eval_args.batch_size),
                      0, max_nodes + 1 + timestamp2] = 1
            rnn_input[torch.arange(eval_args.batch_size),
                      0, 2 * max_nodes + 2 + vertex1] = 1
            rnn_input[torch.arange(eval_args.batch_size), 0,
                      2 * max_nodes + 2 + len_node_vec + edge] = 1
            rnn_input[torch.arange(eval_args.batch_size), 0, 2 *
                      max_nodes + 2 + len_node_vec + len_edge_vec + vertex2] = 1

            pred[:, i, 0] = timestamp1
            pred[:, i, 1] = timestamp2
            pred[:, i, 2] = vertex1
            pred[:, i, 3] = edge
            pred[:, i, 4] = vertex2

        nb = feature_map['node_backward']
        eb = feature_map['edge_backward']
        for i in range(eval_args.batch_size):
            dfscode = []
            for j in range(eval_args.max_num_edges):
                if pred[i, j, 0] == max_nodes or pred[i, j, 1] == max_nodes \
                        or pred[i, j, 2] == len_node_vec - 1 or pred[i, j, 3] == len_edge_vec - 1 \
                        or pred[i, j, 4] == len_node_vec - 1:
                    break

                dfscode.append(
                    (int(pred[i, j, 0].data), int(pred[i, j, 1].data), nb[int(pred[i, j, 2].data)],
                     eb[int(pred[i, j, 3].data)], nb[int(pred[i, j, 4].data)]))

            graph = graph_from_dfscode(dfscode)

            # Remove self loops
            graph.remove_edges_from(graph.selfloop_edges())

            # Take maximum connected component
            if len(graph.nodes()):
                max_comp = max(nx.connected_components(graph), key=len)
                graph = nx.Graph(graph.subgraph(max_comp))

            graphs.append(graph)

    return graphs
