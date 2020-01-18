import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import numpy as np
import networkx as nx

from baselines.graph_rnn.model import create_model
from baselines.graph_rnn.helper import get_attributes_len_for_graph_rnn
from utils import load_model, get_model_attribute

EPS = 1e-9


def evaluate_loss(args, model, data, feature_map):
    x_unsorted = data['x'].to(args.device)

    x_len_unsorted = data['len'].to(args.device)
    x_len_max = max(x_len_unsorted)
    x_unsorted = x_unsorted[:, 0:max(x_len_unsorted), :]

    len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
        len(feature_map['node_forward']), len(feature_map['edge_forward']),
        args.max_prev_node, args.max_head_and_tail)

    batch_size = x_unsorted.size(0)
    # sort input for packing variable length sequences
    x_len, sort_indices = torch.sort(x_len_unsorted, dim=0, descending=True)
    x = torch.index_select(x_unsorted, 0, sort_indices)

    # initialize node_level_rnn hidden according to batch size
    model['node_level_rnn'].hidden = model['node_level_rnn'].init_hidden(
        batch_size=batch_size)

    # Teacher forcing: Feed the target as the next input
    # Start token for graph level RNN decoder is node feature second last bit is 1
    node_level_input = torch.cat(
        (torch.zeros(batch_size, 1, x.size(2), device=args.device), x), dim=1)
    node_level_input[:, 0, len_node_vec - 2] = 1

    # Forward propogation
    node_level_output = model['node_level_rnn'](
        node_level_input, input_len=x_len + 1)

    # Evaluating node predictions
    x_pred_node = model['output_node'](node_level_output)

    # Evaluating edge predictions
    # Make a 2D matrix of edge feature vectors with size = [sum(x_len)] x [min(x_len_max - 1, num_nodes_to_consider) * len_edge_vec]
    # 2D matrix will have edge vectors sorted by time_stamp in graph level RNN
    edge_mat_packed = pack_padded_sequence(
        x[:, :, len_node_vec: min(
            x_len_max - 1, num_nodes_to_consider) * len_edge_vec + len_node_vec],
        x_len, batch_first=True)

    edge_mat, _ = edge_mat_packed.data, edge_mat_packed.batch_sizes

    # Time stamp 'i' corresponds to edge feature sequence of length i (including start token added later)
    # Reverse the matrix in dim 0 (for packing purposes)
    idx = torch.LongTensor(
        [i for i in range(edge_mat.size(0) - 1, -1, -1)]).to(args.device)
    edge_mat = edge_mat.index_select(0, idx)

    # Start token of edge level RNN is 1 at second last position in vector of length len_edge_vector
    # End token of edge level RNN is 1 at last position in vector of length len_edge_vector
    # Convert the edge_mat in a 3D tensor of size
    # [sum(x_len)] x [min(x_len_max, num_nodes_to_consider + 1)] x [len_edge_vec]
    edge_mat = edge_mat.reshape(edge_mat.size(0), min(
        x_len_max - 1, num_nodes_to_consider), len_edge_vec)
    edge_level_input = torch.cat(
        (torch.zeros(sum(x_len), 1, len_edge_vec, device=args.device), edge_mat), dim=1)
    edge_level_input[:, 0, len_edge_vec - 2] = 1

    # Compute descending list of lengths for y_edge
    x_edge_len = []
    # Histogram of y_len
    x_edge_len_bin = torch.bincount(x_len)
    for i in range(len(x_edge_len_bin) - 1, 0, -1):
        # count how many x_len is above and equal to i
        count_temp = torch.sum(x_edge_len_bin[i:]).item()

        # put count_temp of them in x_edge_len each with value min(i, num_nodes_to_consider + 1)
        x_edge_len.extend([min(i, num_nodes_to_consider + 1)] * count_temp)

    x_edge_len = torch.LongTensor(x_edge_len).to(args.device)

    # Get edge-level RNN hidden state from node-level RNN output at each timestamp
    # Ignore the last hidden state corresponding to END
    hidden_edge = model['embedding_node_to_edge'](
        node_level_output[:, 0:-1, :])

    # Prepare hidden state for edge level RNN similiar to edge_mat
    # Ignoring the last graph level decoder END token output (all 0's)
    hidden_edge = pack_padded_sequence(
        hidden_edge, x_len, batch_first=True).data
    idx = torch.LongTensor(
        [i for i in range(hidden_edge.size(0) - 1, -1, -1)]).to(args.device)
    hidden_edge = hidden_edge.index_select(0, idx)

    # Set hidden state for edge-level RNN
    # shape of hidden tensor (num_layers, batch_size, hidden_size)
    hidden_edge = hidden_edge.view(1, hidden_edge.size(0), hidden_edge.size(1))
    hidden_edge_rem_layers = torch.zeros(
        args.num_layers - 1, hidden_edge.size(1), hidden_edge.size(2), device=args.device)
    model['edge_level_rnn'].hidden = torch.cat(
        (hidden_edge, hidden_edge_rem_layers), dim=0)

    # Run edge level RNN
    x_pred_edge = model['edge_level_rnn'](
        edge_level_input, input_len=x_edge_len)

    # cleaning the padding i.e setting it to zero
    x_pred_node = pack_padded_sequence(
        x_pred_node, x_len + 1, batch_first=True)
    x_pred_node, _ = pad_packed_sequence(x_pred_node, batch_first=True)
    x_pred_edge = pack_padded_sequence(
        x_pred_edge, x_edge_len, batch_first=True)
    x_pred_edge, _ = pad_packed_sequence(x_pred_edge, batch_first=True)

    # Loss evaluation & backprop
    x_node = torch.cat(
        (x[:, :, :len_node_vec], torch.zeros(batch_size, 1, len_node_vec, device=args.device)), dim=1)
    x_node[torch.arange(batch_size), x_len, len_node_vec - 1] = 1

    x_edge = torch.cat((edge_mat, torch.zeros(
        sum(x_len), 1, len_edge_vec, device=args.device)), dim=1)
    x_edge[torch.arange(sum(x_len)), x_edge_len - 1, len_edge_vec - 1] = 1

    loss1 = F.binary_cross_entropy(x_pred_node, x_node, reduction='sum')
    loss2 = F.binary_cross_entropy(x_pred_edge, x_edge, reduction='sum')

    # Avg (node prediction + edge prediction) error per example
    loss = (loss1 + loss2) / batch_size

    return loss


def predict_graphs(eval_args):
    """
    Generate graphs (networkx format) given a trained generative graphRNN model
    :param eval_args: ArgsEvaluate object
    """

    train_args = eval_args.train_args
    feature_map = get_model_attribute(
        'feature_map', eval_args.model_path, eval_args.device)
    train_args.device = eval_args.device

    model = create_model(train_args, feature_map)
    load_model(eval_args.model_path, eval_args.device, model)

    for _, net in model.items():
        net.eval()

    max_num_node = eval_args.max_num_node
    len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
        len(feature_map['node_forward']), len(feature_map['edge_forward']),
        train_args.max_prev_node, train_args.max_head_and_tail)
    feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

    graphs = []

    for _ in range(eval_args.count // eval_args.batch_size):
        model['node_level_rnn'].hidden = model['node_level_rnn'].init_hidden(
            batch_size=eval_args.batch_size)

        # [batch_size] * [num of nodes]
        x_pred_node = np.zeros(
            (eval_args.batch_size, max_num_node), dtype=np.int32)
        # [batch_size] * [num of nodes] * [num_nodes_to_consider]
        x_pred_edge = np.zeros(
            (eval_args.batch_size, max_num_node, num_nodes_to_consider), dtype=np.int32)

        node_level_input = torch.zeros(
            eval_args.batch_size, 1, feature_len, device=eval_args.device)
        # Initialize to node level start token
        node_level_input[:, 0, len_node_vec - 2] = 1
        for i in range(max_num_node):
            # [batch_size] * [1] * [hidden_size_node_level_rnn]
            node_level_output = model['node_level_rnn'](node_level_input)
            # [batch_size] * [1] * [node_feature_len]
            node_level_pred = model['output_node'](node_level_output)
            # [batch_size] * [node_feature_len] for torch.multinomial
            node_level_pred = node_level_pred.reshape(
                eval_args.batch_size, len_node_vec)
            # [batch_size]: Sampling index to set 1 in next node_level_input and x_pred_node
            # Add a small probability for each node label to avoid zeros
            node_level_pred[:, :-2] += EPS
            # Start token should not be sampled. So set it's probability to 0
            node_level_pred[:, -2] = 0
            # End token should not be sampled if i less than min_num_node
            if i < eval_args.min_num_node:
                node_level_pred[:, -1] = 0
            sample_node_level_output = torch.multinomial(
                node_level_pred, 1).reshape(-1)
            node_level_input = torch.zeros(
                eval_args.batch_size, 1, feature_len, device=eval_args.device)
            node_level_input[torch.arange(
                eval_args.batch_size), 0, sample_node_level_output] = 1

            # [batch_size] * [num of nodes]
            x_pred_node[:, i] = sample_node_level_output.cpu().data

            # [batch_size] * [1] * [hidden_size_edge_level_rnn]
            hidden_edge = model['embedding_node_to_edge'](node_level_output)

            hidden_edge_rem_layers = torch.zeros(
                train_args.num_layers -
                1, eval_args.batch_size, hidden_edge.size(2),
                device=eval_args.device)
            # [num_layers] * [batch_size] * [hidden_len]
            model['edge_level_rnn'].hidden = torch.cat(
                (hidden_edge.permute(1, 0, 2), hidden_edge_rem_layers), dim=0)

            # [batch_size] * [1] * [edge_feature_len]
            edge_level_input = torch.zeros(
                eval_args.batch_size, 1, len_edge_vec, device=eval_args.device)
            # Initialize to edge level start token
            edge_level_input[:, 0, len_edge_vec - 2] = 1
            for j in range(min(num_nodes_to_consider, i)):
                # [batch_size] * [1] * [edge_feature_len]
                edge_level_output = model['edge_level_rnn'](edge_level_input)
                # [batch_size] * [edge_feature_len] needed for torch.multinomial
                edge_level_output = edge_level_output.reshape(
                    eval_args.batch_size, len_edge_vec)

                # [batch_size]: Sampling index to set 1 in next edge_level input and x_pred_edge
                # Add a small probability for no edge to avoid zeros
                edge_level_output[:, -3] += EPS
                # Start token and end should not be sampled. So set it's probability to 0
                edge_level_output[:, -2:] = 0
                sample_edge_level_output = torch.multinomial(
                    edge_level_output, 1).reshape(-1)
                edge_level_input = torch.zeros(
                    eval_args.batch_size, 1, len_edge_vec, device=eval_args.device)
                edge_level_input[:, 0, sample_edge_level_output] = 1

                # Setting edge feature for next node_level_input
                node_level_input[:, 0, len_node_vec + j * len_edge_vec: len_node_vec + (j + 1) * len_edge_vec] = \
                    edge_level_input[:, 0, :]

                # [batch_size] * [num of nodes] * [num_nodes_to_consider]
                x_pred_edge[:, i, j] = sample_edge_level_output.cpu().data

        # Save the batch of graphs
        for k in range(eval_args.batch_size):
            G = nx.Graph()

            for v in range(max_num_node):
                # End node token
                if x_pred_node[k, v] == len_node_vec - 1:
                    break
                elif x_pred_node[k, v] < len(feature_map['node_forward']):
                    G.add_node(
                        v, label=feature_map['node_backward'][x_pred_node[k, v]])
                else:
                    print('Error in sampling node features')
                    exit()

            for u in range(len(G.nodes())):
                for p in range(min(num_nodes_to_consider, u)):
                    if x_pred_edge[k, u, p] < len(feature_map['edge_forward']):
                        if train_args.max_prev_node is not None:
                            v = u - p - 1
                        elif train_args.max_head_and_tail is not None:
                            if p < train_args.max_head_and_tail[1]:
                                v = u - p - 1
                            else:
                                v = p - train_args.max_head_and_tail[1]

                        G.add_edge(
                            u, v, label=feature_map['edge_backward'][x_pred_edge[k, u, p]])
                    elif x_pred_edge[k, u, p] == len(feature_map['edge_forward']):
                        # No edge
                        pass
                    else:
                        print('Error in sampling edge features')
                        exit()

            # Take maximum connected component
            if len(G.nodes()):
                max_comp = max(nx.connected_components(G), key=len)
                G = nx.Graph(G.subgraph(max_comp))

            graphs.append(G)

    return graphs
