from graphgen.model import RNN, MLP_Plain, MLP_Softmax
from baselines.graph_rnn.helper import get_attributes_len_for_graph_rnn


def create_model(args, feature_map):
    len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(len(
        feature_map['node_forward']), len(feature_map['edge_forward']), args.max_prev_node, args.max_head_and_tail)
    feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

    node_level_rnn = RNN(
        input_size=feature_len, embedding_size=args.embedding_size_node_level_rnn,
        hidden_size=args.hidden_size_node_level_rnn, num_layers=args.num_layers,
        device=args.device).to(device=args.device)

    embedding_node_to_edge = MLP_Plain(
        input_size=args.hidden_size_node_level_rnn, embedding_size=args.embedding_size_node_level_rnn,
        output_size=args.hidden_size_edge_level_rnn).to(device=args.device)

    edge_level_rnn = RNN(
        input_size=len_edge_vec, embedding_size=args.embedding_size_edge_level_rnn,
        hidden_size=args.hidden_size_edge_level_rnn, num_layers=args.num_layers,
        output_size=len_edge_vec, output_embedding_size=args.embedding_size_edge_output,
        device=args.device).to(device=args.device)

    output_node = MLP_Softmax(
        input_size=args.hidden_size_node_level_rnn, embedding_size=args.embedding_size_node_output,
        output_size=len_node_vec).to(device=args.device)

    model = {
        'node_level_rnn': node_level_rnn,
        'embedding_node_to_edge': embedding_node_to_edge,
        'edge_level_rnn': edge_level_rnn,
        'output_node': output_node
    }

    return model
