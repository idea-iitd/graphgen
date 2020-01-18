import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class MLP_Softmax(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            MLP_Plain(input_size, embedding_size, output_size, dropout),
            nn.Softmax(dim=2)
        )

    def forward(self, input):
        return self.mlp(input)


class MLP_Log_Softmax(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Log_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            MLP_Plain(input_size, embedding_size, output_size, dropout),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, input):
        return self.mlp(input)


class MLP_Plain(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Plain, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.Linear(embedding_size, embedding_size),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self.mlp(input)


class RNN(nn.Module):
    """
    Custom GRU layer
    :param input_size: Size of input vector
    :param embedding_size: Embedding layer size (finally this size is input to RNN)
    :param hidden_size: Size of hidden state of vector
    :param num_layers: No. of RNN layers
    :param rnn_type: Currently only GRU and LSTM supported
    :param dropout: Dropout probability for dropout layers between rnn layers
    :param output_size: If provided, a MLP softmax is run on hidden state with output of size 'output_size'
    :param output_embedding_size: If provided, the MLP softmax middle layer is of this size, else 
        middle layer size is same as 'embedding size'
    :param device: torch device to instanstiate the hidden state on right device
    """

    def __init__(
        self, input_size, embedding_size, hidden_size, num_layers, rnn_type='GRU',
        dropout=0, output_size=None, output_embedding_size=None,
        device=torch.device('cpu')
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.output_size = output_size
        self.device = device

        self.input = nn.Linear(input_size, embedding_size)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout
            )
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout
            )

        # self.relu = nn.ReLU()

        self.hidden = None  # Need initialization before forward run

        if self.output_size is not None:
            if output_embedding_size is None:
                self.output = MLP_Softmax(
                    hidden_size, embedding_size, self.output_size)
            else:
                self.output = MLP_Softmax(
                    hidden_size, output_embedding_size, self.output_size)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(
                    param, gain=nn.init.calculate_gain('sigmoid'))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        if self.rnn_type == 'GRU':
            # h0
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        elif self.rnn_type == 'LSTM':
            # (h0, c0)
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))

    def forward(self, input, input_len=None):
        input = self.input(input)
        # input = self.relu(input)

        if input_len is not None:
            input = pack_padded_sequence(
                input, input_len, batch_first=True, enforce_sorted=False)

        output, self.hidden = self.rnn(input, self.hidden)

        if input_len is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)

        if self.output_size is not None:
            output = self.output(output)

        return output


def create_model(args, feature_map):
    max_nodes = feature_map['max_nodes']
    len_node_vec, len_edge_vec = len(
        feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1

    if args.note == 'DFScodeRNN':
        feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_edge_vec
    else:
        feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec

    if args.loss_type == 'BCE':
        MLP_layer = MLP_Softmax
    elif args.loss_type == 'NLL':
        MLP_layer = MLP_Log_Softmax

    dfs_code_rnn = RNN(
        input_size=feature_len, embedding_size=args.embedding_size_dfscode_rnn,
        hidden_size=args.hidden_size_dfscode_rnn, num_layers=args.num_layers,
        rnn_type=args.rnn_type, dropout=args.dfscode_rnn_dropout,
        device=args.device).to(device=args.device)

    output_timestamp1 = MLP_layer(
        input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_timestamp_output,
        output_size=max_nodes + 1, dropout=args.dfscode_rnn_dropout).to(device=args.device)

    output_timestamp2 = MLP_layer(
        input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_timestamp_output,
        output_size=max_nodes + 1, dropout=args.dfscode_rnn_dropout).to(device=args.device)

    output_vertex1 = MLP_layer(
        input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_vertex_output,
        output_size=len_node_vec, dropout=args.dfscode_rnn_dropout).to(device=args.device)

    output_vertex2 = MLP_layer(
        input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_vertex_output,
        output_size=len_node_vec, dropout=args.dfscode_rnn_dropout).to(device=args.device)

    model = {
        'dfs_code_rnn': dfs_code_rnn,
        'output_timestamp1': output_timestamp1,
        'output_timestamp2': output_timestamp2,
        'output_vertex1': output_vertex1,
        'output_vertex2': output_vertex2
    }

    if args.note == 'DFScodeRNN':
        output_edge = MLP_layer(
            input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_edge_output,
            output_size=len_edge_vec, dropout=args.dfscode_rnn_dropout).to(device=args.device)
        model['output_edge'] = output_edge

    return model
