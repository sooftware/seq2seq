import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.beamsearch import BeamSearch
from .attention import MultiHeadAttention

supported_rnns = {
    'lstm': nn.LSTM,
    'gru': nn.GRU,
    'rnn': nn.RNN
}


class DecoderRNN(nn.Module):
    r"""
    Converts higher level features (from encoder) into output sequence.

    Args:
        n_class (int): the number of class
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_dim (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_type (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        k (int) : size of beam

    Inputs: inputs, encoder_outputs, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_dim): tensor with containing the outputs of the listener.
          Used for attention mechanism (default is `None`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Returns: hypothesis, logits
        - **hypothesis** (batch, seq_len): predicted y values (y_hat) by the model
        - **logits** (batch, seq_len, class_num): predicted log probability by the model

    Examples::

        >>> decoder = DecoderRNN(n_class, max_len, hidden_dim, sos_id, eos_id, n_layers)
        >>> hypothesis, logits = decoder(inputs, encoder_outputs, teacher_forcing_ratio=0.90)
    """

    def __init__(self, n_class, max_len, hidden_dim, sos_id, eos_id, pad_id, n_layers=1,
                 rnn_type='gru', dropout_p=0.5, device=None, use_beam_search=False, k=5):

        super(DecoderRNN, self).__init__()
        assert rnn_type.lower() in supported_rnns.keys(), 'RNN type not supported.'

        rnn_cell = supported_rnns[rnn_type]
        self.rnn = rnn_cell(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout_p)
        self.output_size = n_class
        self.max_length = max_len
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.use_beam_search = use_beam_search
        self.k = k
        self.embedding = nn.Embedding(self.output_size, hidden_dim)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
        self.attention = MultiHeadAttention(in_features=hidden_dim, dim=128, n_head=4)

    def forward_step(self, input_var, h_state, encoder_outputs=None):
        embedded = self.embedding(input_var).to(self.device)
        embedded = self.input_dropout(embedded)

        if len(embedded.size()) == 2:  # if beam search
            embedded = embedded.unsqueeze(1)

        if self.training:
            self.rnn.flatten_parameters()

        output, h_state = self.rnn(embedded, h_state)
        context = self.attention(output, encoder_outputs)

        predicted_softmax = F.log_softmax(self.fc(context.contiguous().view(-1, self.hidden_dim)), dim=1)
        return predicted_softmax, h_state

    def forward(self, inputs, encoder_outputs, teacher_forcing_ratio=0.90, use_beam_search=False):
        hypothesis, logits = None, None

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_outputs)
        h_state = self.init_state(batch_size)

        decode_results = list()
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_beam_search:
            search = BeamSearch(self, batch_size)
            hypothesis = search(inputs, encoder_outputs, k=self.k)

        else:
            if use_teacher_forcing:  # if teacher_forcing, Infer all at once
                inputs = inputs[inputs != self.eos_id].view(batch_size, -1)
                predicted_softmax, h_state = self.forward_step(inputs, h_state, encoder_outputs)
                predicted_softmax = predicted_softmax.view(batch_size, inputs.size(1), -1)

                for di in range(predicted_softmax.size(1)):
                    step_output = predicted_softmax[:, di, :]
                    decode_results.append(step_output)

            else:
                input_var = inputs[:, 0].unsqueeze(1)

                for di in range(max_length):
                    predicted_softmax, h_state = self.forward_step(input_var, h_state, encoder_outputs)
                    step_output = predicted_softmax.view(batch_size, 1, -1).squeeze(1)
                    decode_results.append(step_output)
                    input_var = decode_results[-1].topk(1)[1]

            logits = torch.stack(decode_results, dim=1).to(self.device)
            hypothesis = logits.max(-1)[1]

        return hypothesis, logits

    def init_state(self, batch_size):
        if isinstance(self.rnn, nn.LSTM):
            h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
            c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
            h_state = (h_0, c_0)

        else:
            h_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)

        return h_state

    def _validate_args(self, inputs, encoder_outputs):
        batch_size = encoder_outputs.size(0)

        if inputs is None:
            inputs = torch.empty(batch_size, 1).type(torch.long)
            inputs[:, 0] = self.sos_id
            max_length = self.max_length

        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length
