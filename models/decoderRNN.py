import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.beam import Beam
from .attention import Attention

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(nn.Module):
    r"""
    Converts higher level features (from encoder) into output sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        layer_size (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention (bool, optional): flag indication whether to use attention mechanism or not (default: false)
        k (int) : size of beam

    Inputs: inputs, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the listener.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Returns: y_hats, logits
        - **y_hats** (batch, seq_len): predicted y values (y_hat) by the model
        - **logits** (batch, seq_len, vocab_size): predicted log probability by the model

    Examples::

        >>> decoder = DecoderRNN(vocab_size, max_len, hidden_size, sos_id, eos_id, n_layers)
        >>> y_hats, logits = decoder(inputs, encoder_outputs, teacher_forcing_ratio=0.90)
    """
    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru', dropout_p=0.5,
                 use_attention=True, device=None, use_beam_search=True, k=8):
        super(DecoderRNN, self).__init__()
        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU if rnn_cell.lower() == 'gru' else nn.RNN
        self.rnn = self.rnn_cell(hidden_size , hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.n_layers = n_layers
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.device = device
        self.use_beam_search = use_beam_search
        self.k = k
        if use_attention:
            self.attention = Attention(hidden_size)

    def _forward_step(self, input, decoder_hidden, encoder_outputs=None, function=F.log_softmax):
        """ forward one time step """
        batch_size = input.size(0)
        output_size = input.size(1)
        embedded = self.embedding(input).to(self.device)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()
        decoder_output = self.rnn(embedded, decoder_hidden)[0]

        if self.use_attention:
            output = self.attention(decoder_output, encoder_outputs)
        else:
            output = decoder_output

        predicted_softmax = function(self.w(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        return predicted_softmax

    def forward(self, inputs, encoder_outputs, function=F.log_softmax, teacher_forcing_ratio=0.90, use_beam_search=False):
        y_hats, logits = None, None
        decode_results = []
        batch_size = inputs.size(0)
        max_len = inputs.size(1) - 1  # minus the start of sequence symbol
        decoder_hidden = torch.FloatTensor(self.n_layers, batch_size, self.hidden_size).uniform_(-0.1, 0.1).to(self.device)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_beam_search:
            """ Beam-Search Decoding """
            inputs = inputs[:, 0].unsqueeze(1)
            beam = Beam(
                k = self.k,
                decoder_hidden = decoder_hidden,
                decoder = self,
                batch_size = batch_size,
                max_len = max_len,
                function = function,
                device = self.device
            )
            y_hats = beam.search(inputs, encoder_outputs)
        else:
            if use_teacher_forcing:
                # if teacher_forcing, Infer all at once
                inputs = inputs[inputs != self.eos_id].view(batch_size, -1)
                predicted_softmax = self._forward_step(
                    input = inputs,
                    decoder_hidden = decoder_hidden,
                    encoder_outputs = encoder_outputs,
                    function = function
                )
                for di in range(predicted_softmax.size(1)):
                    step_output = predicted_softmax[:, di, :]
                    decode_results.append(step_output)
            else:
                input = inputs[:, 0].unsqueeze(1)
                for di in range(max_len):
                    predicted_softmax = self._forward_step(
                        input = input,
                        decoder_hidden = decoder_hidden,
                        encoder_outputs = encoder_outputs,
                        function = function
                    )
                    step_output = predicted_softmax.squeeze(1)
                    decode_results.append(step_output)
                    input = decode_results[-1].topk(1)[1]

            logits = torch.stack(decode_results, dim=1).to(self.device)
            y_hats = logits.max(-1)[1]
        return y_hats, logits