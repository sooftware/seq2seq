import torch.nn as nn
from torch import Tensor
from typing import Tuple
from seq2seq.model.sublayers import BaseRNN


class Seq2seqEncoder(BaseRNN):
    r"""
    Converts low level features into higher level features

    Args:
        input_size (int): size of input
        hidden_dim (int): the number of features in the hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: False)
        rnn_type (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    Inputs: inputs
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens

    Returns: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden
    """

    def __init__(self, input_size: int, hidden_dim: int = 256,
                 dropout_p: float = 0.5, num_layers: int = 3,
                 bidirectional: bool = True, rnn_type: str = 'lstm'):
        super(Seq2seqEncoder, self).__init__(hidden_dim, hidden_dim, num_layers, rnn_type, dropout_p, bidirectional)
        self.embedding = nn.Embedding(input_size, hidden_dim)
        self.input_dropout = nn.Dropout(dropout_p)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        embedded = self.embedding(inputs)
        embedded = self.input_dropout(embedded)

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden
