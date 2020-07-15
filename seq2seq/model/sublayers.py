import torch.nn as nn
from torch import Tensor
from seq2seq.model.modules import LayerNorm, Linear


class BaseRNN(nn.Module):
    """
    Applies a multi-layer RNN to an input sequence.

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        input_size (int): size of input
        hidden_dim (int): the number of features in the hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: False)
        rnn_type (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability (default: 0)
        device (torch.device): device - 'cuda' or 'cpu'

    Attributes:
          supported_rnns = Dictionary of supported rnns
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN
    }

    def __init__(self, input_size: int, hidden_dim: int = 512, num_layers: int = 3,
                 rnn_type: str = 'lstm', dropout_p: float = 0.3,
                 bidirectional: bool = True, device: str = 'cuda') -> None:
        super(BaseRNN, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_size, hidden_dim, num_layers, True, True, dropout_p, bidirectional)
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class AddNorm(nn.Module):
    """
    Add & Normalization layer proposed in "Attention Is All You Need".
    Transformer employ a residual connection around each of the two sub-layers,
    (Multi-Head Attention & Feed-Forward) followed by layer normalization.
    """
    def __init__(self, sublayer: nn.Module, d_model: int = 512) -> None:
        super(AddNorm, self).__init__()
        self.sublayer = sublayer
        self.layer_norm = LayerNorm(d_model)

    def forward(self, *args):
        residual = args[0]
        output = self.sublayer(*args)

        if isinstance(output, tuple):
            return self.layer_norm(output[0] + residual), output[1]
        else:
            return self.layer_norm(output + residual)
