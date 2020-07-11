import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from seq2seq.modules import Linear


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, value
        - **query** (batch, q_len, hidden_dim): tensor containing projection vector for decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing projection vector for encoder.

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_model: int) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Applies a multi-headed scaled dot mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.

    Args:
        d_model (int): dimension of model
        num_heads (int): The number of heads. (default: )

    Inputs: query, value
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.

    Returns: context
        - **context** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.

    Reference:
        - **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
        - **State-Of-The-Art Speech Recognition with Sequence-to-Sequence Models**: https://arxiv.org/abs/1712.01769

    Contributor:
        - Soohwan Kim @sooftware
        - Deokjin Seo @qute012
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "hidden_dim % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.linear_q = Linear(d_model, self.d_head * num_heads)
        self.linear_k = Linear(d_model, self.d_head * num_heads)
        self.linear_v = Linear(d_model, self.d_head * num_heads)
        self.scaled_dot_attn = ScaledDotProductAttention(d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_head)      # BxK_LENxNxD
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)      # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        context, attn = self.scaled_dot_attn(query, key, value)
        context = context.view(self.num_heads, batch_size, -1, self.d_head)

        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND
        return context, attn
