import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Applies a multi-headed scaled dot mechanism on the output features from the decoder.
    Multi-head attention proposed in "Attention Is All You Need" paper.

    Args:
        hidden_dim (int): The number of expected features in the output
        num_heads (int): The number of heads. (default: )

    Inputs: query, value
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.

    Returns: context
        - **context** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
    """
    def __init__(self, hidden_dim: int = 1024, num_heads: int = 4):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = int(hidden_dim / num_heads)
        self.linear_q = nn.Linear(hidden_dim, self.dim * num_heads)
        self.linear_v = nn.Linear(hidden_dim, self.dim * num_heads)

    def forward(self, query: torch.Tensor, value: torch.Tensor):
        batch_size = value.size(0)
        residual = query

        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.dim)  # BxTxNxD
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.dim)  # BxTxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)  # BNxTxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.dim)  # BNxTxD

        score = torch.bmm(query, value.transpose(1, 2)) / np.sqrt(self.dim)
        attn = F.softmax(score, dim=-1)

        context = torch.bmm(attn, value)
        context = context.view(self.num_heads, batch_size, -1, self.dim)

        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.dim)  # BxTxND
        context = torch.cat((context, residual), dim=2)

        return context
