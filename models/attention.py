import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Applies an dot product attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * encoder_output) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: decoder_output, encoder_output
        - **decoder_output** (batch, output_len, hidden_size): tensor containing the output features from the decoder.
        - **encoder_output** (batch, input_len, hidden_size): tensor containing features of the encoded input sequence.Steps to be maintained at a certain number to avoid extremely slow learning

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
    """
    def __init__(self, decoder_hidden_size):
        super(Attention, self).__init__()
        self.w = nn.Linear(decoder_hidden_size*2, decoder_hidden_size)

    def forward(self, decoder_output, encoder_outputs):
        batch_size = decoder_output.size(0)
        input_size = encoder_outputs.size(1)
        hidden_size = decoder_output.size(2)

        # get attention score
        attn_score = torch.bmm(decoder_output, encoder_outputs.transpose(1, 2))
        # get attention distribution
        attn_distribution = F.softmax(attn_score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        # get attention value
        context = torch.bmm(attn_distribution, encoder_outputs) # get attention value
        # concatenate attn_val & decoder_output
        combined = torch.cat((context, decoder_output), dim=2)
        output = torch.tanh(self.w(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        return output