import torch
import torch.nn as nn


class Seq2seq(nn.Module):
    r"""
    Sequence to Sequence Model

    Args:
        encoder (torch.nn.Module): encoder of seq2seq
        decoder (torch.nn.Module): decoder of seq2seq

    Inputs: inputs, targets, teacher_forcing_ratio, use_beam_search
        - **inputs** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **targets** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0.90)
        - **use_beam_search** (bool): flag indication whether to use beam-search or not (default: false)

    Returns: output, ret_dict
        - **output** (seq_len, batch, num_classes): list of tensors containing the outputs of the decoding function.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_ATTENTION_SCORE* : list of scores
          representing encoder outputs, *KEY_SEQUENCE_SYMBOL* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, teacher_forcing_ratio: float = 0.90):
        encoder_outputs = self.encoder(inputs, input_lengths)
        output, ret_dict = self.decoder(targets, encoder_outputs, teacher_forcing_ratio)

        return output, ret_dict

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
