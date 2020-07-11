import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class Seq2seq(nn.Module):
    """
    Standard sequence-to-sequence architecture with configurable encoder and decoder.

    Args:
        encoder (torch.nn.Module): encoder of seq2seq
        decoder (torch.nn.Module): decoder of seq2seq

    Inputs: inputs, input_lengths, targets, teacher_forcing_ratio
        - **inputs** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (torch.Tensor): tensor of sequences, whose contains length of inputs.
        - **targets** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0.90)

    Returns: decoder_outputs, ret_dict
        - **decoder_outputs** (seq_len, batch, num_classes): list of tensors containing
          the outputs of the decoding function.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_ATTENTION_SCORE* : list of scores
          representing encoder outputs, *KEY_SEQUENCE_SYMBOL* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: Tensor, input_lengths: Tensor, targets: Optional[Tensor] = None,
                teacher_forcing_ratio: float = 1.0) -> Tuple[Tensor, dict]:
        encoder_outputs, hidden = self.encoder(inputs, input_lengths)
        result = self.decoder(targets, encoder_outputs, teacher_forcing_ratio)
        return result

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def set_speller(self, decoder: nn.Module):
        self.decoder = decoder
