import torch.nn as nn
import torch.nn.functional as F


class Seq2seq(nn.Module):
    r"""
    Sequence to Sequence Model

    Args:
        encoder (torch.nn.Module): encoder of seq2seq
        decoder (torch.nn.Module): decoder of seq2seq
        function (torch.nn.functional): A function used to generate symbols from RNN hidden state

    Inputs: inputs, targets, teacher_forcing_ratio, use_beam_search
        - **inputs** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **targets** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0.90)
        - **use_beam_search** (bool): flag indication whether to use beam-search or not (default: false)

    Returns: y_hats, logits
        - **y_hats** (batch, seq_len): predicted y values (y_hat) by the model
        - **logits** (batch, seq_len, vocab_size): logit values by the model

    Examples::
        >>> encoder = EncoderRNN(input_size, ...)
        >>> decoder = DecoderRNN(class_num, ...)
        >>> model = Seq2seq(encoder, decoder)
        >>> y_hats, logits = model()
    """
    def __init__(self, encoder, decoder, function=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.function = function

    def forward(self, inputs, targets, input_lengths, teacher_forcing_ratio=0.90, use_beam_search=False):
        encoder_outputs, encoder_hidden = self.encoder(inputs, input_lengths)
        y_hats, logits = self.decoder(
            inputs=targets,
            encoder_outputs=encoder_outputs,
            function=self.function,
            teacher_forcing_ratio=teacher_forcing_ratio,
            use_beam_search=use_beam_search
        )

        return y_hats, logits

    def set_beam_size(self, k):
        self.speller.k = k

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
