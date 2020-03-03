"""
Copyright 2020- Sooftware
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.beam import Beam
from .attention import SelfAttention

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class Decoder(nn.Module):
    """ Converts higher level features (from encoder) into output """
    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id,
                 layer_size=1, rnn_cell='gru', dropout_p=0,
                 use_attention=True, device=None, use_beam_search=True, k=8):
        super(Decoder, self).__init__()
        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU if rnn_cell.lower() == 'gru' else nn.RNN
        self.rnn = self.rnn_cell(hidden_size , hidden_size, layer_size, batch_first=True, dropout=dropout_p)
        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.layer_size = layer_size
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.device = device
        self.use_beam_search = use_beam_search
        self.k = k
        if use_attention:
            self.attention = SelfAttention(hidden_size)

    def _forward_step(self, decoder_input, decoder_hidden, encoder_outputs, last_alignment, function):
        batch_size = decoder_input.size(0)
        output_size = decoder_input.size(1)
        embedded = self.embedding(decoder_input)
        embedded = self.input_dropout(embedded)
        if self.training:
            self.rnn.flatten_parameters()
        decoder_output, hidden = self.rnn(embedded, decoder_hidden) # decoder output

        alignment = None
        if self.use_attention:
            context, alignment = self.attention(decoder_output, encoder_outputs, last_alignment)
        else:
            context = decoder_output
        predicted_softmax = function(self.out(context.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)                                                                                              -1)
        return predicted_softmax, alignment

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax, teacher_forcing_ratio=0.99, use_beam_search=False):
        y_hats, logit = None, None
        decode_results = []
        # Validate Arguments
        batch_size = inputs.size(0)
        max_length = inputs.size(1) - 1  # minus the start of sequence symbol
        # Initiate decoder Hidden State to zeros  :  LxBxH
        decoder_hidden = torch.FloatTensor(self.layer_size, batch_size, self.hidden_size).uniform_(-0.1, 0.1)#.cuda()
        # Decide Use Teacher Forcing or Not
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_beam_search:
            """Implementation of Beam-Search Decoding"""
            decoder_input = inputs[:, 0].unsqueeze(1)
            beam = Beam(k=self.k, decoder_hidden=decoder_hidden, decoder=self,
                        batch_size=batch_size, max_len=max_length, decode_func=function)
            y_hats = beam.search(decoder_input, encoder_outputs)
        else:
            if use_teacher_forcing:
                decoder_input = inputs[:, :-1]  # except </s>
                last_alignment = None
                """ Fix to non-parallel process even in teacher forcing to apply location-aware attention """
                for di in range(len(decoder_input[0])):
                    predicted_softmax, last_alignment = self._forward_step(
                        decoder_input=decoder_input[:, di].unsqueeze(1),
                        decoder_hidden=decoder_hidden,
                        encoder_outputs=encoder_outputs,
                        last_alignment=last_alignment,
                        function=function)
                    step_output = predicted_softmax.squeeze(1)
                    decode_results.append(step_output)
            else:
                decoder_input = inputs[:, 0].unsqueeze(1)
                last_alignment = None
                for di in range(max_length):
                    predicted_softmax, last_alignment = self._forward_step(decoder_input=decoder_input,
                                                                           decoder_hidden=decoder_hidden,
                                                                           encoder_outputs=encoder_outputs,
                                                                           last_alignment=last_alignment,
                                                                           function=function)
                    step_output = predicted_softmax.squeeze(1)
                    decode_results.append(step_output)
                    decoder_input = decode_results[-1].topk(1)[1]

            logit = torch.stack(decode_results, dim=1).to(self.device)
            y_hats = logit.max(-1)[1]

        return y_hats, logit if self.training else y_hats