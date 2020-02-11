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

import torch.nn as nn
import torch.nn.functional as F

class Seq2seq(nn.Module):
    """ Seq2seq with Attention """
    def __init__(self, encoder, decoder, decode_function = F.log_softmax, use_pyramidal = False):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function
        self.use_pyramidal = use_pyramidal

    def flatten_parameters(self):
        if self.use_pyramidal:
            self.encoder.bottom_rnn.flatten_parameters()
            self.encoder.middle_rnn.flatten_parameters()
            self.encoder.top_rnn.flatten_parameters()
        else:
            self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def beam_search(self, use = True):
        self.decoder.use_beam_search = use

    def forward(self, feats, targets=None, teacher_forcing_ratio=0.99):
        encoder_outputs, encoder_hidden = self.encoder(feats)
        y_hat, logit = self.decoder(inputs = targets,
                                    encoder_hidden = encoder_hidden,
                                    encoder_outputs = encoder_outputs,
                                    function = self.decode_function,
                                    teacher_forcing_ratio = teacher_forcing_ratio)
        return y_hat, logit