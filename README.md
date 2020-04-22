# Pytorch Sequence-to-Sequence
  
### Seq2seq Architecture in Pytorch  
[<img src="https://github.com/gentaiscool/end2end-asr-pytorch/raw/master/img/pytorch-logo-dark.png" height=18>](https://pytorch.org/) <img src="https://img.shields.io/badge/License-Apache--2.0-yellow" height=20>
  
### [**Documentation**](https://sooftware.github.io/Pytorch-Seq2seq/)
  
## Intro
  
This is a framework for Attention based Sequence-to-Sequence (seq2seq) models implemented in [Pytorch](https://pytorch.org/).  
We appreciate any kind of feedback or contribution.  
   
![model](https://camo.githubusercontent.com/9e88497fcdec5a9c716e0de5bc4b6d1793c6e23f/687474703a2f2f73757269796164656570616e2e6769746875622e696f2f696d672f736571327365712f73657132736571322e706e67)
  
## How To Use  
  
### Training

```python
from models.encoderRNN import EncoderRNN
from models.decoderRNN import DecoderRNN
from models.seq2seq import Seq2seq

encoder = EncoderRNN(
    in_features = in_features, 
    hidden_size = config.hidden_size, 
    dropout_p = config.dropout_p, 
    n_layers = config.encoder_layer_size, 
    bidirectional = bidirectional, 
    rnn_type = 'gru'
)
decoder = DecoderRNN(
    class_num = class_num, 
    max_len = config.max_len, 
    hidden_size = config.hidden_size if bidirectional else config.hidden_size << 1,
    sos_id = SOS_token, 
    eos_id = EOS_token,
    n_layers = config.decoder_layer_size, 
    rnn_type = 'gru', 
    dropout_p = config.dropout_p,
    use_attention = config.use_attention, 
    device = device, 
    use_beam_search = False, 
    k = 8
 )
 model = Seq2seq(encoder, decoder)

 y_hats, logits = model(inputs, targets, teacher_forcing_ratio=teacher_forcing_ratio)
```
  
### Performance Test
```python
model = torch.load('weight_path')
model.set_beam_size(k=5)

y_hat, _ = model(inputs, targets, teacher_forcing_ratio=0.0, use_beam_search=True)
```

## Installation
This project recommends Python 3.7 or higher.   
I recommend creating a new virtual environment for this project (using virtualenv or conda).  

### Prerequisites
  
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* PyTorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sh951011/PyTorch-Seq2seq/issues) on Github.  
or Contacts sh951011@gmail.com please.
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  

### Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
  
## Reference
[[1]   IBM pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)   
[[2]   Pytorch-End-to-End-Korean-Speech-Recognition](https://github.com/sooftware/End-to-End-Korean-Speech-Recognition)      
[[3]   RNN Language Model](https://github.com/sooftware/char-rnnlm)      
  
## License
```
Copyright (c) 2020 sooftware

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
