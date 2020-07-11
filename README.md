# Seq2seq-pytorch
  
### Sequence to Sequence Architecture in Pytorch  
[<img src="https://github.com/gentaiscool/end2end-asr-pytorch/raw/master/img/pytorch-logo-dark.png" height=18>](https://pytorch.org/) <img src="https://img.shields.io/badge/License-Apache--2.0-yellow" height=20>
  
### [**Documentation**](https://sooftware.github.io/Seq2seq-pytorch/)
  
`Seq2seq-pytorch` is a framework for attention based sequence-to-sequence models implemented in [Pytorch](https://pytorch.org/).  
The framework has modularized and extensible components for seq2seq models, training, inference, checkpoints, etc.  
  
## Intro
  
<img src="https://user-images.githubusercontent.com/42150335/87226235-6517ec00-c3cd-11ea-9c96-021f6b827a5e.png" width=400>
  
Seq2seq turns one sequence into another sequence. It does so by use of a recurrent neural network (RNN) or more often LSTM or GRU to avoid the problem of vanishing gradient. The context for each item is the output from the previous step. The primary components are one encoder and one decoder network. The encoder turns each item into a corresponding hidden vector containing the item and its context. The decoder reverses the process, turning the vector into an output item, using the previous output as the input context.
  


## Installation
This project recommends Python 3.6 or higher.   
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
  
[[2]   Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)  
  
[[3]   Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)  
   
## Citing
```
@github{
  title={Seq2seq-pytorch},
  author={Soohwan Kim},
  publisher={github},
  docs={https://sooftware.github.ioSeq2seq-pytorch/},
  url={https://github.com/sooftware/Seq2seq-pytorch},
  year={2020}
}
```
