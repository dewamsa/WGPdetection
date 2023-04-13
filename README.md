# WGPdetection

This repo is the official implementation of Efficient Multi-Branch Convolutional Neural Networks (**accepted** on Computers and Electronics in Agriculture). 

## Requirements
Install the libraries below to use the model.
1. PyTorch and Torchvision => https://pytorch.org/ (the latest and GPU version are recommended)
2. Segmentation Models Pytorch (https://github.com/qubvel/segmentation_models.pytorch)

## Models
Our model can be loaded as follows.
```python
from model.model import MBGDRA

model = MBGDRA(
    c=2,        # number of classes
    satt = [True, True, True],     # True if you want to use spatial attention on the decoder layer
    catt = [True, False, True],     # True if you want to use channel attention on the decoder layer
)
```
## Citations
Our published paper can be found in here. \n
Please cite our paper if you find our code help your work.
'''python
@article{ARSA2023107830,
title = {Eco-friendly weeding through precise detection of growing points via efficient multi-branch convolutional neural networks},
journal = {Computers and Electronics in Agriculture},
volume = {209},
pages = {107830},
year = {2023},
issn = {0168-1699},
doi = {https://doi.org/10.1016/j.compag.2023.107830},
author = {Dewa Made Sri Arsa and Talha Ilyas and Seok-Hwan Park and Okjae Won and Hyongsuk Kim},
}
'''
