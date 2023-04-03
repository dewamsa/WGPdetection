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
