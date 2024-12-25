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
## Acknowledgement
We would like to acknowledge the contributions of the following authors to this work:
1. Dewa Made Sri Arsa
2. Talha Ilyas
3. Seok-Hwan Park
4. Okjae Won
5. Hyongsuk Kim

This work is the intellectual property of the Core Research Institute of Intelligent Robots, Jeonbuk National University, Korea.

This work was supported in part by the Crop and Weed Project administered through the Agricultural Science and Technology Development Cooperation Research Program (PJ015720) and by the National Research Foundation of Korea (NRF) grant funded by the Korea government (NRF-2019R1A2C1011297 and NRF-2019R1A6A1A09031717).

## Citations
Our published paper can be found in <a href="https://www.sciencedirect.com/science/article/pii/S0168169923002181">HERE</a>.

Please cite our paper if you find our code help your work.
```latex
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
```
