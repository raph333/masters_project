"""
Parameters used in presumably every experiment
"""
import torch
from torch import nn

CONFIG = {
    'optimizer': torch.optim.Adam,
    'batch_size': 64,
    'ds_fractions': [0.7, 0.15, 0.15],  # training-, validation-, test-set fractions
    'random_seed': 42,
    'loss_function': nn.L1Loss()
}
