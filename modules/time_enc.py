"""
Time Encoding Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import torch
from torch import nn
import numpy as np
from torch import Tensor
from torch.nn import Linear


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int, requires_grad: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

        if not requires_grad:
            self.lin.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, out_channels, dtype=np.float32))).reshape(out_channels, -1))
            self.lin.bias = nn.Parameter(torch.zeros(out_channels))
            self.lin.weight.requires_grad = False
            self.lin.bias.requires_grad = False

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()
