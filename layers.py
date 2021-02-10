# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

FLAG_RELU_INPLACE = True


def get_norm(name: Optional[str], **kwargs):
    if name is None or name.lower() in ['', 'id', 'identity']:
        return nn.Identity()
    elif name.lower() in ['bn', 'batch']:
        return nn.BatchNorm2d(num_features=kwargs.get('num_features'))
    else:
        raise NotImplementedError


def get_act(name: Optional[str], **kwargs):
    if name is None or name.lower() in ['', 'id', 'identity']:
        return nn.Identity()
    elif name.lower() in ['relu']:
        return nn.ReLU(inplace=FLAG_RELU_INPLACE)
    else:
        raise NotImplementedError


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        norm: Optional[str] = None,
        act: Optional[str] = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        )
        self.norm = get_norm(norm, num_features=out_channels)
        self.act = get_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Residual(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + super().forward(x)
