import warnings
from functools import partial

import torch
import torch.nn as nn
from mmseg.registry import MODELS


class PlainBlock(nn.Module):

    def __init__(self, dim, kernel_size, norm='bn'):
        super(PlainBlock, self).__init__()
        if norm == 'bn':
            norm_cls = nn.BatchNorm2d
        elif norm == 'gn' or norm == 'ln':
            norm_cls = partial(nn.GroupNorm, num_groups=1)
        else:
            raise NotImplementedError(f'norm={norm} is not supported')
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.norm1 = norm_cls(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.norm2 = norm_cls(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x += identity
        x = self.act(x)
        return x


class BottleneckBlock(nn.Module):

    def __init__(self, dim, kernel_size, norm='bn', downsample_ratio=0.25):
        super(BottleneckBlock, self).__init__()
        if norm == 'bn':
            norm_cls = nn.BatchNorm2d
        elif norm == 'gn' or norm == 'ln':
            norm_cls = partial(nn.GroupNorm, num_groups=1)
        else:
            raise NotImplementedError(f'norm={norm} is not supported')
        hidden_dim = int(dim * downsample_ratio)
        self.conv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.norm1 = norm_cls(hidden_dim)
        self.conv2 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size,
            padding=kernel_size // 2)
        self.norm2 = norm_cls(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, dim, 1)
        self.norm3 = norm_cls(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x += identity
        x = self.act(x)
        return x


@MODELS.register_module()
class SimpleResNet(nn.Module):

    def __init__(self,
                 depth,
                 channels,
                 kernel_size=3,
                 norm='bn',
                 block='plain',
                 in_channels=3,
                 **kwargs):
        super(SimpleResNet, self).__init__()
        if block == 'plain':
            block_cls = PlainBlock
        elif block == 'bottleneck':
            block_cls = BottleneckBlock
        else:
            raise NotImplementedError(f'block={block} is not supported')
        self.conv = nn.Conv2d(in_channels, channels, 1)
        self.blocks = nn.Sequential()
        for _ in range(depth):
            self.blocks.append(block_cls(channels, kernel_size, norm, **kwargs))

    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        return x
