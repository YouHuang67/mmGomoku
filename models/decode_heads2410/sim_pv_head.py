import torch
import torch.nn as nn
from mmseg.models.builder import MODELS

from projects.cppboard import Board


@MODELS.register_module()
class SimplePolicyValueHead(nn.Module):

    def __init__(self,
                 depth,
                 channels,
                 kernel_size,
                 norm='bn'):
        super(SimplePolicyValueHead, self).__init__()
        if norm == 'bn':
            norm_cls = nn.BatchNorm2d
        elif norm == 'gn' or norm == 'ln':
            norm_cls = nn.GroupNorm
        else:
            raise NotImplementedError(f'norm={norm} is not supported')
        self.action_stem = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size,
                          padding=kernel_size // 2),
                norm_cls(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(depth)
        ])
        self.action_stem.append(nn.Conv2d(channels, 1, 1))
        self.value_stem = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size,
                          padding=kernel_size // 2),
                norm_cls(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(depth)
        ])
        self.value_stem.append(nn.AdaptiveAvgPool2d(1))
        self.value_stem.append(nn.Conv2d(channels, 1, 1))

    def forward(self, x):
        action_probs = self.action_stem(x)
        values = self.value_stem(x)
        return dict(action_probs=action_probs, values=values.flatten())


@MODELS.register_module()
class FCPolicyValueHead(nn.Module):

    def __init__(self, size=Board.BOARD_SIZE):
        super(FCPolicyValueHead, self).__init__()
        in_dim = size ** 2
        self.size = size
        self.bn = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(in_dim, 1, bias=True)

    def forward(self, x):
        out = torch.tanh(self.fc(self.bn(x.flatten(1))))
        action_probs = x.reshape(-1, self.size, self.size)
        values = out.flatten()
        return dict(action_probs=action_probs, values=values)
