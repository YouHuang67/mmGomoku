import torch
import torch.nn as nn
from mmseg.models.builder import MODELS


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
