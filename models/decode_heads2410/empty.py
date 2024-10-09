import torch
import torch.nn as nn
from mmseg.models.builder import MODELS


@MODELS.register_module()
class EmptyHead(nn.Module):

    def forward(self, x):
        if isinstance(x, dict):
            return x
        elif isinstance(x, (list, tuple)):
            action_probs, values = x
            return dict(action_probs=action_probs, values=values.flatten())
        else:
            raise NotImplementedError(f'input with type {type(x)} is not supported')
