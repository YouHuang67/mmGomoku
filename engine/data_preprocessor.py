import torch

from mmengine.model import BaseDataPreprocessor
from mmseg.registry import MODELS


@MODELS.register_module()
class MMGKDataPreProcessor(BaseDataPreprocessor):

    def forward(self, data, training=False):
        data = self.cast_data(data)
        data_samples = data.get('data_samples')
        inputs = data['inputs']
        inputs = [_input.float() for _input in inputs]
        inputs = torch.stack(inputs, dim=0)
        targets = data['targets']
        targets = [_target.long() for _target in targets]
        targets = torch.stack(targets, dim=0).flatten()
        return dict(inputs=inputs, targets=targets, data_samples=data_samples)
