import torch
from torch import Tensor

from mmseg.models.builder import MODELS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.utils import SampleList


class BaseTrainer(BaseSegmentor):

    def __init__(self,
                 backbone,
                 neck,
                 decode_head,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 pretrained=None,
                 init_cfg=None):
        super(BaseTrainer, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.decode_head = MODELS.build(decode_head)
        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self, inputs, targets, data_samples, mode='loss', **kwargs):  # noqa
        if mode == 'loss':
            return self.forward_train(inputs, targets, data_samples, **kwargs)
        elif mode == 'predict':
            return self.forward_test(inputs, targets, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               f'Only supports loss and predict')

    def forward_train(self, inputs, targets, data_samples, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def forward_test(self, inputs, targets, data_samples, **kwargs):
        raise NotImplementedError

    def _forward(self, inputs, data_samples=None):
        raise NotImplementedError

    def encode_decode(self, inputs, batch_data_samples):
        raise NotImplementedError

    def extract_feat(self, inputs):
        raise NotImplementedError

    def predict(self, inputs, data_samples=None):
        raise NotImplementedError

    def loss(self, inputs, data_samples):
        raise NotImplementedError
