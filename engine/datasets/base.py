import logging

import numpy as np
from mmengine.dataset import BaseDataset, Compose
from mmengine.logging import print_log
from mmcv.transforms.base import BaseTransform


class BasePackInputs(BaseTransform):

    def __init__(self, meta_keys):
        self.meta_keys = meta_keys

    def transform(self, results):
        """
        :param results:
        :return: a dict
        """
        raise NotImplementedError

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


class InvalidGomokuSampleError(Exception):

    pass


class NotFoundValidGomokuSampleError(Exception):

    pass


class BaseGomokuDataset(BaseDataset):

    default_meta_root = ''

    def __init__(self,
                 pipeline,
                 data_root,
                 meta_root=None,
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000):

        self.backend_args = None
        self.data_root = data_root
        self.meta_root = meta_root or self.default_meta_root
        self.filter_cfg = None
        self._indices = None
        self.serialize_data = serialize_data
        self.max_refetch = max_refetch
        self.data_list = []
        self.data_bytes: np.ndarray

        self.pipeline = Compose(pipeline)
        if not lazy_init:
            self.full_init()

    def load_data_list(self):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        info = None
        for _ in range(self.max_refetch):
            try:
                data = self.prepare_data(idx)
            except InvalidGomokuSampleError as error:
                info = str(error)
                continue
            return data
        else:
            raise NotFoundValidGomokuSampleError(
                f'Cannot find valid Gomoku sample after '
                f'{self.max_refetch} retries, due to {info}')

    def get_data_info(self, idx):
        data_info = super(BaseGomokuDataset, self).get_data_info(idx)
        data_info['dataset'] = \
            self.__class__.__name__.lower().replace('dataset', '')
        return data_info

    @property
    def metainfo(self):
        return dict()
