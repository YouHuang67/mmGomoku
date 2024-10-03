from pathlib import Path

import mmengine
from mmengine.dist import barrier, get_dist_info
from mmseg.registry import TRANSFORMS, DATASETS

from .base import BaseGomokuDataset


def load_vct_actions(file='data/gomocup/vct_actions.json'):
    data = mmengine.load(file)
    actions = [list(map(tuple, acts)) for acts in data['actions']]
    vct_actions = [
        (index, step, tuple(act)) for index, step, act in data['vct_actions']
    ]
    return actions, vct_actions


@DATASETS.register_module()
class VCTActionDataset(BaseGomokuDataset):

    def __init__(self,
                 pipeline,
                 train,
                 data_root='data/gomocup',
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000):
        self.train = train
        super(VCTActionDataset, self).__init__(
            pipeline=pipeline,
            data_root=data_root,
            meta_root=None,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def load_data_list(self):
        mode = 'train' if self.train else 'eval'
        data_root = Path(self.data_root)
        actions, vct_actions = \
            load_vct_actions(str(data_root / 'vct_actions.json'))
        split = mmengine.load(data_root / 'split.json')[mode]
        data_list = []
        for index in split:
            idx, step, action = vct_actions[index]
            data_list.append(dict(history=actions[idx][:step], action=action))
        return data_list
