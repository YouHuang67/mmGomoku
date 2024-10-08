from pathlib import Path

import mmengine
from mmengine.dist import barrier, get_dist_info
from mmseg.registry import DATASETS

from .base import BaseGomokuDataset


@DATASETS.register_module()
class GeneratedActionsV0Dataset(BaseGomokuDataset):

    def __init__(self,
                 pipeline,
                 data_root='work_dirs/data_generation/main_v0',
                 meta_root='data/meta-info/generated_actions_v0.json',
                 serialize_data=True,
                 lazy_init=False,
                 max_refetch=1000):
        super(GeneratedActionsV0Dataset, self).__init__(
            pipeline=pipeline,
            data_root=data_root,
            meta_root=meta_root,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    def load_data_list(self):
        data_root = Path(self.data_root)
        meta_root = Path(self.meta_root)
        if not meta_root.is_file():
            if get_dist_info()[0] == 0:
                history_list = []
                paths = sorted(data_root.rglob('?????.json'), key=str)
                for path in paths:
                    history_list.append(mmengine.load(path)['history'])
                meta_root.parent.mkdir(parents=True, exist_ok=True)
                mmengine.dump(dict(history_list=history_list), str(meta_root))
        barrier()
        data_list = []
        history_list = mmengine.load(meta_root)['history_list']
        for history in history_list:
            for step in range(len(history)):
                left_step = len(history) - step
                data_list.append(dict(history=history[:step],
                                      action=history[step],
                                      left_step=left_step))
        return data_list
