import random
import warnings
from pathlib import Path

import numpy as np
import mmengine
from mmseg.registry import TRANSFORMS, DATASETS
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import BaseDataElement

from projects.cppboard import Board
from .base import BaseGomokuDataset, BasePackInputs


def load_vct_actions(file='data/gomocup/vct_actions.json'):
    data = mmengine.load(file)
    actions = [list(map(tuple, acts)) for acts in data['actions']]
    vct_actions = [
        (index, step, tuple(act)) for index, step, act in data['vct_actions']
    ]
    return actions, vct_actions


@TRANSFORMS.register_module()
class GomokuAugmentation(BaseTransform):

    def transform(self, results):
        history = results.pop('history')
        action = results.pop('action')
        actions = Board.get_homogenous_actions(action)
        index = random.randint(0, len(actions) - 1)
        results['ori_history'] = history
        results['ori_action'] = action
        results['history'] = \
            [Board.get_homogenous_actions(act)[index] for act in history]
        results['action'] = actions[index]
        results['homogenous_index'] = index
        return results


@TRANSFORMS.register_module()
class PackGomokuInputs(BasePackInputs):

    def __init__(self,
                 meta_keys=('history', 'action',
                            'ori_history', 'ori_action',
                            'homogenous_index')):
        super(PackGomokuInputs, self).__init__(meta_keys)

    def transform(self, results):
        packed_results = dict()

        board = self.create_board(results['history'], size=Board.BOARD_SIZE)
        if not board.flags.c_contiguous:
            board = np.ascontiguousarray(board)
            board = to_tensor(board)
        else:
            board = to_tensor(board).contiguous()
        packed_results['inputs'] = board

        row, col = results['action']
        target = row * Board.BOARD_SIZE + col
        packed_results['targets'] = to_tensor(int(target))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
            else:
                warnings.warn(f'Key {key} not in results')
        data_sample = BaseDataElement()
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    @staticmethod
    def create_board(history, size=Board.BOARD_SIZE):
        board = np.zeros((3, size, size), dtype=int)

        if len(history) % 2 == 0:
            first_channel_indices = history[0::2]
            second_channel_indices = history[1::2]
        else:
            first_channel_indices = history[1::2]
            second_channel_indices = history[0::2]

        if first_channel_indices:
            r1, c1 = zip(*first_channel_indices)
            board[0, r1, c1] = 1
        if second_channel_indices:
            r2, c2 = zip(*second_channel_indices)
            board[1, r2, c2] = 1

        board[2] = 1 - (board[0] + board[1])
        return board


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
