from torch.utils.data.dataloader import DataLoader
from mmengine.config import Config
from mmengine.registry.root import FUNCTIONS
from mmengine.registry import DefaultScope
from mmseg.registry import DATASETS, DATA_SAMPLERS

import engine
DefaultScope.get_instance('mmseg', scope_name='mmseg')


def convert_config_to_dataloader(cfg_path, mode='test'):
    if mode not in ['test', 'train']:
        raise ValueError('mode must be one of ["test", "train"]')
    if mode == 'train':
        dataloader_cfg = Config.fromfile(cfg_path).train_dataloader
    else:
        dataloader_cfg = Config.fromfile(cfg_path).test_dataloader
    dataset = DATASETS.build(dataloader_cfg.pop('dataset'))
    sampler = DATA_SAMPLERS.build(
        dataloader_cfg.pop('sampler'), default_args=dict(dataset=dataset))
    collate_fn = FUNCTIONS.get('pseudo_collate')
    if hasattr(dataset, 'full_init'):
        dataset.full_init()
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_sampler=None,
        collate_fn=collate_fn,
        worker_init_fn=None,
        **dataloader_cfg)
    return dataloader, dataset
