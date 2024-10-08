data_preprocessor = dict(type='MMGKDataPreProcessor')

model = dict(
    type='PolicyValueTrainerV0',
    init_cfg=None,
    backbone=dict(
        type='SimpleResNet',
        block='bottleneck',
        depth=16,
        channels=64,
        kernel_size=7,
        downsample_ratio=1.0),
    neck=None,
    decode_head=dict(
        type='SimplePolicyValueHead',
        depth=3,
        channels=64,
        kernel_size=3),
    train_cfg=dict(),
    test_cfg=dict()
)
find_unused_parameters = False
