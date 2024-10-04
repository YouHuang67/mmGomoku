data_preprocessor = dict(type='MMGKDataPreProcessor')

model = dict(
    type='PolicyValueTrainerV0',
    init_cfg=None,
    backbone=dict(
        type='SimpleResNet',
        depth=16,
        channels=64,
        kernel_size=3),
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
