data_preprocessor = dict(type='MMGKDataPreProcessor')

model = dict(
    type='PolicyValueTrainerV0',
    init_cfg=None,
    backbone=dict(type='SEWideResnet16_1'),
    neck=None,
    decode_head=dict(type='FCPolicyValueHead'),
    train_cfg=dict(),
    test_cfg=dict()
)
find_unused_parameters = False
