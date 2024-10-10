_base_ = ['../../_base_/optimizer_adamw_40k.py',
          '../../datasets/train_simaug_gen_actions_v0_main_v1.py',
          '../resnet_bottleneck_d16_c64_k7.py']
model = dict(
    type='PolicyValueTrainerV2',
    init_cfg=None,
    train_cfg=dict(gamma=1.00)
)
batch_size = 16
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size // 4
)
