_base_ = ['../../_base_/optimizer_adamw_40k.py',
          '../../datasets/train_simaug_vct_actions.py',
          '../resnet_bottleneck_d16_c64_k7.py']
model = dict()
batch_size = 16
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size // 4
)
