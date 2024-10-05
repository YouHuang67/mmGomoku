_base_ = ['../../_base_/optimizer_adamw_40k.py',
          '../../datasets/train_simaug_vct_actions.py',
          '../resnet_plain_d16_c64_k3.py']
model = dict(type='PolicyValueTrainerV1')
batch_size = 16
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size // 4
)
