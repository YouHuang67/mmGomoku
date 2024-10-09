_base_ = ['../../_base_/optimizer_adamw_160k.py',
          '../../datasets/train_simaug_vct_actions.py',
          '../gkformer_v0_d16_c64_h4_r1.py']
model = dict()
batch_size = 16
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size // 4
)
