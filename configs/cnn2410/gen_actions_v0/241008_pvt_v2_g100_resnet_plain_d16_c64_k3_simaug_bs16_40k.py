_base_ = ['../../_base_/optimizer_adamw_40k.py',
          '../../datasets/train_simaug_gen_actions_v0.py',
          '../resnet_plain_d16_c64_k3.py']
model = dict(
    type='PolicyValueTrainerV2',
    init_cfg=dict(_delete_=True,
                  type='Pretrained',
                  checkpoint='work_dirs/cnn2410/vct_actions/'
                             '241004_pvt_v0_resnet_plain_d16_c64_k3_'
                             'simaug_vct_actions_bs16_40k/iter_40000.pth'),
    train_cfg=dict(gamma=1.00)
)
batch_size = 16
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=batch_size // 4
)
