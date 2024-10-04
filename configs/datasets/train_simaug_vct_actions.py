batch_size = 16
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(
        type='VCTActionDataset',
        train=True,
        pipeline=[
            dict(type='GomokuAugmentation'),
            dict(type='PackGomokuInputs')
        ], data_root='data/gomocup'
    )
)
