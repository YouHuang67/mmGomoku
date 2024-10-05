test_cfg = dict(_delete_=True, type='SimpleTestLoop')
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VCTActionDataset',
        train=False,
        pipeline=[
            dict(type='GomokuAugmentation'),
            dict(type='PackGomokuInputs')
        ], data_root='data/gomocup'
    )
)
test_evaluator = dict()
