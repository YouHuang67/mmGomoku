batch_size = 16
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(
        type='GeneratedActionsV0Dataset',
        pipeline=[
            dict(type='GomokuAugmentation'),
            dict(type='PackGomokuInputs',
                 meta_keys=('history', 'action',
                            'ori_history', 'ori_action',
                            'homogenous_index',
                            'left_step')),  # additional key
        ], data_root='work_dirs/data_generation/main_v0'
    )
)
