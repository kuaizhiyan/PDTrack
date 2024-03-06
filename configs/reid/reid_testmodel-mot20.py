_base_ = [
    './reid_testmodel.py'
]

model = dict(head=dict(num_classes=1701))
data_root = 'data/MOT20/'
train_dataloader = dict(dataset=dict(data_root=data_root,
                                     triplet_sampler=dict(num_ids=16, ins_per_id=4)),
)
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader
# train, val, test setting
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=1400000,
    val_interval=100000,
)

