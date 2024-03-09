_base_ = [
    '../_base_/datasets/mot_challenge_reid.py', '../_base_/default_runtime.py'
]


model = dict(
    type='BaseReID',
    data_preprocessor=dict(
        type='ReIDDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    backbone=dict(
        type='mmpretrain.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint="https://download.openmmlab.com/mmclassification/v0/fp16/resnet50_batch256_fp16_imagenet_20210320-b3964210.pth"
        )
        ),
    neck=
 dict(
    type='TestModel_neck',
        num_queries=33,
        with_encoder=False,
        with_decoder=True,
        with_conditionpos=True,
        with_agg=False,
        channel_mapper=dict(
            in_channels=[2048],   # the output feature map dim
            out_channels=256,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='LeakyReLU')
            ),
        encoder=dict(
            num_layers=4,
            layer_cfg=dict(  # DetrTransformerEncoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True)))
        ),
        decoder=dict(
            num_layers=4,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1,
                    cross_attn=False),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1,
                    cross_attn=True))
        ),
        positional_encoding=dict(num_feats=128, normalize=True),    # num_feats = len(x)+len(y)
        # init_cfg=dict(
        #     type="Pretrained",
        #     checkpoint="/home/kzy/project/mmdetection/work_dirs/reid_testmodel/iter_2000.pth"
        # )

),
    head=dict(
        type='LinearReIDHead',
        num_fcs=1,
        in_channels=256,
        fc_channels=1024,
        out_channels=128,
        num_classes=380,
        loss_cls=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
        loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
        norm_cfg=dict(type='BN1d'),
        # norm_cfg=dict(type='IN1d'),
        act_cfg=dict(type='ReLU')),
    # init_cfg=dict(
    #     type='Pretrained',
    #     checkpoint=  # noqa: E251
    #     'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth'  # noqa: E501
    # )
    )

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=None,
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=6,
        by_epoch=True,
        milestones=[5],
        gamma=0.1)
]


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=None,
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,          # IterBased 修改 !
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=6,
        # by_epoch=True,
        by_epoch=False,          # IterBased 修改 !
        milestones=[5],
        gamma=0.1)
]

train_dataloader = dict(
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(
        triplet_sampler=dict(num_ids=16, ins_per_id=4),
))
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=2000,
        save_best='auto',
        max_keep_ckpts=3,  # only keep latest 5 checkpoints
    )
)

log_processor = dict(by_epoch=False)
# train, val, test setting
# train_cfg = dict(
#     type='EpochBasedTrainLoop',
#     max_epochs=80,
#     val_interval=1)
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=100000,
    val_interval=10000,
)
#     type='EpochBasedTrainLoop',
#     max_epochs=80,
#     val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
