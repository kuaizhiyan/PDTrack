_base_ = [
    './yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py',  # noqa: E501
]

dataset_type = 'MOTChallengeDataset'
detector = _base_.model
detector.pop('data_preprocessor')
del _base_.model

model = dict(
    type='StrongSORT',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(576, 1024),
                size_divisor=32,
                interval=10)
        ]),
    detector=detector,
    reid=dict(
        type='PartQuerier_neck',
            num_queries=129,
            embed_dims=256,
            with_agg=False,
            channel_mapper=dict(
                in_channels=[512,1024,2048],   # the output feature map dim
                out_channels=256,
                kernel_size=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='LeakyReLU')
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
            # init_config=dict(
            #     type='Pretrained',
            #     checkpoint='/home/kzy/project/mmdetection/Experiments/part_decoder_b64/best_reid-metric_mAP_iter_3000.pth'
            # )
        ),
    cmc=dict(
        type='CameraMotionCompensation',
        warp_mode='cv2.MOTION_EUCLIDEAN',
        num_iters=100,
        stop_eps=0.00001),
    tracker=dict(
        type='StrongSORTTracker',
        motion=dict(type='KalmanFilter', center_only=False, use_nsa=True),
        obj_score_thr=0.6,
        reid=dict(
            num_samples=None,
            img_scale=(256, 128),
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            match_score_thr=0.3,
            motion_weight=0.02,
        ),
        match_iou_thr=0.7,
        momentums=dict(embeds=0.1, ),
        num_tentatives=2,
        num_frames_retain=100),
    postprocess_model=dict(
        type='AppearanceFreeLink',
        checkpoint=  # noqa: E251
        'https://download.openmmlab.com/mmtracking/mot/strongsort/mot_dataset/aflink_motchallenge_20220812_190310-a7578ad3.pth',  # noqa: E501
        temporal_threshold=(0, 30),
        spatial_threshold=50,
        confidence_threshold=0.95,
    ))

train_pipeline = None
test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
            dict(type='Resize', scale=_base_.img_scale, keep_ratio=True),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='LoadTrackAnnotations'),
        ]),
    dict(type='PackTrackInputs')
]

train_dataloader = None
val_dataloader = dict(
    # Now StrongSORT only support video_based sampling
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=_base_.data_root,
        ann_file='annotations/half-val_cocoformat.json',
        data_prefix=dict(img_path='train'),
        # when you evaluate track performance, you need to remove metainfo
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

train_cfg = None
optim_wrapper = None

# evaluator
val_evaluator = dict(
    _delete_=True,
    type='MOTChallengeMetric',
    metric=['HOTA', 'CLEAR', 'Identity'],
    # use_postprocess to support AppearanceFreeLink in val_evaluator
    use_postprocess=True,
    postprocess_tracklet_cfg=[
        dict(
            type='InterpolateTracklets',
            min_num_frames=5,
            max_num_frames=20,
            use_gsi=True,
            smooth_tau=10)
    ])
test_evaluator = val_evaluator

default_hooks = dict(logger=dict(type='LoggerHook', interval=1))

del _base_.param_scheduler
del _base_.custom_hooks
