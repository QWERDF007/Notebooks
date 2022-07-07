_base_ = './deeplabv3plus_r50-d8_512x512_40k_voc12aug.py'

num_classes = 3
classes = ('background', 'unknown', 'fg')
palette = [[0, 0, 0], [128, 128, 128], [192, 128, 128]]

dataset_type = 'CustomDataset'
data_root = '/home/myuser/Workspaces/data/trimap/'

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained='http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
    backbone=dict(
        _delete_=True,
        type='Xception',
        # depth=65,
        out_indices=(1, 2, 10, 20),
        dilations=(1, 2, 2, 4),
        strides=(2, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=256,
        dilations=(1, 6, 12, 18),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=728,
        in_index=1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/masks',
        img_suffix='.png',
        ignore_index=255,
        split=None,
        classes=classes,
        palette=palette,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(2052, 513), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(513, 513), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(513, 513), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/masks',
        img_suffix='.png',
        ignore_index=255,
        split=None,
        classes=classes,
        palette=palette,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2052, 513),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/masks',
        img_suffix='.png',
        ignore_index=255,
        split=None,
        classes=classes,
        palette=palette,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2052, 513),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

workflow = [('train', 1)]

runner = dict(type='EpochBasedRunner', max_iters=None, max_epochs=200)
checkpoint_config = dict(by_epoch=True, interval=1)
evaluation = dict(interval=2, metric='mIoU', pre_eval=True)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=True)

log_config = dict(
    interval=50, 
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=True)],
    )
