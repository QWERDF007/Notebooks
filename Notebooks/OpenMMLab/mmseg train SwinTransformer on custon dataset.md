# Train SwinTransformer on custom

[MMSegmentation Doc](https://mmsegmentation.readthedocs.io/en/latest/)

## platform & environment

- Ubuntu 16.04.6 LTS

- Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
- GeForce GTX 1080 Ti
- 32 GB RAM
- CUDA 10.2

create conda 

```shell
conda create -n openmmlab python=3.8
conda activate openmmlab
```

install pytorch

```shell
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

install mmcv

```shell
python -m pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
```

install MMSegmentation

```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
python -m pip install -e .
```

## prepare data

```shell
data/
└── custom_dataset
    ├── img
    │   ├── train
    │   └── val
    └── mask
        ├── train
        └── val
```

A training pair will consist of the files with same suffix in img/mask directory.

## Download Pretrained Model

Download Model：[upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_20210526_211650-762e2178.pth)

save model to `pretrain/`, rename the model file as `swin_base_patch4_window7_224_22k.pth`

## Config

copy `configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K.py`, and rename as `configs/swin/upernet_swin_base_patch4_window7_512x512_160k_custom.py`. (or any name you want). Then, overwirte this file as below:

```python
_base_ = [
    './upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py'
]

# classes name and palette
classes = ["bg", "jewelry", "headwear", "other", "tattoo", "mask", "earphone", "hand", "bag"]
palette = [[0,0,0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0]]

# dataset type
dataset_type = 'CustomDataset'
# dataset root
data_root = 'data/custom_dataset'

# add/remove any process you want,
train_pipeline=  [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
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
            dict(type='Collect', keys=['img']),
        ]
    ),
]


data = dict(
    samples_per_gpu=2,  # batch size for each gpu
    workers_per_gpu=4,  # 
    train=dict(
        type=dataset_type,
        data_root=data_root,
        classes=classes,
        palette=palette,
        img_dir='img/train',
        img_suffix='.jpg',
        ann_dir='mask/train',
        seg_map_suffix='.png',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        classes=classes,
        palette=palette,
        img_dir='img/val',
        img_suffix='.jpg',
        ann_dir='mask/val',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        classes=classes,
        palette=palette,
        img_dir='img/train',
        img_suffix='.jpg',
        ann_dir='mask/val',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
    ),
)

log_config = dict(
    interval=50, 
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook', by_epoch=True),
        # dict(type='WandbLoggerHook', by_epoch=True, init_kwargs=dict(project='mmsegmentation', name='swin_custom')),
    ]
)

# single gpu
norm_cfg = dict(type='BN', requires_grad=True)
# multi gpu
# norm_cfg=dict(type='SyncBN', requires_grad=True)

model = dict(
    pretrained='pretrain/swin_base_patch4_window7_224_22k.pth',
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=9,  # num
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0
        ),
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=9, # 辅助头的类别数
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
)
```

[Learn about Configs](https://mmsegmentation.readthedocs.io/en/latest/tutorials/config.html)

you need to correctly set `classes`, `palette`,  `dataset_type`,  `data_root`, `data` and `num_classes`。

And you can use `TensorboardLoggerHook` and `WandbLoggerHook` to trace you model and visualize the metrics.



install wandb & tensorboard

```shell
python -m pip install wandb tensorboard
```

- [WandbLoggerHook](https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/hooks/logger/wandb.html)
- [TensorboardLoggerHook](https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/hooks/logger/tensorboard.html)

## Train

```shell
python tools/train.py configs/swin/upernet_swin_base_patch4_window7_512x512_160k_custom.py  --work-dir ${WORK_DIR}
```

logs and checkpoints will be saved on `WORK_DIR`