# 自定义数据训练 SwinTransformer

[MMSegmentation 中文文档](https://mmsegmentation.readthedocs.io/zh_CN/latest/)

## 平台&环境

- Ubuntu 16.04.6 LTS

- Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
- GeForce GTX 1080 Ti
- 32 GB RAM
- CUDA 10.2

创建 conda 环境

```shell
conda create -n openmmlab python=3.8
conda activate openmmlab
```

安装 pytorch

```shell
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

安装 mmcv

```shell
python -m pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
```

安装 MMSegmentation

```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
python -m pip install -e .
```

## 数据准备

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

`img` 存放 rgb 图像，`mask` 存放标注图像，标注图像是与 rgb 图像同样大小的，只包含 `[0, num_classes-1]` 像素值的图像。

## 模型准备

下载预训练模型：[upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K_20210526_211650-762e2178.pth)

将下载好的模型文件保存到 `pretrain/` 目录下，并重命名为 `swin_base_patch4_window7_224_22k.pth`

## 配置文件

复制一份 `configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_22K.py` 改名为 `configs/swin/upernet_swin_base_patch4_window7_512x512_160k_custom.py`，然后重载里面的相应的配置

```python
_base_ = [
    './upernet_swin_base_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py'
]

# 类别名称和对应的调色板
classes = ["bg", "jewelry", "headwear", "other", "tattoo", "mask", "earphone", "hand", "bag"]
palette = [[0,0,0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0]]

# 数据类型和数据根目录，根目录与img_dir/ann_dir结合构成完整目录
dataset_type = 'CustomDataset'
data_root = 'data/custom_dataset'

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
    samples_per_gpu=2,  # 单个 GPU 的 Batch size
    workers_per_gpu=4,  # 单个 GPU 分配的数据加载线程数
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
        num_classes=9,  # 类别数
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0
        ),
    ),
)
```

**注意：** 需要修改 `classes`，`palette`， `dataset_type`， `data_root`，`data` 和 `num_classes`。

如果需要 `tensorboard`、`wandb` 可视化追踪，需要安装对应的包，并在配置文件中的 `log_config` 的 `hooks` 中添加 `TensorboardLoggerHook` 和 `WandbLoggerHook`。

```shell
python -m pip install wandb tensorboard
```

- [WandbLoggerHook](https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/hooks/logger/wandb.html)
- [TensorboardLoggerHook](https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/hooks/logger/tensorboard.html)

## 训练

```shell
python tools/train.py configs/swin/upernet_swin_base_patch4_window7_512x512_160k_custom.py  --work-dir ${WORK_DIR}
```

日志和 Checkpoints 会保存到指定的 `WORK_DIR` 内。