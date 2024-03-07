# 自定义数据训练

## 数据准备

```
├── data
│   ├── my_dataset
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{img_suffix}
│   │   │   │   ├── yyy{img_suffix}
│   │   │   │   ├── zzz{img_suffix}
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{seg_map_suffix}
│   │   │   │   ├── yyy{seg_map_suffix}
│   │   │   │   ├── zzz{seg_map_suffix}
│   │   │   ├── val
```

`img_dir` 存放 RGB 图像的目录，`ann_dir` 存放标注的目录。每个训练对的前缀相同，后缀可以在配置文件中配置。

**注意：**标注是跟图像同样的形状 (H, W) 单通道图像，其中的像素值的范围是 `[0, num_classes - 1]`

## 配置文件定义

从 `mmsegmentation` 项目中的 `configs/` 目录下复制对应模型的配置文件，重命名后，修改对应的字段，或继承基础配置文件后自定义配置。

`mmsegmentation` 在配置文件中整合了继承和模块化，可以通过继承已经存在的配置文件来快速搭建模型，直接在配置文件中重载相应的字段，剩余字段将使用继承得到的。配置文件的详解 [Here](https://mmsegmentation.readthedocs.io/zh_CN/latest/tutorials/config.html#)。可以通过 `tools/print_config.py` 来查看完整的配置，可以传递参数 `--options xxx.yyy=zzz` 查看更新后的配置。

示例：

`configs/segformer/segformer_mit-b5_512x512_160k_ade20k.py`：

```python
_base_ = ['./segformer_mit-b0_512x512_160k_ade20k.py']

# model settings
model = dict(
    pretrained='pretrain/mit_b5.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
```

通过 `tools/print_config.py` 打印 `configs/segformer/segformer_mit-b5_512x512_160k_ade20k.py` 的结果：

```python
Config:
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/mit_b5.pth',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
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
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
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
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
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
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)
```

**需要修改的字段：**

model：`dict`，定义模型，继承相应的配置，修改 `num_classes`，如果单卡训练则将 `norm_cfg` 的 `type` 改成 `BN`，否则报错。

data：`dict`，指定 `train`、`val` 和 `test` 的数据。

- `data_root`：数据根目录
- `img_dir`：图像目录，相对于数据根目录
- `img_suffix`：图像后缀，默认 `.jpg`
- `ann_dir`：标注目录，相对于数据根目录
- `seg_map_suffix`：标注后缀，默认 `.png`
- `split`：指定加载部分，仅包含要加载项的stem，`None` 则加载 `img_dir`/`ann_dir` 下的全部图像
- `pipeline`：[自定义数据流程](https://mmsegmentation.readthedocs.io/zh_CN/latest/tutorials/data_pipeline.html)
- `classes`：用于可视化结果显示，字符串列表
- `palette`：颜色表，用于可视化

**可修改字段：**

runner：两种选择，`IterBasedRunner` 和 `EpochBasedRunner`，通过 `max_iters` 或 `max_epochs` 指定训练多少个 `iter` 或者 `epoch`。

optimizer：优化器配置。

optimizer_config：

lr_config：学习率配置。根据[线性扩展原则](https://arxiv.org/pdf/1706.02677.pdf)，需要设置学习率正比于 `batch size`。

checkpoint_config：保存 checkpoints 的相关配置，详细 [Here](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook)

log_config：日志配置，可在 `hooks` 添加 `tensorboard`、`wandb` 等可视化的 hooks。[Here](https://mmsegmentation.readthedocs.io/zh_CN/latest/tutorials/customize_runtime.html#log-config)

evaluation：指定评估时所用的指标，以及评估的间隔。

load_from：训练时指定加载的训练文件，仅加载权重。

resume_from：恢复训练时所指定加载的恢复文件，加载权重和优化器状态，包含迭代次数。

workflow：工作流。

## 训练

单卡：

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

示例：

```shell
python tools/train.py configs/segformer/segformer_mit-b5_512x512_160k_ade20k.py
```

多卡：

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

示例：

```shell
./tools/dist_train.sh configs/segformer/segformer_mit-b5_512x512_160k_ade20k.py 4 
```

**可选参数：**

- --no-validate：不在训练中进行评估
- --work-dir：日志和模型文件保存目录
- --resume-from：从 checkpoint 恢复训练，加载权重及优化器状态
- --load-from：从 checkpoint 加载权重

## 测试

**单卡：**

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
```

**多卡：**

```shell
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

**可选参数：**

- --out：pickle格式的输出结果的文件名，`.pkl` 后缀
- --eval：评估指标
- --show：可视化分割结果
- --show-dir：可视化结果保存到指定目录
- --eval-options：评估时的可选参数





<!-- 完成标志, 看不到, 请忽略! -->
