# 自定义数据训练

## 数据准备

有三种方法在 `mmdetection` 中支持新的数据集：

- 将数据集重新组织为COCO格式。
- 将数据集重新组织为中间格式。
- 实现一个新的数据集。

**中间格式的目录结构：**

```shell
data
├── ann.pkl
└── images
```

`images` 为存放图像的目录，目录名任意。`ann.pkl` 是标注文件，文件名任意，后缀 `.pkl`，是一个 `list` 包含多个 `dict`，每个 `dict` 为一张图像的标注，结构如下：

```python
[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        'ann': {
            'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
            'labels': <np.ndarray> (n, ),
            'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
            'labels_ignore': <np.ndarray> (k, 4) (optional field)
        }
    },
    ...
]
```

可以通过 `pickle` 将标注内容序列化到文件。

[背景处理](https://github.com/open-mmlab/mmdetection/issues/2956)：`filter_gt_empty` 设为 `False`，并把 `bboxes` 设为 `[0,0,0,0]`。[Here](https://github.com/open-mmlab/mmdetection/blob/8951a8a70f36d442d8f5a9ecfe6361bbd6dc2adb/mmdet/datasets/coco.py#L133)

## 配置文件定义

从 `mmdetection` 项目中的 `configs/` 目录下复制对应模型的配置文件，重命名后，修改对应的字段，或继承基础配置文件后自定义配置。

`mmdetection` 在配置文件中整合了继承和模块化，可以通过继承已经存在的配置文件来快速搭建模型，直接在配置文件中重载相应的字段，剩余字段将使用继承得到的。配置文件的详解 [Here](https://mmdetection.readthedocs.io/zh_CN/latest/tutorials/config.html)。可以通过 `tools/misc/print_config.py` 来查看完整的配置。

**示例：**

`configs/yolox/yolox_l_8x8_300e_coco.py`：

```python
_base_ = './yolox_s_8x8_300e_coco.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))
```

通过 `tools/misc/print_config.py` 打印  `configs/yolox/yolox_l_8x8_300e_coco.py` 的结果：

```python
Config:
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=15,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(
        type='SyncRandomSizeHook',
        ratio_range=(14, 26),
        img_scale=(640, 640),
        interval=10,
        priority=48),
    dict(type='SyncNormHook', num_last_epochs=15, interval=10, priority=48),
    dict(type='ExpMomentumEMAHook', resume_from=None, priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='YOLOX',
    backbone=dict(type='CSPDarknet', deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3),
    bbox_head=dict(
        type='YOLOXHead', num_classes=80, in_channels=256, feat_channels=256),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (640, 640)
train_pipeline = [
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomAffine', scaling_ratio_range=(0.1, 2),
        border=(-320, -320)),
    dict(
        type='MixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False),
    pipeline=[
        dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
        dict(
            type='RandomAffine',
            scaling_ratio_range=(0.1, 2),
            border=(-320, -320)),
        dict(
            type='MixUp',
            img_scale=(640, 640),
            ratio_range=(0.8, 1.6),
            pad_val=114.0),
        dict(
            type='PhotoMetricDistortion',
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Resize', keep_ratio=True),
        dict(type='Pad', pad_to_square=True, pad_val=114.0),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    dynamic_scale=(640, 640))
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=(640, 640), pad_val=114.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file='data/coco/annotations/instances_train2017.json',
            img_prefix='data/coco/train2017/',
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=[
            dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-320, -320)),
            dict(
                type='MixUp',
                img_scale=(640, 640),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', keep_ratio=True),
            dict(type='Pad', pad_to_square=True, pad_val=114.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        dynamic_scale=(640, 640)),
    val=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Pad', size=(640, 640), pad_val=114.0),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Pad', size=(640, 640), pad_val=114.0),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
interval = 10
evaluation = dict(interval=10, metric='bbox')
```

**需要修改的字段：**

model：`dict`，定义模型，继承相应的配置，修改 `num_classes`。

data：`dict`，指定 `train`、`val` 和 `test` 的数据。

- `img_prefix`：用于指定图像的存储目录。
- `classes`：用于指定对应的类别的名称，用于结果可视化时显示标签。 
- `pipeline`：用于[自定义数据预处理流程](https://mmdetection.readthedocs.io/zh_CN/latest/tutorials/data_pipeline.html)。
- `type`：`CustomDataset`，或者可以使用自定义数据类型，在 `datasets`目录下定义 `mydataset.py`，继承 `CustomDataset` 并重写 `load_annotations(self, ann_file)` 和 `get_ann_info(self, idx)`，并添加至 `__init__.py`，可参考 `coco.py` 等。

**可修改字段：**

runner：两种选择，`IterBasedRunner` 和 `EpochBasedRunner`，通过 `max_iters` 或 `max_epochs` 指定训练多少个 `iter` 或者 `epoch`。

optimizer：优化器配置

optimizer_config：

lr_config：学习率配置。根据[线性扩展原则](https://arxiv.org/pdf/1706.02677.pdf)，需要设置学习率正比于 `batch size`。[Here](https://mmdetection.readthedocs.io/zh_CN/latest/1_exist_data_model.html#id13)

checkpoint_config：保存 checkpoints 的相关配置，详细 [Here](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook)

log_config：日志配置，可在 `hooks` 添加 `tensorboard`、`wandb` 等可视化的 hooks

evaluation：指定评估时所用的指标，以及评估的间隔，`CustomDataset` 只支持 `mAP` 和 `recall`。

load_from：训练时指定加载的训练文件，仅加载权重。

resume_from：恢复训练时所指定加载的恢复文件，加载权重和优化器状态，包含迭代次数。

workflow：工作流。

## 训练

单卡训练：

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

示例：

```shell
python tools/train.py configs/yolox/yolox_l_8x8_300e_coco.py
```

train.py 参数：

- `--work-dir`：保存日志和模型文件的目录
- `--resume-from`：恢复训练时加载的 `checkpoint`
- `--no-validate `：训练时是否进行评估
- `--gpus`：训练使用的 gpu 数量（仅适用于非分布式训练）
- `--gpu-ids`：训练所使用的 gpu 的 id（仅适用于非分布式训练），索引从 0 开始
- `--seed`：随机种子
- `--deterministic`：设置 `cudnn` 寻找适合硬件的最快的算法，但会带来随机性
- `--options`：用于修改部分配置文件中的参数
- `--cfg-options`：用于修改配置文件中的参数
- `--launcher`：可选 `{none, pytorch, slurm, mpi}`
- `--local_rank`：

多卡训练：

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

示例：

```shell
./tools/dist_train.sh configs/yolox/yolox_l_8x8_300e_coco.py 4
```

可加 train.py 的部分参数

## 测试

单张图像测试：

```shell
python demo/image_demo.py \
    ${IMAGE_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--device ${GPU_ID}] \
    [--score-thr ${SCORE_THR}]
```

多卡测试：

```shell
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}]
```

**可选参数：**

- --out：pickle格式的输出结果的文件名
- --eval：评估指标
- --show：可视化分割结果
- --show-dir：可视化结果保存到指定目录
- --show-score-thr：如果指明，低于阈值的检测结果将被移除
- --cfg-options：配置文件的参数
- --eval-options：评估时的可选参数
- --async-test：异步推理

[more](https://mmdetection.readthedocs.io/zh_CN/latest/1_exist_data_model.html)





<!-- 完成标志, 看不到, 请忽略! -->
