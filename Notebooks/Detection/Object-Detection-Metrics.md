# [目标检测指标](https://github.com/rafaelpadilla/Object-Detection-Metrics)

这个项目的动机是不同的工作和实现在**在目标检测问题的评估指标**方面缺乏共识。虽然在线竞赛使用自己的指标来评估目标检测任务，但只有一些竞赛提供参考代码片段来计算检测到的对象的准确性。

想要使用不同于竞赛提供的数据集评估自己的工作的研究人员需要实现自己版本的指标。有时错误或不同的实现会产生不同和有偏差的结果。理想情况下，为了在不同方法之间进行可靠的基准测试，需要有一个灵活的实现，每个人都可以使用，而不管使用的数据集如何。

该项目提供了易于使用的函数，实现了目标检测最流行竞赛使用的相同指标。我们的实现不需要修改检测模型以复杂的输入格式，避免转换为XML或JSON文件。我们简化了输入数据（地面真实边界框和检测边界框），并在单个项目中收集了学术界和挑战赛中使用的主要指标。我们的实现经过仔细比较官方实现，结果完全相同。

在下面的主题中，您可以找到不同竞赛和作品中使用的最流行指标的概述，以及展示如何使用我们代码的示例。

## Table of contents

- [Motivation](#metrics-for-object-detection)
- [Different competitions, different metrics](#different-competitions-different-metrics)
- [Important definitions](#important-definitions)
- [Metrics](#metrics)
  - [Precision x Recall curve](#precision-x-recall-curve)
  - [Average Precision](#average-precision)
    - [11-point interpolation](#11-point-interpolation)
    - [Interpolating all  points](#interpolating-all-points)
- [**How to use this project**](#how-to-use-this-project)
- [References](#references)

## Different competitions, different metrics

- [PASCAL VOC Challenge](http://host.robots.ox.ac.uk/pascal/VOC/) 提供了一个 Matlab 脚本，以评估检测到的对象的质量。竞赛的参与者可以使用提供的 Matlab 脚本在提交结果之前测量其检测的准确性。官方文档解释了他们的目标检测指标标准，可以在[此处](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00050000000000000000)访问。当前 PASCAL VOC 目标检测挑战使用的指标是 Precision x Recall 曲线和 AP (**area under the curve (AUC) of the Precision x Recall curve**)。

  PASCAL VOC Matlab 评估代码从 XML 文件中读取真实边界框，如果您想将其应用于其他数据集或特定情况，则需要更改代码。即使像 Faster-RCNN 这样的项目实现了 PASCAL VOC 评估指标，也需要将检测到的边界框转换为其特定格式。Tensorflow 框架也有其 PASCAL VOC 指标实现。

- [COCO Detection Challenge](https://competitions.codalab.org/competitions/5181) 使用不同的指标来评估不同算法的目标检测准确性。在[这里](http://cocodataset.org/#detection-eval)，您可以找到一个文档，解释了用于表征 COCO 上对象检测器性能的 12 个指标。该竞赛提供 Python 和 Matlab 代码，以便用户在提交结果之前验证其分数。也需要将结果转换为竞赛所需的格式。

- [Google Open Images Dataset V4 Competition](https://storage.googleapis.com/openimages/web/challenge.html) 也使用平均精度（mAP）来评估 500  个类别上的目标检测任务。

- [ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-detection-challenge) 考虑真实边界框与检测到的边界框之间的重叠区域和类别,为每个图像定义一个错误。总误差计算为所有测试数据集图像中所有最小误差的平均值。[这里](https://www.kaggle.com/c/imagenet-object-localization-challenge#evaluation)有关于他们评估方法更多细节。

## Important definitions

### Intersection Over Union (IOU)

交并比 (IOU) 是一种基于 Jaccard 指数的度量，用于评估两个边界框之间的重叠。它需要一个真实边界框和一个预测边界框。通过应用 IoU，我们可以判断检测是否有效（真正例）或无效（假正例）。

IOU 由预测边界框和真实边界框之间的重叠区域除以它们之间的联合区域面积给出：

<img src="./assets/iou.gif" />

下图展示了真实边界框 (绿色) 和预测边界框 (红色) 之间的 IOU。

<img src="./assets/iou.png" />

### True Positive, False Positive, False Negative and True Negative

指标中使用的一些基本概念:

- **真正例 (TP)**: 正确的检测。IOU ≥ 阈值的检测
- **假正例 (FP)**: 错误的检测。IOU < 阈值的检测
- **假反例 (FN)**: 未检测到的真实目标
- **真反例 (TN)**: 不适用。它表示一个被纠正的错误检测。在目标检测任务中，图像中有许多不应检测到的可能的边界框。因此，TN 将是所有在图像中被正确不检测到的可能的边界框 (图像中许多可能的框)。这就是为什么指标不使用它的原因。

阈值: 根据指标的不同，通常设置为 50%、75% 或 95%。

### 精确率 (Precision)

精确率是模型仅识别相关对象的能力。它是正确的正类预测的百分比，计算公式如下:

<img src="./assets/precision.gif" />

### 召回率 (Recall)

召回率是模型找到所有相关案例 (所有真实边界框) 的能力。它是所有相关真实目标中被检测出的真正例 (TP) 的百分比，计算公式如下:

<img src="./assets/recall.gif" />

## Metrics

在下面的主题中，有关于目标检测中使用的最流行指标的一些评论。

### Precision x Recall curve

Precision x Recall 曲线是评估对象检测器性能的好方法，通过为每个对象类更改置信度来绘制曲线。如果一个特定类别的对象检测器的精确率在召回率增加时仍保持高水平，则认为其是好的，这意味着如果您变化置信度阈值，精度和召回率仍将保持高水平。判断对象检测器好坏的另一种方法是查看只能识别相关对象（0 假正例 = 高精度）的检测器，找到所有真实对象（0 假反例 = 高召回率）。

一个差的对象检测器需要增加检测对象的数量 (增加假正例 = 降低精确率) 来获取所有真实目标 (高召回率)。这就是为什么 Precision x Recall 曲线通常以高精度值开始，随着召回率的增加而降低。您可以在下一个主题 (Average Precision) 中看到 Prevision x Recall 曲线的示例。这种曲线用于 PASCAL VOC 2012 挑战赛，我们的实现中也可使用。