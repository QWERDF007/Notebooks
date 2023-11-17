

# [Fast R-CNN](https://arxiv.org/abs/1504.08083)

## Abstract

本文提出了一种快速的基于区域 (Fast Region-based) 的卷积网络方法（Fast R-CNN）进行目标检测。Fast R-CNN 以先前的工作为基础，使用深度卷积网络对目标提案进行有效分类。与之前的工作相比，Fast R-CNN 采用了一些创新技术来提高训练和测试速度，同时也提高了检测精度。Fast R-CNN 训练非常深的 VGG16 网络比 R-CNN 快 9倍，测试比 R-CNN 快了 213 倍，并在 PASCAL VOC 2012 上实现了更高的 mAP，与 SPP-net 相比，Fast R-CNN 在 VGG16 网络上的训练的速度快了 3 倍，测试速度提高了 10 倍，并且更加准确。Fast R-CNN 的 Python 和 C ++（使用Caffe）实现以 MIT 开源许可证发布在：https://github.com/rbgirshick/fast-rcnn。

