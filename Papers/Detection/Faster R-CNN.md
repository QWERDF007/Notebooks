# [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)

## Abstract

最先进的目标检测网络依赖于区域提议算法来假设物体位置。像SPPnet [1]和Fast R-CNN [2]这样的进步减少了这些检测网络的运行时间，暴露了区域提议计算作为瓶颈。在这项工作中，我们引入了一个区域提议网络（RPN），它与检测网络共享全图像卷积特征，从而实现了几乎免费的区域提议。RPN是一个完全卷积的网络，它在每个位置同时预测对象边界和对象性分数。RPN被端到端地训练以生成高质量的区域提议，这些提议由Fast R-CNN用于检测。我们进一步通过共享它们的卷积特征将RPN和Fast R-CNN合并为单个网络——使用最近流行的神经网络术语“注意”机制，RPN组件告诉统一网络要查找哪里。对于非常深的VGG-16模型[3]，我们的检测系统在GPU上每秒5帧（包括所有步骤），同时在PASCAL VOC 2007、2012和MS COCO数据集上实现了最先进的目标检测精度，每个图像仅使用300个提议。在ILSVRC和COCO 2015比赛中，Faster R-CNN和RPN是多个赛道中获得第一名的基础。代码已经公开发布。

**索引词：** Object Detection, Region Proposal, Convolutional Neural Network