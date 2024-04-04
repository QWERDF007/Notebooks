# [YOLO-World: Real-Time Open-Vocabulary Object Detection](https://arxiv.org/abs/2401.17270)

You Only Look Once (YOLO) 系列检测器已经证明了自己是一种高效实用的工具。然而，它们依赖于预定义和训练好的物体类别，限制了它们在开放场景中的适用性。为了解决这一限制，我们引入了一种创新方法 YOLO-World，它通过视觉—语言建模和大规模数据集预训练，赋予 YOLO 开放词汇检测能力。具体来说，我们提出了一种新的可重新参数化 (Re-parameterizable) 视觉语言路径聚合网络 (RepVL-PAN) 和区域文本对比损失函数，以促进视觉和语言信息之间的交互。我们的方法在零样本场景下高效率检测各种物体表现出色。在具有挑战性的 LVIS 数据集上，YOLO-World 在 V100 上实现了 35.4 AP 和 52.0 FPS 的性能，在精度和速度方面都优于许多最先进的方法。此外，经过微调的 YOLO-World 在一些下游任务（例如目标检测和开放词汇实例分割）上也取得了显著的性能提升。

## 1. Introduction

