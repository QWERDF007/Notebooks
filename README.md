# Notebooks

一些记录、论文翻译、心得

[TOC]

| 图像分类                 | 图像分割 | 目标检测 | 超分辨率 | 自监督 |
| ------------------------ | -------- | -------- | -------- | -------- |
| [:cat:](#cat-图像分类) | [:dog:](#dog-图像分割) | [:car:](#car-目标检测) | [:cow:](#cow-超分辨率) | [:cake:](#cake-自监督) |

| 阴影去除                       | 图像变形                 | 环境依赖               |        |         |
| ------------------------------ | ------------------------ | ---------------------- | ------ | ------- |
| [:hamster:](#hamster-阴影去除) | [:deer:](#deer-图像变形) | [:dragon:](#dragon-环境依赖) | :pear: | :peach: |



## What's New

- 2022.05.04 - ResNet 论文翻译解读

- 2022.04.19 - gcc & g++ 8.4 安装、pytorch 1.10.0 c++ 编译、Nvidia 驱动、CUDA、cudnn 安装

- 2022.04.19 - tensorflow 2.5.0 c++ 编译

- 2022.04.12 - AlexNet 论文翻译解读

- 2022.03.17 - mmseg 自定义数据训练 SwinTransformer

- 2021.12.02 - ESRGAN 论文翻译解读 (未完 - 补充材料)
- 2021.11.30 - MAE 论文翻译解读 (未完)
- 2021.09.14 - mmseg, mmdet 使用说明; 目标检测数据格式 (COCO、VOC、YOLO)
- 2021.02.24 - MLS 代码
- 2021.02.09 - 移动最小二乘法的图像变形 (MLS)



## :cat: 图像分类


- [x] [AlexNet 论文翻译解读](./Papers/Classification/AlexNet.md)
- [x] [ResNet 论文翻译解读](./Papers/Classification/ResNet.md)


## :dog: 图像分割

- [x] [MMSegmentation 使用说明](./MachineLearning/OpenMMLab/mmseg自定义数据训练.md)
- [x] [MMSegmentation 在自定义数据集上训练 SwinTransformer](./MachineLearning/OpenMMLab/mmseg自定义数据训练SwinTransformer.md)



## :car: 目标检测

- [x] [目标检测数据格式](./MachineLearning/ObjectDection/DataFormat.md)
- [x] [MMdetection 使用说明](./MachineLearning/OpenMMLab/mmdet自定义数据训练.md)



## :cow: 超分辨率

- [ ] [ESRGAN-Enhanced Super-Resolution Generative Adversarial Networks](./Papers/SuperResolution/ESRGAN-Enhanced Super-Resolution Generative Adversarial Networks.md) (ESRGAN) 论文翻译解读 (未完)



## :cake: 自监督

- [x] [Masked Autoencoders Are Scalable Vision Learners](./Papers/SelfSupervise/Masked Autoencoders Are Scalable Vision Learners.md) (MAE) 论文翻译解读



## :hamster: 阴影去除

- [ ] [Towards Ghost-free Shadow Removal via Dual Hierarchical Aggregation Network and Shadow Matting GAN]() (DHAN) 论文翻译解读 (未开始)



## :deer: 图像变形

- [x] [Image Deformation Using Moving Least Squares]() (MLS) 论文翻译解读; [code](./Code/mls)



## :dragon: 环境依赖

- [x] [TensorFlow 2.5.0 C++ 编译](./MachineLearning/Env/libtensorflow编译.md)
- [x] [PyTorch 1.10.0 C++ 编译](./MachineLearning/Env/libtorch编译.md)
- [x] [gcc & g++ 8.4 安装](./MachineLearning/Env/gcc&g++安装.md)
- [x] [Nvidia 驱动、CUDA、cudnn 安装](./MachineLearning/Env/NVIDIA驱动&CUDA&CUDNN安装.md)