# Notebooks

一些记录、论文翻译、心得

[TOC]

| 图像分类                 | 图像分割 | 目标检测 | 超分辨率 | 自监督 |
| ------------------------ | -------- | -------- | -------- | -------- |
| [:cat:](#cat-图像分类) | [:dog:](#dog-图像分割) | [:car:](#car-目标检测) | [:cow:](#cow-超分辨率) | [:fox_face:](#fox_face-自监督) |

| 阴影去除                       | 图像变形                 | 图像生成           | 自然语言处理 | 环境依赖 |
| ------------------------------ | ------------------------ | ---------------------- | ------ | ------- |
| [:hamster:](#hamster-阴影去除) | [:deer:](#deer-图像变形) | [:fire:](#fire-图像生成) | [:wolf:](#wolf-自然语言处理) | [:dragon:](#dragon-环境依赖) |

✅❌

## What's New

- ❌ 2022.06.29 - 卷积、池化等算子输出尺寸计算 (未完 - 转置卷积)

- ✅ 2022.06.29 - DeepLabv3 论文翻译解读

- ✅ 2022.06.22 - DeepLabv3+ 论文翻译解读

- ✅ 2022.06.17 - DatasetGAN 使用指北

- ❌ 2022.06.16 - SemanticStyleGAN 占坑 

- ✅ 2022.06.14 - DatasetGAN 论文翻译解读

- ✅ 2022.06.01 - StyleGAN 论文翻译解读

- ✅ 2022.05.23 - Transformer 论文翻译解读

- ✅ 2022.05.04 - ResNet 论文翻译解读

- ✅ 2022.04.19 - gcc & g++ 8.4 安装、pytorch 1.10.0 c++ 编译、Nvidia 驱动、CUDA、cudnn 安装

- ✅ 2022.04.19 - tensorflow 2.5.0 c++ 编译

- ✅ 2022.04.12 - AlexNet 论文翻译解读

- ✅ 2022.03.17 - mmseg 自定义数据训练 SwinTransformer

- ❌ 2021.12.02 - ESRGAN 论文翻译解读 (未完 - 补充材料)
- ❌ 2021.11.30 - MAE 论文翻译解读 (未完)
- ✅ 2021.09.14 - mmseg, mmdet 使用说明; 目标检测数据格式 (COCO、VOC、YOLO)
- ✅ 2021.02.24 - MLS 代码实现
- ✅ 2021.02.09 - 移动最小二乘法的图像变形 (MLS) 论文翻译



## :cat: 图像分类


- [x] [ImageNet Classification with Deep Convolutional Neural Networks](./Papers/Classification/AlexNet.md) (AlexNet) 论文翻译解读
- [x] [Deep Residual Learning for Image Recognition](./Papers/Classification/ResNet.md) (ResNet) 论文翻译解读
- [ ] [卷积、汇聚、反卷积、空洞卷积输出尺寸计算](./MachineLearning/Classification/卷积池化等算子输出尺寸计算.md)




## :dog: 图像分割

- [x] [MMSegmentation 使用说明](./MachineLearning/OpenMMLab/mmseg自定义数据训练.md)
- [x] [MMSegmentation 在自定义数据集上训练 SwinTransformer](./MachineLearning/OpenMMLab/mmseg自定义数据训练SwinTransformer.md)
- [x] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](./Papers/Segmentation/DeepLabv3Plus.md) (DeepLabv3+) 论文解读
- [x] [Rethinking Atrous Convolution for Semantic Image Segmentation](./Papers/Segmentation/DeepLabv3.md) (DeepLabv3) 论文解读



## :car: 目标检测

- [x] [目标检测数据格式](./MachineLearning/ObjectDection/DataFormat.md)
- [x] [MMdetection 使用说明](./MachineLearning/OpenMMLab/mmdet自定义数据训练.md)



## :cow: 超分辨率

- [ ] [ESRGAN-Enhanced Super-Resolution Generative Adversarial Networks](./Papers/SuperResolution/ESRGAN.md) (ESRGAN) 论文翻译解读 (未完)



## :fox_face: 自监督

- [x] [Masked Autoencoders Are Scalable Vision Learners](./Papers/SelfSupervised/MAE.md) (MAE) 论文翻译解读



## :hamster: 阴影去除

- [ ] [Towards Ghost-free Shadow Removal via Dual Hierarchical Aggregation Network and Shadow Matting GAN]() (DHAN) 论文翻译解读 (未开始)



## :deer: 图像变形

- [x] [Image Deformation Using Moving Least Squares](./Papers/Deformation/MLS.md) (MLS) 论文翻译解读; [code](./Code/mls)



## :fire: 图像生成

- [x] [Barbershop 发型替换指北](./MachineLearning/GAN/Barbershop替换发型指北.md)
- [x] [A Style-Based Generator Architecture for Generative Adversarial Networks](./Papers/GAN/StyleGAN.md) (StyleGAN) 论文翻译解读
- [x] [DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort](./Papers/GAN/DatasetGAN.md) (DatasetGAN) 论文翻译解读 (补充材料未完)
- [x] [DatasetGAN 使用指北](./MachineLearning/GAN/DatasetGAN使用指北.md)



## :wolf: 自然语言处理

- [x] [Transformer 论文翻译](./Papers/NLP/Transformer.md)



## :dragon: 环境依赖

- [x] [TensorFlow 2.5.0 C++ 编译](./MachineLearning/Env/libtensorflow编译.md)
- [x] [PyTorch 1.10.0 C++ 编译](./MachineLearning/Env/libtorch编译.md)
- [x] [gcc & g++ 8.4 安装](./MachineLearning/Env/gcc&g++安装.md)
- [x] [Nvidia 驱动、CUDA、cudnn 安装](./MachineLearning/Env/NVIDIA驱动&CUDA&CUDNN安装.md)