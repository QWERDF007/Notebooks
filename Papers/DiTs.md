# [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

## Abstract

我们探索了一类基于 Transformer 架构的扩散模型。我们训练了图像的潜空间扩散模型，使用了一个在潜空间 patch 上运行 Transformer 替换了通常使用的 U-Net 骨干网络。我们通过以 Gflops 衡量的前向传播的复杂度的角度分析了我们的 Diffusion Transformers (DiTs) 的可扩展性。我们发现，具有更高 Gflops 的 DiTs (通过增加 Transformer 的深度/宽度或增加输入 tokens 的数量) 的FID值一致更低。除了具有良好的可扩展性之外，我们最大的 DiT-XL/2 模型还超过了所有先前的扩散模型在有类别条件的 ImageNet $512 \times 512$ 和 $256 \times 256$ 基准测试上的性能，在后者上取得了最先进的 2.27 FID。

## 1. Introduction

机器学习正在经历一个由 Transformer 驱动的复兴。在过去的五年中，用于自然语言处理[8, 42]、视觉[10]以及其他几个领域的神经架构在很大程度上已被 Transformer [60]所取代。然而，许多图像级生成模型仍然坚持原有的趋势——尽管 Transformer 在自回归模型 [3,6,43,47] 中得到了广泛的应用，但在其他生成建模框架中的采用率较低。例如，扩散模型一直是图像级生成模型最新进展的前沿[9,46]，但它们都采用了卷积 U-Net 架构作为 backbone。Ho 等人的开创性工作[19]首次为扩散模型引入了 U-Net 骨干网络。U-Net 最初在像素级自回归模型和条件GANs[23]中取得了成功，它是从 PixelCNN++ [52, 58] 继承过来的，但有一些改变。该模型是卷积的，主要由 ResNet [15]块组成。与标准的 U-Net [49]不同，额外的空间自注意力块 (Transformer 的关键组件) 被插入到较低的分辨率中。Dhariwal 和 Nichol[9] 对 U-Net 的几种架构选择进行了消融实验，例如使用自适应归一化层 [40] 来注入条件信息和卷积层的通道计数。然而，Ho 等人的 U-Net 的总体设计基本保持不变。

这项工作旨在揭示扩散模型中架构选择的重要性，并为未来的生成建模研究提供经验基线。我们展示了 U-Net 归纳偏差对扩散模型性能并不至关重要，它们可以轻松地被标准设计 (如 transformers) 替代。因此，扩散模型有望从最近的架构统一化趋势中受益，例如从其他领域继承最佳实践和训练方法，同时保留可扩展性、鲁棒性和效率等优良特性。标准化的架构还将为跨领域研究开辟新的可能性。

在本文中，我们专注于一类基于 transformers 的新型扩散模型，我们称之为 Diffusion Transformers，简称 DiTs。DiTs 遵循 Vision Transformers (ViTs) 的最佳实践，ViTs 已被证明其在视觉识别任务上的可扩展性优于传统卷积网络 (例如 ResNet)。

更具体地说，我们研究了 transformer 关于网络复杂度与样本质量的可扩展行为。我们展示了通过在潜在扩散模型 (LDMs) 框架下构建和基准测试 DiT 的设计空间，其中扩散模型在 VAE 的潜在空间内进行训练，我们可以用 transformer 成功地替换 U-Net 骨干网络。我们进一步展示了 DiT 是可扩展的扩散模型架构：网络复杂度 (以 Gflops 衡量) 与样本质量 (以 FID 衡量) 之间存在强烈的相关性。通过简单地扩大 DiT 并训练具有高容量骨干网络 (118.6 Gflops) 的 LDM，我们能够在有类别条件的 $256 \times 256$ ImageNet 生成基准上达到最先进的 2.27 FID。

## 6. Conclusion

我们引入了 Diffusion Transformers (DiTs)，这是一种简单的基于 transformer 的扩散模型的骨干网络，优于之前的 U-Net 模型，并继承了 transformer 模型类的出色扩展性能。鉴于本文中有希望的扩展结果，未来的工作应继续将 DiTs 扩展到更大的模型和 token 数量。DiT 还可以作为文本到图像模型 (例如 DALL·E 2 和 Stable Diffusion) 的一个插拔式 (drop-in) 骨干进行探索。