# [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)

## Abstract

我们使用扩散概率模型（diffusion probabilistic models）进行高质量图像合成，这是一类受非平衡热力学考虑启发的潜变量模型。我们通过训练一个加权变分界限（weighted variational bound），该界限根据扩散概率模型和Langevin动力学的去噪分数匹配之间的新颖联系设计，并且我们的模型自然地采用了一种渐进式有损解压缩方案，可以解释为自回归解码的一般化。在无条件CIFAR10数据集上，我们获得了9.46的Inception分数和3.17的最先进FID分数。在256x256 LSUN上，我们获得了类似于 ProgressiveGAN 的样本质量。我们的实现可在 https://github.com/hojonathanho/diffusion 上找到。