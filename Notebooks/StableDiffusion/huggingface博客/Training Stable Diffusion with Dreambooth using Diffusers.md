# Training Stable Diffusion with Dreambooth using 🧨 Diffusers

Dreambooth 是一种使用专门的微调形式来教授新概念给 [Stable Diffusion](https://huggingface.co/blog/stable_diffusion) 的技术。有些人已经在使用它来将自己放置在奇幻的情境中，而另一些人则在使用它来融合新的风格。Diffusers 提供了 Dreambooth [训练脚本](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)。训练时间不长，但选择正确的超参数集合很难，而且很容易过拟合。

我们进行了大量实验来分析 Dreambooth 中不同设置的影响。本文介绍了我们的发现以及一些技巧，可以在使用Dreambooth 对稳定扩散进行微调时改进您的结果。

在开始之前，请注意，这种方法绝不能用于恶意目的，以任何方式生成伤害或未经他们知情地冒充他人。使用它训练的模型仍受 CreativeML Open RAIL-M 许可证约束，该许可证管理稳定扩散模型的分发。

注：此帖子的早期版本已发布为 [W&B 报告](https://wandb.ai/psuraj/dreambooth/reports/Dreambooth-Training-Analysis--VmlldzoyNzk0NDc3)。

## TL;DR: Recommended Settings

- Dreambooth 很容易过拟合。为了获得高质量的图像，我们必须在训练 steps 的数量和学习率之间找到一个“甜点”。我们建议使用较低的学习率，并逐步增加 steps 的数量，直到结果令人满意。
- 对于人脸，Dreambooth 需要更多的训练 steps。在我们的实验中，当使用批量大小为 2 和 LR 为 1e-6 时，800-1200 个 steps 效果很好。
- 在训练人脸时，先验保留非常重要，以避免过拟合。对于其他主题，似乎没有太大的区别。
- 如果您发现生成的图像是噪音或质量下降，则很可能意味着过拟合。首先，请尝试上述步骤以避免它。如果生成的图像仍然是噪音，请使用 DDIM scheduler 或更多的推理 steps（在我们的实验中，约 100 个 steps 效果很好）。
- 除 UNet 外，训练文本编码器也对质量有很大影响。我们最好的结果是使用文本编码器微调、低 LR 和适当数量的 steps 的组合获得的。但是，微调文本编码器需要更多显存，因此至少具有 24 GB RAM 的 GPU 是理想的。使用 8 位 Adam、fp16 训练或梯度累积等技术，可以在 16 GB GPU上进行训练，例如 Google Colab 或 Kaggle 提供的 GPU。
- 使用或不使用 EMA 产生了类似的结果。
- 没有必要使用 sks 单词来训练 Dreambooth。最初的一些实现之一使用它是因为它是词汇表中的一个罕见标记，但实际上它是一种 rifle。我们的实验以及@nitrosocke等人进行的实验表明，选择您自然用于描述目标的术语是可以接受的。
