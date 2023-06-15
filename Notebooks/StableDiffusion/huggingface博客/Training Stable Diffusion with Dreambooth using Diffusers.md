# [Training Stable Diffusion with Dreambooth using 🧨 Diffusers](https://huggingface.co/blog/dreambooth)

Dreambooth 是一种使用专门的微调形式来教授新概念给 [Stable Diffusion](https://huggingface.co/blog/stable_diffusion) 的技术。有些人已经在使用它来将自己放置在奇幻的情境中，而另一些人则在使用它来融合新的风格。Diffusers 提供了 Dreambooth [训练脚本](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)。训练时间不长，但选择正确的超参数集合很难，而且很容易过拟合。

我们进行了大量实验来分析 Dreambooth 中不同设置的影响。本文介绍了我们的发现以及一些技巧，可以在使用Dreambooth 对稳定扩散进行微调时改进您的结果。

在开始之前，请注意，这种方法绝不能用于恶意目的，以任何方式生成伤害或未经他们知情地冒充他人。使用它训练的模型仍受 CreativeML Open RAIL-M 许可证约束，该许可证管理稳定扩散模型的分发。

注：此帖子的早期版本已发布为 [W&B 报告](https://wandb.ai/psuraj/dreambooth/reports/Dreambooth-Training-Analysis--VmlldzoyNzk0NDc3)。

## TL;DR: Recommended Settings

- ==Dreambooth 很容易过拟合==。为了获得高质量的图像，我们必须在训练 steps 的数量和学习率之间找到一个“甜点”。我们建议使用较低的学习率，并逐步增加 steps 的数量，直到结果令人满意。
- 对于人脸，Dreambooth 需要更多的训练 steps。在我们的实验中，当使用批量大小为 2 和 LR 为 1e-6 时，800-1200 个 steps 效果很好。
- 在==训练人脸时，先验保留非常重要，以避免过拟合。对于其他主题，似乎没有太大的区别==。
- 如果您发现生成的图像是噪音或质量下降，则很可能意味着过拟合。首先，请尝试上述步骤以避免它。如果生成的图像仍然是噪音，请使用 DDIM scheduler 或更多的推理 steps（在我们的实验中，约 100 个 steps 效果很好）。
- 除 UNet 外，训练文本编码器也对质量有很大影响。我们最好的结果是使用文本编码器微调、低 LR 和适当数量的 steps 的组合获得的。但是，微调文本编码器需要更多显存，因此至少具有 24 GB RAM 的 GPU 是理想的。使用 8 位 Adam、fp16 训练或梯度累积等技术，可以在 16 GB GPU上进行训练，例如 Google Colab 或 Kaggle 提供的 GPU。
- 使用或不使用 EMA 产生了类似的结果。
- 没有必要使用 sks 单词来训练 Dreambooth。最初的一些实现之一使用它是因为它是词汇表中的一个罕见标记，但实际上它是一种 rifle。我们的实验以及@nitrosocke等人进行的实验表明，选择您自然用于描述目标的术语是可以接受的。

## Learning Rate Impact

Dreambooth 很快就会过拟合。为了获得良好的结果，请以对您的数据集有意义的方式调整学习率和训练 steps 的数量。在我们的实验中（详见下文），我们使用高低学习率在四个不同的数据集上进行了微调。==在所有情况下，我们都使用低学习率获得了更好的结果==。

## Experiments Settings

我们的所有实验都是使用 [train_dreambooth.py](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) 脚本在 2x 40GB A100 上使用 AdamW 优化器进行的。我们使用相同的种子，并在所有运行中保持所有超参数相等，除了 LR、steps 的数量和是否使用先验保留（prior preservation）。

对于前 3 个示例（各种对象），我们使用批量大小为 4（每个 GPU 2 个）进行了400 steps 的模型微调。我们使用了高学习率 5e-6 和低学习率 2e-6。没有使用先验保留。

最后一个实验尝试将人类主体添加到模型中。在这种情况下，我们使用了先验保留，批量大小为 2（每个GPU 1 个），800  和 1200 steps。我们使用了高学习率 5e-6 和低学习率 2e-6。

请注意，您可以使用 8 位 Adam、fp16 训练或梯度累积来减少内存要求，并在具有 16 GB 内存的GPU上运行类似的实验。

### Cat Toy

高学习率 (`5e-6`)

<img src="">

低学习率 (`2e-6`)

<img src="">

### Pighead

高学习率 (`5e-6`). 注意，颜色伪影是噪声残留——运行更多推理 steps 可以帮助解决其中一些细节。

<img src="">

低学习率 (`2e-6`)

<img src="">

### Mr. Potato Head

高学习率 (`5e-6`). 注意，颜色伪影是噪声残留——运行更多推理 steps 可以帮助解决其中一些细节。

<img src="">

低学习率 (`2e-6`)

<img src="">

### Human Face

我们试图将 Seinfeld 中的 Kramer 角色融入到 Stable Diffusion 中。正如先前提到的，我们使用更少的批次大小进行了更多 steps 的训练。即便如此，结果并不出色。为了简洁起见，我们省略了这些样本图像，并将读者推荐到下一节，其中面部训练成为我们努力的重点。

## Summary of Initial Results

为了在 Dreambooth 中训练 Stable Diffusion 并获得良好的结果，调整学习率和训练 steps 对你的数据集非常重要。

学习率过高和训练 steps 过多会导致过拟合。无论使用什么提示，模型都会主要生成来自你的训练数据的图像。 学习率过低和 steps 过少会导致欠拟合：模型将无法生成我们试图融入的概念。 面部更难训练。在我们的实验中，2e-6的学习率和400个训练 steps 对于对象效果很好，但面部需要1e-6（或2e-6）和约1200个 steps 。

如果模型过度拟合，图像质量会大大降低，这种情况发生在：

学习率过高。 我们运行了太多的训练 steps 。 在面部的情况下，当没有使用先前的保留时，如下一节所示。



















