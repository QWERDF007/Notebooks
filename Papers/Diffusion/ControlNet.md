# [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)

## Abstract

我们提出了一种神经网络结构ControlNet，用于控制预训练的大型扩散模型以支持附加输入条件。ControlNet以端到端的方式学习任务特定条件，即使训练数据集很小（<50k），学习也是稳健的。此外，训练ControlNet的速度与微调扩散模型的速度相同，并且该模型可以在个人设备上进行训练。或者，如果有强大的计算集群可用，则该模型可以扩展到大量（数百万至数十亿）的数据。我们报告说，像稳定扩散这样的大型扩散模型可以通过ControlNets进行增强，以实现像边缘映射、分割映射、关键点等条件输入。这可能会丰富控制大型扩散模型的方法，并进一步促进相关应用。