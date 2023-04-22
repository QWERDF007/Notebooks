# [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752.pdf)

## Abstract

通过将图像形成过程分解为去噪自动编码器的顺序应用，扩散模型（DM）在图像数据及其它领域实现了最先进的合成结果。此外，它们的公式允许引导机制来控制图像生成过程而无需重新训练。然而，由于这些模型通常直接在像素空间中操作，因此强大的DM优化通常需要消耗数百个GPU天，并且由于顺序评估而导致推理代价高昂。为了在保留其质量和灵活性的同时在有限的计算资源上进行DM训练，我们将它们应用于强大的预训练自动编码器的潜在空间中。与以前的工作相比，对这种表示进行扩散模型训练首次允许达到复杂度降低和细节保留之间的近乎最优点，极大地提高了视觉保真度。通过将交叉关注层引入模型架构中，我们将扩散模型转化为通用调节输入（如文本或边界框）的强大而灵活的生成器，并且高分辨率合成变得可能。我们的潜在扩散模型（LDM）在图像修复和类条件图像合成方面取得了新的最先进分数，并在各种任务中表现出高度竞争力，包括文本到图像合成、无条件图像生成和超分辨率，同时与基于像素的DM相比显著降低了计算要求。