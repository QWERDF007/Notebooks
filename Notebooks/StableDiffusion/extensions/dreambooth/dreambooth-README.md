# [Dreambooth Extension for Stable-Diffusion-WebUI](https://github.com/d8ahazard/sd_dreambooth_extension)

这是 Shivam Shriao 的 [Diffusers Repo](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth) 的 WIP 端口，它是默认的 [Huggingface Diffusers Repo](https://github.com/huggingface/diffusers) 的修改版本，针对低 VRAM GPU 的性能进行了优化。

此外，还借鉴了 BMaltais 的 [Koyha SS](https://github.com/bmaltais/kohya_ss) 的部分内容。

它还添加了其他几个功能，包括同时训练多个概念和（即将推出）Inpainting 训练。

## 安装





## Usage

### 创建模型

1. 转到 `Dreambooth` 选项卡。

2. 在 `Create Model` 子选项卡下，输入新的模型名称并选择要训练的源检查点。
   如果您想使用 HF Hub 中的模型，请指定模型 URL 和 token。URL格式应为`runwayml/stable-diffusion-v1-5`

   源检查点将被提取到 `models\dreambooth\MODELNAME\working`。

3. 单击 `Create`。这将需要一两分钟，但完成后，UI应指示已设置新的模型目录。

## 顶部各种按钮

- *Save Params* - 保存当前模型的当前训练参数。
- *Load Params* - 从当前选择的模型中加载训练参数。使用此功能将参数从一个模型复制到另一个模型。
- *Generate Ckpt* - 从当前版本的当前保存权重生成检查点。
- *Generate Samples* - 在训练时单击此按钮以在下一个间隔之前生成样本。
- *Cancel* - 取消当前步骤后的训练。
- *Train* - 开始训练。

## 模型选择

- *Model* - 要使用的模型。更改模型时，训练参数不会自动加载到UI中。
- *Lora Model* - 如果恢复训练，则加载现有的 lora 检查点，或者如果生成检查点，则与基本模型合并。
- *Half Model* - 启用此选项以使用半精度保存模型。结果是一个较小的检查点，图像输出几乎没有明显差异。
- *Save Checkpoint to Subdirectory* - 使用模型名称将检查点保存到子目录中。

## 训练参数

- ==*Performance Wizard (WIP)* - 尝试根据 GPU 的 VRAM 数量和实例图像数量设置最佳训练参数==。可能不完美，但至少是一个很好的起点。

### Intervals

- *Training Steps Per Image (Epochs)* - 如名称所示，一个 epoch 是对整个实例图像集训练一次。因此，如果我们想要每个图像训练 100 steps，我们可以将此值设置为100，然后就可以开始了。不需要数学计算。
- *Pause After N Epochs* - 当设置为大于0的值时，训练将在指定时间后暂停。
- *Amount of time to pause between Epochs, in Seconds* - 当 N 大于零时，在 N 个 epochs 之间暂停多长时间，以秒为单位。
- *Use Concepts* - 是否使用具有多个概念的JSON文件或字符串，或下面的单独设置。
- *Save Model/Preview Frequency (Epochs)* - 保存检查点和预览频率将是每个 epoch，而不是每个 steps。

### Batching

- *Batch size* - 同时处理多少个训练步骤。您可能希望将其保留为1。
- *Gradient Accumulation Steps* - 这应该设置为与训练 Batch size 相同的值。
- *Class batch size* - 同时生成多少分类图像。将其设置为您可以使用 Txt2Image 一次安全处理的任何内容，或者只是将其保持不变。
- *Set Gradients to None When Zeroing* - 将梯度设置为 None 而不是零。这通常具有更低的内存占用量，并且可以适度提高性能。
  https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
- *Gradient Checkpointing* - 启用此功能可节省VRAM，但速度会稍微降低。
  https://arxiv.org/abs/1604.06174v2
- *Max Grad Norms* - 梯度规范化的最大数量

### Learning Rate

- *Learning rate* - 训练影响新模型的强度。较高的学习率需要较少的训练步骤，但更容易导致过度拟合。推荐在 0.000006 和 0.00000175 之间

- *Scale Learning Rate* - 随时间调整学习率。
- *Learning Rate Scheduler* - 与学习率一起使用的调度程序。
- *Learning Rate Warmup Steps* - 在缩放学习率之前运行多少步骤。我想。

### Image Processing

- *Resolution* - 您的实例图像的分辨率。应该是 512 或 768。使用高于 512 的分辨率将导致更多的 vram 使用。

- *Center Crop* - 启用此功能可在输入图像大于指定分辨率时自动使用 “dumb cropping”。
- *Apply Horizontal Flip* - 启用后，训练期间将随机水平翻转实例图像。这可以提高可编辑性，但可能需要更多的训练步骤，因为我们实际上正在增加数据集大小。

### Miscellaneous

- *Pretrained VAE Name or Path* - 输入现有 vae .bin 文件的完整路径，它将被用于替代源检查点中的 VAE。
- *Use Concepts List* - 启用此功能可忽略概念选项卡，并从 JSON 文件中加载训练数据。
- *Concepts List* - 包含要训练的概念的 json 文件的路径。

## Advanced Settings

在这里，您将找到更多与性能相关的设置。改变这些可能会影响训练所需的 VRAM 量。

### Tuning

- *Use CPU Only* - 如字面意义，如果您无法使用其他任何设置进行训练，这是最后的手段。此外，它将非常缓慢。此外，您不能在 CPU 训练中使用 8 Bit-Adam，否则您会度过糟糕的时光。
- *Use EMA* - ==在训练 unet 时使用估计的移动平均值。据称，这对于生成图像更好，但似乎对训练结果影响很小==。使用更多 VRAM。
- *Mixed Precision* - 当使用 8 bit AdamW 时，您*必须*将其设置为 fp16 或 bf16。Bf16 精度仅受较新的 GPU 支持，并且默认情况下启用/禁用。
- *Memory Attention* - 要使用的注意力类型。可选项是：
  - 'default'：通常最快，但使用最多 VRAM；
  - 'xformers'：较慢，使用较少VRAM，只能与 *Mixed Precision* ='fp16'一起使用（对Apple Silicon没有影响）；
  - 'flash_attention'：最慢，需要最低 VRAM。
- *Don't Cache Latents* - 为什么这叫做“cache latents”？因为这就是原始脚本使用的内容，我正在尽可能轻松地更新它。无论如何...当此框选中时，latents 将不会被缓存。当 latents 未被缓存时，您将节省一些 VRAM，但训练速度会稍慢。
- *Train Text Encoder* - 不是必需的，但建议使用。需要更多VRAM，可能无法在<12 GB的GPU上工作。极大地改善了输出结果。
- *Prior Loss Weight* - 计算先前损失时要使用的权重。您可能希望将其保留为1。
- *Center Crop* - 如果图像尺寸不正确，则裁剪图像？我不使用此功能，并建议您只是“正确”地裁剪图像。
- *Pad Tokens* - 由于某种原因，将文本标记填充到更长的长度。
- *Shuffle Tags* - 启用此功能可将输入提示视为逗号分隔列表，并对该列表进行洗牌，从而可以实现更好的可编辑性。
- *Max Token Length* - 将标记器的默认限制提高到 75 以上。需要Pad Tokens> 75。
- *AdamW Weight Decay* - 用于训练的 AdamW Optimizer 的权重衰减。值越接近 0，越接近您的训练数据集，值越接近 1，越具有一般性并偏离您的训练数据集。默认值为1e-2，==建议使用低于0.1的值==。



## Concepts

这个 UI 暴露了三个概念，这似乎是一次训练的合理数量。

如果您希望同时使用三个以上的概念，则可以完全忽略此部分，而是使用 “Parameters” 选项卡下 “Miscellaneous” 部分中的 "Use Concepts List" 选项。

您可以参考[示例概念列表](https://github.com/d8ahazard/sd_dreambooth_extension/blob/main/dreambooth/concepts_list.json)以获取JSON格式的示例。您可以理论上以此方法使用任意数量的概念。

### Concept Parameters

以下是用于训练概念的各种参数列表。

- *Maximum Training Steps* - 训练概念的总寿命训练步骤数。将其保留为-1以使用全局值。
- *Dataset Directory* - 实例图像所在的目录。
- *Classification Dataset Directory* - 存储类图像的目录。留空以保存到模型目录。

#### Filewords

以下值将与提示中的 [filewords] 标签一起使用，以添加/删除标签。有关更多信息，请参见下面的 'Using [filewords]' 部分。

- *Instance Token* - 您的主题的唯一标识符。 (sks，xyz)。留空以进行微调。
- *Class Token* - 您的主题是什么。如果 xyz 是人，则可以是 person/man/woman。

### Prompts

- *Instance Prompt* - 用于您的实例图像的提示。使用 [filewords] 插入或组合现有标签与 tokens。
- *Class Prompt* - 用于生成和训练类图像的提示。使用 [filewords] 插入或组合现有标签与 tokens。
- *Classification Image Negative Prompt* - 在生成类图像时，将使用此负面提示来指导图像生成。
- *Sample Image Prompt* - 生成示例图像时使用的提示。使用 [filewords] 插入或组合现有标签与 tokens。
- *Sample Prompt Template File* - 用于生成示例图像的现有txt文件。[filewords] 和 [names] 将替换为 instance token。
- *Sample Image Negative Prompt* - 在生成示例图像时，将使用此负面提示来指导图像生成。

### Image Generation

- *Total Number of Class/Reg Images* - 将生成多少分类图像。将其保留在 0 以禁用先验保留。
- *Classification/Sample CFG Scale* - 生成图像时的 Classifier Free Guidance scale
- *Classification/Sample Steps* - 生成每个图像时要使用的 steps 数量。
- *Number of Samples to Generate* - 要生成多少个样本图像。
- *Sample Seed* - 用于一致样本生成的种子。设置为-1以使用随机种子。

#### Using [filewords]

每个概念都允许您使用来自实例和类图像的图像文件名或附带的文本文件的提示。

要指示训练器使用现有文件中的提示，请在 instance/class/sample prompts 中使用 `[filewords]`。

为了正确插入和删除现有提示中的单词，我们需要让训练器知道哪些单词表示我们的主题名称和类别。

为此，我们指定了一个 instance token 和 class token。如果您的主题称为 'zxy'，并且它是一个男人，则您的 instance token 将是'zxy'，而您的 class token 将是'man'。

现在，在构建提示时，可以根据需要插入或删除主题和类别。

## Debugging

这里是一堆我添加的随机东西，看起来很有用，但似乎没有其他地方适合。

*Preview Prompts* - 返回将用于训练的提示的JSON字符串。它不太好看，但您可以知道是否会正常工作。

*Generate Sample Image* - 使用下面指定的种子和提示生成示例。

*Sample Prompt* - 示例应该是什么。

*Sample Seed* - 用于您的示例的种子。将其保留在 -1 以使用随机种子。

*Train Imagic Only* - Imagic is basically dreambooth,，但仅使用一个图像，速度显着更快。

如果使用 Imagic，则将使用第一个概念的实例数据目录中的第一个图像进行训练。

有关更多详细信息，请参见：https://github.com/ShivamShrirao/diffusers/tree/main/examples/imagic

### Continuing Training

一旦模型已经训练了任意数量的步骤，就会保存一个包含 UI 中所有参数的配置文件。

如果您希望继续训练模型，只需从下拉列表中选择模型名称，然后单击模型名称下拉列表旁边的蓝色按钮以加载先前的参数。

![image-20230423160355577](C:\Users\wt\Desktop\image-20230423160355577.png)