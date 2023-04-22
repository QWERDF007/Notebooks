参数的含义：

| 参数            | 说明                                                         |
| --------------- | ------------------------------------------------------------ |
| Prompt          | 提示词（正向）                                               |
| Negative prompt | 消极的提示词（反向）                                         |
| Width & Height  | 要生成的图片尺寸。尺寸越大，越耗性能，耗时越久。             |
| CFG scale       | AI 对描述参数（Prompt）的倾向程度。值越小生成的图片越偏离你的描述，但越符合逻辑；值越大则生成的图片越符合你的描述，但可能不符合逻辑。 |
| Sampling method | 采样方法。有很多种，但只是采样算法上有差别，没有好坏之分，选用适合的即可。 |
| Sampling steps  | 采样步长。太小的话采样的随机性会很高，太大的话采样的效率会很低，拒绝概率高(可以理解为没有采样到,采样的结果被舍弃了)。 |
| Seed            | 随机数种子。生成每张图片时的随机种子，这个种子是用来作为确定扩散初始状态的基础。不懂的话，用随机的即可。 |

- 批量打标签软件 BooruDataset

- Civitai Helper：LORA 缩略图和管理

- LORA 按需分类：进到存放lora的文件夹，在此目录下，建立你想要分类的文件夹，把相应的lora拖进去就行

- 神器 [Tiled VAE](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)

- [Segment Anything for Stable Diffusion WebUI](https://github.com/continue-revolution/sd-webui-segment-anything)
