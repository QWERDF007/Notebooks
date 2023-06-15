# 参数的含义

| 参数                        | 说明                                                         |
| --------------------------- | ------------------------------------------------------------ |
| Prompt                      | 提示词（正向）                                               |
| Negative prompt             | 消极的提示词（反向）                                         |
| Width & Height              | 要生成的图片尺寸。尺寸越大，越耗性能，耗时越久。             |
| CFG scale                   | AI 对描述参数（Prompt）的倾向程度。值越小生成的图片越偏离你的描述，但越符合逻辑；值越大则生成的图片越符合你的描述，但可能不符合逻辑。 |
| Sampling method             | 采样方法。有很多种，但只是采样算法上有差别，没有好坏之分，选用适合的即可。 |
| Sampling steps              | 采样步长。太小的话采样的随机性会很高，太大的话采样的效率会很低，拒绝概率高(可以理解为没有采样到,采样的结果被舍弃了)。 |
| Seed                        | 随机数种子。生成每张图片时的随机种子，这个种子是用来作为确定扩散初始状态的基础。不懂的话，用随机的即可。 |
| Only masked padding, pixels | 当选择 “Only Masked” 时，输入到 Stable Diffusion 的图像将被裁剪到修复区域的大小，再加上滑块指定的填充。如果您选择了一个 100 x 100 的区域，并设置了 32 像素的填充，那么输入到 Stable Diffusion 的图像将被裁剪到 164 x 164，然后放大到您的图像生成大小，最后在粘贴回原始图像时缩小回 164 x 164。 |
| Mask blur                   | 在有 mask 的部分图像上应用滑块对应大小的高斯模糊，[这里](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1526#discussioncomment-4259479) |



## Sampling method





- [BooruDataset](https://github.com/starik222/BooruDatasetTagManager)：批量打标签软件

- [birme](https://www.birme.net/?target_width=512&target_height=512)：批量裁剪网页

- Civitai Helper：LORA 缩略图和管理

- LORA 按需分类：进到存放lora的文件夹，在此目录下，建立你想要分类的文件夹，把相应的lora拖进去就行

- 神器 [Tiled VAE](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)

- [Segment Anything for Stable Diffusion WebUI](https://github.com/continue-revolution/sd-webui-segment-anything)

- 



./testModels/detectron2_seg/mask_rcnn_R_50_FPN_3x.yaml



![image-20230426135607277](C:\Users\wt\AppData\Roaming\Typora\typora-user-images\image-20230426135607277.png)



![image-20230426135600435](C:\Users\wt\AppData\Roaming\Typora\typora-user-images\image-20230426135600435.png)

![image-20230426135627450](C:\Users\wt\AppData\Roaming\Typora\typora-user-images\image-20230426135627450.png)

![image-20230426135639626](C:\Users\wt\AppData\Roaming\Typora\typora-user-images\image-20230426135639626.png)

![image-20230426135653650](C:\Users\wt\AppData\Roaming\Typora\typora-user-images\image-20230426135653650.png)
