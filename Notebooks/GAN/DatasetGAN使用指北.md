# 环境平台

- Ubuntu 16.04
- CUDA 10.0
- Tensorflow 1.14
- Pytorch 1.3
- GTX 1080Ti

# 训练 StyleGAN2

1. 下载 [StyleGAN2](https://github.com/NVlabs/stylegan2)

   ```bash
   git clone https://github.com/NVlabs/stylegan2
   ```

2. 准备 StyleGAN2 数据

   首先将训练数据裁剪对齐放到同一目录下，然后使用 StyleGAN2 项目下的 dataset_tool.py 制作 tfrecord 格式的数据。

   ```bash
   python dataset_tool.py create_from_images ${OUTPUT} ${INPUT}
   ```

   其中 `INPUT` 是图像目录，`OUTPUT` 是 tfrecord 的输出目录。要求图像是正方形，且图像大小是 2 的幂，也可参考 `create_lsun_wide`  处理长方形形状的图像。本指北使用 512 大小的图像。

3. 训练 StyleGAN2

   `run_training.py` 中将 `sched.minibatch_size` 和 `sched.minibatch_gpu_base` 设置为固定值 32 和 4，注意调整。

   ```bash
   python run_training.py --num-gpus=2 --data-dir=mydata --dataset=mydataset --total-kimg=100000 --config=config-f --mirror-augment=True 
   ```

   可以通过在 `run_training.py` 中指定 `train.resume_pkl` 和 `train.resume_kimg` 来恢复训练，其中 `train.resume_kimg` 与训练过程中的 `scheduler` 相关。

4. 转换 tensorflow checkpoint 到 pytorch

   下载 stylegan2-pytorch

   ```bash
   git clone https://github.com/rosinality/stylegan2-pytorch
   ```

   转换 tensorflow checkpoint 到 pytorch，`config-f` 需要指定 `--channel_multiplier=2`

   ```bash
   python convert_weight.py ${CKPT} --repo ${PATH_TO_STYLEGAN2} --gen --channel_multiplier 2
   ```

# 生成潜在编码

使用 [EditGAN](https://github.com/nv-tlabs/editGAN_release) 中的 DatasetGAN，它支持 StyleGAN2。可以参考 EditGAN 的 README 中的步骤训练一个 `StyleGAN Encoder`，对真实图像进行编码得到潜在编码。

但实际编码效果较差，没办法反演得到与真实图像一致的图像。所以直接使用 StyleGAN 生成图像并同时得到潜在编码，然后对生成图像进行标注。通过实验证明 `latent_to_image()` 返回的 `latent` 即为生成图像的潜在编码，最后通过 `latent_to_image()` 生成一批图像和潜在编码。

   ```python
   import os
   from pathlib import Path
   
   os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
   import numpy as np
   import torch
   import random
   torch.manual_seed(0)
   import json
   
   from tqdm import tqdm, trange
   from PIL import Image
   
   from utils.data_utils import *
   from utils.model_utils import *
   
   import argparse
   import imageio
   
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
   
   def main(args, total):
       np.random.seed(41)
       g_all, _, upsamplers, _, avg_latent = prepare_model(args)
       outdir = Path(args['training_data_path'])
       with torch.no_grad():
           for i in trange(total):
               latent = np.random.randn(1, 512)
               latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
               sample_img, sample_latnet = latent_to_image(
                       g_all, upsamplers, latent, return_only_im=True, process_out=True)
               latent_out = sample_latnet.detach().cpu().numpy()[0]
               np.save(str(outdir / 'latents_image_{}.npy'.format(i)), latent_out)
               imageio.imsave(str(outdir / 'image_{}.png'.format(i)), sample_img[0])
   
   
   if __name__ == '__main__':
       parser = argparse.ArgumentParser()
   
       parser.add_argument('--exp', type=str)
       parser.add_argument('--total', type=int, default=100)
   
       args = parser.parse_args()
       opts = json.load(open(args.exp, 'r'))
   
       main(opts, args.total)
   ```

修改 `--exp` 指定的 json 文件中的 `stylegan_checkpoint` 为相应的 **pytorch stylegan checkpoint**

```bash
python generate.py --exp ${EXP} --total 1000
```

# 标注分割数据

使用 labelme 等工具对生成图像进行标注，并转换为与生成图像大小一致的 mask 图像，保存为 `image_mask{x}.npy` 文件，`x` 对应为生成图像的标号。( `uint8` 的 `ndarray`， `np.save()` )

# 训练 DatasetGAN

修改 `--exp` 中指定的 json 文件中参数：

- `exp_dir` 实验目录，存放模型及一些东西
- `batch_size` 训练的批量大小
- `category` 修改为 `car` 外的任意值即可，`car` 在代码中会进行特殊处理。 
- `number_class` 为类别数，不包含背景
- `max_training` 为训练数据集的大小
- `dim` 的前两位为 512 或其他相应的大小
- `annotation_mask_path` 修改为图像及标注的目录
- `model_num` 为多少个 MLP 做 ensemble
- `optimized_latent_path` 为图像的潜在编码的目录

将生成图像和对应的标注图像放到 `annotation_mask_path` 指定的目录下，且文件名以 *image_0.jpg* 和 *image_mask0.npy* 开始，否则需要将文件重新命名。处理 `.png` 或者其他类型图像需要修改 `prepare_data()` 中的 

````python
im_name = os.path.join( args['annotation_mask_path'], 'image_%d.jpg' % i)
````

同样，将 `latents` 文件放到 `optimized_latent_path` 指定的目录下，文件名以 *latents_image_0.npy* 开始。

数据准备完毕后，调用 `train_interpreter.py` 

```bash
python train_interpreter.py --exp ${EXP}
```

# 使用 DatasetGAN 生成图像-标注对

在 `train_interpreter.py` 中添加函数 `generate_data()`

```python
def generate_data(args,  total, seed=41, start_step=0, vis=True):
    np.random.seed(seed)

    result_path = os.path.join(args['exp_dir'], 'vis')
    g_all, _, upsamplers, _, avg_latent = prepare_model(args)
    classifier_list = []
    for MODEL_NUMBER in trange(args['model_num']):
        classifier = pixel_classifier(numpy_class=(args['number_class'] + 1), dim=args['dim'][-1])
        checkpoint = torch.load(os.path.join(args['exp_dir'], 'model_' + str(MODEL_NUMBER) + '.pth'))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier = classifier.to(device)
        classifier.eval()
        classifier_list.append(classifier)

    with torch.no_grad():
        for i in trange(total):
            latent = np.random.randn(1, 512)
            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
            img, affine_layers = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1], return_upsampled_layers=True)

            if args['dim'][0] != args['dim'][1]:
                img = img[:, 64:448]
            img = img[0]

            if args['dim'][0] != args['dim'][1]:
                affine_layers = affine_layers[:, :, 64:448]
            affine_layers = affine_layers[0]
            affine_layers = affine_layers.reshape(args['dim'][-1], -1).transpose(1, 0)

            seg_mode_ensemble = []
            for MODEL_NUMBER in range(args['model_num']):
                classifier = classifier_list[MODEL_NUMBER]
                img_seg = classifier(affine_layers)
                img_seg = img_seg.squeeze()

                img_seg_final = oht_to_scalar(img_seg)
                img_seg_final = img_seg_final.reshape(args['dim'][0], args['dim'][1], 1)
                img_seg_final = img_seg_final.cpu().detach().numpy()

                seg_mode_ensemble.append(img_seg_final)
            

            img_seg_final = np.concatenate(seg_mode_ensemble, axis=-1)
            img_seg_final = scipy.stats.mode(img_seg_final, 2)[0].reshape(args['dim'][0], args['dim'][1])
            del (affine_layers)
            if vis:
                color_mask = 0.7 * colorize_mask(img_seg_final, palette) + 0.3 * img

                image_label_name = os.path.join(result_path, "vis_" + str(i + start_step) + '.png')
                image_name = os.path.join(result_path, "vis_" + str(i + start_step) + '_image.png')

                imageio.imwrite(image_label_name,  color_mask.astype(np.uint8))
                imageio.imwrite(image_name, img.astype(np.uint8))
            else:
                image_label_name = os.path.join(result_path, 'label_' + str(i + start_step) + '.png')
                image_name = os.path.join(result_path,  str(i + start_step) + '.png')

                imageio.imwrite(image_label_name,  img_seg_final.astype(np.uint8))
                imageio.imwrite(image_name, img.astype(np.uint8))
```

在 `if __name__ == '__main__':` 中调用 `generate_data()` 生成图像-标注对。

注意，上述代码将论文中的计算不确定性的部分删除。

   





<!-- 完成标志, 看不到, 请忽略! -->