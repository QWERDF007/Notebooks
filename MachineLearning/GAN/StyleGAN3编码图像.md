# StyleGAN3 编码图像、图像反演、图像插值

## 训练 StyleGAN3

克隆 StyleGAN3 仓库，可以根据官方仓库提供的 [README](https://github.com/NVlabs/stylegan3) 进行训练

```bash
git clone https://github.com/NVlabs/stylegan3.git
```

```bash
cd stylegan3
```

准备数据集

```bash
python dataset_tool.py --source=/tmp/images --dest=/tmp/dataset.zip --resolution=512x512
```

训练

```bash
python train.py --outdir=stylegan3-runs --cfg=stylegan3-r --data=/tmp/dataset.zip --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=20000 --snap=5
```

## 训练 stylegan3-encoder

下载预训练模型放到 `pretrained` 目录：

- [model_ir_se50.pth](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view)

将 `training/loss_encoder.py` 的 22 行

```python
self.facenet.load_state_dict(torch.load('pretrained/model_ir_se50.pth'))
```

修改为

```python
self.facenet.load_state_dict(torch.load('pretrained/model_ir_se50.pth', map_location='cpu'))
```

多卡训练时，此处会在 gpu 0 上加载多个 `model_ir_se50.pth` 可能会导致 `OOM`。

训练

```bash
python train.py --encoder base --data /tmp/images --gpus 8 --batch 32 --generator stylegan3.pkl --training_steps 20000 --outdir stylegan3-encoder-runs
```

需要注意是否使用 `w_avg`，即是否对编码后的 `w` 叠加 `G.mapping.w_avg`，需要在训练与预测阶段保持一致。

训练过程会在 `stylegan3-encoder-runs/00000-base-images-gpus8-batch32/image_snapshots` 中保存训练过程中的反演效果。

## 编码图像、图像反演

在 `training/dataset_encoder.py` 中添加

```python
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        self.paths = sorted(make_dataset(dataset_dir))
        self.transforms = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = self.transforms(PIL.Image.open(self.paths[i]).convert('RGB'))
        stem = os.path.basename(self.paths[i]).split('.')[0]
        return x,stem
```

然后编写 [`encode_images.py`](../../Code/stylegan3-encoder/encode_images.py)，调用并生成 `w`，shape：[16, 512]。shape 与生成图像分辨率有关，1024 对应shape：[18, 512]，此处生成图像分辨率为 512。[`encode_images.py`](../../Code/stylegan3-encoder/encode_images.py) 部分代码如下：

```python
infer_set = InferenceDataset(srcdir)
E = Encoder(pretrained=encoder_pkl, w_avg=latent_avg)
infer_loader = torch.utils.data.DataLoader(dataset=infer_set, batch_size=batch_size, num_workers=16)

    with torch.no_grad():
        for X, stems in tqdm(infer_loader, total=len(infer_loader)):
            X = X.cuda()
            w = E(X)
            w = w.cpu().numpy()
            for i,stem in enumerate(stems):
                np.save(str(latents_dir / (stem + '.npy')), w[i])
```

图像反演，使用编码的 `w` 通过 stylegan3 生成图像即可。[`encode_images.py`](../../Code/stylegan3-encoder/encode_images.py) 部分代码如下：

```python
synth = G.synthesis(w)
    for stem in stems:
        save_image(synth, str(generated_dir / (stem + '.png')), image_size, image_size)
```

## 图像插值

对两张图像编码生成的 `w` 进行混合 $w_o = \alpha \cdot w_1 + (1 - \alpha) \cdot w_2$ 。