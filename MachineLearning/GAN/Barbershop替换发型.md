# Barbershop 替换发型、发色

[Barbershop Github](https://github.com/ZPdesu/Barbershop)

## 平台&环境

- Ubuntu 16.04.6 LTS

- Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
- GeForce GTX 1080 Ti
- 32 GB RAM
- CUDA 10.2

下载仓库

```bash
git clone https://github.com/ZPdesu/Barbershop.git
```

进入仓库

```bash
cd Barbershop
```

创建 conda 环境，略微修改，将 yml 文件中的 cuda 改为 10.2，pytorch 改为 cuda10.2 对应的 1.7.1

```bash
conda env create --file environment/environment.yml
```

激活环境

```bash
conda activate Barbershop
```

## 准备数据

下载 [II2S](https://drive.google.com/drive/folders/15jsR9yy_pfDHiS9aE3HcYDgwtBbAneId) 图像，已经对齐，可以用来作模板，放到 `input/face` 目录。

准备一堆待处理的正脸照片，将其放到项目下的目录 `unprocessed` 中，可以通过参数修改至其他目录。

正脸照片要求：脸部不要有任何遮挡，眼镜如无必要也尽可能摘掉。(歪脸或者遮挡会导致效果变差)

对齐数据

```bash
python align_face.py
```

通过参数指定输入/输出目录，其中 `INPUT_DIR` 和 `OUTPUT_DIR` 为指定的输入和输出目录

```bash
python align_face.py -unprocessed_dir ${INPUT_DIR} -output_dir ${OUTPUT_DIR}
```

执行上述命令会自动下载 dlib 的脸部检测模型，若因为网络原因无法下载，可通过科学手段下载，或者[百度网盘](https://pan.baidu.com/s/1Eq7egs0-jRr5IH8HlS_7jA) (6dfr)

## 运行

生成真实的结果

```bash
python main.py --input_dir input/face --im_path1 90.png --im_path2 15.png --im_path3 117.png --sign realistic --smooth 5 --output_dir output
```

其中 `--input_dir` 是存放待处理的图像的目录，`--im_path1` 是待处理的图像，`--im_path2` 是结构图像，即发型模板，`--im_path3` 是外观图像，即发色模板，`--output_dir` 是输出目录。

生成忠于 mask 的图像

```bash
python main.py --im_path1 90.png --im_path2 15.png --im_path3 117.png --sign fidelity --smooth 5
```

执行上述命令，会自动下载 stylegan2 的 pretrained 模型，可以手动下载后放到 `pretrained_models` 目录，也可[手动下载](https://github.com/ZPdesu/Barbershop/issues/7)。

## 效果

```bash
python main.py --input_dir input/face --im_path1 41.png --im_path2 7.png --im_path3 44.png --sign realistic --smooth 5 --output_dir output
```



