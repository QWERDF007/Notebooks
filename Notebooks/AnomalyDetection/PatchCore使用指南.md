# PatchCore 使用指南

## 1. 训练

PatchCore 仓库：https://github.com/amazon-science/patchcore-inspection

示例：

```shell
python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results patch_core -b wideresnet50 -le layer2 -le layer3 --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 -d cable mvtec F:/data/mvtec_anomaly_detection
```

### 参数组

```python
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
```

- `--gpu`：gpu 设备序号
- `--seed`：随机种子
- `--log_group`：结果目录构成之一，组目录
- `--log_project`：结果目录构成之一，项目目录
- `results_path`：结果目录构成之一，顶级目录

- `--save_patchcore_model`：是否保存 patchcore 模型
- `--save_segmentation_images`：是否保存分割图像

#### patch_core

<details><summary><em>[Click to expand]</em></summary>
<br>

```python
@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
```

- `--backbone_names`，`-b`：指定抽取特征的骨干网络
- `--layers_to_extract_from`，`-le`：提取的特征层级
- `--pretrain_embed_dimension`：
- `--target_embed_dimension`：
- `--preprocessing`：
- `--aggregation`：
- `--anomaly_scorer_num_nn`：
- `--patchsize`：
- `--patchscore`：
- `--patchoverlap`：
- `--patchsize_aggregate`：
- `--faiss_on_gpu`：在 gpu 上进行 faiss 搜索
- `--faiss_num_workers`：

</details>

#### sampler

<details><summary><em>[Click to expand]</em></summary>
<br>

```python
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
```

- `name`：采样算法名称，可选 `identity`，`greedy_coreset`，`approx_greedy_coreset`
- `--percentage`，`-p`：特征采样比例

</details>

#### dataset

<details><summary><em>[Click to expand]</em></summary>
<br>

```python
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
```

- `name`：数据集名称，支持 `mvtec`，其他需要添加
- `data_path`：数据集的目录构成之一，顶级目录
- `--subdatasets`，`-d`：数据集的目录构成之一，子数据集目录
- `--train_val_split`：训练验证划分比例
- `--batch_size`：批量大小
- `--num_workers`：加载数据的进程数
- `--resize`：加载图像最初被缩放调整的大小
- `--imagesize`：加载图像被缩放后裁剪为(居中)的大小
- `--augment`：是否增强，无用

</details>