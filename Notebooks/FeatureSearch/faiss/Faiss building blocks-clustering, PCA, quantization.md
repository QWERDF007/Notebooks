# [Faiss 构建模块：聚类、PCA、量化](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization)

Faiss 建立在一些具有非常高效实现的基本算法之上：k-means 聚类、PCA、PQ 编码/解码。

## 聚类

Faiss 提供了一种高效的 k-means 实现。对存储在给定二维张量 `x` 中的一组向量进行聚类操作如下：

```python
ncentroids = 1024
niter = 20
verbose = True
d = x.shape[1]
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
kmeans.train(x)
```

得到的质心位于 `kmeans.centroids`。

随着迭代的目标函数的值（k-means 情况下为总的平方误差）存储在变量 `kmeans.obj` 中，更广泛的统计数据存储在 `kmeans.iteration_stats` 中。

要在 GPU 上运行它，请将选项 `gpu=True` 添加到 Kmeans 构造函数中。这将使用机器上所有可用的 GPU。

## 其他选项

`Kmeans` 对象主要是 C++ 对象 [`Clustering`](https://github.com/facebookresearch/faiss/blob/master/faiss/Clustering.h) 的一个层，该对象的所有字段都可以通过构造函数设置。这些字段包括：

- `nredo`：运行聚类的次数，并保留最佳质心（根据聚类目标函数选择）
- `verbose`：使聚类输出信息更加详细
- `spherical`：执行球形 k-means —— 每次迭代后对质心进行 L2 归一化
- `int_centroids`：将质心坐标取整 (round)
- `update_index`：是否每次迭代后重新训练索引？
- `min_points_per_centroid` / `max_points_per_centroid`：低于设定值，您会收到警告，高于设定值，训练集被子采样
- seed：随机数生成器的种子

## 任务

要在 kmeans 完成训练后计算从一组向量 `x` 到聚类质心的映射，请使用：

```python
D, I = kmeans.index.search(x, 1)
```

这将返回每个线向量在 `x` 中的最近质心 I。D 包含 L2 距离的平方。

对于相反的操作，例如。要找到距计算质心最近的 15 个 `x` 中的点，必须使用一个新的 Index：

```python
index = faiss.IndexFlatL2(d)
index.add(x)
D, I = index.search(kmeans.centroids, 15)
```