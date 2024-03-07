# [MetricType 和距离](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)

## `MetricType` 和距离的一些说明

Faiss 的 indices 支持两种主要方法：L2 和内积。其他的则由 `IndexFlat` 提供支持。

有关指标的完整列表，[请参阅此处](https://github.com/facebookresearch/faiss/blob/master/faiss/MetricType.h#L44)。

## METRIC_L2

Faiss 使用欧氏距离（L2 距离）的平方，避免了开方操作。这仍然是与欧氏距离单调等价的，但如果需要精确的距离，还需要对结果进行额外的开方操作。

这个指标对数据的旋转（正交矩阵变换）是不变的。

## METRIC_INNER_PRODUCT

这通常用于推荐系统中的最大内积搜索。查询向量的范数不影响结果的排名（当然，数据库向量的范数确实很重要）。这本身不是余弦相似度，除非向量已经被归一化（位于单位超球面的表面上；请参阅下面的余弦相似度）。

## 如何为向量的余弦相似度建立向量索引？

向量 $x$ 和 $y$ 之间的余弦相似度定义为：

$$
\cos(x,y) = \frac{\left< x,y \right>}{\left| x \right| \times \left| y \right|}
$$
它是相似度，而不是距离，人们通常会搜索具有较大相似度的向量。

通过预先归一化查询和数据库向量，可以将问题映射回最大内积搜索。具体步骤如下：

- 使用 `METRIC_INNER_PRODUCT` 构建 Index。
- 在将向量添加到 Index 之前，对它们进行归一化（在 Python 中使用 `faiss.normalize_L2`）。
- 在搜索向量之前，对它们进行归一化。

需要注意的是，这与使用 `METRIC_L2` 的 Index 等效，只是对于归一化向量来说距离通过下式相关： $\left| x - y \right| ^2 = 2 - 2 \times \left< x, y \right>$ 

## 额外指标

`IndexFlat`、`IndexHNSW` 和 `GpuIndexFlat` 支持额外的指标。

支持 [`METRIC_L1`](https://en.wikipedia.org/wiki/Taxicab_geometry)、[`METRIC_Linf`](https://en.wikipedia.org/wiki/Chebyshev_distance) 和 [`METRIC_Lp`](https://en.wikipedia.org/wiki/Lp_space) 指标。`METRIC_Lp` 包括使用 `Index::metric_arg` (C++) 和 `index.metric_arg` (Python) 来设置幂。

[`METRIC_Canberra`](https://en.wikipedia.org/wiki/Canberra_distance)、[`METRIC_BrayCurtis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html) 和 [`METRIC_JensenShannon`](https://en.wikipedia.org/wiki/Jensen–Shannon_divergence) 同样可用。对于 Mahalanobis 见下述

## 如何为马氏距离索引向量？

[马氏距离](https://en.wikipedia.org/wiki/Mahalanobis_distance)相当于变换空间中的 L2 距离。要转换到该空间：

- 计算数据的协方差矩阵
- 将所有向量（查询和数据库）乘以协方差矩阵的 Cholesky 分解的逆矩阵
- 在 METRIC_L2 索引中进行索引

示例：[mahalnobis_to_L2.ipynb](https://gist.github.com/mdouze/6cc12fa967e5d9911580ef633e559476)

如何使用 Faiss 白化数据并计算马氏距离：[demo_whitening.ipynb](https://gist.github.com/mdouze/33fc39927c343c4ca003f1d8f5a412ef)

## 如何对仅支持 L2 的 Indexes 进行最大内积搜索？

可以通过添加一个维度来将向量进行转换，使得最大内积搜索等效于 L2 距离搜索。请参阅 Bachrach 等人于 2014 年在 ACM 推荐系统会议上发表的论文[《使用欧几里得变换加速 Xbox 推荐系统》](http://ulrichpaquet.com/Papers/SpeedUp.pdf)，其中第 3 节将内积计算转换为 L2 距离计算。一个实现示例可以参见 [demo_IP_to_L2.ipynb](https://gist.github.com/mdouze/e4bdb404dbd976c83fe447e529e5c9dc)。

然而，需要注意的是，尽管在数学上是等价的，但这种转换可能与量化（以及可能的其他 Index 结构）交互不佳，请参阅 Morozov 和 Babenko 在 Neurips'18 上发表的论文[《非度量相似图用于最大内积搜索》](https://proceedings.neurips.cc/paper/2018/hash/229754d7799160502a143a72f6789927-Abstract.html)。

## 如何对仅支持最大内积的 Indexes 进行 L2 距离搜索？

逆向转换也使用向量的附加维度，请参阅 Hong 等人在 PAMI'20 上发表的论文[《用于最近邻搜索的不对称映射量化》](https://cse.buffalo.edu/~jsyuan/papers/2020/Asymmetric_Mapping_Quantization_for_Nearest_Neighbor_Search.pdf)。

## 如何找到最远距离的向量，而不是最相近的向量？

对于余弦相似度和内积，只需查询相反的向量即可。

对于 L2 距离，有一个涉及一个附加维度的技巧：[demo_farthest_L2.ipynb](https://gist.github.com/mdouze/c7653aaa8c3549b28bad75bd67543d34#file-demo_farthest_l2-ipynb)

# 进一步阅读
- 上一章：[在GPU上运行](<Running on GPUs.md>)
- 下一章: []()





<!-- 完成标志, 看不到, 请忽略! -->