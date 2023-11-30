# [Faiss 向量检索](https://github.com/facebookresearch/faiss/wiki)

**Faiss** 是一个用于稠密向量的高效的相似性搜索和聚类的库。它包含的算法可以在任意大小的向量集合中搜索，一直到可能无法放入内存的向量集合。它还包含用于评估和参数调优的支持代码。Faiss 是用 C++ 编写的，有完整的 Python/numpy 接口。一些最有用的算法在 GPU 上实现。

**ps：** Faiss 的工作，就是把我们自己的候选向量集封装成一个 index 数据库，它可以加速我们检索相似向量 Top-K 的过程，其中有些检索还支持GPU构建。

## 什么是相似性搜索

给定一个 $d$ 维向量 $x_i$，Faiss 在内存中构建一个数据结构。构建完数据结构后，当给定一个新的 $d$ 维向量 $x$ 时，它高效地执行操作：
$$
i = \mathop{\arg\min}\limits_{i} \lVert x - x_i \rVert
$$
其中 $\lVert \cdot \rVert$ 是欧拉距离 (L2)。

在 Faiss 术语中，数据结构是一个 `index`，一个具有 `add` 方法来添加向量 $x_i$ 的对象。请注意，$x_i$ 的维度是固定的。计算 $\mathop{\arg\min}$ 是在 index 上的搜索操作。这就是 Faiss 的工作。它还可以：

- 不仅返回最近邻，还可以返回第 2 近，第 3 近，...，第 k 近邻
- 一次性搜索几个向量而不是一个 (批处理)。对于多个 `index` 类型，这比一个向量接一个地搜索要快
- 以精度换速度，例如，对于速度快 10 倍或者内存少 10 倍的方法，给出错误结果的几率为 10%
- 执行最大内积搜索 $\mathop{\arg\max}\limits_i <x,x_i>$ 代替最小欧拉距离搜索。对于其他距离 (L1、Linf 等等) 有有限的支持
- 返回查询点给定半径内的所有元素 (范围搜索)
- 将 `index` 存储在磁盘上而不是内存
- 索引二进制向量而不是浮点向量

## Faiss 的研究基础





## Faiss 文档目录

### Tutorial

- [Installing Faiss](<Installing Faiss.md>)
- [Getting started](<Getting started.md>)
- [Faster search](<Faster search.md>)
- [Lower memory footprint](<Lower memory footprint.md>)
- [Running on GPUs](<Running on GPUs.md>)

### Basics

