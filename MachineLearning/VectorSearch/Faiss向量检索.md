# Faiss 向量检索

**Faiss** 是一个用于稠密向量的高效的相似性搜索和聚类的库。它包含的算法可以在任意大小的向量集合中搜索，一直到可能无法放入内存的向量集合。它还包含用于评估和参数调优的支持代码。Faiss 是用 C++ 编写的，有完整的 Python/numpy 接口。一些最有用的算法在 GPU 上实现。

**ps：** Faiss 的工作，就是把我们自己的候选向量集封装成一个 index 数据库，它可以加速我们检索相似向量 Top-K 的过程，其中有些检索还支持GPU构建。

## 相似性搜索

给定一个 $d$ 维向量 $x_i$，Faiss 在内存中构建一个数据结构。构建完数据结构后，当给定一个新的 $d$ 维向量 $x$ 时，它高效地执行操作：
$$
i = \mathop{\arg\min}\limits_{i} \lVert x - x_i \rVert
$$
其中 $\lVert \cdot \rVert$ 是欧拉距离 (L2)。

在 Faiss 术语中，数据结构是一个 `index`，一个具有 `add` 方法来添加向量 $x_i$ 的对象。请注意，$x_i$ 的维度是固定的。计算 $\mathop{\arg\min}$ 是在 index 上的搜索操作。这就是 Faiss 的工作。它还可以：

- 不仅返回最近邻，还可以返回第 2 近，第 3 近，...，第 k 近邻
- 一次性搜索几个向量而不是一个 (批处理)。对于多个 index 类型，这比一个向量接一个地搜索要快
- 以精度换速度，例如，对于速度快 10 倍或者内存少 10 倍的方法，给出错误结果的几率为 10%
- 执行最大内积搜索 $\mathop{\arg\max}\limits_i <x,x_i>$ 代替最小欧拉距离搜索。对于其他距离 (L1、Linf 等等) 有有限的支持
- 返回查询点给定半径内的所有元素 (范围搜索)
- 将 index 存储在磁盘上而不是内存
- 索引二进制向量而不是浮点向量

## 安装

使用 conda 安装最新稳定版本：

```bash
# CPU-only version
$ conda install -c pytorch faiss-cpu

# GPU(+CPU) version
$ conda install -c pytorch faiss-gpu

# or for a specific CUDA version
$ conda install -c pytorch faiss-gpu cudatoolkit=10.2 # for CUDA 10.2
```

## 入门指南

### 准备数据

Faiss 处理固定维度 $d$ 的向量集合，典型的是一些 10 维到 100 维。这些集合能被存储在矩阵中。我们假设行为主存储，即第 i 个向量的第 j 个分量存储在矩阵的第 i 行第 j 列。Faiss 仅使用 32 位浮点矩阵。

我们需要两个矩阵：

- `xb` 表示数据库，包含所有必须被检索的向量，以及我们要搜索的向量。它的大小是 $n_b \times d$
- `xq` 表示查询向量，我们要找到最近邻向量的向量。它的大小是 $n_q \times d$。如果我们只有单个查询向量，$n_q = 1$

在下面的示例中，我们要处理的向量是从一个 $d = 64$ 维的均匀分布中生成的。只是为了好玩，我们在第一个维度上添加一个取决于向量索引的平移。

#### In Python

```python
import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.
```

#### In C++

```c++
int d = 64;                            // dimension
int nb = 100000;                       // database size
int nq = 10000;                        // nb of queries
float *xb = new float[d * nb];
float *xq = new float[d * nq];
for(int i = 0; i < nb; i++) {
	for(int j = 0; j < d; j++) xb[d * i + j] = drand48();
	xb[d * i] += i / 1000.;
}
for(int i = 0; i < nq; i++) {
	for(int j = 0; j < d; j++) xq[d * i + j] = drand48();
	xq[d * i] += i / 1000.;
}
```

这个例子使用普通的数组，因为这是所有 C++ 矩阵库支持的最小的共同标准。Faiss 能适应任意矩阵库，只要它提供一个指向底层数据的指针。例如 `std::vector<float>` 的内部指针由 `data()` 方法给出。

### 构建一个 index 并给它添加向量

Faiss 是围绕 `Index` 对象构建的。它封装了数据库向量的集合，并可选地对他们进行预处理，使搜索高效。索引有很多种类型，我们将使用最简单的版本暴力 L2 距离搜索：`IndexFlatL2`。

所有的索引在构建时需要知道它们作用的向量的维度，在我们的例子中是 $d$。然后，大多数索引需要一个训练阶段，来分析向量的分布。对于 `IndexFlatL2`，我们可以跳过这个操作。

当索引构建好并且训练后，在索引上可执行两个操作：`add` 和 `search`。

要向索引中添加元素，我们在 `xb` 上调用 `add`。我们也可以显示索引的两个状态变量：`is_trained`，一个表示是否需要训练的布尔值，`ntotal`，索引化后的向量的数量。

一些索引 (但不是 `IndexFlatL2`) 还可以存储对应每个向量的整型 IDs。如果不提供 IDs，`add` 仅使用向量的序号作为 id，例如第一个向量是 0，第二个是 1，依此类推。

#### In Python

```python
import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)
```

#### In C++

```c++
faiss::IndexFlatL2 index(d);           // call constructor
printf("is_trained = %s\n", index.is_trained ? "true" : "false");
index.add(nb, xb);                     // add vectors to the index
printf("ntotal = %ld\n", index.ntotal);
```

### 搜索

可以在一个索引上执行的基本操作是 k 近邻搜索，例如，对于每个查询向量，在数据库中找到它的 k 个最近邻。

这个操作的结果能够方便地存储在一个大小为 $n_q \times k$ 的矩阵中，第 i 行包含查询向量 i 的邻居的 id，按距离递增排序。除了这个矩阵，`search` 操作返回一个有对应的平方距离的 $n_q \times k$ 的浮点矩阵。

作为完整性检查，我们可以首先搜索一些数据库向量，以确保最近邻确实是向量本身。

#### In Python

```python
k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

#### In C++

```c++
int k = 4;
{       // sanity check: search 5 first vectors of xb
    idx_t *I = new idx_t[k * 5];
    float *D = new float[k * 5];
    index.search(5, xb, k, D, I);
    printf("I=\n");
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < k; j++) printf("%5ld ", I[i * k + j]);
        printf("\n");
    }
    ...
        delete [] I;
    delete [] D;
}
{       // search xq
    idx_t *I = new idx_t[k * nq];
    float *D = new float[k * nq];
    index.search(nq, xq, k, D, I);
    ...
}
```

完整的 C++ 代码参见 [tutorial/cpp](https://github.com/facebookresearch/faiss/tree/main/tutorial/cpp)

### 结果

```python
[[  0 393 363  78]
 [  1 555 277 364]
 [  2 304 101  13]
 [  3 173  18 182]
 [  4 288 370 531]]
[[0.        7.1751738 7.20763   7.2511625]
 [0.        6.3235645 6.684581  6.799946 ]
 [0.        5.7964087 6.391736  7.2815123]
 [0.        7.2779055 7.527987  7.6628466]
 [0.        6.7638035 7.2951202 7.3688145]]

[[ 381  207  210  477]
 [ 526  911  142   72]
 [ 838  527 1290  425]
 [ 196  184  164  359]
 [ 526  377  120  425]]
[[ 9900 10500  9309  9831]
 [11055 10895 10812 11321]
 [11353 11103 10164  9787]
 [10571 10664 10632  9638]
 [ 9628  9554 10036  9582]]
```

