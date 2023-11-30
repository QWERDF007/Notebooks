# [入门](https://github.com/facebookresearch/faiss/wiki/Getting-started)

对于以下内容，我们假设已安装 Faiss。我们提供 C++ 和 Python 代码示例。该代码可以通过复制/粘贴或从 Faiss 发行版的 [`tutorial/`](https://github.com/facebookresearch/faiss/tree/master/tutorial) 子目录运行来运行。

## 准备一些数据

Faiss 处理固定维度 $d$ 的向量集合，典型的是一些 10 维到 100 维。这些集合能被存储在矩阵中。我们假设行为主存储，即第 i 个向量的第 j 个分量存储在矩阵的第 i 行第 j 列。Faiss 仅使用 32 位浮点矩阵。

我们需要两个矩阵：

- `xb` 表示数据库，包含所有必须被检索的向量，以及我们要搜索的向量。它的大小是 $n_b \times d$
- `xq` 表示查询向量，我们要找到最近邻向量的向量。它的大小是 $n_q \times d$。如果我们只有单个查询向量，$n_q = 1$

在下面的示例中，我们要处理的向量是从一个 $d = 64$ 维的均匀分布中生成的。只是为了好玩，我们在第一个维度上添加一个取决于向量索引的平移。

### In Python

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

### In C++

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

## 构建一个 index 并给它添加向量

Faiss 是围绕 Index 对象构建的。它封装了数据库向量的集合，并可选地对他们进行预处理，使搜索高效。`Index` 有很多种类型，我们将使用最简单的版本暴力 L2 距离搜索：`IndexFlatL2`。

所有的 Index 在构建时需要知道它们作用的向量的维度，在我们的例子中是 $d$ 。然后，大多数 Index 需要一个训练阶段，来分析向量的分布。对于 `IndexFlatL2`，我们可以跳过这个操作。

当 Index 构建好并且训练后，在 Index 上可执行两个操作：`add` 和 `search`。

要向 Index 中添加元素，我们在 `xb` 上调用 `add`。我们也可以显示 Index 的两个状态变量：`is_trained`，一个表示是否需要训练的布尔值，`ntotal`，索引化后的向量的数量。

一些 Index (但不是 `IndexFlatL2`) 还可以存储对应每个向量的整型 IDs。如果不提供 IDs，`add` 仅使用向量的序号作为 id，例如第一个向量是 0，第二个是 1，依此类推。

### In Python

```python
import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)
```

### In C++

```c++
faiss::IndexFlatL2 index(d);           // call constructor
printf("is_trained = %s\n", index.is_trained ? "true" : "false");
index.add(nb, xb);                     // add vectors to the index
printf("ntotal = %ld\n", index.ntotal);
```

### 结果

这应该只显示 true（ Index 已训练）和 100000（向量存储在 Index 中）。

## 搜索

可以在一个 Index 上执行的基本操作是 k 近邻搜索，例如，对于每个查询向量，在数据库中找到它的 `k` 个最近邻。

这个操作的结果能够方便地存储在一个大小为 $n_q \times k$ 的整型矩阵中，第 i 行包含查询向量 i 的邻居的 id，按距离递增排序。除了这个矩阵，`search` 操作返回一个有对应的平方距离的 $n_q \times k$ 的浮点矩阵。

作为完整性检查，我们可以首先搜索一些数据库向量，以确保最近邻确实是向量本身。

### In Python

```python
k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

### In C++

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

这部分内容进行了编辑，否则 C++ 版本会变得很冗长，完整代码可以在 Faiss 的 [`tutorial/cpp`]([tutorial/cpp](https://github.com/facebookresearch/faiss/tree/main/tutorial/cpp)) 子目录中查看。 

### 结果

健全性检查的输出应该类似于

```
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
```

每个查询的最近邻确实是向量的索引，对应的距离为 0。在同一行内，距离是单调递增的。

实际搜索的输出类似于

```
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

由于向向量的第一个分量添加了值，数据集沿着 d 维空间中的第一轴扩散。所以前几个向量的邻居在数据集开始附近，索引约 10000 左右的向量的邻居也在数据集索引 10000 左右。

在 2016 年的机器上执行上述搜索大约需要 3.3 秒。

# 进一步阅读

- 上一章：[安装 Faiss](<Installing Faiss.md>)
- 下一章：[更快的搜索](<Faster search.md>)

