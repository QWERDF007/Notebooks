# [更快的搜索](https://github.com/facebookresearch/faiss/wiki/Faster-search)

## 这太慢了，怎样才能让它更快呢？

为了加快搜索速度，可以将数据集分成若干部分。我们在 d 维空间中定义 Voronoi cells，每个数据库向量落在其中一个单元中。在搜索时，只有包含查询向量 `x` 的单元中的数据库向量 `y` 以及一些相邻的向量与查询向量进行比较。

这是通过 `IndexIVFFlat` Index 完成的。这种类型的 Index 需要一个训练阶段，可以在与数据库向量具有相同分布的任何向量集合上执行。在这种情况下，我们只使用数据库向量本身。

`IndexIVFFlat` 还需要另一个 Index，即量化器，用于将向量分配到 Voronoi 单元。每个单元都由一个质心定义，找到向量所属的 Voronoi 单元包括在质心集合中找到向量的最近邻。这是另一个索引的任务，通常是 `IndexFlatL2`。

搜索方法有两个参数：`nlist`，单元格数量，以及 `nprobe`，为执行搜索而访问的单元格数量 (out of `nlist`)。搜索时间大致随着探测次数的增加而线性增加，加上由于量化而产生的一些常数。

### In Python

```python
nlist = 100
k = 4
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist)
assert not index.is_trained
index.train(xb)
assert index.is_trained

index.add(xb)                  # add may be a bit slower as well
D, I = index.search(xq, k)     # actual search
print(I[-5:])                  # neighbors of the 5 last queries
index.nprobe = 10              # default nprobe is 1, try a few more
D, I = index.search(xq, k)
print(I[-5:])                  # neighbors of the 5 last queries
```

### In C++

```c++
    int nlist = 100;
    int k = 4;
    faiss::IndexFlatL2 quantizer(d);       // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    assert(!index.is_trained);
    index.train(nb, xb);
    assert(index.is_trained);
    index.add(nb, xb);
    {       // search xq
        idx_t *I = new idx_t[k * nq];
        float *D = new float[k * nq];
        index.search(nq, xq, k, D, I);
        printf("I=\n");                    // print neighbors of 5 last queries
        ...
        index.nprobe = 10;                 // default nprobe is 1, try a few more
        index.search(nq, xq, k, D, I);
        printf("I=\n");
        ...
    }
```

### 结果

对于 `nprobe=1`，结果如下所示

```
[[ 9900 10500  9831 10808]
 [11055 10812 11321 10260]
 [11353 10164 10719 11013]
 [10571 10203 10793 10952]
 [ 9582 10304  9622  9229]]

```

这些值与暴力搜索相似，但不完全相同 (见上文)。这是因为某些结果不在完全相同的 Voronoi 单元中。因此，访问更多的单元可能会有所帮助。

将 `nprobe` 增加到 10 正是这样做的：

```
[[ 9900 10500  9309  9831]
 [11055 10895 10812 11321]
 [11353 11103 10164  9787]
 [10571 10664 10632  9638]
 [ 9628  9554 10036  9582]]
```

这是正确的结果。请注意，在这种情况下获得完美结果仅仅是数据分布的产物，因为它在 x 轴上有一个强大的成分，这使得它更容易处理。`nprobe` 参数始终是调整结果速度和准确性之间权衡的一种方式。设置 `nprobe = nlist` 给出与暴力搜索相同的结果 (但速度较慢)。

# 进一步阅读

- 上一章：[入门](<Getting started.md>)
- 下一章：[更低的内存占用](<Lower memory footprint.md>)





<!-- 完成标志, 看不到, 请忽略! -->
