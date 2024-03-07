# [更低的内存占用](https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint)

## 这占用了太多内存，如何缩小存储空间？

我们看到的 Index `IndexFlatL2` 和 `IndexIVFFlat` 都存储完整的向量。为了扩展到非常大的数据集，Faiss 提供了基于乘积量化器的有损压缩来压缩存储的向量的变体。

向量仍然存储在 Voronoi 单元中，但它们的大小减小到可配置的字节数 m (d 必须是 m 的倍数)。

压缩基于 [Product Quantizer](https://hal.archives-ouvertes.fr/file/index/docid/514462/filename/paper_hal.pdf)，它可以被视为一个额外的量化级别，应用于要编码的向量的子向量。

在这种情况下，由于向量没有被精确存储，因此搜索方法返回的距离也是近似值。

### In Python

```python
nlist = 100
m = 8                             # number of subquantizers
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                    # 8 specifies that each sub-vector is encoded as 8 bits
index.train(xb)
index.add(xb)
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
index.nprobe = 10              # make comparable with experiment above
D, I = index.search(xq, k)     # search
print(I[-5:])
```

### In C++

```c++
    int nlist = 100;
    int k = 4;
    int m = 8;                             // number of subquantizers
    faiss::IndexFlatL2 quantizer(d);       // the other index
    faiss::IndexIVFPQ index(&quantizer, d, nlist, m, 8);

    index.train(nb, xb);
    index.add(nb, xb);
    {       // sanity check
        ...
        index.search(5, xb, k, D, I);
        printf("I=\n");
        ...
        printf("D=\n");
        ...
    }
    {       // search xq
        ...
        index.nprobe = 10;
        index.search(nq, xq, k, D, I);
        printf("I=\n");
        ...
    }
```

### 结果

结果看起来像：

```
[[   0  608  220  228]
 [   1 1063  277  617]
 [   2   46  114  304]
 [   3  791  527  316]
 [   4  159  288  393]]

[[ 1.40704751  6.19361687  6.34912491  6.35771513]
 [ 1.49901485  5.66632462  5.94188499  6.29570007]
 [ 1.63260388  6.04126883  6.18447495  6.26815748]
 [ 1.5356375   6.33165455  6.64519501  6.86594009]
 [ 1.46203303  6.5022912   6.62621975  6.63154221]]
```

我们可以观察到，最近邻居被正确找到 (它是向量ID本身)，但是向量到其自身的估计距离不为 0，尽管它明显低于到其他邻居的距离。这是由于有损压缩造成的。

这里我们将 64 个 32 位浮点数压缩为 8 个字节，因此压缩因子为 32。

当搜索实际查询时，结果如下所示：

```
[[ 9432  9649  9900 10287]
 [10229 10403  9829  9740]
 [10847 10824  9787 10089]
 [11268 10935 10260 10571]
 [ 9582 10304  9616  9850]]
```

可以将它们与之前的 `IVFFlat` 结果进行比较。大多数结果都是错误的，但它们位于空间的正确区域，位于 ID 10000 的周围。对于真实数据，情况更好，因为：

- 均匀分布的数据很难索引，因为不存在可用于聚类或降维的规律性
- 对于自然数据，语义最近邻通常比不相关的结果更接近 query

## 简化指数构建

由于构建索引可能会变得复杂，因此有一个工厂函数可以根据给定的字符串构建它们。上述索引可以通过以下简写获得：

```python
index = faiss.index_factory(d, "IVF100,PQ8")
```

```c++
faiss::Index *index = faiss::index_factory(d, "IVF100,PQ8");
```

替换 `PQ8` 为 `Flat` 以获得 `IndexFlat`。 当对输入向量应用预处理 (PCA) 时，工厂特别有用。例如，将向量缩减到 32 维的 PCA 投影的工厂字符串是：`"PCA32,IVF100,Flat"`。

# 进一步阅读

探索接下来的部分，获取有关索引类型、GPU faiss、编码结构等的更多具体信息。

上一章：[更快的搜索](<Faster search.md>)

下一节：[在 GPU 上运行](<Running on GPUs.md>)





<!-- 完成标志, 看不到, 请忽略! -->