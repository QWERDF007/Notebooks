# [在 GPU 上运行](https://github.com/facebookresearch/faiss/wiki/Running-on-GPUs)

Faiss 可以几乎无缝地利用您的 nvidia GPU。

首先，声明一个 GPU 资源，它封装了一块 GPU 内存：

## In Python 声明 GPU 资源

```python
res = faiss.StandardGpuResources()  # use a single GPU
```

## In C++ 声明 GPU 资源

```c++
faiss::gpu::StandardGpuResources res;  // use a single GPU
```

然后使用 GPU 资源构建 GPU Index。

## In Python 构建 GPU Index

```python
# build a flat (CPU) index
index_flat = faiss.IndexFlatL2(d)
# make it into a gpu index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
```

## In C++ 构建 GPU Index

```c++
faiss::gpu::GpuIndexFlatL2 gpu_index_flat(&res, d);
```

注意：单个 GPU 资源对象可以被多个 Index 使用，只要它们不发起并发查询。

获得的 GPU Index 的使用方式与 CPU Index 完全相同：

## In Python 使用 GPU Index

```
gpu_index_flat.add(xb)         # add vectors to the index
print(gpu_index_flat.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = gpu_index_flat.search(xq, k)  # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

## In C++ 使用 GPU Index

```c++
    gpu_index_flat.add(nb, xb);  // add vectors to the index
    printf("ntotal = %ld\n", gpu_index_flat.ntotal);

    int k = 4;
    {       // search xq
        idx_t *I = new idx_t[k * nq];
        float *D = new float[k * nq];

        gpu_index_flat.search(nq, xq, k, D, I);

        // print results
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }
```

## 结果

结果与 CPU 版本相同。另请注意，在小型数据集上，性能提升并不明显。

# 使用多个 GPU

使用多个 GPU 主要是声明多个 GPU 资源。在 python 中，这可以使用 helper `index_cpu_to_all_gpus` 隐式完成。

示例：

## In Python

```python
ngpus = faiss.get_num_gpus()

print("number of GPUs:", ngpus)

cpu_index = faiss.IndexFlatL2(d)

gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    cpu_index
)

gpu_index.add(xb)              # add vectors to the index
print(gpu_index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = gpu_index.search(xq, k) # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

## In C++

```c++
    int ngpus = faiss::gpu::getNumDevices();

    printf("Number of GPUs: %d\n", ngpus);

    std::vector<faiss::gpu::GpuResources*> res;
    std::vector<int> devs;
    for(int i = 0; i < ngpus; i++) {
        res.push_back(new faiss::gpu::StandardGpuResources);
        devs.push_back(i);
    }

    faiss::IndexFlatL2 cpu_index(d);

    faiss::Index *gpu_index =
        faiss::gpu::index_cpu_to_gpu_multiple(
            res,
            devs,
            &cpu_index
        );

    printf("is_trained = %s\n", gpu_index->is_trained ? "true" : "false");
    gpu_index->add(nb, xb);  // vectors to the index
    printf("ntotal = %ld\n", gpu_index->ntotal);

    int k = 4;

    {       // search xq
        idx_t *I = new idx_t[k * nq];
        float *D = new float[k * nq];

        gpu_index->search(nq, xq, k, D, I);

        // print results
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }

    delete gpu_index;

    for(int i = 0; i < ngpus; i++) {
        delete res[i];
    }
```



# 进一部分阅读

- 上一章：[更低的内存占用](<Lower memory footprint.md>)
- 下一章：

