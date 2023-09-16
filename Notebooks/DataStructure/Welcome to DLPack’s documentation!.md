# Welcome to DLPack’s documentation!

## Purpose

为了使 `ndarray` 系统能够与各种框架交互，需要一个稳定的内存数据结构。

`DLPack` 是一个可以在主要框架之间交换的数据结构。它是在许多深度学习系统核心开发者的输入下开发的。亮点包括:

* 最小和稳定：简单的头文件
* 设计跨硬件：CPU、CUDA、OpenCL、Vulkan、Metal、VPI、ROCm、WebGPU、Hexagon
* 已经是一个具有广泛社区采用和支持的标准:
  * NumPy
  * CuPy
  * PyTorch
  * Tensorflow
  * MXNet
  * TVM
  * mpi4py

* 干净的 C ABI 兼容。
  * 意味着您可以从任何语言创建和访问它。
  * 对于构建支持这些数据类型的 JIT 和 AOT 编译器也是必不可少的。

## Scope

DLPack 的主要设计理念是最小主义。DLPack 放弃了 `allocator`、`device` API 的考虑，专注于最小的数据结构。同时仍考虑到了跨硬件支持的需求 (例如，对不支持普通寻址的平台，数据字段是不透明的)。

它也通过删除一些遗留问题来简化设计 (例如，假定所有内容都是行主顺序的 (row major)，可以使用 strides 来支持其他情况，避免考虑更多布局的复杂性)。

## Roadmap

* 可以将C API 暴露为新的 Python 属性 `__dlpack_info__` 以返回 API 和 ABI 版本。(参见 [#34](https://github.com/dmlc/dlpack/issues/34), [#72](https://github.com/dmlc/dlpack/issues/34))
* 澄清对齐要求。(参见 [data-apis/array-api#293](https://github.com/data-apis/array-api/issues/293), [numpy/numpy#20338](https://github.com/numpy/numpy/issues/20338), [data-apis/array-api#293 (评论)](https://github.com/data-apis/array-api/issues/293#issuecomment-964434449))
* 添加对布尔数据类型的支持 (参见 [#75](https://github.com/dmlc/dlpack/issues/75))
* 添加只读标志 (ABI 中断) 或在规范中作出硬性要求,即导入的数组应被视为只读。(参见 [data-apis/consortium-feedback#1(评论)](https://github.com/data-apis/consortium-feedback/issues/1#issuecomment-675857753), [data-apis/array-api#191](https://github.com/data-apis/array-api/issues/191))
* 标准化C接口的流交换。(参见 [#74](https://github.com/dmlc/dlpack/issues/74), [#65](https://github.com/dmlc/dlpack/issues/65))

## DLPack Documentation

- [C API (`dlpack.h`)](https://dmlc.github.io/dlpack/latest/c_api.html)
  - [Macros](https://dmlc.github.io/dlpack/latest/c_api.html#macros)
  - [Enumerations](https://dmlc.github.io/dlpack/latest/c_api.html#enumerations)
  - [Structs](https://dmlc.github.io/dlpack/latest/c_api.html#structs)
- [Python Specification for DLPack](https://dmlc.github.io/dlpack/latest/python_spec.html)
  - [Syntax for data interchange with DLPack](https://dmlc.github.io/dlpack/latest/python_spec.html#syntax-for-data-interchange-with-dlpack)
  - [Semantics](https://dmlc.github.io/dlpack/latest/python_spec.html#semantics)
  - [Implementation](https://dmlc.github.io/dlpack/latest/python_spec.html#implementation)
  - [Reference Implementations](https://dmlc.github.io/dlpack/latest/python_spec.html#reference-implementations)

# Indices and tables

- [Index](https://dmlc.github.io/dlpack/latest/genindex.html)
- [Search Page](https://dmlc.github.io/dlpack/latest/search.html)



# Reference

[下一篇：C API (`dlpack.h`)](<./C API (dlpack.h).md>)
