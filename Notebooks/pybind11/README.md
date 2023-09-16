# pybind11 — Seamless operability between C++11 and Python

**pybind11** 是一个轻量级的仅头文件的库，它在 Python 和 C++ 类型之间相互暴露，主要用于创建现有C++代码的Python绑定。它的目标和语法与 David Abrahams 的优秀 Boost.Python 库类似:通过使用编译时反射推断类型信息来最小化传统扩展模块中的样板代码。

Boost.Python 的主要问题以及创建这样一个类似项目的原因——是 Boost。Boost 是一个庞大而复杂的实用程序库套件，可与几乎所有存在的 C++ 编译器一起使用。这种兼容性是有代价的:需要使用神秘的模板技巧和解决方法来支持最古老和最错误补丁的编译器样本。现在 C++11 兼容的编译器已经广泛可用，这种笨重的机器已经成为一个过于庞大和不必要的依赖项。

可以将此库视为 Boost.Python 的一个微小的自包含版本，其中与绑定生成无关的所有内容都被剔除。如果不考虑注释，核心头文件只需要大约 4K 行代码，并依赖于Python (3.6+ 或 PyPy) 和 C++ 标准库。这个紧凑的实现在一定程度上得益于一些新的 C++11 语言特性 (具体来说:元组、lambda 函数和变参模板)。自其创建以来，这个库在许多方面已经超越了 Boost.Python，在许多常见情况下导致了大大简化的绑定代码。

参考文档和教程提供在 https://pybind11.readthedocs.io/en/latest。本手册的PDF版本可在[此处](https://pybind11.readthedocs.io/_/downloads/en/latest/pdf/)获得。源代码始终可在 https://github.com/pybind/pybind11 获得。

## Core features

pybind11可以将以下核心 C++ 特性映射到Python:

* 接受和返回按值、引用或指针传递的自定义数据结构的函数
* 实例方法和静态方法 
* 重载函数
* 实例属性和静态属性
* 任意异常类型
* 枚举
* 回调函数
* 迭代器和范围
* 自定义运算符
* 单继承和多继承
* STL 数据结构
* 带有引用计数的智能指针，如 `std::shared_ptr`
* 正确引用计数的内部引用
* 具有虚 (纯虚) 方法的 C++ 类可以在 Python 中扩展

## Goodies

除了核心功能，pybind11 还提供了一些额外的好处:

* 支持与实现无关的接口的 Python 3.6+ 和 PyPy3 7.3 (pybind11 2.9 是最后一个支持 Python 2 和 3.5 的版本)。
* 可以绑定捕获了变量的 C++11 lambda 函数。lambda 的捕获数据存储在结果 Python 函数对象中。
* pybind11 在可能的情况下使用 C++11 移动构造函数和移动赋值运算符来高效传输自定义数据类型。
* 通过 Python 的缓冲区协议轻松暴露自定义数据类型的内部存储。这非常方便，例如在类似 Eigen 的 C++ 矩阵类和 NumPy 之间快速转换时，无需昂贵的复制操作。
* pybind11 可以自动向量化函数，以便它们透明地应用于一个或多个 NumPy 数组参数的所有条目。
* 通过几行代码就可以支持 Python 的基于切片的访问和赋值操作。
* 一切都包含在几个头文件中; 不需要链接任何额外的库。
* 与 Boost.Python 生成的等效绑定相比，二进制文件通常至少缩小 2 倍。PyRosetta 的一个最近的 pybind11 转换 (一个巨大的 Boost.Python 绑定项目) 报告了5.4 倍的二进制大小缩减和 5.8 倍的编译时间缩减。
* 函数签名在编译时(使用 `constexpr` )预计算，导致更小的二进制文件。
* 通过少量额外工作，C++ 类型可以像常规 Python 对象一样被打包和解包。

## Supported compilers

1. Clang/LLVM 3.3 或更高版本 (对于Apple Xcode 的 clang,这是 5.0.0 或更高版本)

2. GCC 4.8 或更高版本

3. Microsoft Visual Studio 2017 或更高版本

4. Intel 经典 C++ 编译器 18 或更高版本 (ICC 20.2 在 CI 中测试)

5. Cygwin/GCC (以前在 2.5.1 上测试过)

6. NVCC (CUDA 11.0 在 CI 中测试)

7. NVIDIA PGI (20.9 在 CI 中测试)

## Contents:

THE BASICS

- [Installing the library](<./THE BASICS/Installing the library.md>)

- [First steps]()

