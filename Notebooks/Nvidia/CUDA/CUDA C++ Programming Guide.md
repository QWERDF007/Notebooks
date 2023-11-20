# [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

# 1. Introduction

## 1.1. The Benefits of Using GPUs

GPU (图形处理器) 在相差不大价格和功率的情况下，提供比 CPU 更高的指令吞吐量和内存带宽。许多应用程序利用这些更高的能力在 GPU 上比在 CPU 上运行得更快 (请参见 GPU 应用程序)。其他计算设备，如 FPGA，也非常节能，但提供的编程灵活性远不如 GPU。

GPU 和 CPU 之间的这种能力差异是因为它们的设计目标不同。虽然 CPU 旨在尽可能快地执行称为线程的一系列操作，并且可以并行执行几十个这些线程，但 GPU 旨在并行执行数千个线程 (通过分摊较慢的单线程性能以实现更大的吞吐量)。

GPU 专门用于高度并行的计算，因此设计时更多的晶体管用于数据处理而不是数据缓存和流控制。图 1 显示了 CPU 与 GPU 的芯片资源分配示例分布。

<img src="./assets/CUDA-C++-guide-fig1.jpg">

**图 1**：GPU 将更多晶体管用于数据处理

将更多的晶体管用于数据处理，例如浮点运算，有利于高度并行的计算；GPU 可以通过计算来隐藏内存访问延迟，而不是依赖于大型数据缓存和复杂的流控制来避免长时间的内存访问延迟，这两者对于晶体管而言都是昂贵的。

通常，一个应用程序既有并行部分也有顺序部分，因此系统采用 GPU 和 CPU 的组合以最大限度地提高整体性能。具有高度并行性的应用程序可以利用 GPU 的大规模并行特性以实现比在 CPU 上更高的性能。

## 1.2. CUDA®: A General-Purpose Parallel Computing Platform and Programming Model

2006 年 11 月，NVIDIA® 推出了 CUDA®，这是一个通用的并行计算平台和编程模型，利用 NVIDIA GPU 中的并行计算引擎以比 CPU 更高效的方式解决许多复杂的计算问题。

CUDA 附带一个软件环境，允许开发人员使用 C++ 作为高级编程语言。如图2所示，还支持其他语言、应用程序编程接口或基于指令的方法，例如 FORTRAN、DirectCompute、OpenACC。

<img src="./assets/CUDA-C++-guide-fig2.jpg">

## 1.3. A Scalable Programming Model

多核 CPU 和多核 GPU 的出现意味着主流处理器芯片现在都是并行系统。面临的挑战是开发能够透明地扩展其并行性的应用程序软件，以利用数量不断增加的处理器核心，就像 3D 图形应用程序透明地将其并行性扩展到具有不同数量核心的多核 GPU 一样。

CUDA 并行编程模型旨在克服这一挑战，同时为熟悉 C 等标准编程语言的程序员保持较低的学习曲线。

其核心是三个关键抽象——线程组的层次结构、共享内存和栅障同步 (barrier synchronization) ——它们作为一组最小的语言扩展简单地暴露给程序员。

这些抽象提供了细粒度的数据并行性和线程并行性，嵌套在粗粒度的数据并行性和任务并行性中。它们引导程序员将问题划分为可以由线程块独立并行解决的粗略子问题，并将每个子问题划分为可以由块内的所有线程并行协作解决的更精细的部分。

这种分解通过允许线程在解决每个子问题时进行合作来保留语言表达能力，同时实现自动可扩展性。事实上，每个线程块都可以以任何顺序（同时或顺序）调度到 GPU 内的任何可用多处理器上，以便编译后的 CUDA 程序可以在任意数量的多处理器上执行，如图 3 所示，并且运行时系统仅需要知道物理多处理器的数量。

这种可扩展的编程模型允许 GPU 架构通过简单地扩展多处理器和内存分区的数量来跨越广泛的市场范围：从高性能发烧友 GeForce GPU 和专业的 Quadro 和 Tesla 计算产品到各种廉价的主流 GeForce GPU（有关所有[支持 CUDA 的 GPU 的](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus)列表，请参阅支持 CUDA 的 GPU）。

<img src="./assets/CUDA-C++-guide-fig3.jpg">

**图 3**：自动的可扩展性

> Note：
>
> GPU 围绕流式多处理器 (SM) 阵列构建（有关更多详细信息，请参阅[硬件实现](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)）。多线程程序被划分为彼此独立执行的线程块，因此具有更多多处理器的 GPU 会比具有更少多处理器的 GPU 在更短的时间内自动执行程序。

## 1.4. Document Structure

- [Introduction](#1-Introduction) 是对 CUDA 的一般介绍。
- [Programming Model](#2-Programming-Model) 概述了 CUDA 编程模型。

# 2. Programming Model

本章介绍了 CUDA 编程模型背后的主要概念，概述了它们在 C++ 中的体现。

CUDA C++ 的详细描述见 [Programming Interface](#3-Programming-Interface)。

本章和下一章中使用的向量加法示例的完整代码可在 [vectorAdd CUDA 示例](https://docs.nvidia.com/cuda/cuda-samples/index.html#vector-addition)中找到。

## 2.1. Kernels

CUDA C++ 通过允许程序员定义称为内核的 C++ 函数来扩展 C++，这些函数在调用时由 N 个不同的CUDA 线程并行执行 N 次，而不是像常规 C++ 函数那样只执行一次。

内核使用 `__global__` 声明说明符定义，并使用新的 `<<<...>>>` 执行配置语法 (参见 [C++ 语言扩展](#7.-C++-Language-Extensions)) 指定执行该内核的 CUDA 线程数。**每个执行内核的线程都会被赋予一个唯一的线程 ID**，该 ID 可以在内核中通过内置变量 `threadIdx` 进行访问。

作为说明，下面的示例代码使用内置变量 `threadIdx`，对两个大小为 N 的向量 A 和 B 进行加法，并将结果存储到向量 C 中：

```c++
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

这里，执行 `VecAdd()` 的 N 个线程每个都执行一个成对的加法。

## 2.2. Thread Hierarchy

为方便起见，`threadIdx` 是一个 3 分量的向量，因此可以使用一维、二维或三维的线程索引来标识线程，形成一维、二维或三维的线程块，称为线程块 (thread block)。这提供了一种自然的方式来调用域中元素（例如向量、矩阵或体积）的计算。

线程的索引和线程 ID 之间的关系很简单：对于一维块，它们是相同的；对于大小为 (Dx,Dy) 的二维块，索引为 (x,y) 的线程的线程 ID 为 (x + y Dx)； 对于大小为 (Dx,Dy,Dz) 的三维块，索引为 (x,y,z) 的线程的线程 ID 为 (x + y Dx + z Dx Dy)。

例如，下面的代码对大小为 NxN 的两个矩阵 A 和 B 进行加法，并将结果存储到矩阵 C 中：

```c++
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

每个块的线程数量是有限的，因为块中的所有线程预期驻留在同一个流式多处理器核心上，并且必须共享该核心的有限内存资源。在当前的 GPU 上，**一个线程块最多可包含 1024 个线程**。

然而，一个内核可以由多个形状相同的线程块来执行，因此线程总数等于每个块的线程数乘以块数。

如图 4 所示，块被组织成一维、二维或三维线程块网格。网格中的线程块数量通常由要处理的数据大小决定，该数据通常超过了系统中的处理器数量。

<img src="./assets/CUDA-C++-guide-fig4.jpg">

**图 4**：线程块网格

在 `<<<...>>>` 语法中指定的每个块中的线程数和网格中的块数可以是 `int` 或 `dim3` 类型。 可以如上例所示，指定二维块或网格。

网格中的每个块可以通过一个一维、二维或三维的唯一索引来标识，该索引可以通过内置变量 `blockIdx` 在内核中访问。线程块的维度可以通过内置变量 `blockDim` 在内核中访问。

将前面的 `MatAdd()` 示例扩展为处理多个块，代码如下:

```c++
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

16x16 (256 个线程) 的线程块大小虽然在本例中是任意的，但却是常见的选择。像前面一样，网格是用足够多的块创建的，以便每个矩阵元素都有一个线程处理。为简单起见，这个示例假设每个维度的网格的线程数能被该维度中每个块的线程数整除，但实际上不一定如此。

线程块需要独立执行：必须能够以任何顺序（并行或串行）执行它们。这一独立性要求使得线程块可以按图 3 所示在任意数量的核心上以任何顺序调度，使程序员可以编写随核心数量扩展的代码。

块内的线程可以通过某些共享内存共享数据并同步其执行来协调内存访问来进行协作。更准确地说，可以通过调用内在函数 `__syncthreads()` 在内核中指定同步点；`__syncthreads()` 充当所有线程必须等待的屏障，然后才允许继续执行。[Shared Memory](#3.2.4.-Shared-Memory) 给出了使用共享内存的示例。除了`__syncthreads()` 之外，[Cooperative Groups API](#8.-Cooperative-Groups) 还提供了一组丰富的线程同步基元。

为了高效协作，共享内存预期是一个低延迟内存，靠近每个处理器核心 (有点像L1缓存)，并且预期`__syncthreads()` 开销很小。

### 2.2.1. Thread Block Clusters



# 3. Programming Interface



## 3.2. CUDA Runtime



### 3.2.4. Shared Memory





# 7. C++ Language Extensions

## 7.1. Function Execution Space Specifiers

### 7.1.1. \_\_global\_\_



# 8. Cooperative Groups











































