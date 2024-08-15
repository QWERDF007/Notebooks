# [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

# 1. 介绍

## 1.1. 使用 GPU 的好处

GPU (图形处理器) 在相差不大价格和功率的情况下，提供比 CPU 更高的指令吞吐量和内存带宽。许多应用程序利用这些更高的能力在 GPU 上比在 CPU 上运行得更快 (请参见 GPU 应用程序)。其他计算设备，如 FPGA，也非常节能，但提供的编程灵活性远不如 GPU。

GPU 和 CPU 之间的这种能力差异是因为它们的设计目标不同。虽然 CPU 旨在尽可能快地执行称为线程的一系列操作，并且可以并行执行几十个这些线程，但 GPU 旨在并行执行数千个线程 (通过分摊较慢的单线程性能以实现更大的吞吐量)。

GPU 专门用于高度并行的计算，因此设计时更多的晶体管用于数据处理而不是数据缓存和流控制。图 1 显示了 CPU 与 GPU 的芯片资源分配示例分布。

<img src="./assets/CUDA-C++-guide-fig1.jpg">

**图 1**：GPU 将更多晶体管用于数据处理

将更多的晶体管用于数据处理，例如浮点运算，有利于高度并行的计算；GPU 可以通过计算来隐藏内存访问延迟，而不是依赖于大型数据缓存和复杂的流控制来避免长时间的内存访问延迟，这两者对于晶体管而言都是昂贵的。

通常，一个应用程序既有并行部分也有顺序部分，因此系统采用 GPU 和 CPU 的组合以最大限度地提高整体性能。具有高度并行性的应用程序可以利用 GPU 的大规模并行特性以实现比在 CPU 上更高的性能。

## 1.2. CUDA®: 一个通用的并行计算平台和编程模型

2006 年 11 月，NVIDIA® 推出了 CUDA®，这是一个通用的并行计算平台和编程模型，利用 NVIDIA GPU 中的并行计算引擎以比 CPU 更高效的方式解决许多复杂的计算问题。

CUDA 附带一个软件环境，允许开发人员使用 C++ 作为高级编程语言。如图2所示，还支持其他语言、应用程序编程接口或基于指令的方法，例如 FORTRAN、DirectCompute、OpenACC。

<img src="./assets/CUDA-C++-guide-fig2.jpg">

## 1.3. 一个可扩展的编程模型

多核 CPU 和多核 GPU 的出现意味着主流处理器芯片现在都是并行系统。面临的挑战是开发能够透明地扩展其并行性的应用程序软件，以利用数量不断增加的处理器核心，就像 3D 图形应用程序透明地将其并行性扩展到具有不同数量核心的多核 GPU 一样。

CUDA 并行编程模型旨在克服这一挑战，同时为熟悉 C 等标准编程语言的程序员保持较低的学习曲线。

其核心是三个关键抽象——线程组的层次结构、共享内存和栅障同步 (barrier synchronization) ——它们作为一组最小的语言扩展简单地暴露给程序员。

这些抽象提供了细粒度的数据并行性和线程并行性，嵌套在粗粒度的数据并行性和任务并行性中。它们引导程序员将问题划分为可以由线程块独立并行解决的粗略子问题，并将每个子问题划分为可以由块内的所有线程并行协作解决的更精细的部分。

这种分解通过允许线程在解决每个子问题时进行合作来保留语言表达能力，同时实现自动可扩展性。事实上，每个线程块都可以以任何顺序（同时或顺序）调度到 GPU 内的任何可用多处理器上，以便编译后的 CUDA 程序可以在任意数量的多处理器上执行，如图 3 所示，并且运行时系统仅需要知道物理多处理器的数量。

这种可扩展的编程模型允许 GPU 架构通过简单地扩展多处理器和内存分区的数量来跨越广泛的市场范围：从高性能发烧友 GeForce GPU 和专业的 Quadro 和 Tesla 计算产品到各种廉价的主流 GeForce GPU（有关所有[支持 CUDA 的 GPU 的](#6. 支持 CUDA 的 GPUs)列表，请参阅支持 CUDA 的 GPU）。

<img src="./assets/CUDA-C++-guide-fig3.jpg">

**图 3**：自动的可扩展性

> Note：
>
> GPU 围绕流式多处理器 (SM) 阵列构建（有关更多详细信息，请参阅[硬件实现](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation)）。多线程程序被划分为彼此独立执行的线程块，因此具有更多多处理器的 GPU 会比具有更少多处理器的 GPU 在更短的时间内自动执行程序。

## 1.4. 文档结构

- [介绍](#1. 介绍) 是对 CUDA 的一般介绍。
- [编程模型](#2. 编程模型) 概述了 CUDA 编程模型。

# 2. 编程模型

本章介绍了 CUDA 编程模型背后的主要概念，概述了它们在 C++ 中的体现。

CUDA C++ 的详细描述见 [编程接口](#3-编程接口)。

本章和下一章中使用的向量加法示例的完整代码可在 [vectorAdd CUDA 示例](https://docs.nvidia.com/cuda/cuda-samples/index.html#vector-addition) 中找到。

## 2.1. Kernels

CUDA C++ 通过允许程序员定义称为 *kernels* 的 C++ 函数来扩展 C++，这些函数在调用时由 N 个不同的CUDA 线程并行执行 N 次，而不是像常规 C++ 函数那样只执行一次。

内核使用 `__global__` 声明说明符定义，并使用新的 `<<<...>>>` 执行配置语法 (参见 [C++ 语言扩展](#7. C++ 语言扩展)) 指定执行该内核的 CUDA 线程数。**每个执行内核的线程都会被赋予一个唯一的线程 ID**，该 ID 可以在内核中通过内置变量 `threadIdx` 进行访问。

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

## 2.2. 线程层次结构

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

块内的线程可以通过某些共享内存共享数据并同步其执行来协调内存访问来进行协作。更准确地说，可以通过调用内在函数 `__syncthreads()` 在内核中指定同步点；`__syncthreads()` 充当所有线程必须等待的屏障 (barrier)，然后才允许继续执行。[共享内存](#3.2.4. 共享内存) 给出了使用共享内存的示例。除了`__syncthreads()` 之外，[Cooperative Groups API](#8.-Cooperative-Groups) 还提供了一组丰富的线程同步基元。

为了高效协作，共享内存预期是一个低延迟内存，靠近每个处理器核心 (有点像L1缓存)，并且预期`__syncthreads()` 开销很小。

### 2.2.1. 线程块集群

随着 NVIDIA [Compute Capability 9.0](#16.8. 计算能力 9.0) 的推出，CUDA 编程模型引入了一个可选的层次结构级别，称为由线程块组成的线程块集群。类似于线程块中的线程被保证在流处理器上同时调度，线程块集群中的线程块也被保证在 GPU 处理集群 (GPC) 上同时调度。

类似于线程块，集群也可以组织成一维、二维或三维，如图 5 所示。集群中的线程块数量可以由用户定义，并且集群中最多支持 8 个线程块作为 CUDA 中一个可移植的集群大小。注意，在 GPU 硬件或 MIG 配置上太小上而无法支持 8 个多处理器时，最大集群大小将相应减小。这些较小配置以及支持超过 8 个线程块集群大小的更大配置的标识是特定于架构的，可以通过 `cudaOccupancyMaxPotentialClusterSize` API 查询。

<img src="./assets/CUDA-C++-guide-fig5.jpg">

**图 5**：线程块集群

> Note：
>
> 在使用集群支持启动的内核中，出于兼容性目的，gridDim 变量仍然表示线程块数量的大小。可以使用 Cluster Group API 找到集群中块的排名。

可以使用编译器时期内核属性 `__cluster_dims__(X,Y,Z)` 或使用 CUDA 内核启动 API  `cudaLaunchKernelEx` 在内核中启用线程块集群。下面的示例展示了如何使用编译器时期内核属性启动集群。使用属性定义的集群大小在编译时固定，然后可以使用经典的 `<<< , >>>` 启动内核。如果内核使用编译器时期的集群大小，则在启动内核时无法修改集群大小。

```c++
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

线程块集群大小也可以在运行时设置，内核可以使用 CUDA 内核启动 API `cudaLaunchKernelEx` 启动。下面的代码示例展示了如何使用可扩展 API 启动集群内核。

```c++
// Kernel definition
// No compile time attribute attached to the kernel
__global__ void cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Kernel invocation with runtime cluster size
    {
        cudaLaunchConfig_t config = {0};
        // The grid dimension is not affected by cluster launch, and is still enumerated
        // using number of blocks.
        // The grid dimension should be a multiple of cluster size.
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}
```

在计算能力 9.0 的 GPU 中，集群中的所有线程块都保证在单个 GPU 处理集群 (GPC) 上共同调度，并允许集群中的线程块使用 Cluster Group API  `cluster.sync()` 执行硬件支持的同步。集群组还提供了成员函数来查询集群组大小，即线程数目的 `num_threads()` 和块数目的 `num_blocks()` API。集群组中线程或块的 rank 可以分别通过 `dim_threads()` 和 `dim_blocks()` API查询。

属于一个集群的线程块可以访问分布式共享内存 (Distributed Shared Memory)。集群中的线程块有能力对分布式共享内存中的任意地址进行读、写和原子操作。[分布式共享内存](#3.2.5. 分布式共享内存)给出了在分布式共享内存中执行直方图的示例。

## 2.3. 内存层次结构

CUDA 线程在执行期间可以访问多个内存空间中的数据，如图 6 所示。每个线程都有私有的本地内存。每个线程块都有对该块的所有线程可见的共享内存，并且与该块具有相同的生命周期。线程块集群 (thread block cluster) 中的线程块可以对彼此的共享内存执行读、写和原子操作。所有线程都可以访问相同的全局内存。

还有两个额外的只读内存空间供所有线程访问：常量内存空间和纹理内存空间。全局、常量和纹理内存空间针对不同的内存使用 (见[设备内存访问](#5.3.2. 设备内存访问)) 进行了优化。纹理内存还为某些特定的数据格式提供不同的寻址模式以及数据过滤 (见[纹理和表面内存](#3.2.14. 纹理和表面内存))。

全局、常量和纹理内存空间在同一应用程序启动的内核之间是持久的。

<img src="./assets/CUDA-C++-guide-fig6.jpg">

**图 6**：内存层次结构

## 2.4. 异构编程

如图 7 所示，CUDA 编程模型假设 CUDA 线程在与运行 C++ 程序的主机物理分离的设备上执行。例如，当内核在 GPU 上执行而 C++ 程序的其余部分在 CPU 上执行时就是这种情况。

CUDA 编程模型还假设主机和设备都在 DRAM 中维护各自独立的内存空间，分别称为主机内存和设备内存。因此，程序通过调用 CUDA 运行时 (在[编程接口](#3. 编程接口)中描述) 来管理内核可见的全局、常量和纹理内存空间。这包括设备内存的分配和释放以及主机和设备内存之间的数据传输。

统一内存提供了托管内存 (managed memory) 来桥接主机和设备内存空间。托管内存可作为具有公共地址空间的单一的、一致的内存映像，可从系统中的所有 CPU 和 GPU 进行访问。这种功能支持设备内存的超额订阅，并且无需在主机和设备上显式镜像数据，从而大大简化移植应用程序的任务。请参阅[统一内存编程](#19. 统一内存编程)。

<img src="./assets/CUDA-C++-guide-fig7.jpg">

**图 7**：异构编程

> Note：
>
> 串行代码在主机上执行，而并行代码在设备上执行。

## 2.5. 异步 SIMT 编程模型

在 CUDA 编程模型中，线程是进行计算或内存操作的最低级抽象。从基于 NVIDIA Ampere GPU 架构的设备开始，CUDA 编程模型通过异步编程模型为内存操作提供加速。异步编程模型定义了与 CUDA 线程相关的异步操作的行为。

异步编程模型定义了用于 CUDA 线程之间同步的[异步屏障](#7.26. 异步屏障)的行为。该模型还解释和定义了如何使用 `cuda::memcpy_async` 在 GPU 中计算时异步地从全局内存中移动数据。

### 2.5.1. 异步操作

一个异步操作被定义为由一个 CUDA 线程发起并像由另一个线程异步执行的操作。在一个合理的程序中，一个或多个 CUDA 线程与该异步操作同步。发起异步操作的 CUDA 线程不需要是参与同步的线程之一。

这样的一个异步线程 (as-if 线程) 总是与发起异步操作的 CUDA 线程相关联。异步操作使用同步对象来同步操作的完成。这样的同步对象可以由用户显式管理 (例如 `cuda::memcpy_async`)，或在库中隐式管理 (例如 `cooperative_groups::memcpy_async`)。

同步对象可以是一个 `cuda::barrier` 或 `cuda::pipeline`。这些对象在[异步屏障](#7.26. 异步屏障)和[使用 cuda::pipeline 的异步数据复制](#7.27. 异步数据复制)中有详细解释。这些同步对象可以在不同的线程作用域使用。作用域定义了可以使用同步对象来与异步操作同步的线程集合。下表定义了 CUDA C++ 中可用的线程作用域以及可以与每个线程同步的线程。

| 线程作用域                                | 描述                                                    |
| ----------------------------------------- | ------------------------------------------------------- |
| `cuda::thread_scope::thread_scope_thread` | 只有发起异步操作的 CUDA 线程才会同步。                  |
| `cuda::thread_scope::thread_scope_block`  | 与发起线程相同线程块中的所有或任何 CUDA 线程同步。      |
| `cuda::thread_scope::thread_scope_device` | 与发起线程相同 GPU 设备中的所有或任何 CUDA 线程同步。   |
| `cuda::thread_scope::thread_scope_system` | 与发起线程相同系统中的所有或任何 CUDA 或 CPU 线程同步。 |

这些线程作用域作为 CUDA 标准 C++ 库中标准 C++ 的扩展来实现。

## 2.6. 计算能力

设备的计算能力由一个版本号表示，有时也称为它的 "SM 版本"。这个版本号标识了 GPU 硬件支持的功能，并由应用程序在运行时使用来确定当前 GPU 上可用的硬件功能和/或指令。

计算能力包含主版本号 `X` 和次版本号 `Y`，表示为 `X.Y`。

主版本号相同的设备属于同一核心架构。主版本号为 9 的设备基于 NVIDIA Hopper GPU 架构，8 为基于 NVIDIA Ampere GPU 架构的设备，7 为基于 Volta 架构的设备，6 为基于 Pascal 架构的设备，5 为基于 Maxwell 架构的设备，3 为基于 Kepler 架构的设备。

次版本号对应于核心架构的增量改进，可能包括新特性。

计算能力为 7.5 的设备的架构是 Turing，它是基于 Volta 架构的增量更新。

[支持 CUDA 的 GPUs](#6. 支持 CUDA 的 GPUs) 列出了所有支持 CUDA 的设备及其计算能力。[计算能力](#16. 计算能力)给出了每种计算能力的技术规格。

> Note
>
> 不要将特定 GPU 的计算能力版本与 CUDA 版本 (例如 CUDA 7.5、CUDA 8、CUDA 9) 相混淆，后者是 CUDA 软件平台的版本。应用程序开发人员使用 CUDA 平台来创建在多代 GPU 架构 (包括未来尚未发明的 GPU 架构) 上运行的应用程序。尽管新的 CUDA 平台版本通常通过支持该架构的计算能力版本来添加对新 GPU 架构的本地支持，但新的 CUDA 平台版本通常也包括与硬件生成无关的软件功能。

从 CUDA 7.0 和 CUDA 9.0 开始，分别不再支持 Tesla 和 Fermi 架构。

# 3. 编程接口

CUDA C++ 为熟悉 C++ 编程语言的用户提供了一条简单的路径，可以轻松编写供设备执行的程序。

它由一组最小的 C++ 语言扩展和一个运行时库组成。

核心语言扩展已经在[编程模型](#2. 编程模型)中引入。它们允许程序员将内核定义为 C++ 函数，并使用一些新语法在每次调用函数时指定网格和块的维度。所有扩展的完整描述可以在 [C++ 语言扩展](#7. C++ 语言扩展)中找到。包含这些扩展中的任何一个的源文件都必须按照[使用 NVCC编译](3.1. 使用 NVCC 编译)中的说明使用 `nvcc` 进行编译。

运行时在 [CUDA 运行时](#<3.2. CUDA 运行时>)中介绍。它提供了在主机上执行的 C 和 C++ 函数，用于分配和释放设备内存、在主机内存和设备内存之间传输数据、管理具有多个设备的系统等。运行时的完整描述可以在 CUDA 参考手册中找到。

运行时是建立在一个更低级的 C API 之上，即 CUDA 驱动程序 API，应用程序也可以访问该 API。驱动程序 API 通过暴露更低级别的概念（如 CUDA 上下文——设备的主机进程的模拟，以及 CUDA 模块——设备的动态加载库的模拟）提供额外的控制级别。大多数应用程序不使用驱动程序 API，因为它们不需要这种额外的控制级别，并且在使用运行时时，上下文和模块管理是隐式的，从而产生更简洁的代码。由于运行时可与驱动程序 API 互操作，因此大多数需要某些驱动程序 API 功能的应用程序可以默认使用运行时 API，并且仅在需要时使用驱动程序 API。驱动程序 API 在[驱动程序 API](#17. 驱动程序 API) 中介绍，并在参考手册中完整描述。

## 3.1. 使用 NVCC 编译

Kernels 可以使用称为 PTX 的 CUDA 指令集架构来编写，在 PTX 参考手册中有描述。但是，使用像 C++ 这样的高级编程语言通常会更有效。在这两种情况下，kernels 都必须由 `nvcc` 编译成二进制代码才能在设备上执行。

`nvcc` 是一个编译器驱动程序，它简化了编译 C++ 或 PTX 代码的过程：它提供了简单和熟悉的命令行选项，并通过调用实现不同编译阶段的工具集合来执行这些选项。本节概述了 `nvcc` 工作流程和命令选项。完整的描述可以在 `nvcc` 用户手册中找到。

### 3.1.1. 编译工作流程

#### 3.1.1.1. 离线编译

使用 `nvcc` 编译的源文件可以包含主机代码 (即在主机上执行的代码) 和设备代码 (即在设备上执行的代码) 的混合。`nvcc` 的基本工作流程包括将设备代码与主机代码分离，然后:

* 将设备代码编译成汇编形式 (PTX 代码) 和/或二进制形式 (cubin 对象)，

* 修改主机代码，用必要的 CUDA 运行时函数调用替换 [Kernels](#2.1. Kernels) 中引入的 `<<<...>>>` 语法 (在执行配置中有更详细的描述)，以从 PTX 代码和/或 cubin 对象加载和启动每个编译好的 kernel。

修改后的主机代码要么以 C++ 代码的形式输出，留待使用其他工具编译，要么通过让 `nvcc` 在最后一个编译阶段调用主机编译器直接输出为对象代码。

应用程序然后可以：

* 链接到编译后的主机代码 (这是最常见的情况)，

* 或者忽略修改后的主机代码 (如果有的话)，并使用 CUDA 驱动程序API (见[驱动程序 API](#17. 驱动程序 API)) 来加载和执行 PTX 代码或 cubin 对象。

#### 3.1.1.2. 即时编译

应用程序在运行时加载的任何 PTX 代码都会由设备驱动程序进一步编译为二进制代码。这称为"即时编译"。即时编译会增加应用程序的加载时间，但允许应用程序受益于随每个新设备驱动程序带来的任何编译器改进。正如[应用程序兼容性](#3.1.4. 应用兼容性)中所详细解释的，这也是应用程序在编译时不存在的设备上运行的唯一方法。

当设备驱动程序为某个应用程序即时编译某些 PTX 代码时，它会自动缓存生成二进制代码的副本，以避免在后续调用应用程序时重复编译。缓存（称为计算缓存）在设备驱动程序升级时会自动失效，以便应用程序可以受益于内置于设备驱动程序的新即时编译器的改进。

环境变量可用于控制即时编译，如[CUDA 环境变量](#18. CUDA 环境变量)中所述。

作为使用 `nvcc` 来编译 CUDA C++ 设备代码的替代方法，可以使用 NVRTC 在运行时编译 CUDA C++ 设备代码为 PTX。NVRTC 是 CUDA C++ 的运行时编译库；更多信息可以在 NVRTC 用户指南中找到。

### 3.1.2. 二进制兼容性

二进制代码是特定于架构的。使用指定目标架构的编译器选项 `-code` 生成 cubin 对象：例如，使用 `-code=sm_80` 编译会为计算能力为 8.0 的设备生成二进制代码。从一个次要版本到下一个次要版本保证二进制兼容性，但从一个次要版本到前一个次要版本或跨主要版本不保证兼容性。换句话说，为计算能力 X.y 生成的 cubin 对象仅能在计算能力 X.z（其中 z≥y）的设备上执行。

> Note
> 二进制兼容性仅对桌面平台支持。它不支持 Tegra。桌面平台和 Tegra 之间也不支持二进制兼容性。

### 3.1.3. PTX 兼容性

一些 PTX 指令只在更高计算能力的设备上支持。例如，[Warp Shuffle 函数](#7.22. Warp Shuffle 函数)仅在计算能力为 5.0 及更高的设备上支持。`-arch` 编译器选项指定将 C++ 编译为 PTX 代码时假定的计算能力。因此，包含 warp shuffle 的代码必须使用 `-arch=compute_50` (或更高版本) 进行编译。

为某些特定计算能力生成的 PTX 代码总是可以编译成更高或相等计算能力的二进制代码。注意，从早期的 PTX 版本编译的二进制代码可能无法利用某些硬件特性。例如，目标是 Volta 架构 (计算能力 7.0) 设备的二进制代码，如果是从 Pascal 架构 (计算能力 6.0) 生成的 PTX 编译的，则不会利用 Tensor Core 指令,因为这在 Pascal 上不可用。因此，最终的二进制代码的性能可能不如使用最新版本的 PTX 生成二进制代码时的潜在性能。

### 3.1.4. 应用兼容性

为了在特定计算能力的设备上执行代码，应用程序必须加载与该计算能力兼容的二进制代码或 PTX 代码。特别是，为了能够在计算能力更高的未来架构 (目前还无法为其生成二进制代码) 的设备上执行代码，应用程序必须加载将为这些设备即时编译的 PTX 代码。

嵌入在 CUDA C++ 应用程序中的 PTX 和二进制代码由 `-arch` 和 `-code` 编译器选项或 `-gencode` 编译器选项控制，详见 `nvcc` 用户手册。例如，

```shell
nvcc x.cu 
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_60,code=sm_60 
        -gencode arch=compute_70,code=\"compute_70,sm_70\"
```

嵌入与计算能力 5.0 和 6.0 兼容的二进制代码 (第一个和第二个 `-gencode` 选项) 以及与计算能力 7.0 兼容的 PTX 和二进制代码 (第三个 `-gencode` 选项)。

生成主机代码，以便在运行时自动选择最合适的代码进行加载和执行。在上例中，将是：

- 适用于计算能力为 5.0 和 5.2 的设备的 5.0 二进制代码
- 适用于计算能力为 6.0 和 6.1 的设备的 6.0 二进制代码
- 适用于计算能力为 7.0 和 7.5 的设备的 7.0 二进制代码
- 针对计算能力为 8.0 和 8.6 的设备在运行时编译成二进制代码的 PTX 代码

`x.cu` 可以具有优化的代码路径，例如使用 warp reduction operations，这只在计算能力为 8.0 及更高的设备上支持。可以使用 `__CUDA_ARCH__` 宏根据计算能力区分各种代码路径。它仅针对设备代码定义。例如，使用 `-arch=compute_80` 编译时，`__CUDA_ARCH__` 等于 `800`。

使用驱动程序 API 的应用程序必须将代码编译到单独的文件中，并在运行时显式加载和执行最合适的文件。

Volta 架构引入了"独立线程调度"，这改变了 GPU 上线程的调度方式。对于依赖于先前架构中 [SIMT 调度](#4.1. SIMT 架构)的特定行为的代码，独立线程调度可能会改变参与线程的集合，导致错误的结果。为了帮助迁移，同时实现[独立线程调度](#16.6.2. 独立线程调度)中详细说明的更正措施，Volta 开发人员可以通过组合编译器选项 `-arch=compute_60 -code=sm_70` 选择 Pascal 的线程调度。

`nvcc` 用户手册列出了 `-arch`、`-code` 和 `-gencode` 编译器选项的各种缩写。例如，`-arch=sm_70` 是 `-arch=compute_70 -code=compute_70,sm_70` 的缩写 (与 `-gencode arch=compute_70,code=\"compute_70,sm_70\"` 相同)。

### 3.1.5. C++ 兼容性

编译器的前端根据 C++ 语法规则处理 CUDA 源文件。主机代码完全支持完整的 C++。但是，如[C++ 语言支持](#14. C++ 语言支持)中所述，设备代码只完全支持 C++ 的一个子集。

### 3.1.6. 64 位兼容性

`nvcc` 的 64 位版本以 64 位模式编译设备代码 (即指针为 64 位)。以 64 位模式编译的设备代码仅与以 64 位模式编译的主机代码一起支持。

## 3.2. CUDA 运行时

运行时在 `cudart` 库中实现，可以通过 `cudart.lib` 或 `libcudart.a` 静态链接，也可以通过 `cudart.dll` 或 `libcudart.so` 动态链接。需要 `cudart.dll` 和/或 `cudart.so` 进行动态链接的应用程序通常会将它们作为应用程序安装包的一部分包含在内。只有链接到相同版本的 CUDA 运行时的组件之间传递 CUDA 运行时符号的地址才是安全的。

所有入口点都带有 `cuda` 前缀。

如[异构编程](#2.4. 异构编程)中所述，CUDA 编程模型假设系统由一个主机和一个设备组成，主机和设备都有自己独立的内存。[设备内存](#3.2.2. 设备内存)概述了用于管理设备内存的运行时函数。  

[共享内存](#3.2.4. 共享内存)阐述了如何利用[线程层次结构](#2.2. 线程层次结构)中引入的共享内存来最大限度地提高性能。  

[锁页主机内存](#3.2.6. 锁页主机内存)介绍了锁页主机内存，以便在主机与设备内存之间的数据传输期间重叠内核执行。

[异步并发执行](#3.2.8. 异步并发执行)描述了在系统的各个级别实现异步并发执行所使用的概念和 API。  

[多设备系统](#3.2.9. 多设备系统)展示了当多个设备连接到同一主机时，编程模型如何扩展。

[错误检查](#3.2.12. 错误检查)描述了如何正确检查运行时生成的错误。

[调用堆栈](#3.2.13. 调用堆栈)提到了用于管理 CUDA C++ 调用堆栈的运行时函数。

[纹理和表面内存](#3.2.14. 纹理和表面内存)介绍了纹理和表面内存空间，它们提供了另一种访问设备内存的方式；它们还暴露了 GPU 纹理硬件的一部分。

[图形互操作性](#3.2.15. 图形互操作性)介绍了运行时提供的各种函数，用于与两种主要的图形 API（OpenGL 和 Direct3D）进行互操作。

### 3.2.1. 初始化

从 CUDA 12.0 开始，调用 `cudaInitDevice()` 和 `cudaSetDevice()` 初始化运行时和指定设备关联的主上下文。如果没有这些调用，运行时将隐式地使用设备 0 并根据需要进行自初始化以处理其他运行时 API 请求。在计时运行时函数调用和解释对运行时的第一次调用的错误代码时，需要记住这一点。在 12.0 之前，`cudaSetDevice()` 不会初始化运行时，应用程序通常使用无操作运行时调用 cudaFree(0) 来隔离运行时初始化与其他 API 活动（为了计时和错误处理）。

运行时为系统中的每个设备创建一个 CUDA 上下文（有关 CUDA 上下文的更多详细信息，请参阅[上下文](#<17.1. 上下文>)）。此上下文是该设备的*主上下文*，并在需要该设备上的活动上下文的第一个运行时函数中初始化。它由应用程序的所有主机线程共享。作为此上下文创建的一部分，如果需要，设备代码将被即时编译（请参阅即时编译）并加载到设备内存中。所有这些都是透明发生的。如果需要（例如，为了驱动程序 API 的互操作性），可以从驱动程序 API 访问设备的主上下文，如运行时和驱动程序 API 之间的互操作性中所述。

当主机线程调用 `cudaDeviceReset()` 时，这将销毁主机线程当前操作的设备的主上下文（即，[设备选择](#<3.2.9.2. 设备选择>)中定义的当前设备）。任何具有该设备作为当前设备的主线程的下一个运行时函数调用将为此设备创建一个新的主上下文。

>**注意**
>
>CUDA 接口使用在主机程序初始化期间初始化并在主机程序终止期间销毁的全局状态。CUDA 运行时和驱动程序无法检测此状态是否无效，因此在程序初始化期间或 main 之后终止期间（隐式或显式）使用任何这些接口将导致未定义的行为。
>
>从 CUDA 12.0 开始，`cudaSetDevice()` 现在将在更改主机线程的当前设备后显式初始化运行时。以前的 CUDA 版本延迟了新设备上的运行时初始化，直到在 `cudaSetDevice()` 之后进行第一次运行时调用。此更改意味着现在检查 `cudaSetDevice()` 的返回值以获取初始化错误非常重要。
>
>参考手册中错误处理和版本管理部分的运行时函数不会初始化运行时。

### 3.2.2. 设备内存

如异构编程中所述，CUDA 编程模型假设一个由主机和设备组成的系统，每个系统都有自己的独立内存。内核从设备内存中运行，因此运行时提供了分配、释放和复制设备内存以及在主机内存和设备内存之间传输数据的函数。

设备内存可以分配为线性内存或 CUDA 数组。

CUDA 数组是针对纹理获取优化的不透明内存布局。它们在纹理和表面内存中描述。

线性内存分配在一个统一的地址空间中，这意味着单独分配的实体可以通过指针相互引用，例如在二叉树或链表中。地址空间的大小取决于主机系统（CPU）和所用 GPU 的计算能力：

表 1 线性内存地址空间

|                              | x86_64 (AMD64) | POWER (ppc64le) | ARM64        |
| ---------------------------- | -------------- | --------------- | ------------ |
| 最高至计算能力 5.3 (Maxwell) | 40bit          | 40bit           | 40bit        |
| 计算能力 6.0 (Pascal) 或更新 | 最高至 47bit   | 最高至 49bit    | 最高至 48bit |

> **注意**
>
> 在计算能力为 5.3（Maxwell）及更早的设备上，CUDA 驱动程序创建一个未提交的 40 位虚拟地址保留，以确保内存分配（指针）落在支持的范围内。此保留显示为保留的虚拟内存，但直到程序实际分配内存才会占用任何物理内存。

线性内存通常使用 `cudaMalloc()` 分配，使用 `cudaFree()` 释放，主机内存和设备内存之间的数据传输通常使用 `cudaMemcpy()` 完成。在内核的向量加法代码示例中，需要将向量从主机内存复制到设备内存：

```c++
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Host code
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    ...

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    ...
}
```

线性内存还可以通过 `cudaMallocPitch()` 和 `cudaMalloc3D()` 分配。对于二维或三维数组的分配，推荐使用这些函数，因为它们可以确保分配适当填充以满足设备内存访问中描述的对齐要求，从而在访问行地址或在二维数组和其他设备内存区域之间执行复制（使用 `cudaMemcpy2D()` 和 `cudaMemcpy3D()` 函数）时确保最佳性能。返回的 pitch（或步幅）必须用于访问数组元素。

以下代码示例分配了一个 `width` x `height` 的二维浮点值数组，并展示了如何在设备代码中循环遍历数组元素：

```c++
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch,
                width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}
```

以下代码示例分配了一个 `width` x `height` x `depth` 的浮点值 3D 数组，并展示了如何在设备代码中循环遍历数组元素：

```c++
// Host code
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float),
                                    height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth)
{
    char* devPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}
```

>**注意**
>
>为了避免过度分配内存并影响系统整体性能，请根据问题规模向用户请求分配参数。如果分配失败，您可以回退到其他较慢的内存类型（`cudaMallocHost()`、`cudaHostRegister()` 等），或返回错误消息告知用户所需的内存量。如果您的应用程序由于某种原因无法请求分配参数，我们建议在支持的平台上使用 `cudaMallocManaged()`。

参考手册列出了用于在 `cudaMalloc()` 分配的线性内存、`cudaMallocPitch()` 或 `cudaMalloc3D()` 分配的线性内存、CUDA 数组以及为全局或常量内存空间声明的变量分配的内存之间复制内存的各种函数。 

以下代码示例说明了通过运行时 API 访问全局变量的各种方法：

```c++
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```

`cudaGetSymbolAddress()` 用于检索指向全局内存空间中为变量分配的内存的地址。分配内存的大小可以通过 `cudaGetSymbolSize()` 获取。

### 3.2.3. 设备内存 L2 访问管理

当 CUDA 内核重复访问全局内存中的数据区域时，可以认为这些数据访问是持久的。另一方面，如果数据只被访问一次，则可以认为这些数据访问是流式的。 从 CUDA 11.0 开始，计算能力为 8.0 及以上的设备能够影响数据在 L2 缓存中的持久性，从而潜在地提供更高的带宽和更低的全局内存访问延迟。

#### 3.2.3.1. L2 缓存保留用于持久访问

L2 缓存的一部分可以被保留用于持久访问全局内存的数据。持久访问优先使用 L2 缓存的这部分，而普通或流式访问全局内存只有在持久访问未使用这部分 L2 缓存时才能使用。 用于持久访问的 L2 缓存保留大小可以在一定范围内调整：

```c++
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/
```

当 GPU 配置为多实例 GPU (MIG) 模式时，L2 缓存保留功能被禁用。 当使用多进程服务 (MPS) 时，无法通过 `cudaDeviceSetLimit` 更改 L2 缓存保留大小。相反，保留大小只能在 MPS 服务器启动时通过环境变量 `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT` 指定。

#### 3.2.3.2. 持久访问的 L2 策略

访问策略窗口指定了全局内存的一个连续区域和该区域内访问在 L2 缓存中的持久性属性。 

下面的代码示例展示了如何使用 CUDA Stream 设置 L2 持久访问窗口。

**CUDA Stream 示例**

```c++
cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                              // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

当一个内核随后在 CUDA `stream` 中执行时，全局内存范围 `[ptr..ptr+num_bytes)` 内的内存访问比其他全局内存位置的访问更有可能保留在 L2 缓存中。 

L2 持久性也可以为 CUDA 图形内核节点设置，如下例所示：

**CUDA GraphKernelNode 示例**

```c++
cudaKernelNodeAttrValue node_attribute;                                     // Kernel level attributes data structure
node_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
node_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                            // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
node_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
node_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
node_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA Graph Kernel node of type cudaGraphNode_t
cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);
```

`hitRatio` 参数可以用来指定获得 `hitProp` 属性的访问比例。在上面的两个例子中，全局内存区域 `[ptr..ptr+num_bytes)` 中 60% 的内存访问具有持久性属性，40% 的内存访问具有流式属性。哪些特定的内存访问被分类为持久性（`hitProp`）是随机的，概率大约为 `hitRatio`；概率分布取决于硬件架构和内存范围。

例如，如果 L2 保留缓存大小为 16KB，而 `accessPolicyWindow` 中的 `num_bytes` 为 32KB：

- 当 `hitRatio` 为 0.5 时，硬件会随机选择 32KB 窗口中的 16KB 作为持久性数据，并将其缓存到保留的 L2 缓存区域中。
- 当 `hitRatio` 为 1.0 时，硬件将尝试将整个 32KB 窗口缓存到保留的 L2 缓存区域中。由于保留区域小于窗口，缓存行将被逐出以保持最近使用的 32KB 数据中的 16KB 在 L2 缓存的保留部分。

因此，`hitRatio` 可以用来避免缓存行的抖动，并总体上减少进出 L2 缓存的数据量。

低于 1.0 的 `hitRatio` 值可以用来手动控制不同 CUDA streams 的 `accessPolicyWindows` 在 L2 中缓存的数据量。例如，假设 L2 保留缓存大小为 16KB；两个不同的 CUDA streams 中的两个并发内核，每个内核都有 16KB 的 `accessPolicyWindow`，并且两个内核的 `hitRatio` 值都为 1.0，可能会在竞争共享 L2 资源时逐出彼此的缓存行。但是，如果两个 `accessPolicyWindow 的 hitRatio` 值为 0.5，它们不太可能逐出自己或彼此的持久性缓存行。

#### 3.2.3.3. L2 访问属性

针对不同的全局内存数据访问定义了三种访问属性：

- `cudaAccessPropertyStreaming`：以流式属性发生的内存访问不太可能保留在 L2 缓存中，因为这些访问优先被逐出。
- `cudaAccessPropertyPersisting`：以持久性属性发生的内存访问更有可能保留在 L2 缓存中，因为这些访问优先保留在 L2 缓存的保留部分。
- `cudaAccessPropertyNormal`：此访问属性强制将先前应用的持久性访问属性重置为正常状态。来自先前 CUDA 内核的具有持久性属性的内存访问可能会在其预期使用后长时间保留在 L2 缓存中。这种使用后的持久性减少了可用于后续不使用持久性属性的内核的 L2 缓存量。使用 `cudaAccessPropertyNormal` 属性重置访问属性窗口会删除先前访问的持久性（优先保留）状态，就好像先前访问没有访问属性一样。

#### 3.2.3.4. L2 持久性示例

以下示例演示了如何为持久访问保留 L2 缓存，通过 CUDA Stream 在 CUDA 内核中使用保留的 L2 缓存，然后重置 L2 缓存。

```C++
cudaStream_t stream;
cudaStreamCreate(&stream);                                                                  // Create CUDA stream

cudaDeviceProp prop;                                                                        // CUDA device properties variable
cudaGetDeviceProperties( &prop, device_id);                                                 // Query GPU properties
size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size);                                  // set-aside 3/4 of L2 cache for persisting accesses or the max allowed

size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);                        // Select minimum of user defined num_bytes and max window size.

cudaStreamAttrValue stream_attribute;                                                       // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);               // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // Number of bytes for persistence access
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                                        // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;               // Persistence Property
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;                // Type of access property on cache miss

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Set the attributes to a CUDA Stream

for(int i = 0; i < 10; i++) {
    cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);                                 // This data1 is used by a kernel multiple times
}                                                                                           // [data1 + num_bytes) benefits from L2 persistence
cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);                                     // A different kernel in the same stream can also benefit
                                                                                            // from the persistence of data1

stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // Setting the window size to 0 disable it
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Overwrite the access policy attribute to a CUDA Stream
cudaCtxResetPersistingL2Cache();                                                            // Remove any persistent lines in L2

cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);                                     // data2 can now benefit from full L2 in normal mode
```

#### 3.2.3.5. 重置 L2 访问为正常

来自先前 CUDA 内核的持久性 L2 缓存行可能会在使用后长时间保留在 L2 中。因此，将 L2 缓存重置为正常对于流式或正常内存访问以正常优先级利用 L2 缓存非常重要。有三种方法可以将持久性访问重置为正常状态。

- 使用访问属性 `cudaAccessPropertyNormal` 重置之前的持久性内存区域。
- 通过调用 `cudaCtxResetPersistingL2Cache()` 将所有持久性 L2 缓存行重置为正常。
- 未被使用的行最终会自动重置为正常。强烈不建议依赖自动重置，因为自动重置所需的时间是不确定的。

#### 3.2.3.6. 管理 L2 保留缓存的利用率

在不同 CUDA streams 中并发执行的多个 CUDA 内核可能为其流分配不同的访问策略窗口。但是，L2 保留缓存部分由所有这些并发 CUDA 内核共享。因此，该保留缓存部分的净利用率是所有并发内核的个体使用之和。当持久性访问量超过保留的 L2 缓存容量时，指定内存访问为持久性的好处会减弱。

为了管理保留的 L2 缓存部分的利用率，应用程序必须考虑以下因素：

- L2 保留缓存的大小。
- 可能并发执行的 CUDA 内核。
- 所有可能并发执行的 CUDA 内核的访问策略窗口。
- 何时以及如何重置 L2 以允许正常或流式访问以相同的优先级利用先前保留的 L2 缓存。

#### 3.2.3.7. 查询 L2 缓存属性

与 L2 缓存相关的属性是 `cudaDeviceProp` 结构的一部分，可以使用 CUDA 运行时 API `cudaGetDeviceProperties` 进行查询。 CUDA 设备属性包括：

- `l2CacheSize`：GPU 上可用的 L2 缓存量。
- `persistingL2CacheMaxSize`：可以为持久性内存访问保留的最大 L2 缓存量。
- `accessPolicyMaxWindowSize`：访问策略窗口的最大大小。

#### 3.2.3.8. 控制持久性内存访问的 L2 缓存保留大小

用于持久性内存访问的 L2 保留缓存大小可以使用 CUDA 运行时 API `cudaDeviceGetLimit` 查询，并使用 CUDA 运行时 API `cudaDeviceSetLimit` 作为 `cudaLimit` 设置。设置此限制的最大值为 `cudaDeviceProp::persistingL2CacheMaxSize`。

```c++
enum cudaLimit {
    /* other fields not shown */
    cudaLimitPersistingL2CacheSize
};
```

### 3.2.4. 共享内存

如[变量内存空间说明符](#<7.2. 变量内存空间说明符>)中所述，共享内存使用 `__shared__` 内存空间说明符进行分配。

如线程层次结构中所述，共享内存比全局内存快得多，并在共享内存中有详细介绍。它可以作为 scratchpad memory（或软件管理的缓存）来最小化来自 CUDA 块的全局内存访问，如下面的矩阵乘法示例所示。

下面的代码示例是矩阵乘法的简单实现，没有利用共享内存。每个线程读取 A 的一行和 B 的一列，并计算 C 的相应元素，如图 8 所示。因此，A 从全局内存中读取 B.width 次，B 从全局内存中读取 A.height 次。

```c++
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}
```

<img src="./assets/matrix-multiplication-without-shared-memory.jpg">

图 8：不使用共享内存的矩阵乘法

下面的代码示例是矩阵乘法的实现，它利用了共享内存。在这个实现中，每个线程块负责计算 C 的一个方形子矩阵 Csub，块中的每个线程负责计算 Csub 的一个元素。如图 9 所示，Csub 等于两个矩形矩阵的乘积：A 的子矩阵维度为 (A.width, block_size)，具有与 Csub 相同的行索引，B 的子矩阵维度为 (block_size, A.width)，具有与 Csub 相同的列索引。为了适应设备的资源，这两个矩形矩阵被分割成尽可能多的维度为 block_size 的方形矩阵，Csub 被计算为这些方形矩阵乘积的和。这些乘积中的每一个都是通过首先将两个对应的方形矩阵从全局内存加载到共享内存（每个线程加载每个矩阵的一个元素），然后让每个线程计算乘积的一个元素来执行的。每个线程将这些乘积的每个结果累加到一个寄存器中，完成后将结果写入全局内存。

通过这种方式阻塞计算，我们可以利用快速的共享内存并节省大量全局内存带宽，因为 A 只从全局内存读取 (B.width / block_size) 次，B 只读取 (A.height / block_size) 次。

前一个代码示例中的 Matrix 类型增加了 stride 字段，以便可以使用相同类型有效地表示子矩阵。\_\_device\_\_ 函数用于获取和设置元素，并从矩阵构建任何子矩阵。

```c++
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;
// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}
// Thread block size
#define BLOCK_SIZE 16
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
```

<img src="./assets/matrix-multiplication-with-shared-memory.jpg">

图 9：使用共享内存的矩阵乘法

### 3.2.5. 分布式共享内存

计算能力 9.0 引入了线程块集群，使线程块集群中的线程能够访问集群中所有参与线程块的共享内存。这种分区共享内存称为分布式共享内存，相应的地址空间称为分布式共享内存地址空间。属于线程块集群的线程可以读取、写入或在分布式地址空间中执行原子操作，而不考虑地址是否属于本地线程块或远程线程块。无论内核是否使用分布式共享内存，共享内存大小规范（静态或动态）仍然是每个线程块的。分布式共享内存的大小只是每个集群的线程块数量乘以每个线程块的共享内存大小。

访问分布式共享内存中的数据需要所有线程块存在。用户可以使用 Cluster Group API 中的 `cluster.sync()` 来保证所有线程块都已开始执行。用户还需要确保所有分布式共享内存操作在线程块退出之前发生，例如，如果远程线程块试图读取给定线程块的共享内存，用户需要确保远程线程块读取的共享内存完成之后才能退出。

CUDA 提供了一种访问分布式共享内存的机制，应用程序可以受益于利用其功能。让我们来看一个简单的直方图计算以及如何使用线程块集群在 GPU 上优化它。计算直方图的一种标准方法是在每个线程块的共享内存中进行计算，然后执行全局内存原子操作。这种方法的一个限制是共享内存容量。一旦直方图箱不再适合共享内存，用户需要直接计算直方图，因此需要在全局内存中进行原子操作。使用分布式共享内存，CUDA 提供了一个中间步骤，根据直方图箱的大小，直方图可以计算在共享内存、分布式共享内存或直接在全局内存中。

下面的 CUDA 内核示例展示了如何根据直方图箱的数量计算共享内存或分布式共享内存中的直方图。

```c++
#include <cooperative_groups.h>

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input,
                                   size_t array_size)
{
  extern __shared__ int smem[];
  namespace cg = cooperative_groups;
  int tid = cg::this_grid().thread_rank();

  // Cluster initialization, size and calculating local bin offsets.
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();
  int cluster_size = cluster.dim_blocks().x;

  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    smem[i] = 0; //Initialize shared memory histogram to zeros
  }

  // cluster synchronization ensures that shared memory is initialized to zero in
  // all thread blocks in the cluster. It also ensures that all thread blocks
  // have started executing and they exist concurrently.
  cluster.sync();

  for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
  {
    int ldata = input[i];

    //Find the right histogram bin.
    int binid = ldata;
    if (ldata < 0)
      binid = 0;
    else if (ldata >= nbins)
      binid = nbins - 1;

    //Find destination block rank and offset for computing
    //distributed shared memory histogram
    int dst_block_rank = (int)(binid / bins_per_block);
    int dst_offset = binid % bins_per_block;

    //Pointer to target block shared memory
    int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    //Perform atomic update of the histogram bin
    atomicAdd(dst_smem + dst_offset, 1);
  }

  // cluster synchronization is required to ensure all distributed shared
  // memory operations are completed and no thread block exits while
  // other thread blocks are still accessing distributed shared memory
  cluster.sync();

  // Perform global memory histogram, using the local distributed memory histogram
  int *lbins = bins + cluster.block_rank() * bins_per_block;
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    atomicAdd(&lbins[i], smem[i]);
  }
}
```

上面的内核可以在运行时以集群大小启动，具体取决于所需的分布式共享内存量。如果直方图足够小，可以放入一个块的共享内存中，用户可以使用集群大小 1 启动内核。下面的代码片段展示了如何根据共享内存需求动态启动集群内核。

```c++
// Launch via extensible launch
{
  cudaLaunchConfig_t config = {0};
  config.gridDim = array_size / threads_per_block;
  config.blockDim = threads_per_block;

  // cluster_size depends on the histogram size.
  // ( cluster_size == 1 ) implies no distributed shared memory, just thread block local shared memory
  int cluster_size = 2; // size 2 is an example here
  int nbins_per_block = nbins / cluster_size;

  //dynamic shared memory size is per block.
  //Distributed shared memory size =  cluster_size * nbins_per_block * sizeof(int)
  config.dynamicSmemBytes = nbins_per_block * sizeof(int);

  CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = cluster_size;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, input, array_size);
}
```

### 3.2.6. 锁页主机内存

运行时提供函数允许使用锁定页面（也称为固定）主机内存（与通过 malloc() 分配的常规可分页主机内存相反）：

- `cudaHostAlloc()` 和 `cudaFreeHost()` 分配和释放锁定页面主机内存；
- `cudaHostRegister()` 锁定 `malloc()` 分配的一系列内存（有关限制，请参阅参考手册）。

使用锁定页面主机内存有几个优点：

- 在某些设备上，锁定页面主机内存和设备内存之间的复制可以与内核执行并发进行，如[异步并发执行](#<3.2.8. 异步并发执行>)中所述。
- 在某些设备上，锁定页面主机内存可以映射到设备的地址空间，无需将其复制到或从设备内存中复制，如映射内存中所述。
- 在具有前端总线的系统上，如果主机内存分配为锁定页面，则主机内存和设备内存之间的带宽更高，如果此外分配为写组合，则带宽更高，如写组合内存中所述。

> **注意**
>
> 非 I/O 相干 Tegra 设备上没有缓存锁定页面主机内存。此外，非 I/O 相干 Tegra 设备不支持 `cudaHostRegister()`。 
>

简单的零拷贝 CUDA 示例附带有关锁定页面内存 API 的详细文档。

#### 3.2.6.1. 可移植内存

锁定页面内存块可与系统中的任何设备一起使用（有关多设备系统的更多详细信息，请参阅[多设备系统](#<3.2.9. 多设备系统>)），但默认情况下，上述使用锁定页面内存的优点仅可与分配块时当前的设备一起使用（以及与共享相同统一地址空间的所有设备，如果有的，如[统一虚拟地址空间](#<3.2.10. 统一虚拟地址空间>)中所述）。要使这些优点适用于所有设备，需要通过将标志 `cudaHostAllocPortable` 传递给 `cudaHostAlloc()` 或通过将标志 `cudaHostRegisterPortable` 传递给 `cudaHostRegister()` 来分配块。

#### 3.2.6.2. 写合并内存

默认情况下，锁定页面主机内存被分配为可缓存的。可以通过将标志 `cudaHostAllocWriteCombined` 传递给 `cudaHostAlloc()` 来选择性地将其分配为写合并。写合并内存释放了主机的 L1 和 L2 缓存资源，使更多缓存可用于应用程序的其余部分。此外，写合并内存不会在跨 PCI Express 总线的传输期间被窥探，这可以将传输性能提高多达 40%。

从主机读取写合并内存的速度慢得令人望而却步，因此一般来说，写合并内存应该用于主机只写入的内存。

应避免在写合并内存上使用 CPU 原子指令，因为并非所有 CPU 实现都保证该功能。

#### 3.2.6.3. 映射内存

可以通过将标志 `cudaHostAllocMapped` 传递给 `cudaHostAlloc()` 或通过将标志 `cudaHostRegisterMapped` 传递给 `cudaHostRegister()` 将锁定页面主机内存块映射到设备的地址空间。因此，这样的块通常有两个地址：一个由 `cudaHostAlloc()` 或 `malloc()` 返回的主机内存中的地址，另一个可以通过 `cudaHostGetDevicePointer()` 获取并在内核中用于访问块的设备内存中的地址。唯一的例外是使用 `cudaHostAlloc()` 分配的指针以及当主机和设备使用统一地址空间时，如统一虚拟地址空间中所述。

直接从内核访问主机内存无法提供与设备内存相同的带宽，但有一些优点：

- 无需在设备内存中分配块并在该块和主机内存中的块之间复制数据；数据传输根据内核需要隐式执行；
- 无需使用流（参见并发数据传输）来重叠数据传输和内核执行；内核发起的数据传输自动与内核执行重叠。

然而，由于映射的锁定页面内存在主机和设备之间共享，因此应用程序必须使用流或事件（参见异步并发执行）同步内存访问，以避免任何潜在的读后写、写后读或写后写危险。

为了能够检索任何映射锁定页面内存的设备指针，必须在执行任何其他 CUDA 调用之前，使用 `cudaDeviceMapHost` 标志调用 `cudaSetDeviceFlags()` 来启用锁定页面内存映射。否则，`cudaHostGetDevicePointer()` 将返回错误。

如果设备不支持映射锁定页面主机内存，`cudaHostGetDevicePointer()` 也会返回错误。应用程序可以通过检查 `canMapHostMemory` 设备属性（参见[设备枚举](#<3.2.9.1. 设备枚举>)）来查询此功能，对于支持映射锁定页面主机内存的设备，该属性等于 1。

请注意，对映射锁定页面内存执行的原子函数（参见原子函数）从主机或其他设备的角度来看不是原子的。

还要注意，CUDA 运行时要求从设备启动的 1 字节、2 字节、4 字节和 8 字节自然对齐的加载和存储到主机内存从主机和其他设备的角度保留为单个访问。在某些平台上，对内存的原子操作可能被硬件拆分为单独的加载和存储操作。这些组件加载和存储操作对保留自然对齐访问具有相同的要求。例如，CUDA 运行时不支持 PCI Express 总线拓扑，其中 PCI Express 桥将 8 字节自然对齐写入拆分为设备和主机之间的两个 4 字节写入。

### 3.2.7. 内存同步域



### 3.2.8. 异步并发执行

CUDA 将以下操作作为独立任务公开，这些任务可以彼此并发运行：

- 主机上的计算
- 设备上的计算
- 从主机到设备的内存传输
- 从设备到主机的内存传输
- 给定设备内存内的内存传输
- 设备之间的内存传输

这些操作之间实现的并发级别取决于设备的功能集和计算能力，如下所述。

#### 3.2.8.1. 主机和设备之间的并发执行

异步库函数通过在设备完成请求的任务之前将控制权返回给主机线程来促进并发主机执行。使用异步调用，可以将许多设备操作排队，以便在有可用设备资源时由 CUDA 驱动程序执行。这减轻了主机线程管理设备的大部分责任，使其可以自由地执行其他任务。以下设备操作相对于主机是异步的：

- 内核启动
- 单个设备内存内的内存复制
- 从主机到设备的 64 KB 或更小的内存块的内存复制
- 由后缀为 Async 的函数执行的内存复制
- 内存设置函数调用

程序员可以通过将 `CUDA_LAUNCH_BLOCKING` 环境变量设置为 1 来全局禁用所有在系统上运行的 CUDA 应用程序的内核启动的异步性。此功能仅用于调试目的，不应作为使生产软件可靠运行的方法。

如果通过分析器（Nsight、Visual Profiler）收集硬件计数器，除非启用并发内核分析，否则内核启动是同步的。如果涉及非锁定页面主机内存，异步内存复制也可能是同步的。

#### 3.2.8.2. 并发内核执行

某些计算能力为 2.x 或更高的设备可以同时执行多个内核。应用程序可以通过检查 `concurrentKernels` 设备属性（参见设备枚举）来查询此功能，对于支持它的设备，该属性等于 1。

设备可以同时执行的内核启动最大数量取决于其计算能力，并在表 21 中列出。

一个 CUDA 上下文的内核不能与另一个 CUDA 上下文的内核并发执行。GPU 可以进行时间切片以向每个上下文提供向前进度。如果用户希望从多个进程同时在 SM 上运行内核，必须启用 MPS。

使用大量纹理或大量本地内存的内核不太可能与其他内核并发执行。

#### 3.2.8.3. 数据传输和内核执行的重叠

某些设备可以同时执行到或从 GPU 的异步内存复制和内核执行。应用程序可以通过检查 `asyncEngineCount` 设备属性（参见设备枚举）来查询此功能，对于支持它的设备，该属性大于零。如果主机内存参与复制，则它必须是锁定页面。

还可以同时执行设备内复制和内核执行（在支持 `concurrentKernels` 设备属性的设备上）和/或与到或从设备的复制（对于支持 `asyncEngineCount` 属性的设备）。设备内复制是使用具有相同设备上的目标地址和源地址的标准内存复制函数启动的。

#### 3.2.8.4. 并发数据传输

某些计算能力为 2.x 或更高的设备可以重叠到和从设备的复制。应用程序可以通过检查 `asyncEngineCount` 设备属性（参见设备枚举）来查询此功能，对于支持它的设备，该属性等于 2。为了重叠，参与传输的任何主机内存都必须是锁定页面。

#### 3.2.8.5. 流

应用程序通过流管理上述并发操作。流是一系列命令（可能由不同的主机线程发出）按顺序执行。另一方面，不同的流可以以相对于彼此的任意顺序或并发地执行它们的命令；这种行为没有保证，因此不应依赖于正确性（例如，内核间通信是未定义的）。流上发出的命令可以在满足命令的所有依赖项时执行。依赖项可能是同一流上先前启动的命令或来自其他流的依赖项。同步调用的成功完成保证了所有启动的命令都已完成。

##### 3.2.8.5.1. 流的创建和销毁

流通过创建流对象并将其指定为一系列内核启动和主机 `<->` 设备内存复制的流参数来定义。以下代码示例创建两个流并分配一个锁定页面内存中的 `float` 数组 `hostPtr`。

```c++
cudaStream_t stream[2];
for (int i = 0; i < 2; ++i)
    cudaStreamCreate(&stream[i]);
float* hostPtr;
cudaMallocHost(&hostPtr, 2 * size);
```

每个流由以下代码示例定义为一个从主机到设备的内存复制、一个内核启动和一个从设备到主机的内存复制的序列：

```c++
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel <<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
```

每个流将其输入数组 `hostPtr` 的一部分复制到设备内存中的数组 `inputDevPtr`，通过调用 `MyKernel()` 在设备上处理 `inputDevPtr`，并将结果 `outputDevPtr` 复制回 `hostPtr` 的相同部分。重叠行为描述了这些流在此示例中根据设备功能的重叠情况。请注意，对于任何重叠发生，`hostPtr` 必须指向锁定页面主机内存。

通过调用 `cudaStreamDestroy()` 释放流。

```c++
for (int i = 0; i < 2; ++i)
    cudaStreamDestroy(stream[i]);
```

如果在调用 `cudaStreamDestroy()` 时设备仍在流中工作，该函数将立即返回，并且一旦设备完成流中的所有工作，与流关联的资源将被自动释放。

##### 3.2.8.5.2. 默认流

不指定任何流参数或等效地将流参数设置为零的内核启动和主机 `<->` 设备内存复制将被发送到默认流。因此，它们按顺序执行。

对于使用 `--default-stream per-thread` 编译标志（或在包含 CUDA 头文件（`cuda.h` 和 `cuda_runtime.h`）之前定义 `CUDA_API_PER_THREAD_DEFAULT_STREAM` 宏）编译的代码，默认流是常规流，每个主机线程都有自己的默认流。

> **注意**
>
> `#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1` 不能用于在代码由 `nvcc` 编译时启用此行为，因为 `nvcc` 在翻译单元的顶部隐式包含 `cuda_runtime.h`。在这种情况下，需要使用 `--default-stream per-thread` 编译标志或使用` -DCUDA_API_PER_THREAD_DEFAULT_STREAM=1` 编译器标志定义 `CUDA_API_PER_THREAD_DEFAULT_STREAM` 宏。

对于使用 `--default-stream legacy` 编译标志编译的代码，**默认流是一个称为 NULL stream 的特殊流**，每个设备都有一个用于所有主机线程的单个 NULL stream。NULL stream 是特殊的，因为它会导致隐式同步，如[隐式同步](#<3.2.8.5.4. 隐式同步>)中所述。  

对于没有指定 `--default-stream` 编译标志的代码，假设 `--default-stream legacy` 是默认值。

##### 3.2.8.5.3. 显式同步

有多种方法可以显式地同步流。

- `cudaDeviceSynchronize()` 等待所有主机线程的所有流中的所有先前命令完成。
- `cudaStreamSynchronize()` 以流作为参数，等待给定流中的所有先前命令完成。它可以用于将主机与特定流同步，允许其他流继续在设备上执行。
- `cudaStreamWaitEvent()` 以流和事件作为参数（有关事件的描述，请参阅事件），并使在调用 `cudaStreamWaitEvent()` 之后添加到给定流的所有命令延迟其执行，直到给定事件完成。
- `cudaStreamQuery()` 为应用程序提供了一种了解流中所有先前命令是否完成的方法。

##### 3.2.8.5.4. 隐式同步

如果主机线程在两个命令之间发出以下任何一项操作，则来自不同流的两个命令不能并发运行：

- 锁定页面主机内存分配
- 设备内存分配
- 设备内存设置
- 两个地址之间的内存复制到同一设备内存
- 对 [NULL stream](#<3.2.8.5.2. 默认流>) 的任何 CUDA 命令
- 计算能力 7.x 中描述的 L1/共享内存配置之间的切换

需要依赖性检查的操作包括同一流中的任何其他命令以及对该流的 `cudaStreamQuery()` 的任何调用。因此，应用程序应遵循以下指南以提高其并发内核执行的潜力：

- 所有独立操作应在依赖操作之前发出
- 应尽可能延迟任何类型的同步

##### 3.2.8.5.5. 重叠行为

两个流之间的执行重叠量取决于命令发出到每个流的顺序，以及设备是否支持数据传输和内核执行的重叠（参见数据传输和内核执行的重叠）、并发内核执行（参见并发内核执行）和/或并发数据传输（参见并发数据传输）。

例如，在不支持并发数据传输的设备上，[创建和销毁](#<3.2.8.5.1 流的创建和销毁>)的代码示例中的两个流完全不重叠，因为从主机到设备的内存复制发出到流 [1]，从设备到主机的内存复制发出到流 [0] 之后，因此它只能在从设备到主机发出的内存复制完成后开始到流 [0]。如果代码以以下方式重写（并假设设备支持数据传输和内核执行的重叠）

```c++
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
for (int i = 0; i < 2; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
```

那么发出到流 [1] 的从主机到设备的内存复制与发出到流 [0] 的内核启动重叠。

在支持并发数据传输的设备上，创建和销毁的代码示例中的两个流确实重叠：发出到流 [1] 的从主机到设备的内存复制与发出到流 [0] 的从设备到主机的内存复制重叠，甚至与发出到流 [0] 的内核启动重叠（假设设备支持数据传输和内核执行的重叠）。

##### 3.2.8.5.6. 主机函数（回调）

运行时提供了一种通过 `cudaLaunchHostFunc()` 在流中的任何点插入 CPU 函数调用的方法。提供的函数将在主机上执行，一旦之前发出到流的所有命令完成。

以下代码示例在向每个流发出主机到设备的内存复制、内核启动和设备到主机的内存复制之后，将主机函数 `MyCallback` 添加到每个流中。该函数将在每个设备到主机内存复制完成后开始在主机上执行。

```c++
void CUDART_CB MyCallback(void *data){
    printf("Inside callback %d\n", (size_t)data);
}
...
for (size_t i = 0; i < 2; ++i) {
    cudaMemcpyAsync(devPtrIn[i], hostPtr[i], size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(devPtrOut[i], devPtrIn[i], size);
    cudaMemcpyAsync(hostPtr[i], devPtrOut[i], size, cudaMemcpyDeviceToHost, stream[i]);
    cudaLaunchHostFunc(stream[i], MyCallback, (void*)i);
}
```

在流中发出的主机函数后的命令不会在函数完成之前开始执行。

排队到流中的主机函数不得进行 CUDA API 调用（直接或间接），因为如果它进行这样的调用，它最终可能会等待自身并导致死锁。

##### 3.2.8.5.7. 流优先级

可以在创建时使用 `cudaStreamCreateWithPriority()` 指定流的相对优先级。允许的优先级范围，按 [最高优先级，最低优先级] 顺序排列，可以使用 `cudaDeviceGetStreamPriorityRange()` 函数获得。在运行时，高优先级流中的待处理工作优先于低优先级流中的待处理工作。

以下代码示例获取当前设备允许的优先级范围，并创建具有最高和最低可用优先级的流。

```c++
// get the range of stream priorities for this device
int priority_high, priority_low;
cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
// create streams with highest and lowest available priorities
cudaStream_t st_high, st_low;
cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);
```

#### 3.2.8.6. 程序化依赖启动和同步

程序化依赖启动机制允许依赖的二级内核在依赖它的主内核在同一 CUDA 流中完成执行之前启动。从计算能力 9.0 的设备开始可用，当二级内核可以完成不依赖于主内核结果的重大工作时，此技术可以提供性能优势。

##### 3.2.8.6.1. 背景

CUDA 应用程序通过在 GPU 上启动和执行多个内核来利用 GPU。图 10 显示了典型的 GPU 活动时间线。

<img src="./assets/gpu-activity.jpg">

图 10：GPU 活动时间线

在这里，`secondary_kernel` 在 `primary_kernel` 完成执行后启动。序列化执行通常是必要的，因为 `secondary_kernel` 依赖于 `primary_kernel` 生成的结果数据。如果 `secondary_kernel` 不依赖于 `primary_kernel`，则可以通过使用 CUDA 流并发启动两者。即使 `secondary_kernel` 依赖于 `primary_kernel`，也存在一些并发执行的可能性。例如，几乎所有内核在执行过程中都有一些 *preamble* 部分，在此期间执行诸如清零缓冲区或加载常数值等任务。

<img src="./assets/secondary-kernel-preamble.jpg">

图 11：`secondary_kernel` 的前言部分

图 11 显示了可以并发执行而不会影响应用程序的 `secondary_kernel` 的部分。请注意，并发启动还可以让我们隐藏 `secondary_kernel` 的启动延迟在 `primary_kernel` 的执行背后。

<img src="./assets/preamble-overlap.jpg">

图 12：并发执行 `primary_kernel` 和 `secondary_kernel`

图 12 显示了 primary_kernel 和 secondary_kernel 的并发执行，可以使用程序化依赖启动实现。

程序化依赖启动对 CUDA 内核启动 API 引入了更改，如下节所述。这些 API 需要至少计算能力 9.0 才能提供重叠执行。

##### 3.2.8.6.2. API 描述

在程序化依赖启动中，主内核和次内核在同一个 CUDA 流中启动。当主内核准备好启动次内核时，它应该使用所有线程块执行 `cudaTriggerProgrammaticLaunchCompletion`。次内核必须使用可扩展的启动 API 启动，如下所示。

```c++
__global__ void primary_kernel() {
   // Initial work that should finish before starting secondary kernel

   // Trigger the secondary kernel
   cudaTriggerProgrammaticLaunchCompletion();

   // Work that can coincide with the secondary kernel
}

__global__ void secondary_kernel()
{
   // Independent work

   // Will block until all primary kernels the secondary kernel is dependent on have completed and flushed results to global memory
   cudaGridDependencySynchronize();

   // Dependent work
}

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;
configSecondary.attrs = attribute;
configSecondary.numAttrs = 1;

primary_kernel<<<grid_dim, block_dim, 0, stream>>>();
cudaLaunchKernelEx(&configSecondary, secondary_kernel);
```

当使用 `cudaLaunchAttributeProgrammaticStreamSerialization` 属性启动次级内核时，CUDA 驱动程序可以安全地提前启动次级内核，而不必等待主内核的完成和内存刷新才能启动次级内核。

CUDA 驱动程序可以在所有主线程块启动并执行 `cudaTriggerProgrammaticLaunchCompletion` 后启动次级内核。如果主内核不执行触发器，它会在主内核中的所有线程块退出后隐式发生。

在任一情况下，次级线程块可能在主内核写入的数据可见之前启动。因此，当次级内核配置为程序化依赖启动时，它必须始终使用 `cudaGridDependencySynchronize` 或其他方式来验证主内核的结果数据可用。

请注意，这些方法提供了主内核和次级内核并发执行的机会，但是这种行为是机会性的，不能保证导致并发内核执行。以这种方式依赖并发执行是不安全的，可能导致死锁。

##### 3.2.8.6.3. 在 CUDA 图中使用

程序化依赖启动可以通过流捕获或直接通过边缘数据在 CUDA 图中使用。要在具有边缘数据的 CUDA 图中编程此功能，请在连接两个内核节点的边上使用 `cudaGraphDependencyType` 值 `cudaGraphDependencyTypeProgrammatic`。这种边缘类型使上游内核对下游内核中的 `cudaGridDependencySynchronize()` 可见。这种类型必须与 `cudaGraphKernelNodePortLaunchCompletion` 或 `cudaGraphKernelNodePortProgrammatic` 的输出端口一起使用。

流捕获的结果图等效项如下[HERE](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#use-in-cuda-graphs)：



#### 3.2.8.7. CUDA 图

CUDA 图呈现了 CUDA 中工作提交的新模型。图是一系列操作，例如内核启动，通过依赖关系连接，与执行分开定义。这允许一次定义图并重复启动它。将图的定义与执行分开可以实现许多优化：首先，与流相比，CPU 启动成本降低，因为大部分设置是预先完成的；其次，将整个工作流呈现给 CUDA 可以实现流的分段工作提交机制可能无法实现的优化。

要查看图中可能的优化，请考虑流中发生的情况：当您将内核放入流中时，主机驱动程序执行一系列操作来准备在 GPU 上执行内核。这些用于设置和启动内核的操作是一项开销，必须为发出的每个内核支付。对于执行时间较短的 GPU 内核，此开销成本可能占整体端到端执行时间的重要部分。

使用图的工作提交分为三个不同的阶段：定义、实例化和执行。

- 在定义阶段，程序创建图中操作的描述以及它们之间的依赖关系。
- 实例化对图模板进行快照，验证它，并执行大部分工作设置和初始化，目的是最大限度地减少启动时需要执行的操作。生成的实例称为可执行图。
- 可执行图可以像任何其他 CUDA 工作一样启动到流中。它可以启动任意次数，无需重复实例化。

##### 3.2.8.7.1. 图结构

操作形成图中的节点。操作之间的依赖关系是边。这些依赖关系限制了操作的执行顺序。

一旦依赖的节点完成，就可以随时调度操作。调度由 CUDA 系统决定。

###### 3.2.8.7.1.1. 节点类型

图节点可以是：

- 内核
- CPU 函数调用
- 内存复制
- memset
- 空节点
- 等待事件
- 记录事件
- 信号外部信号量
- 等待外部信号量
- 条件节点
- 子图：执行单独的嵌套图，如下图所示。

<img src="./assets/child-graph.jpg">

图 13：子图示例

###### 3.2.8.7.1.2. 边缘数据

CUDA 12.3 引入了 CUDA 图上的边缘数据。边缘数据修改由边指定的依赖关系，由三个部分组成：输出端口、输入端口和类型。输出端口指定何时触发关联的边。输入端口指定节点的哪个部分依赖于关联的边。类型修改端点之间的关系。

端口值特定于节点类型和方向，边缘类型可能限于特定节点类型。在所有情况下，零初始化的边缘数据表示默认行为。输出端口 0 等待整个任务，输入端口 0 阻塞整个任务，边缘类型 0 与具有内存同步行为的完整依赖相关联。

边缘数据可以通过与关联节点并行的数组在各种图 API 中可选指定。如果它作为输入参数省略，则使用零初始化数据。如果它作为输出（查询）参数省略，则 API 在忽略的边缘数据全部为零初始化时接受它，并在调用将丢弃信息时返回 `cudaErrorLossyQuery`。

边缘数据也可用于某些流捕获 API：`cudaStreamBeginCaptureToGraph()`、`cudaStreamGetCaptureInfo()` 和 `cudaStreamUpdateCaptureDependencies()`。在这些情况下，还没有下游节点。数据与悬挂边（半边）关联，该边将连接到未来的捕获节点或在流捕获终止时被丢弃。请注意，某些边缘类型不会等待上游节点的完全完成。在考虑流捕获是否已完全重新加入原始流时，忽略这些边，并且不能在捕获结束时丢弃。请参阅创建使用流捕获的图。

目前，没有节点类型定义额外的输入端口，只有内核节点定义额外的输出端口。有一种非默认依赖类型 `cudaGraphDependencyTypeProgrammatic`，它可以在两个内核节点之间启用程序化依赖启动。

##### 3.2.8.7.2. 使用图 API 创建图

图可以通过两种机制创建：显式 API 和流捕获。以下是创建和执行以下图的示例。

<img src="./assets/create-a-graph.jpg">

```c++
// Create the graph - it starts out empty
cudaGraphCreate(&graph, 0);

// For the purpose of this example, we'll create
// the nodes separately from the dependencies to
// demonstrate that it can be done in two stages.
// Note that dependencies can also be specified
// at node creation.
cudaGraphAddKernelNode(&a, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&b, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&c, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&d, graph, NULL, 0, &nodeParams);

// Now set up dependencies on each node
cudaGraphAddDependencies(graph, &a, &b, 1);     // A->B
cudaGraphAddDependencies(graph, &a, &c, 1);     // A->C
cudaGraphAddDependencies(graph, &b, &d, 1);     // B->D
cudaGraphAddDependencies(graph, &c, &d, 1);     // C->D
```

##### 3.2.8.7.3. 使用流捕获创建图

流捕获提供了一种从现有基于流的 API 创建图的机制。可以通过调用 `cudaStreamBeginCapture()` 和 `cudaStreamEndCapture()` 来括起将工作启动到流中的代码段，包括现有代码。见下文。

```c++
cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);
```

调用 `cudaStreamBeginCapture()` 将流置于捕获模式。当流被捕获时，启动到流中的工作不会排队执行。而是附加到正在逐步构建的内部图中。然后通过调用 `cudaStreamEndCapture()` 返回此图，该调用也结束流的捕获模式。由流捕获正在积极构建的图称为捕获图。

流捕获可用于除 `cudaStreamLegacy`（NULL Stream）之外的任何 CUDA 流。请注意，它可以用于 `cudaStreamPerThread`。如果程序正在使用旧流，则可以将流 0 重新定义为每线程流，而无需功能更改。参见默认流。

可以使用 `cudaStreamIsCapturing()` 查询流是否正在被捕获。

可以使用 `cudaStreamBeginCaptureToGraph()` 将工作捕获到现有图中。工作不是捕获到内部图，而是捕获到用户提供的图。

###### 3.2.8.7.3.1. 跨流依赖和事件

流捕获可以处理使用 `cudaEventRecord()` 和 `cudaStreamWaitEvent()` 表达的跨流依赖关系，前提是等待的事件被记录到同一捕获图中。

当在处于捕获模式的流中记录事件时，会产生捕获事件。捕获事件表示捕获图中的一组节点。

当流等待捕获事件时，如果它尚未处于捕获模式，它会将流置于捕获模式，并且流中的下一个项目将对捕获事件中的节点具有额外的依赖关系。然后将两个流捕获到同一捕获图中。

当流捕获中存在跨流依赖关系时，必须在调用 `cudaStreamBeginCapture()` 的同一流中调用 `cudaStreamEndCapture()`；这是原始流。由于基于事件的依赖关系，被捕获到同一捕获图中的任何其他流也必须重新加入原始流。如下所示。所有捕获到同一捕获图的流在 `cudaStreamEndCapture()` 时退出捕获模式。未能重新加入原始流将导致整个捕获操作失败。

```c++
// stream1 is the origin stream
cudaStreamBeginCapture(stream1);

kernel_A<<< ..., stream1 >>>(...);

// Fork into stream2
cudaEventRecord(event1, stream1);
cudaStreamWaitEvent(stream2, event1);

kernel_B<<< ..., stream1 >>>(...);
kernel_C<<< ..., stream2 >>>(...);

// Join stream2 back to origin stream (stream1)
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);

kernel_D<<< ..., stream1 >>>(...);

// End capture in the origin stream
cudaStreamEndCapture(stream1, &graph);

// stream1 and stream2 no longer in capture mode
```

上面代码返回的图如图 14 所示。

>**注意**
>
>当流退出捕获模式时，流中的下一个未捕获项目（如果有）仍将依赖于最近的先前未捕获项目，尽管中间项目已被删除。

###### 3.2.8.7.3.2. 禁止和未处理的操作

同步或查询正在捕获的流或捕获事件的执行状态是无效的，因为它们不代表计划执行的项目。查询或同步包含活动流捕获的更广泛句柄（例如，当任何关联流处于捕获模式时）的设备或上下文句柄也是无效的。

当同一上下文中的任何流正在被捕获并且不是使用 `cudaStreamNonBlocking` 创建时，任何尝试使用旧流都是无效的。这是因为旧流句柄始终包含这些其他流；向旧流排队将创建对正在捕获的流的依赖，查询或同步它将查询或同步正在捕获的流。

因此，在这种情况下调用同步 API 也是无效的。同步 API（例如 cudaMemcpy()）将工作排队到旧流并同步它，然后再返回。

> **注意**
>
> 一般来说，当依赖关系将捕获的内容与未捕获的内容连接起来并排队执行时，CUDA 更倾向于返回错误而不是忽略依赖关系。进入或退出捕获模式是一个例外；这会切断在模式转换之前和之后添加到流中的项目之间的依赖关系。

通过等待来自正在捕获的流的捕获事件来合并两个独立的捕获图是无效的，该捕获事件与事件关联的捕获图不同。等待来自未捕获流的非捕获事件而不指定 cudaEventWaitExternal 标志是无效的。

一些将异步操作排队到流中的 API 当前在图中不受支持，如果调用带有正在捕获的流的 API（例如 `cudaStreamAttachMemAsync()`），将返回错误。

###### 3.2.8.7.3.3. 无效化

当尝试在流捕获期间执行无效操作时，任何关联的捕获图都会被无效化。当捕获图被无效化时，进一步使用任何正在捕获的流或与图关联的捕获事件都是无效的，并将返回错误，直到使用 `cudaStreamEndCapture()` 结束流捕获。此调用将使关联的流退出捕获模式，但也会返回错误值和 NULL 图。

##### 3.2.8.7.4. CUDA 用户对象

CUDA 用户对象可用于帮助管理 CUDA 中异步工作使用的资源的生命周期。特别是，此功能对 CUDA 图和流捕获很有用。

各种资源管理方案与 CUDA 图不兼容。例如，考虑基于事件的池或同步创建、异步销毁方案。

```c++
// Library API with pool allocation
void libraryWork(cudaStream_t stream) {
    auto &resource = pool.claimTemporaryResource();
    resource.waitOnReadyEventInStream(stream);
    launchWork(stream, resource);
    resource.recordReadyEvent(stream);
}
```

```c++
// Library API with asynchronous resource deletion
void libraryWork(cudaStream_t stream) {
    Resource *resource = new Resource(...);
    launchWork(stream, resource);
    cudaStreamAddCallback(
        stream,
        [](cudaStream_t, cudaError_t, void *resource) {
            delete static_cast<Resource *>(resource);
        },
        resource,
        0);
    // Error handling considerations not shown
}
```

这些方案对于 CUDA 图来说是困难的，因为资源的非固定指针或句柄需要间接寻址或图更新，并且每次提交工作都需要同步 CPU 代码。如果这些考虑对库的调用者隐藏，它们也不适用于流捕获，并且由于在捕获期间使用了不允许的 API。存在各种解决方案，例如将资源暴露给调用者。CUDA 用户对象提供了另一种方法。

CUDA 用户对象将用户指定的析构回调与内部引用计数关联，类似于 C++ 的 `shared_ptr`。引用可以由 CPU 上的用户代码和 CUDA 图拥有。请注意，对于用户拥有的引用，与 C++ 智能指针不同，没有表示引用的对象；用户必须手动跟踪用户拥有的引用。一个典型的用例是在创建用户对象后立即将唯一的用户拥有引用移动到 CUDA 图。

当引用与 CUDA 图关联时，CUDA 将自动管理图操作。克隆的 `cudaGraph_t` 保留源 `cudaGraph_t` 拥有的每个引用的副本，具有相同的数量。实例化的 `cudaGraphExec_t` 保留源 `cudaGraph_t` 中每个引用的副本。当 `cudaGraphExec_t` 在未同步的情况下销毁时，引用将保留直到执行完成。

下面是一个使用示例。

```c++
cudaGraph_t graph;  // Preexisting graph

Object *object = new Object;  // C++ object with possibly nontrivial destructor
cudaUserObject_t cuObject;
cudaUserObjectCreate(
    &cuObject,
    object,  // Here we use a CUDA-provided template wrapper for this API,
             // which supplies a callback to delete the C++ object pointer
    1,  // Initial refcount
    cudaUserObjectNoDestructorSync  // Acknowledge that the callback cannot be
                                    // waited on via CUDA
);
cudaGraphRetainUserObject(
    graph,
    cuObject,
    1,  // Number of references
    cudaGraphUserObjectMove  // Transfer a reference owned by the caller (do
                             // not modify the total reference count)
);
// No more references owned by this thread; no need to call release API
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);  // Will retain a
                                                               // new reference
cudaGraphDestroy(graph);  // graphExec still owns a reference
cudaGraphLaunch(graphExec, 0);  // Async launch has access to the user objects
cudaGraphExecDestroy(graphExec);  // Launch is not synchronized; the release
                                  // will be deferred if needed
cudaStreamSynchronize(0);  // After the launch is synchronized, the remaining
                           // reference is released and the destructor will
                           // execute. Note this happens asynchronously.
// If the destructor callback had signaled a synchronization object, it would
// be safe to wait on it at this point.
```

图中子图节点拥有的引用与子图相关，而不是与父图相关。如果子图被更新或删除，引用相应地更改。如果可执行图或子图使用 `cudaGraphExecUpdate` 或 `cudaGraphExecChildGraphNodeSetParams` 更新，则新源图中的引用被克隆并替换目标图中的引用。在任何一种情况下，如果先前的启动未同步，任何将被释放的引用都将保留，直到启动完成执行。

目前没有通过 CUDA API 等待用户对象析构器的机制。用户可以从析构器代码手动发出同步对象信号。此外，不允许从析构器调用 CUDA API，类似于 `cudaLaunchHostFunc` 的限制。这是为了避免阻塞 CUDA 内部共享线程并阻止向前进度。如果依赖项是一路的，并且执行调用的线程不能阻止 CUDA 工作的向前进度，则允许信号另一个线程执行 API 调用。

用户对象是使用 `cudaUserObjectCreate` 创建的，这是浏览相关 API 的良好起点。

##### 3.2.8.7.5. 更新实例化图

使用图的工作提交分为三个不同的阶段：定义、实例化和执行。在工作流不变的情况下，定义和实例化的开销可以摊销到多次执行中，图比流具有明显的优势。

图是工作流的快照，包括内核、参数和依赖关系，以便尽可能快速高效地重放。在工作流更改的情况下，图会过时，必须修改。图结构（例如拓扑或节点类型）的重大更改将需要重新实例化源图，因为必须重新应用各种与拓扑相关的优化技术。

重复实例化的成本可能会降低图执行的整体性能优势，但通常只有节点参数（例如内核参数和 `cudaMemcpy` 地址）会更改，而图拓扑保持不变。对于这种情况，CUDA 提供了一种称为“图更新”的轻量级机制，允许就地修改某些节点参数，而无需重建整个图。这比重新实例化效率高得多。

更新将在下一次启动图时生效，因此不会影响之前的图启动，即使它们在更新时正在运行。图可以重复更新和重新启动，因此可以在流上排队多个更新/启动。

CUDA 提供了两种更新实例化图参数的机制：整个图更新和单个节点更新。整个图更新允许用户提供一个拓扑相同的 `cudaGraph_t` 对象，其节点包含更新的参数。单个节点更新允许用户显式更新单个节点的参数。当更新大量节点或调用者不知道图拓扑时（即图是库调用的流捕获的结果），使用更新的 `cudaGraph_t` 更方便。当更改数量较少且用户拥有需要更新的节点的句柄时，首选使用单个节点更新。单个节点更新跳过未更改节点的拓扑检查和比较，因此在许多情况下可以更有效率。

CUDA 还提供了一种启用和禁用单个节点而不影响其当前参数的机制。

以下部分将更详细地解释每种方法。

###### 3.2.8.7.5.1. 图更新限制

内核节点：
- 函数的所有者上下文不能更改。
- 原本不使用 CUDA 动态并行的节点不能更新为使用 CUDA 动态并行的函数。

`cudaMemset` 和 `cudaMemcpy` 节点：

- 操作数分配/映射到的 CUDA 设备不能更改。
- 源/目标内存必须从与原始源/目标内存相同的上下文分配。
- 只能更改一维 `cudaMemset` / `cudaMemcpy` 节点。

额外的 memcpy 节点限制：
- 更改源或目标内存类型（即 `cudaPitchedPtr`、`cudaArray_t` 等）或传输类型（即 `cudaMemcpyKind`）不受支持。

外部信号量等待节点和记录节点：
- 不支持更改信号量数量。

条件节点：
- 句柄创建和分配的顺序必须在图之间匹配。
- 不支持更改节点参数（即条件中的图数、节点上下文等）。
- 更改条件体图中节点的参数受上述规则限制。

对主机节点、事件记录节点或事件等待节点的更新没有限制。

###### 3.2.8.7.5.2. 整个图更新

`cudaGraphExecUpdate()` 允许使用拓扑相同的图（“更新”图）的参数更新实例化图（“原始图”）。更新图的拓扑必须与用于实例化 `cudaGraphExec_t` 的原始图相同。此外，指定依赖关系的顺序必须匹配。最后，CUDA 需要一致地排序接收器节点（没有依赖关系的节点）。CUDA 依赖于特定 API 调用的顺序来实现一致的接收器节点排序。

更明确地说，遵循以下规则将导致 `cudaGraphExecUpdate()` 确定性地配对原始图和更新图中的节点：

1. 对于任何捕获流，必须按相同顺序进行操作该流的 API 调用，包括事件等待和其他与节点创建不直接对应的 API 调用。
2. 直接操作给定图节点的传入边的 API 调用（包括捕获流 API、节点添加 API 和边添加/删除 API）必须按相同顺序进行。此外，当在这些 API 的数组中指定依赖关系时，依赖关系在这些数组中指定的顺序必须匹配。
3. 接收器节点必须一致排序。接收器节点是在 `cudaGraphExecUpdate()` 调用时最终图中没有依赖节点/输出边的节点。以下操作会影响接收器节点排序（如果存在）并且必须（作为一个组合集）按相同顺序进行：
   - 导致接收器节点的节点添加 API。
   - 导致节点成为接收器节点的边移除。
   - `cudaStreamUpdateCaptureDependencies()`，如果它从捕获流的依赖集中删除接收器节点。
   - `cudaStreamEndCapture()`。

以下示例展示了如何使用 API 更新实例化图：

### 3.2.9. 多设备系统



### 3.2.12. 错误检查



### 3.2.13. 调用堆栈



### 3.2.14. 纹理和表面内存



### 3.2.15. 图形互操作性




## 4.1. SIMT 架构



# 6. 支持 CUDA 的 GPUs



# 7. C++ 语言扩展

## 7.1. 函数执行空间指定符

函数执行空间指定符用于指示函数是在主机上还是设备上执行，以及是否可以从主机或设备调用。

### 7.1.1. \_\_global\_\_

`__global__` 执行空间指定符声明一个函数为内核函数。这种函数：

- 在设备上执行，
- 可以从主机调用，
- 对于计算能力为 5.0 或更高的设备，可以从设备调用（有关更多详细信息，请参阅 [CUDA 动态并行](#<9. CUDA 动态并行性>)）。

`__global__` 函数必须具有 void 返回类型，并且不能是类的成员。

对 `__global__` 函数的任何调用都必须指定其执行配置，如[执行配置](#<7.37. 执行配置>)中所述。

对 `__global__` 函数的调用是异步的，这意味着它在设备完成执行之前返回。

### 7.1.2. \_\_device\_\_

`__device__` 执行空间指定符声明一个函数：

- 在设备上执行的函数，
- 只能从设备调用。

`__global__`  和 `__device__`  执行空间指定符不能一起使用。

### 7.1.3. \_\_host\_\_

`__host__` 执行空间指定符声明一个函数：

- 在主机上执行的函数，
- 只能从主机调用。

仅使用 `__host__` 执行空间指定符声明函数或不使用任何 `__host__`、`__device__`  或 `__global__` 执行空间指定符声明函数是等效的；在任一情况下，该函数仅编译为主机。

`__global__` 和 `__host__` 执行空间指定符不能一起使用。

`__device__` 和 `__host__` 执行空间指定符可以一起使用，在这种情况下，该函数同时编译为主机和设备。在[应用程序兼容性](#<3.1.4. 应用程序兼容性>)中引入的 `__CUDA_ARCH__` 宏可用于区分主机和设备之间的代码路径：

```c++
__host__ __device__ func()
{
#if __CUDA_ARCH__ >= 800
   // Device code path for compute capability 8.x
#elif __CUDA_ARCH__ >= 700
   // Device code path for compute capability 7.x
#elif __CUDA_ARCH__ >= 600
   // Device code path for compute capability 6.x
#elif __CUDA_ARCH__ >= 500
   // Device code path for compute capability 5.x
#elif !defined(__CUDA_ARCH__)
   // Host code path
#endif
}
```

### 7.1.4. 未定义行为

当发生以下情况时，“跨执行空间”调用具有未定义行为：

- 定义了 `__CUDA_ARCH__`，从 `__global__`、`__device__` 或 `__host__ __device__` 函数内部调用 `__host__` 函数。
- 未定义 `__CUDA_ARCH__`，从  `__host__`  函数内部调用 `__device__` 函数。

### 7.1.5. \_\_noinline\_\_  和 \_\_forceinline\_\_

编译器在认为合适时内联任何 `__device__` 函数。`__noinline__` 函数限定符可以作为编译器不内联函数的提示。`__forceinline__` 函数限定符可用于强制编译器内联函数。`__noinline__` 和 `__forceinline__` 函数限定符不能一起使用，并且这两个函数限定符都不能应用于内联函数。

### 7.1.6. \_\_inline_hint\_\_

`__inline_hint__` 限定符可以在编译器中启用更积极的内联。与 `__forceinline__` 不同，它并不意味着函数是内联的。它可用于在使用 LTO 时改善跨模块的内联。`__noinline__` 或 `__forceinline__` 函数限定符不能与 `__inline_hint__` 函数限定符一起使用。

## 7.22. Warp Shuffle 函数



## 7.26. 异步屏障



## 7.27. 异步数据复制



## 7.37. 执行配置

任何对 `__global__` 函数的调用都必须指定该调用的*执行配置*。执行配置定义了将在设备上执行函数的 grid 和 blocks 的维度，以及关联的流（有关流的描述，请参阅 [CUDA 运行时](#<3.2. CUDA 运行时>)）。

执行配置通过在函数名称和带括号的参数列表之间插入形如 `<<< Dg, Db, Ns, S >>>` 的表达式来指定，其中：

- `Dg` 是 `dim3` 类型（参见 [dim3](#<7.3.2. dim3>)），指定网格的维度和大小，使得 `Dg.x * Dg.y * Dg.z` 等于启动的块数；
- `Db` 是 `dim3` 类型（参见 [dim3](#<7.3.2. dim3>)），指定每个块的维度和大小，使得 `Db.x * Db.y * Db.z` 等于每个块的线程数；
- `Ns` 是 `size_t` 类型，指定每个块为该调用动态分配的共享内存中的字节数，除了静态分配的内存之外；此动态分配的内存用于任何声明为外部数组的变量，如 `__shared__` 中所述；`Ns` 是可选参数，默认为 0；
- `S` 是 `cudaStream_t` 类型，指定关联的流；`S` 是可选参数，默认为 0。

例如，声明为

```c++
 __global__ void Func(float* parameter);
```

的函数必须像这样调用：

```c++
Func<<< Dg, Db, Ns >>>(parameter);
```

执行配置的参数在实际函数参数之前计算。

如果 `Dg` 或 `Db` 大于计算能力中指定的设备的最大允许大小，或者 `Ns` 大于设备上可用的最大共享内存量减去静态分配所需的共享内存量，则函数调用将失败。

计算能力 9.0 及更高版本允许用户指定编译时线程块集群维度，以便内核可以使用 CUDA 中的集群层次结构。编译时集群维度可以使用 `__cluster_dims__([x, [y, [z]]])` 指定。下面的示例显示了 X 维度中编译时集群大小为 2，Y 和 Z 维度中为 1。

```
__global__ void __cluster_dims__(2, 1, 1) Func(float* parameter);
```

线程块集群维度也可以在运行时指定，并且具有集群的内核可以使用 `cudaLaunchKernelEx` API 启动。该 API 采用类型为 `cudaLaunchConfig_t` 的配置参数、内核函数指针和内核参数。运行时内核配置示例如下：

```c++
__global__ void Func(float* parameter);


// Kernel invocation with runtime cluster size
{
    cudaLaunchConfig_t config = {0};
    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension should be a multiple of cluster size.
    config.gridDim = Dg;
    config.blockDim = Db;
    config.dynamicSmemBytes = Ns;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    float* parameter;
    cudaLaunchKernelEx(&config, Func, parameter);
}
```



# 8. Cooperative Groups



# 9. CUDA 动态并行性











# 14. C++ 语言支持

















# 16. 计算能力

## 16.8. 计算能力 9.0



### 16.6.2. 独立线程调度



# 17. 驱动程序 API



# 18. CUDA 环境变量



# 19. 统一内存编程

