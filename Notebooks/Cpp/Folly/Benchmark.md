# [Benchmark](https://github.com/facebook/folly/blob/main/folly/docs/Benchmark.md)

`folly/Benchmark.h` 提供了一个用于编写和执行基准测试的简单框架。目前该框架仅针对单线程测试（尽管您可以在内部使用 fork-join 并行处理并测量总运行时间）。

## 概述

使用 `folly/Benchmark.h` 非常简单，下面是一个示例：

```c++
    #include <folly/Benchmark.h>
    #include <vector>
    using namespace std;
    using namespace folly;
    BENCHMARK(insertFrontVector) {
      // Let's insert 100 elements at the front of a vector
      vector<int> v;
      for (unsigned int i = 0; i < 100; ++i) {
        v.insert(v.begin(), i);
      }
    }
    BENCHMARK(insertBackVector) {
      // Let's insert 100 elements at the back of a vector
      vector<int> v;
      for (unsigned int i = 0; i < 100; ++i) {
        v.insert(v.end(), i);
      }
    }
    int main() {
      runBenchmarks();
    }
```

编译并运行该代码产生标准输出：

```bash
    ===============================================================================
    test.cpp                                              relative ns/iter  iters/s
    ===============================================================================
    insertFrontVector                                                3.84K  260.38K
    insertBackVector                                                 1.61K  622.75K
    ===============================================================================
```

我们稍后再关注空列 "relative"。该表格包含每个基准测试的每次调用所花费的时间以及相反的每秒调用次数。数字以公制表示（K 代表千，M 代表百万等）。正如预期的那样，在这个例子中，第二个函数要快得多（更少的 ns/iter 和更多的 iters/s）。

该宏 `BENCHMARK` 引入了一个函数，并将其添加到包含系统中所有基准的内部数组中。定义的函数不带参数并返回 `void`。

框架多次调用该函数来收集有关它的统计信息。有时函数本身想要进行迭代——例如插入 `n` 元素而不是 100 个元素怎么样？要在内部进行迭代，请使用两个参数的 `BENCHMARK`。第二个参数是迭代次数，由框架传递给函数。计数的类型是隐式的 `unsigned`。考虑一个稍微修改过的例子：

```cpp
    #include <folly/Benchmark.h>
    #include <folly/container/Foreach.h>
    #include <vector>
    using namespace std;
    using namespace folly;
    BENCHMARK(insertFrontVector, n) {
      vector<int> v;
      for (unsigned int i = 0; i < n; ++i) {
        v.insert(v.begin(), i);
      }
    }
    BENCHMARK(insertBackVector, n) {
      vector<int> v;
      for (unsigned int i = 0; i < n; ++i) {
        v.insert(v.end(), i);
      }
    }
    int main() {
      runBenchmarks();
    }
```

产生的数字有很大的不同：

```bash
    ===============================================================================
    Benchmark                                             relative ns/iter  iters/s
    ===============================================================================
    insertFrontVector                                               39.92    25.05M
    insertBackVector                                                 3.46   288.89M
    ===============================================================================
```

现在，这些数字表示单次插入的速度，因为框架假设用户定义的函数使用内部迭代（确实如此）。所以在向量后面插入比在前面插入快 10 倍以上！说到比较...

## 基线

在任何测量中，选择一个或多个良好的基线都是一项至关重要的活动。如果没有基线，就只能从纯粹的数字中得出很少的信息。例如，如果您对算法进行实验，良好的基线通常是一种既定的方法（例如用于排序的内置方法 `std::sort` ）。基本上所有的实验数据都应该与一些基线进行比较。

为了支持基线驱动的测量，`folly/Benchmark.h` 定义了 `BENCHMARK_RELATIVE`，其工作方式与 `BENCHMARK` 非常相似，不同之处在于它考虑最近词汇上出现的 `BENCHMARK` 作为基线，并填充 "relative" 列。举例来说，我们想使用向量的前插入作为基线，然后看看向后插入与它的比较：

```c++
    #include <folly/Benchmark.h>
    #include <folly/container/Foreach.h>
    #include <vector>
    using namespace std;
    using namespace folly;
    BENCHMARK(insertFrontVector, n) {
      vector<int> v;
      for (unsigned int i = 0; i < n; ++i) {
        v.insert(v.begin(), i);
      }
    }
    BENCHMARK_RELATIVE(insertBackVector, n) {
      vector<int> v;
      for (unsigned int i = 0; i < n; ++i) {
        v.insert(v.end(), i);
      }
    }
    int main() {
      runBenchmarks();
    }
```

该程序打印类似的内容：

```bash
    ===============================================================================
    Benchmark                                             relative ns/iter  iters/s
    ===============================================================================
    insertFrontVector                                               42.65    23.45M
    insertBackVector                                     1208.24%    3.53   283.30M
    ===============================================================================
```

显示向后插入与向前插入相比具有 1208.24% 的相对速度优势。选择 scale 的方式是：100% 表示速度相同，小于 100% 的数字表示基准测试比基线更慢，数字大于 100% 表示基准测试比基线更快。例如，如果您看到 42%，则表示基准测试速度是基线速度的 0.42。如果您看到 123%，则意味着基准测试快了 23% 或 1.23 倍。

要关闭当前基准测试组并启动另一个，只需再次使用 `BENCHMARK` 即可。

## Ars Gratia Artis

如果您想绘制一条水平虚线（例如，在组的末尾或出于任何原因），请使用 `BENCHMARK_DRAW_LINE()`。这条线起到了纯粹的美学作用；它不会以任何方式与测量交互。

```c++
    BENCHMARK(foo) {
      Foo foo;
      foo.doSomething();
    }

    BENCHMARK_DRAW_LINE();

    BENCHMARK(bar) {
      Bar bar;
      bar.doSomething();
    }
```

## 暂停基准测试

有时，基准测试代码必须在基准函数内部进行一些物理上的准备工作，但不应占用其时间预算。要暂时暂停基准测试，请按照下述伪代码使用 `BENCHMARK_SUSPEND`：

```cpp
    BENCHMARK(insertBackVector, n) {
      vector<int> v;
      BENCHMARK_SUSPEND {
        v.reserve(n);
      }
      for (unsigned int i = 0; i < n; ++i) {
        v.insert(v.end(), i);
      }
    }
```

执行的预分配 `v.reserve(n)` 不会计入基准测试的总运行时间。

只有主线程应该调用 `BENCHMARK_SUSPEND`（当然，当其他线程正在执行实际工作时，它不应该调用它）。这是因为计时器是应用程序全局的。

如果 `BENCHMARK_SUSPEND ` 引入的作用域不是想要的，您可能需要“手动”使用 `BenchmarkSuspender` 类型。构造这样的对象会暂停时间测量，销毁它会恢复测量。如果您想在析构函数之前恢复时间测量，请对 `BenchmarkSuspender` 对象进行调用 `dismiss`。前面的例子可以这样写：

```cpp
    BENCHMARK(insertBackVector, n) {
      BenchmarkSuspender braces;
      vector<int> v;
      v.reserve(n);
      braces.dismiss();
      for (unsigned int i = 0; i < n; ++i) {
        v.insert(v.end(), i);
      }
    }
```

## `doNotOptimizeAway`

最后，小实用函数 `doNotOptimizeAway` 可以防止编译器优化，从而干扰基准测试。针对用于基准测试但在其他方面无用的变量调用 `doNotOptimizeAway(var)`。编译器往往会很好地消除未使用的变量，并且该函数会欺骗它认为实际上需要一个变量。例子：

```cpp
    BENCHMARK(fpOps, n) {
      double d = 1;
      FOR_EACH_RANGE (i, 1, n) {
        d += i;
        d -= i;
        d *= i;
        d /= i;
      }
      doNotOptimizeAway(d);
    }
```

## 深入了解

`folly/Benchmark.h` 有一种简单、系统的方法来收集计时。

首先，它将测量结果组织成几个大的 epochs，并在所有 epochs 中取最小值。取最小值可以得到最接近实际运行时间的结果。基准时间不是围绕平均值波动的常规随机变量。相反，我们寻找的实际时间是一个存在各种加性噪声的时间（即没有噪声能够将基准测试时间缩短到其实际值以下）。理论上，采样无限次并保持最小值是需要测量的实际时间。这就是为什么随着 epochs 数量的增加，基准测试的准确性也会提高。

当然，在实际运行中也会有噪声和由运行环境引起的各种影响。但是，在基准测试期间（直接设置、简单循环）的噪声对于实际应用中的噪声来说是一个不良模型。因此，在多个 epochs 中取最小值是最具信息量的结果。

在每个时期内，被测量的函数会迭代增加的次数，直到总运行时间足够大，使噪声可以忽略不计。此时收集时间，并计算每次迭代的时间。正如前面提到的，所有 epochs 中每次迭代的最小时间就是最终结果。

所使用的计时器函数是 `clock_gettime`，使用 `CLOCK_REALTIME` 时钟标识。请注意，您必须使用较新的 Linux 内核（2.6.38 或更新版本），否则 `CLOCK_REALTIME` 的分辨率不足。
