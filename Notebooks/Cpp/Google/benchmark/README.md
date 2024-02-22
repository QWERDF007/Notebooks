# [Benchmark](https://github.com/google/benchmark)

用于对代码片段进行基准测试的库，类似于单元测试。例子：

```cpp
#include <benchmark/benchmark.h>

static void BM_SomeFunction(benchmark::State& state) {
  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    SomeFunction();
  }
}
// Register the function as a benchmark
BENCHMARK(BM_SomeFunction);
// Run the benchmark
BENCHMARK_MAIN();
```

## 入门

要开始使用，请参阅[需求](#需求)和[安装](#安装)。有关完整示例， 请参阅[用法](#用法) 。有关更全面的功能概述，请参阅[用户指南](<用户指南.md>)。

阅读 [Google 测试文档](https://github.com/google/googletest/blob/main/docs/primer.md)也可能会有所帮助 ，因为 API 的某些结构方面是相似的。

## 资源

IRC channels:
* [libera](https://libera.chat) #benchmark

[额外的工具文档](https://github.com/google/benchmark/blob/main/docs/tools.md)

[装配测试文档](https://github.com/google/benchmark/blob/main/docs/AssemblyTests.md)

[构建和安装 Python 绑定](https://github.com/google/benchmark/blob/main/docs/python_bindings.md)

## 需求

该库可与 C++03 一起使用。然而，它需要 C++11 来构建，包括编译器和标准库支持。

构建该库需要以下最低版本：

- GCC 4.8
- Clang 3.4
- Visual Studio 14 2015
- Intel 2015 Update 1

请参阅[特定于平台的构建说明](https://github.com/google/benchmark/blob/main/docs/platform_specific_build_instructions.md)。

## 安装

介绍了使用 cmake 的安装过程。作为先决条件，您需要安装 git 和 cmake。

有关受支持的构建工具版本的更多详细信息，请参阅 [dependency.md ](https://github.com/google/benchmark/blob/main/docs/dependencies.md)。

```bash
# 克隆仓库.
$ git clone https://github.com/google/benchmark.git
# 切换目录
$ cd benchmark
# 创建构建目录以存放构建输出
$ cmake -E make_directory "build"
# 使用 cmake 生成构建系统文件，并下载依赖
$ cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../
# 或者使用 CMake 3.13，用更简单的方式
# cmake -DCMAKE_BUILD_TYPE=Release -S . -B "build"
# 构建库
$ cmake --build "build" --config Release
```

这将构建 `benchmark` 和 `benchmark_main` 和测试。在 UNIX 系统上，构建目录现在应该如下所示：

```
/benchmark
  /build
    /src
      /libbenchmark.a
      /libbenchmark_main.a
    /test
      ...
```

接下来，您可以运行测试来检查构建。

```bash
cmake -E chdir "build" ctest --build-config Release
```

如果您想全局安装该库，还可以运行：

```bash
sudo cmake --build "build" --config Release --target install
```

请注意，Google Benchmark 需要 Google Test 来构建和运行测试。这种依赖关系可以通过两种方式提供：

- 将 Google Test 源吗 checkout 到 `benchmark/googletest`。
- 如果在配置过程中指定 `-DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON` ，库将自动下载并构建任何所需的依赖项。

如果您不想构建和运行测试，请添加 `-DBENCHMARK_ENABLE_GTEST_TESTS=OFF` 到 `CMAKE_ARGS`。

### Debug vs Release

默认情况下，benchmark 构建为 debug 库。在这种情况下，您将在输出中看到警告。要将其构建为 release 库，请在生成构建系统文件时添加 `-DCMAKE_BUILD_TYPE=Release`，如上所示。需要在构建命令使用 `--config Release` 来正确支持多配置工具 (例如 Visual Studio)，而对于其他构建系统 (例如 Makefile) 可以跳过。

要启用链接时优化，还要在生成构建系统文件时添加 `-DBENCHMARK_ENABLE_LTO=true`。

如果您使用 gcc，如果自动检测失败，您可能需要设置 cmake 缓存变量 `GCC_AR` 和 `GCC_RANLIB`。

如果您使用 clang，您可能需要设置 cmake 缓存变量`LLVMAR_EXECUTABLE`、`LLVMNM_EXECUTABLE` 和 `LLVMRANLIB_EXECUTABLE`。

要启用检查工具 (例如，`asan` 和 `tsan`)，请添加：

```bash
 -DCMAKE_C_FLAGS="-g -O2 -fno-omit-frame-pointer -fsanitize=address -fsanitize=thread -fno-sanitize-recover=all"
 -DCMAKE_CXX_FLAGS="-g -O2 -fno-omit-frame-pointer -fsanitize=address -fsanitize=thread -fno-sanitize-recover=all "  
```

### 稳定和实验库版本

main 分支包含最新稳定版本的 benchmark 库；其 API 可以被认为基本上是稳定的，只有在新的主要版本发布时才会进行源代码重大更改。

较新的实验性功能已在 [`v2` 分支](https://github.com/google/benchmark/tree/v2)上实现和测试。我们鼓励希望使用、测试新功能并提供反馈的用户尝试此分支。但是，该分支不提供稳定性保证，并保留随时更改和破坏 API 的权利。

## 用法

### 基本用法

定义一个执行要测量的代码的函数，使用 `BENCHMARK` 宏将其注册为基准函数，并确保适当的 `main` 函数可用：

```cpp
#include <benchmark/benchmark.h>

static void BM_StringCreation(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
// Register the function as a benchmark
BENCHMARK(BM_StringCreation);

// Define another benchmark
static void BM_StringCopy(benchmark::State& state) {
  std::string x = "hello";
  for (auto _ : state)
    std::string copy(x);
}
BENCHMARK(BM_StringCopy);

BENCHMARK_MAIN();
```

要运行基准测试，请进行编译和链接 `benchmark` 库 (libbenchmark.a/.so)。如果您按照上面的构建步骤进行操作，则库将位于您创建的构建目录下。

```bash
# Example on linux after running the build steps above. Assumes the
# `benchmark` and `build` directories are under the current directory.
$ g++ mybenchmark.cc -std=c++11 -isystem benchmark/include \
  -Lbenchmark/build/src -lbenchmark -lpthread -o mybenchmark
```

或者，链接 `benchmark_main` 库并删除 `BENCHMARK_MAIN();` 以获得相同的行为。

编译后的可执行文件将默认运行所有基准测试。传递 `--help` 标志以获取选项信息或参阅[用户指南](用户指南.md)。

### 与 CMake 一起使用

如果使用 CMake，建议使用 `target_link_libraries` 链接项目提供的 `benchmark::benchmark` 和 `benchmark::benchmark_main` 库。可以使用 `find_package` 用来导入库的已安装版本。

```cmake
find_package(benchmark REQUIRED)
```

或者，使用 `add_subdirectory` 将库直接合并到 CMake 项目中。

```cmake
add_subdirectory(benchmark)
```

无论哪种方式，请按如下方式链接库。

```cmake
target_link_libraries(MyTarget benchmark::benchmark)
```