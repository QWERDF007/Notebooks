# [Folly](https://github.com/facebook/folly)

## 什么是 `folly`？

Folly（Facebook Open Source Library 的缩写）是一个 C++17 组件库，其设计考虑了实用性和效率。**Folly 包含 Facebook 广泛使用的各种核心库组件**。特别是，它经常是 Facebook 其他开源 C++ 项目的依赖项，这些项目可以在其中共享代码。

它补充（而不是竞争）诸如 `Boost` 等产品，当然还有 `std`。事实上，只有当我们需要的东西不可用或不满足所需的性能配置文件时，我们才会开始定义自己的组件。如果当 `std` 或 `Boost` 废弃了某些东西时，我们会尽力将其从 folly 中删除。

性能问题渗透到 Folly 的大部分内容中，有时会导致设计比原本的设计更加独特（参见例如 `PackedSyncPtr.h`、`SmallLocks.h` ）。大规模场景下的良好性能是 Folly 所有内容的统一主题。

## 介绍视频

https://www.youtube.com/watch?v=Wr_IfOICYSs

## 逻辑设计

Folly 是一个相对独立组件的集合，有些简单到只包含几个符号。内部依赖没有限制，这意味着一个 folly 模块可以使用任何其他 folly 组件。

所有的符号都定义在顶层命名空间 `folly` 中，当然宏例外。宏名称全部为大写，应以 `FOLLY_` 为前缀。命名空间 `folly` 定义了其他内部命名空间，如 `internal` 或 `detail`。用户代码不应依赖这些命名空间中的符号。

Folly 还有一个 `experimental` 目录。这主要意味着我们认为 API 可能会随时间大幅改变。通常，这些代码仍在大规模使用并经过良好测试。

## 里面是什么？

由于 folly 的结构相当扁平，查看顶层 `folly/` 目录中的头文件是了解其内容的最佳方式。您还可以检查 `docs` 文件夹中的文档，从概述开始。

Folly 在 GitHub 上的发布地址是 https://github.com/facebook/folly。

## 构建笔记

由于 folly 不提供从提交到提交的任何 ABI 兼容性保证，因此我们通常建议将 folly 构建为静态库。

Folly 支持 gcc (5.1+)、clang 和 MSVC 编译器。它可以在 Linux (x86-32、x86-64 和 ARM)、iOS、macOS 和 Windows (x86-64) 上运行。CMake 构建仅在部分平台上经过测试；至少，我们的目标是支持 macOS 和 Linux (在最新 Ubuntu LTS 版本或更新的版本上)。

### `getdeps.py`

这个脚本被许多 Meta 的开源工具使用。它将首先下载和构建所有必需的依赖项，然后调用 cmake 等来构建 folly。这将帮助确保您使用所有相关依赖库的相关版本进行构建，同时考虑本地系统上安装的版本。

它使用 python 编写，所以您需要 python3.6 或更高版本在 PATH 中。它可在 Linux、macOS 和 Windows 上工作。

folly 的 cmake 构建设置保存在其 getdeps 清单 `build/fbcode_builder/manifests/folly` 中，如果需要，您可以本地编辑它。

#### 依赖项

如果在 Linux 或 MacOS 上 (安装了 homebrew)，您可以安装系统依赖项以节省构建时间：

```bash
# Clone the repo
git clone https://github.com/facebook/folly
# Install dependencies
cd folly
sudo ./build/fbcode_builder/getdeps.py install-system-deps --recursive
```

如果你想在安装之前查看软件包：

```bash
./build/fbcode_builder/getdeps.py install-system-deps --dry-run --recursive
```

在其他平台上，或者如果在 Linux 上没有安装系统依赖项，`getdeps.py` 将主要在构建步骤中为您下载和构建这些依赖项。

`getdeps.py` 使用和安装的一些依赖项包括：

- 一个使用 C++14 编译的 boost 版本。
- googletest 是构建和运行 folly 测试所必需的。

#### 构建

这个脚本将首先下载和构建所有必需的依赖项，然后调用 cmake 等来构建 folly。这将帮助确保您使用所有相关依赖库的相关版本进行构建，同时考虑本地系统上安装的版本。

`getdeps.py` 当前需要 python 3.6+ 在你的 PATH 上。

`getdeps.py` 将调用 cmake 等。

```
# Clone the repo
git clone https://github.com/facebook/folly
cd folly
# Build, using system dependencies if available
python3 ./build/fbcode_builder/getdeps.py --allow-system-packages build
```

它将输出放入暂存区：

- `installed/folly/lib/libfolly.a`: Library

您还可以指定 `--scratch-path` 参数来控制构建中使用的临时目录的位置。您可以从日志中或者使用 `python3 ./build/fbcode_builder/getdeps.py show-inst-dir` 找到默认的临时安装位置。

还有 `--install-dir` 和 `--install-prefix` 参数可提供更细粒度的安装目录控制。但是，鉴于 folly 在提交之间不提供兼容性保证，我们通常建议将库构建并安装到临时位置，然后将项目的构建指向该临时位置，而不是将 folly 安装到传统的系统安装目录中。例如，如果您使用 CMake 构建，可以使用 `CMAKE_PREFIX_PATH` 变量在构建项目时允许 CMake 在此临时安装目录中找到 folly。

如果您想再次调用 `cmake` 进行迭代，构建目录中会输出一个有用的 `run_cmake.py` 脚本。您可以从日志或使用`python3 ./build/fbcode_builder/getdeps.py show-build-dir` 找到临时构建目录。

#### 运行测试

默认情况下 `getdeps.py` 会构建 folly 的测试。要运行它们：

```bash
cd folly
python3 ./build/fbcode_builder/getdeps.py --allow-system-packages test
```

#### `build.sh`/`build.bat` wrapper

`build.sh` 可以在 Linux 和 MacOS 上用，Windows 上使用 `build.bat` 代替。它是 `getdeps.py` 的一层封装。

### 直接使用 CMake 构建

如果你不想让 getdeps 为你调用 cmake，则默认情况下，构建测试作为 CMake `all` 目标的一部分被禁用。要构建测试，请在配置时向 CMake 指定 `-DBUILD_TESTS=ON`。

注意，如果你想在基于 `getdeps.py` 的构建上再次调用 `cmake` 进行迭代，构建目录中会输出一个有用的 `run_cmake.py`脚本。你可以从日志或使用`python3 ./build/fbcode_builder/getdeps.py show-build-dir` 找到临时构建目录。

如果你切换到构建目录，也可以用 ctests 运行测试，例如 `(cd $(python3 ./build/fbcode_builder/getdeps.py show-build-dir) && ctest)`

#### 在非默认位置查找依赖项

如果您在非默认位置安装了 boost、gtest 或其他依赖项，则可以使用 `CMAKE_INCLUDE_PATH` 和 `CMAKE_LIBRARY_PATH` 变量使 CMAKE 也在非标准位置查找头文件和库。例如，要同时搜索目录 `/alt/include/path1` 和 `/alt/include/path2` 的头文件以及目录 `/alt/lib/path1` 和 `/alt/lib/path2` 的库，您可以按如下方式调用 `cmake`：

```bash
cmake \
  -DCMAKE_INCLUDE_PATH=/alt/include/path1:/alt/include/path2 \
  -DCMAKE_LIBRARY_PATH=/alt/lib/path1:/alt/lib/path2 ...
```

### Ubuntu LTS, CentOS Stream, Fedora



### Windows (Vcpkg)

