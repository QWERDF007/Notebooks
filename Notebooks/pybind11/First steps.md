# [First steps](https://pybind11.readthedocs.io/en/stable/basics.html)

这部分文档展示了 pybind11 的基本特性。在开始之前，请确保您的开发环境已经设置好，能够编译包含的测试用例。

## 编译测试用例

### Linux/macOS

在 Linux 上，您需要安装 **python-dev** 或 **python3-dev** 包以及 **cmake**。在 macOS 上，包含的 Python 版本可以直接使用，但 **cmake** 仍然需要安装。

安装好先决条件后，运行以下命令：

```bash
mkdir build
cd build
cmake ..
make check -j 4
```

### Windows

在 Windows 上，仅支持 **Visual Studio 2017** 及更高版本。

> 注意：
>
> 要在 Visual Studio 2017 (MSVC 14.1) 中使用 C++17，pybind11 需要向编译器传递标志 `/permissive-` 以强制执行标准一致性。在使用 Visual Studio 2019 构建时，这并不绝对必要，但仍然建议使用。

编译并运行测试：

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release --target check
```

这将创建一个 Visual Studio 项目，从命令行编译并运行目标。

> 注意：
>
> 如果所有测试都失败，请确保 Python 二进制文件和测试用例是为相同处理器类型和位宽（即 **i386** 或 **x86_64**）编译的。您可以使用 `cmake -A x64 ..` 指定生成的 Visual Studio 项目的目标架构为 **x86_64**。

> See also：
>
> 已经熟悉 Boost.Python 的高级用户可能希望跳过教程，直接查看 `tests` 目录中的测试用例，这些测试用例涵盖了 pybind11 的所有特性。

## 头文件和命名空间约定

为了简洁，所有代码示例都假设存在以下两行：

```c++
#include <pybind11/pybind11.h>
namespace py = pybind11;
```

某些特性可能需要额外的头文件，但必要时会指定。

## 为简单函数创建绑定

让我们从为一个非常简单的函数创建 Python 绑定开始，该函数将两个数字相加并返回它们的结果：

```
int add(int i, int j) {
    return i + j;
}
```

为了简单起见，我们将此函数和绑定代码放入名为 `example.cpp` 的文件中，内容如下：

```c++
#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // 可选的模块文档字符串

    m.def("add", &add, "A function that adds two numbers");
}
```

> 实际上，实现和绑定代码通常会在不同的文件中

`PYBIND11_MODULE()` 宏创建了一个函数，当在 Python 中发出 `import` 语句时将被调用。模块名称（`example`）作为第一个宏参数（它不应该加引号）。第二个参数（`m`）定义了一个 `py::module_` 类型的变量，这是创建绑定的主要接口。`module_::def()` 方法生成绑定代码，将 `add()` 函数暴露给 Python。

> 注意
>
> 注意到暴露我们的函数到 Python 所需的代码量非常少：所有关于函数参数和返回值的细节都是使用模板元编程自动推断出来的。这种方法和使用的语法借鉴自 Boost.Python，尽管底层实现非常不同。
>

pybind11 是一个仅包含头文件的库，因此不需要链接任何特殊库，也没有中间（魔法）转换步骤。在 Linux 上，上面的示例可以使用以下命令编译：

```bash
$ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
```

> 注意
>
> 如果您使用[作为子模块包含](<./Installing the library.md#作为子模块包含>)来获取 pybind11 源代码，那么在上述编译中使用 `$(python3-config --includes) -Iextern/pybind11/include` 替换 `$(python3 -m pybind11 --includes)`，如手动构建中所解释的。
>

有关 Linux 和 macOS 上所需编译器标志的更多详细信息，请参阅[手动构建](https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually)。有关完整的跨平台编译指南，请参阅[构建系统](https://pybind11.readthedocs.io/en/stable/compiling.html#compiling)页面。

[python_example](https://github.com/pybind/python_example) 和 [cmake_example](https://github.com/pybind/cmake_example) 存储库也是开始的好地方。它们都是具有跨平台构建系统的完整项目示例。两者之间的唯一区别在于 python_example 使用 Python 的 `setuptools` 构建模块，而 cmake_example 使用 CMake（对于现有的 C++ 项目可能更受偏好）。

构建上述 C++ 代码将生成一个二进制模块文件，可以导入到 Python 中。假设编译后的模块位于当前目录中，以下交互式 Python 会话显示了如何加载和执行示例：

```bash
$ python
Python 3.9.10 (main, Jan 15 2022, 11:48:04)
[Clang 13.0.0 (clang-1300.0.29.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import example
>>> example.add(1, 2)
3
>>> 
```

## 关键词参数

通过简单的代码修改，可以通知 Python 关于参数的名称（在本例中为 "i" 和 "j"）。

```c++
m.def("add", &add, "A function which adds two numbers",
      py::arg("i"), py::arg("j"));
```

[`arg`](https://pybind11.readthedocs.io/en/stable/reference.html#_CPPv43arg) 是几种特殊标签类之一，可以用来向 `module_::def()` 传递元数据。通过这种修改后的绑定代码，我们现在可以使用关键词参数调用函数，这是一种更易读的替代方案，特别是对于接受许多参数的函数：

```python
>>> import example
>>> example.add(i=1, j=2)
3L
```

关键词名称也出现在文档中的函数签名里。

```python
>>> help(example)

....
FUNCTIONS
    add(...)
        Signature : (i: int, j: int) -> int

        A function which adds two numbers
```

命名参数的简短表示法也可用：

```c++
// 常规表示法
m.def("add1", &add, py::arg("i"), py::arg("j"));
// 简写
using namespace pybind11::literals;
m.def("add2", &add, "i"_a, "j"_a);
```

`_a` 后缀形成了一个 C++11 字面量，等同于 `arg`。注意，必须先通过指令 `using namespace pybind11::literals` 使字面量运算符可见。这不会从 `pybind11` 命名空间中引入除字面量之外的任何其他内容。

## 默认参数

假设要绑定的函数有默认参数，例如：

```c++
int add(int i = 1, int j = 2) {
    return i + j;
}
```

遗憾的是，pybind11 无法自动提取这些参数，因为它们不是函数类型信息的一部分。但是，它们可以很容易地使用 `arg` 的扩展来指定：

```c++
m.def("add", &add, "A function which adds two numbers",
      py::arg("i") = 1, py::arg("j") = 2);
```

默认值也会出现在文档中。

```python
>>> help(example)

....
FUNCTIONS
    add(...)
        Signature : (i: int = 1, j: int = 2) -> int

        A function which adds two numbers
```

默认参数的简写表示法也可用：

```c++
// 常规表示法
m.def("add1", &add, py::arg("i") = 1, py::arg("j") = 2);
// 简写
m.def("add2", &add, "i"_a=1, "j"_a=2);
```

## 导出变量

要公开 C++ 中的值，使用 `attr` 函数将其注册到模块中，如下所示。内置类型和通用对象（稍后将详细介绍）在分配为属性时会自动转换，也可以使用 `py::cast` 函数进行显式转换。

```c++
PYBIND11_MODULE(example, m) {
    m.attr("the_answer") = 42;
    py::object world = py::cast("World");
    m.attr("what") = world;
}
```

然后，它们可以从 Python 中访问：

```python
>>> import example
>>> example.the_answer
42
>>> example.what
'World'
```

## 支持的数据类型

开箱即用支持大量数据类型，可以无缝地用作函数参数、返回值或通常与 `py::cast` 一起使用。 有关完整概览，请参见[类型转换](https://pybind11.readthedocs.io/en/stable/advanced/cast/index.html)部分。