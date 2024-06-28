# [NumPy](https://pybind11.readthedocs.io/en/latest/advanced/pycpp/numpy.html)

## 缓冲区协议
Python 支持一种非常通用且方便的方法来交换插件库之间的数据。类型可以暴露一个缓冲区视图[1]，这提供了对原始内部数据表示形式的快速直接访问。假设我们想要绑定以下简单的 `Matrix` 类：

```c++
class Matrix {
public:
    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
        m_data = new float[rows*cols];
    }
    float *data() { return m_data; }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
private:
    size_t m_rows, m_cols;
    float *m_data;
};
```

以下绑定代码将 `Matrix` 内容作为缓冲区对象公开，使得可以将矩阵转换为 NumPy 数组。甚至可以使用像 `np.array(matrix_instance, copy=False)` 这样的Python 表达式完全避免复制操作。

```c++
py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
   .def_buffer([](Matrix &m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                               // 缓冲区指针
            sizeof(float),                          // 一个标量的大小
            py::format_descriptor<float>::format(), // Python结构风格格式描述符
            2,                                      // 维度数量
            { m.rows(), m.cols() },                 // 缓冲区维度
            { sizeof(float) * m.cols(),             // 每个索引的步幅（字节）
              sizeof(float) }
        );
    });
```

在一个新类型中支持缓冲区协议涉及在 `py::class_` 构造函数中指定特殊的 `py::buffer_protocol()` 标签，并使用一个 lambda 函数调用 `def_buffer()` 方法，该 lambda 函数按需创建一个 `py::buffer_info` 描述记录，描述给定的矩阵实例。`py::buffer_info` 的内容反映了 Python 缓冲区协议规范。

```c++
struct buffer_info {
    void *ptr;
    py::ssize_t itemsize;
    std::string format;
    py::ssize_t ndim;
    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> strides;
};
```

要创建一个可以接受 Python 缓冲区对象作为参数的 C++ 函数，只需使用类型 `py::buffer` 作为其参数之一。缓冲区可以以多种配置存在，因此在函数体中通常需要进行一些安全检查。下面，你可以看到如何为 Eigen 双精度矩阵（`Eigen::MatrixXd`）类型定义一个自定义构造函数的基本示例，该类型支持从兼容的缓冲区对象（例如 NumPy矩阵）初始化。

```c++
/* 将MatrixXd（或某些其他Eigen类型）绑定到Python */
typedef Eigen::MatrixXd Matrix;

typedef Matrix::Scalar Scalar;
constexpr bool rowMajor = Matrix::Flags & Eigen::RowMajorBit;

py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
    .def(py::init([](py::buffer b) {
        typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;

        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();

        /* Some basic validation checks ... */
        if (info.format != py::format_descriptor<Scalar>::format())
            throw std::runtime_error("Incompatible format: expected a double array!");

        if (info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        auto strides = Strides(
            info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(Scalar),
            info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(Scalar));

        auto map = Eigen::Map<Matrix, 0, Strides>(
            static_cast<Scalar *>(info.ptr), info.shape[0], info.shape[1], strides);

        return Matrix(map);
    }));
```

对于 Eigen 数据类型的 `def_buffer()` 调用应该如下所示：

```c++
.def_buffer([](Matrix &m) -> py::buffer_info {
    return py::buffer_info(
        m.data(),                                // 缓冲区指针
        sizeof(Scalar),                          // 一个标量的大小
        py::format_descriptor<Scalar>::format(), // Python结构风格格式描述符
        2,                                       // 维度数量
        { m.rows(), m.cols() },                  // 缓冲区维度
        { sizeof(Scalar) * (rowMajor ? m.cols() : 1),
          sizeof(Scalar) * (rowMajor ? 1 : m.rows()) } // 每个索引的步幅（字节）
    );
})
```

对于绑定Eigen类型（尽管有一些限制）的更简单方法，请参见 [Eigen](https://pybind11.readthedocs.io/en/latest/advanced/cast/eigen.html) 部分。

> see also
>
> 文件 `tests/test_buffers.cpp` 包含一个完整的示例，演示了使用 pybind11 的缓冲区协议的更多细节

## 数组
通过在上述代码片段中将 `py::buffer` 替换为 `py::array`，我们可以将函数限制为只接受 NumPy 数组（而不是满足缓冲协议的任何类型的 Python 对象）。

在许多情况下，我们希望定义一个函数，它只接受特定数据类型的 NumPy 数组。这可以通过 `py::array_t<T>` 模板实现。例如，以下函数要求参数是一个包含双精度值的 NumPy 数组。

```cpp
void f(py::array_t<double> array);
```

当它被不同类型（例如整数或整数列表）调用时，绑定代码将尝试将输入转换为所需类型的 NumPy 数组。此功能需要包含 `pybind11/numpy.h` 头文件。注意，`pybind11/numpy.h` 不依赖于 NumPy 头文件，因此可以在没有声明对 NumPy 的构建时依赖性的情况下使用；NumPy>=1.7.0 是运行时依赖。

NumPy数组中的数据不一定以密集方式打包；此外，条目可以由任意的列和行步幅分隔。有时，要求函数只接受使用 C（行主序）或 Fortran（列主序）排序的密集数组可能会很有用。这可以通过具有值为 `py::array::c_style` 或 `py::array::f_style` 的第二个模板参数来实现。

```cpp
void f(py::array_t<double, py::array::c_style | py::array::forcecast> array);
```

`py::array::forcecast` 参数是第二个模板参数的默认值，它确保不符合要求的参数被转换为满足指定要求的数组，而不是尝试下一个函数重载。

数组上有几种方法；下面列出的参考方法有效，以及基于 NumPy API 的以下函数：

- `.dtype()` 返回包含值的类型。
- `.strides()` 返回数组的步幅指针（可选传递一个轴索引以获取一个数字）。
- `.flags()` 返回标志设置。`.writable()` 和 `.owndata()` 直接可用。
- `.offset_at()` 返回偏移量（可选传递索引）。
- `.squeeze()` 返回删除长度为 1 的轴的视图。
- `.view(dtype)` 返回具有不同 dtype 的数组视图。
- `.reshape({i, j, ...})` 返回具有不同形状的数组视图。`.resize({...})` 也可用。
- `.index_at(i, j, ...)` 从开头到给定索引的计数。

还有几种获取引用的方法（下面描述）。

## 结构化类型
为了使 `py::array_t` 能够使用结构化（记录）类型，我们首先需要通过 `PYBIND11_NUMPY_DTYPE` 宏注册类型的内存布局，该宏在插件定义代码中调用，期望类型后跟字段名：

```cpp
struct A {
    int x;
    double y;
};

struct B {
    int z;
    A a;
};

// ...
PYBIND11_MODULE(test, m) {
    // ...
    PYBIND11_NUMPY_DTYPE(A, x, y);
    PYBIND11_NUMPY_DTYPE(B, z, a);
    /* 现在A和B都可以用作py::array_t的模板参数 */
}
```

结构应由基本算术类型、`std::complex`、先前注册的子结构以及上述任何类型的数组组成。支持 C++ 数组和 `std::array`。虽然有一个静态断言可以防止许多类型的不支持结构，但用户仍有责任只使用可以作为原始内存安全操作的“简单”结构。

## 向量化函数
假设我们想要将具有以下签名的函数绑定到 Python，以便它可以处理任意 NumPy 数组参数（向量、矩阵、一般 N 维数组）以及其正常参数：

```cpp
double my_func(int x, float y, double z);
```

在包含 `pybind11/numpy.h` 头文件后，这是非常简单的：

```cpp
m.def("vectorized_func", py::vectorize(my_func));
```

像下面这样调用函数将导致对 `my_func` 进行 4 次调用，每次调用都使用数组的其中一个元素。与 `numpy.vectorize()` 等解决方案相比，这种显著的优势是，元素循环完全在 C++ 端运行，并且可以被编译器压缩成一个紧凑的、优化的循环。结果是返回一个类型为 `numpy.dtype.float64` 的 NumPy 数组。

```python
x = np.array([[1, 3], [5, 7]])
y = np.array([[2, 4], [6, 8]])
z = 3
result = vectorized_func(x, y, z)
```

标量参数 `z` 被透明地复制了4次。输入数组 `x` 和 `y` 自动转换为正确的类型（它们是 `numpy.dtype.int64` 类型，但需要分别是 `numpy.dtype.int32` 和 `numpy.dtype.float32`）。

> 注意
>
> 只有通过值传递的算术类型、复数类型和 POD 类型以及通过 `const &` 引用传递的参数才会被向量化；所有其他参数都按原样传递。接受右值引用参数的函数不能被向量化。


在计算过于复杂而无法简化为 `vectorize` 的情况下，将需要手动创建和访问缓冲区内容。下面的代码片段包含了一个完整的示例，展示了如何实现这一点（代码有些牵强，因为它可以更简单地使用 `vector` ）

```c++
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
    py::buffer_info buf1 = input1.request(), buf2 = input2.request();

    if (buf1.ndim != 1 || buf2.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input shapes must match");

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = py::array_t<double>(buf1.size);

    py::buffer_info buf3 = result.request();

    double *ptr1 = static_cast<double *>(buf1.ptr);
    double *ptr2 = static_cast<double *>(buf2.ptr);
    double *ptr3 = static_cast<double *>(buf3.ptr);

    for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        ptr3[idx] = ptr1[idx] + ptr2[idx];

    return result;
}

PYBIND11_MODULE(test, m) {
    m.def("add_arrays", &add_arrays, "Add two NumPy arrays");
}
```

> see also
>
> 文件 `tests/test_numpy_vectorize.cpp` 包含一个完整示例，更详细地演示了如何使用 `vectorize()`

## 直接访问

由于性能原因，特别是在处理非常大的数组时，如果已知索引已经有效，通常希望在每次访问时不进行内部维度和边界检查，直接访问数组元素。为了避免这样的检查，`array` 类和 `array_t<T>` 模板类提供了一个未经检查的代理对象，可以通过 `unchecked<N>` 和 `mutable_unchecked<N>` 方法用于这种未经检查的访问，其中 `N` 指定所需的数组维度：

```c++
m.def("sum_3d", [](py::array_t<double> x) {
    auto r = x.unchecked<3>(); // x 必须具有 ndim = 3; 可以是非可写的
    double sum = 0;
    for (py::ssize_t i = 0; i < r.shape(0); i++)
        for (py::ssize_t j = 0; j < r.shape(1); j++)
            for (py::ssize_t k = 0; k < r.shape(2); k++)
                sum += r(i, j, k);
    return sum;
});
m.def("increment_3d", [](py::array_t<double> x) {
    auto r = x.mutable_unchecked<3>(); // 如果 ndim != 3 或 flags.writeable 为 false，则抛出异常
    for (py::ssize_t i = 0; i < r.shape(0); i++)
        for (py::ssize_t j = 0; j < r.shape(1); j++)
            for (py::ssize_t k = 0; k < r.shape(2); k++)
                r(i, j, k) += 1.0;
}, py::arg().noconvert());
```

要从 `array` 对象获取代理，您必须指定数据类型和维度数作为模板参数，例如 `auto r = myarray.mutable_unchecked<float, 2>()`。

如果编译时不知道维度数，可以省略维度模板参数（即调用 `arr_t.unchecked()` 或 `arr.unchecked<T>()` ）。这将给您一个以相同方式工作的代理对象，但会导致代码优化程度降低，因此在紧密循环中会有轻微的效率损失。

请注意，返回的代理对象直接引用数组的数据，并仅在构造时读取其形状、步长和可写标志。您必须确保引用的数组在返回对象的持续时间内不会被销毁或重新塑形，通常通过限制返回实例的作用域来实现。

返回的代理对象支持一些与 `py::array` 相同的方法，以便它可以作为一些现有的、经过索引检查的 `py::array` 使用的替代品：

- `.ndim()` 返回维度数
- `.data(1, 2, ...)` 和 `r.mutable_data(1, 2, ...)` 分别返回给定索引处的 `const T` 或 `T` 数据的指针。后者仅适用于通过 `a.mutable_unchecked()` 获取的代理。
- `.itemsize()` 返回项目的大小（以字节为单位），即 `sizeof(T)`。
- `.shape(n)` 返回第n维的大小
- `.size()` 返回元素的总数（即形状的乘积）。
- `.nbytes()` 返回引用元素使用的字节数（即 `itemsize()` 乘以 `size()` )。

> see also
>
> 文件 `tests/test_numpy_array.cpp` 包含演示此功能的其他示例。

## 省略号

Python 提供了一个方便的 `...` 省略号符号，通常用于切片多维数组。例如，下面的片段提取张量中间维度，第一和最后一个索引设置为零。

```python
a = ...  # a NumPy数组
b = a[0, ..., 0]
```
`py::ellipsis()` 函数可以在 C++ 端执行相同的操作：

```c++
py::array a = /* 一个NumPy数组 */;
py::array b = a[py::make_tuple(0, py::ellipsis(), 0)];
```

## 内存视图

当我们只是想提供一个直接访问 C/C++ 缓冲区的访问器而没有具体的类对象时，我们可以返回一个 `memoryview` 对象。假设我们希望为 2x4 uint8_t 数组公开一个 `memoryview`，我们可以这样做：

```c++
const uint8_t buffer[] = {
    0, 1, 2, 3,
    4, 5, 6, 7
};
m.def("get_memoryview2d", []() {
    return py::memoryview::from_buffer(
        buffer,                                    // 缓冲区指针
        { 2, 4 },                                  // 形状（行，列）
        { sizeof(uint8_t) * 4, sizeof(uint8_t) }   // 字节中的步长
    );
});
```

这种方法旨在为不受 Python 管理的 C/C++ 缓冲区提供 `memoryview`。用户负责管理缓冲区的生命周期。在 C++ 端删除缓冲区后使用这种方式创建的 `memoryview` 将导致未定义的行为。

我们还可以使用 `memoryview::from_memory` 来处理一个简单的 1D 连续缓冲区：

```c++
m.def("get_memoryview1d", []() {
    return py::memoryview::from_memory(
        buffer,               // 缓冲区指针
        sizeof(uint8_t) * 8   // 缓冲区大小
    );
});
```
版本 2.6 中的变更：添加了 `memoryview::from_memory`。