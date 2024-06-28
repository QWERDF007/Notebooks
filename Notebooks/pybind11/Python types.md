# [Python types](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/object.html)

## 可用的包装器
所有主要的 Python 类型都可作为简洁的 C++ 包装器类使用。这些也可以作为函数参数使用——参见 [Python 对象作为参数](https://pybind11.readthedocs.io/en/stable/advanced/functions.html#python-objects-as-args)。

可用类型包括  `handle`, `object`, `bool_`, `int_`, `float_`, `str`, `bytes`, `tuple`, `list`, `dict`, `slice`, `none`, `capsule`, `iterable`, `iterator`, `function`, `buffer`, `array`, 和 `array_t`。

> 警告
>
> 在大量使用此功能于您的 C++ API 之前，请确保查阅 [Gotchas](#陷阱) 部分。

## 从 C++ 实例化复合 Python 类型

可以使用 [`dict`](https://pybind11.readthedocs.io/en/stable/reference.html#_CPPv44dict) 构造函数初始化字典：

```c++
using namespace pybind11::literals; // 引入 `_a` 字面量
py::dict d("spam"_a=py::none(), "eggs"_a=42);
```

可以使用 `py::make_tuple()` 实例化 Python 对象的元组：

```c++
py::tuple tup = py::make_tuple(42, py::none(), "spam");
```

每个元素都转换为支持的 Python 类型。

可以使用以下代码实例化一个[简单的命名空间](https://docs.python.org/3/library/types.html#types.SimpleNamespace)：

```c++
using namespace pybind11::literals;  // 引入 `_a` 字面量
py::object SimpleNamespace = py::module_::import("types").attr("SimpleNamespace");
py::object ns = SimpleNamespace("spam"_a=py::none(), "eggs"_a=42);
```

可以使用 `py::delattr()`, `py::getattr()`, 和 `py::setattr()` 函数修改命名空间的属性。简单的命名空间可以作为类实例的轻量级替代品。

## 来回转换

在这种混合代码中，通常需要将任意C++类型转换为Python，这可以使用`py::cast()`完成：

```c++
MyClass *cls = ...;
py::object obj = py::cast(cls);
```

反向转换使用以下语法：

```c++
py::object obj = ...;
MyClass *cls = obj.cast<MyClass *>();
```

当转换失败时，两个方向都会抛出`cast_error`异常。

## 从 C++ 访问 Python 库

也可以导入在 Python 标准库中定义的对象或当前 Python 环境中可用的对象（`sys.path`），并在 C++ 中使用这些对象。

此示例获取对 Python `Decimal` 类的引用。

```c++
// 等同于 "from decimal import Decimal"
py::object Decimal = py::module_::import("decimal").attr("Decimal");
// 尝试导入scipy
py::object scipy = py::module_::import("scipy");
return scipy.attr("__version__");
```

## 调用 Python 函数 (functions)

也可以通过 `operator()` 调用 Python 类、函数和方法。

```c++
// 构造Decimal类的Python对象
py::object pi = Decimal("3.14159");
// 使用Python创建我们的目录
py::object os = py::module_::import("os");
py::object makedirs = os.attr("makedirs");
makedirs("/tmp/path/to/somewhere");
```

如果定义了 `py::class_` 或类型转换，可以将从 Python 获得的结果转换为纯 C++ 版本。

```c++
py::function f = <...>;
py::object result_py = f(1234, "hello", some_instance);
MyClass &result = result_py.cast<MyClass>();
```

## 调用 Python 方法 (methods)

要调用对象的方法，可以再次使用 `.attr` 来访问 Python 方法。

```c++
// 用十进制计算e^π
py::object exp_pi = pi.attr("exp")();
py::print(py::str(exp_pi));
```

在上面的例子中，`pi.attr("exp")` 是一个绑定方法：它将始终为该类的同一个实例调用该方法。或者，可以通过 Python 类（而不是实例）创建一个非绑定方法，然后显式传递 `self` 对象，后面跟着其他参数。

```c++
py::object decimal_exp = Decimal.attr("exp");

// 计算e^n，n=0..4
for (int n = 0; n < 5; n++) {
    py::print(decimal_exp(Decimal(n));
}
```

## 关键字参数

支持关键字参数。下面是 Python 中常见的调用语法

```python
def f(number, say, to):
    ...  # 函数代码

f(1234, say="hello", to=some_instance)  # Python 中的关键字调用
```

在 C++ 中，可以使用 pybind11 的 `_a` 模拟类似的关键字调用：

```c++
using namespace pybind11::literals; // 引用 `_a` 文字标识符
f(1234, "say"_a="hello", "to"_a=some_instance); // C++ 中的关键字调用
```

## 参数解包

pybind11 支持使用 `*args` 和 `**kwargs` 解包参数，可以和其他参数混合使用：

```c++
// * unpacking
py::tuple args = py::make_tuple(1234, "hello", some_instance);
f(*args);

// ** unpacking
py::dict kwargs = py::dict("number"_a=1234, "say"_a="hello", "to"_a=some_instance);
f(**kwargs);

// mixed keywords, * and ** unpacking
py::tuple args = py::make_tuple(1234);
py::dict kwargs = py::dict("to"_a=some_instance);
f(*args, "say"_a="hello", **kwargs);
```

支持根据 [PEP448](https://www.python.org/dev/peps/pep-0448/) 的通用解包

```c++
py::dict kwargs1 = py::dict("number"_a=1234);
py::dict kwargs2 = py::dict("to"_a=some_instance);
f(**kwargs1, "say"_a="hello", **kwargs2);
```

> see also
>
> 文件 `tests/test_pytypes.cpp` 包含一个完整的示例，更详细地演示了如何传递原生 Python 类型。文件 `tests/test_callbacks.cpp` 提供了一些从 C++ 调用 Python 函数的示例，包括关键字参数和解包。

## 隐式转换

当使用 Python 类型的 C++ 接口，或者调用 Python 函数时，会返回类型为 `object` 的对象。可以调用隐式转换到像 `dict` 这样的子类。通过 `operator[]` 或 `obj.attr()` 返回的代理对象也是如此。转换为子类型可以提高代码的可读性，并允许将值传递给需要特定子类型而不是通用 `object` 的 C++ 函数。

```c++
#include <pybind11/numpy.h>
using namespace pybind11::literals;

py::module_ os = py::module_::import("os");
py::module_ path = py::module_::import("os.path");  // like 'import os.path as path'
py::module_ np = py::module_::import("numpy");  // like 'import numpy as np'

py::str curdir_abs = path.attr("abspath")(path.attr("curdir"));
py::print(py::str("Current directory: ") + curdir_abs);
py::dict environ = os.attr("environ");
py::print(environ["HOME"]);
py::array_t<float> arr = np.attr("ones")(3, "dtype"_a="float32");
py::print(py::repr(arr + py::int_(1)));
```

这些隐式转换适用于 `object` 的子类；不需要像自定义类那样显式调用 `obj.cast()`，参见[来回转换](#来回转换)。

> Note
>
> 如果通过移动构造函数进行简单的转换不可能，隐式和显式转换（调用 `obj.cast()` ）将尝试进行“丰富”转换。例如，`py::list env = os.attr("environ");` 将成功，并且等同于 Python 代码 `env = list(os.environ)` ，该代码生成一个包含字典键的列表。

## 处理异常 

包装类中的 Python 异常将被抛出为 `py::error_already_set` 。有关在调用C++包装类时处理异常的更多信息，请参见 [C++ 中处理来自 Python 的异常](https://pybind11.readthedocs.io/en/latest/advanced/exceptions.html#handling-python-exceptions-cpp)。

## 陷阱 

### 默认构造的包装器

当一个包装器类型被默认构造时，它不是一个有效的 Python 对象（即它不是 `py::none()` ）。它仅仅是 `PyObject*` 空指针。要检查这一点，请使用 `static_cast<bool>(my_wrapper)` 。

### 将 py::none() 分配给包装器 

你可能会想在 C++ 签名中使用像 `py::str` 和 `py::dict` 这样的类型（无论是纯C++，还是在绑定签名中），并将它们默认值设置为 `py::none()`。然而，在最好的情况下，它会因为 `None` 不能转换为该类型（例如 `py::dict` ）而快速失败，或者在更糟糕的情况下，它将静默工作但破坏你想要使用的数据类型（例如`py::str(py::none()` ) 在 Python 中将产生 `"None"`）。