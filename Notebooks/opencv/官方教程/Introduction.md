# Introduction



OpenCV (开源计算机视觉库: http://opencv.org)是一个开源的库，其中包含几百个计算机视觉算法。该文档描述所谓的OpenCV 2.x API，这本质上是一个 C++ API，与基于 C 语言的 OpenCV 1.x API (C API 已被废弃，自OpenCV 2.4 版本发布以来就没有用"C"编译器测试过了) 相对。

OpenCV 具有模块化结构，这意味着该包包含几个共享的或静态的库。可用的模块如下:

- 核心功能 (**core**) - 一个紧凑的模块，定义了基本的数据结构，包括密集的多维数组Mat和所有其他模块都使用的基本函数。
- 图像处理 (**imgproc**) - 一个图像处理模块，包括线性和非线性图像过滤，几何图像变换(缩放，仿射和透视变形，通用表格重映射)，颜色空间转换，直方图等。
- 视频分析 (**video**) - 一个视频分析模块，包括运动估计，背景减除和目标跟踪算法。 
- 相机标定和三维重建 (**calib3d**) - 基本的多视图几何算法，单目和立体相机标定，对象姿态估计，立体匹配算法和三维重建的元素。
- 二维特征框架 (**features2d**) - 显著的特征检测器，描述符和描述符匹配器。
- 对象检测 (**objdetect**) - 检测对象和预定义类(如人脸，眼睛，杯子，人，汽车等)的实例。
- 高级GUI (**highgui**) - 易于使用的简单UI功能接口。
- 视频IO (**videoio**) - 易于使用的视频捕获和视频编解码器接口。
- 一些其他的辅助模块，如 FLANN和 Google 测试包装器，Python 绑定等。

文档的后续章节描述了每个模块的功能。但首先，请确保熟悉库中广泛使用的通用API概念。

## API Concepts

### cv Namespace

所有 OpenCV 的类和函数都放在 `cv` 命名空间中。因此，要从代码中访问这个功能，请使用 `cv::` 限定符或 `using namespace cv;` 指令:

```c++
#include "opencv2/core.hpp"
...
cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, 5);
...
```

或：

```c++
#include "opencv2/core.hpp"
using namespace cv;
...
Mat H = findHomography(points1, points2, RANSAC, 5 );
...
```

当前或未来的一些 OpenCV 外部名称可能与 STL 或其他库冲突。在这种情况下，请使用显式命名空间说明符来解决名称冲突：

```c++
Mat a(100, 100, CV_32F);
randu(a, Scalar::all(1), Scalar::all(std::rand()));
cv::log(a, a);
a /= std::log(2.);
```

### Automatic Memory Management

OpenCV自动处理所有内存。

首先，`std::vector`、`cv::Mat` 和函数及方法使用的其他数据结构都有析构函数，这些析构函数在需要时释放底层内存缓冲区。这意味着析构函数不总是像 `Mat` 的情况那样释放缓冲区。它们会考虑到可能的数据共享。析构函数会减少与矩阵数据缓冲区相关联的引用计数器。如果且仅当引用计数器达到零时，即当没有其他结构引用相同的缓冲区时，才会释放缓冲区。类似地，当 `Mat` 实例被复制时，实际上没有复制任何数据。相反，引用计数器被递增，以记住有另一个此数据的所有者。还有 `cv::Mat::clone` 方法可以创建矩阵数据的完整副本。请参见下面的示例:

```c++
// create a big 8Mb matrix
Mat A(1000, 1000, CV_64F);
// create another header for the same matrix;
// this is an instant operation, regardless of the matrix size.
Mat B = A;
// create another header for the 3-rd row of A; no data is copied either
Mat C = B.row(3);
// now create a separate copy of the matrix
Mat D = B.clone();
// copy the 5-th row of B to C, that is, copy the 5-th row of A
// to the 3-rd row of A.
B.row(5).copyTo(C);
// now let A and D share the data; after that the modified version
// of A is still referenced by B and C.
A = D;
// now make B an empty matrix (which references no memory buffers),
// but the modified version of A will still be referenced by C,
// despite that C is just a single row of the original A
B.release();
// finally, make a full copy of C. As a result, the big modified
// matrix will be deallocated, since it is not referenced by anyone
C = C.clone();
```

你看 `Mat` 等基本结构的使用很简单。但是，在不考虑自动内存管理的情况下创建的高级类甚至用户数据类型又如何呢？对于他们，OpenCV 提供了类似于 C++11 中的 `std::shared_ptr` 的 `cv::Ptr` 模板类。因此，不要使用普通指针：

```c++
T* ptr = new T(...);
```

您可以使用：

```c++
Ptr<T> ptr(new T(...));
```

或者：

```c++
Ptr<T> ptr = makePtr<T>(...);
```

`Ptr<T>` 封装一个指向 T 实例的指针以及与该指针关联的引用计数器。有关详细信息，请参阅 [cv::Ptr](https://docs.opencv.org/4.8.0/dc/d84/group__core__basic.html#ga6395ca871a678020c4a31fadf7e8cc63) 描述。

### Automatic Allocation of the Output Data

大多数时候，OpenCV 会自动释放内存，并自动为输出函数参数分配内存。因此，如果函数具有一个或多个输入数组 (`cv::Mat` 实例) 和一些输出数组，则输出数组会自动分配或重新分配。输出数组的大小和类型由输入数组的大小和类型确定。如果需要，这些函数会采用额外的参数来帮助确定输出数组的属性。

示例：

```c++
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;
int main(int, char**)
{
    VideoCapture cap(0);
    if(!cap.isOpened()) return -1;
    Mat frame, edges;
    namedWindow("edges", WINDOW_AUTOSIZE);
    for(;;)
    {
        cap >> frame;
        cvtColor(frame, edges, COLOR_BGR2GRAY);
        GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        Canny(edges, edges, 0, 30, 3);
        imshow("edges", edges);
        if(waitKey(30) >= 0) break;
    }
    return 0;
}
```

数组 `frame` 是由 `>>` 运算符自动分配的，因为视频帧分辨率和位深度对视频捕获模块是已知的。数组 `edges` 是由 `cvtColor` 函数自动分配的。它与输入数组具有相同的大小和位深度。通道数为1，因为传入了颜色转换代码  `cv::COLOR\_BGR2GRAY`，表示颜色到灰度的转换。请注意，`frame` 和 `edges` 在循环体第一次执行时仅分配一次，因为所有后续视频帧都具有相同的分辨率。如果你以某种方式改变了视频分辨率，数组会自动重新分配。

这种技术的关键组件是 `cv::Mat::create` 方法。它接受所需的数组大小和类型。如果数组已经具有指定的大小和类型，则该方法不执行任何操作。否则，它会释放先前分配的数据 (如果有的话，这部分涉及递减引用计数器并与零进行比较)，然后分配所需大小的新缓冲区。大多数函数对每个输出数组调用 `cv::Mat::create` 方法，因此自动输出数据分配得到实现。

这种方案的一些显着例外是 `cv::mixChannels`、`cv::RNG::fill` 和其他一些函数和方法。它们无法分配输出数组，所以你必须提前完成这一操作。

### Saturation Arithmetics

作为计算机视觉库，OpenCV 经常处理图像像素，这些像素通常以紧凑的 8 或 16 位每通道形式编码，因此具有有限的值范围。此外，对图像的某些操作，如颜色空间转换、亮度/对比度调整、锐化、复杂的内插 (bi-cubic、Lanczos) 会产生超出可用范围的值。如果只存储结果的最低 8(16) 位，这会导致视觉伪影并可能影响进一步的图像分析。为了解决这个问题，使用了所谓的饱和算术。例如，要将运算结果 r 存储到 8 位图像中，可以找到 0..255 范围内最近的值:

$$
I(x,y) = \min(\max(round(r), 0), 255)
$$

类似的规则适用于有符号的 8 位、有符号的 16 位和无符号的类型。这个语义在整个库中都被使用。在 C++ 代码中，它是使用类似标准 C++ cast 操作的 `cv::saturate_cast<>` 函数来完成的。参见下面提供的公式的实现:

```c++
I.at<uchar>(y, x) = saturate_cast<uchar>(r);
```

其中 `cv::uchar` 是 OpenCV 8 位无符号整数类型。优化后的 SIMD 代码中使用了 `paddusb`、`packuswb` 等 SSE2指令。它们有助于实现与 C++ 代码中完全相同的行为。

> 当结果是 32 位整数时，不应用饱和。

### Fixed Pixel Types. Limited Use of Templates

模板是 C++ 的一个伟大功能，可以实现非常强大、高效且安全的数据结构和算法。然而，广泛使用模板可能会显著增加编译时间和代码大小。此外，在仅使用模板时很难分离接口和实现。对于基本算法这可能没问题，但对于计算机视觉库来说就不好了，因为单个算法可能跨越数千行代码。因此，为了简化为其他语言 (如Python、Java、Matlab) 开发绑定的过程，这些语言根本没有模板或具有有限的模板功能，当前的 OpenCV 实现基于多态性和基于模板的运行时分派。在运行时分派会太慢的地方 (像像素访问运算符)，不可能的地方 (通用 `cv::Ptr<>` 实现)，或只是非常不方便的地方 (`cv::saturate_cast<>()`)，当前的实现引入了小的模板类、方法和函数。在当前OpenCV 版本的其他任何地方，模板的使用都是有限的。

因此，库可以操作的原始数据类型集合是有限定的。也就是说，数组元素应该具有以下类型之一:

- 8 位无符号整数 (uchar)
- 8 位有符号整数 (schar) 
- 16 位无符号整数 (ushort)
- 16 位有符号整数 (short)
- 32 位有符号整数 (int)
- 32 位浮点数 (float)
- 64 位浮点数 (double)
- 几个元素的组合，其中所有元素具有相同类型 (上述之一)。元素为这种组合的数组称为多通道数组，与元素为标量值的单通道数组相对。可能的最大通道数由常量 `CV_CN_MAX` 定义，当前设置为 512。

对这些基本类型，应用以下枚举:

```c++
enum { CV_8U=0, CV_8S=1, CV_16U=2, CV_16S=3, CV_32S=4, CV_32F=5, CV_64F=6 };
```

可以使用以下选项指定多通道（n 通道）类型：

- `CV_8UC1` ... `CV_64FC4` 常量（适用于 1 到 4 的通道数）
- 当通道数大于 4 或编译时未知时，使用 `CV_8UC(n)` ... `CV_64FC(n)` 或 `CV_MAKETYPE(CV_8U, n)` ... `CV_MAKETYPE(CV_64F, n)` 宏。

> `CV_32FC1 == CV_32F, CV_32FC2 == CV_32FC(2) == CV_MAKETYPE(CV_32F, 2)`，和`CV_MAKETYPE(depth, n) == ((depth&7) + ((n-1)<<3)`。这意味着常量类型由深度 (取最低 3 位) 和通道数减 1 (取接下来的 `log2(CV_CN_MAX)` 位) 构成。

示例：

```c++
Mat mtx(3, 3, CV_32F); // make a 3x3 floating-point matrix
Mat cmtx(10, 1, CV_64FC2); // make a 10x1 2-channel floating-point
                           // matrix (10-element complex vector)
Mat img(Size(1920, 1080), CV_8UC3); // make a 3-channel (color) image
                                    // of 1920 columns and 1080 rows.
Mat grayscale(img.size(), CV_MAKETYPE(img.depth(), 1)); // make a 1-channel image of
                                                        // the same size and same
                                                        // channel type as img
```

具有更复杂元素的数组无法使用 OpenCV 构建或处理。此外，每个函数或方法只能处理所有可能的数组类型的一个子集。通常，算法复杂度越高，支持的格式子集越小。以下是此类限制的典型示例:

- 人脸检测算法仅适用于 8 位灰度或彩色图像。

- 线性代数函数和大多数机器学习算法仅适用于浮点数组。 

- 基本函数 (如 `cv::add`) 支持所有类型。

- 颜色空间转换函数支持 8 位无符号、16 位无符号和 32 位浮点类型。

每个函数支持的类型子集都是从实际需求定义的，并可根据用户需求在未来进行扩展。

### InputArray and OutputArray

许多 OpenCV 函数处理稠密的 2 维或多维数值数组。通常，这些函数以 `cv::Mat` 作为参数，但在某些情况下，使用 `std::vector<>` (例如用于点集) 或 `cv::Matx<>` (用于 3x3 齐次矩阵等) 更方便。为避免 API 中的许多重复，引入了特殊的"代理"类。基础"代理"类是 `cv::InputArray`。它用于在函数输入上传递只读数组。派生自 `InputArray` 的类 `cv::OutputArray` 用于为函数指定输出数组。通常，你不需要关心这些中间类型 (也不应该显式声明这些类型的变量) — 它将自动完成所有工作。你可以假定除了 `InputArray/OutputArray`，你可以始终使用 `cv::Mat`、`std::vector<>`、`cv::Matx<>`、`cv::Vec<>` 或 `cv::Scalar`。当一个函数有一个可选的输入或输出数组，并且你没有或不想要一个时，传递 `cv::noArray()`。

### Error Handling

OpenCV 使用异常来指示关键错误。当输入数据格式正确、属于指定值范围时，但由于某种原因算法无法成功 (例如，优化算法没有收敛)，它会返回一个特殊的错误码 (通常只是一个布尔变量)。

异常可以是 `cv::Exception` 类或其派生类的实例。反过来，`cv::Exception` 是 `std::exception` 的派生类。所以它可以在代码中使用其他标准 C++ 库组件优雅地处理。

异常通常是使用 `CV_Error(errcode, description)` 宏抛出的，或其 printf 样式的 `CV_Error_(errcode, (printf-spec, printf-args))` 变体，或者使用宏 `CV\_Assert(condition)` 来检查条件并在不满足时抛出异常。对于性能关键的代码，有 `CV\_DbgAssert(condition)`，它仅在 `Debug` 配置中保留。由于自动内存管理，如果发生突然的错误，所有中间缓冲区都会自动释放。如果需要，你只需要添加一个 `try` 语句来捕获异常:

```c++
try
{
    ... // call OpenCV
}
catch (const cv::Exception& e)
{
    const char* err_msg = e.what();
    std::cout << "exception caught: " << err_msg << std::endl;
}
```

### Multi-threading and Re-enterability

当前的 OpenCV 实现是完全可重入的。也就是说，可以从不同的线程调用同一个函数或不同类实例的同一方法。同样，同一个 `Mat` 可以在不同的线程中使用，因为引用计数操作使用特定体系结构的原子指令。
