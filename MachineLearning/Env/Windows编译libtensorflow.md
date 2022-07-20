# Windows 编译 TensorFlow C++

## 平台 & 环境依赖

- Windows 10
- AMD 3600X
- RAM 16 GB
- Visual Studio 2019
- bazel 3.7.2
- TensorFLow 2.5.0

## TensorFlow 编译

### 安装 bazel

下载 bazel：[Download](https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-windows-x86_64.exe)

将 exe 文件路径添加至环境变量 Path

### 编译 TensorFlow

创建一个 conda 环境

```powershell
conda create -n tf2 python=3.8 
```

```powershell
conda activate tf
```

安装依赖

```powershell
pip3 install six numpy wheel
pip3 install keras_applications==1.0.6 --no-deps
pip3 install keras_preprocessing==1.0.5 --no-deps
```

克隆源码

```powershell
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
```

切换版本至 v2.5.0

```powershell
git checkout v2.5.0
```

配置

```powershell
configure
```

```powershell
You have bazel 3.7.2 installed.
Please specify the location of python. [Default is D:\software\anaconda3\envs\tf2\python.exe]:


Found possible Python library paths:
  D:\software\anaconda3\envs\tf2\lib\site-packages
Please input the desired Python library path to use.  Default is [D:\software\anaconda3\envs\tf2\lib\site-packages]

Do you wish to build TensorFlow with ROCm support? [y/N]: N
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: N
No CUDA support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is /arch:AVX]:


Would you like to override eigen strong inline for some C++ compilation to reduce the compilation time? [Y/n]:
Eigen strong inline overridden.

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]:
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=mkl            # Build with MKL support.
        --config=mkl_aarch64    # Build with oneDNN and Compute Library for the Arm Architecture (ACL).
        --config=monolithic     # Config for mostly static monolithic build.
        --config=numa           # Build with NUMA support.
        --config=dynamic_kernels        # (Experimental) Build kernels into separate shared objects.
        --config=v2             # Build TensorFlow 2.x instead of 1.x.
Preconfigured Bazel build configs to DISABLE default on features:
        --config=noaws          # Disable AWS S3 filesystem support.
        --config=nogcp          # Disable GCP support.
        --config=nohdfs         # Disable HDFS support.
        --config=nonccl         # Disable NVIDIA NCCL support.
Configuration finished
```

编译 dll

```powershell
bazel.exe --output_user_root=D:/DevLib/bazelbuild build --config=opt --config=monolithic //tensorflow:tensorflow_cc.dll 
```

编译 lib

```powershell
bazel.exe --output_user_root=D:/DevLib/bazelbuild build --config=opt --config=monolithic //tensorflow:tensorflow_cc_dll_import_lib
```

安装头文件

```powershell
bazel.exe --output_user_root=D:/DevLib/bazelbuild build --config=opt --config=monolithic //tensorflow:install_headers
```

ps：可以将三个合在一起编译

```powershell
bazel.exe --output_user_root=D:/DevLib/bazelbuild build --config=opt --config=monolithic //tensorflow:tensorflow_cc.dll //tensorflow:tensorflow_cc_dll_import_lib //tensorflow:install_headers
```

**注意：** 16GB 的内存在编译过程中可能会报错，提示堆空间不够，所以在编译中尽量关闭所有不必要的东西，或者加装内存条。

编译完成后在 TensorFlow 的源码目录中会生成 `bazel-bin`，`bazel-out`，`bazel-tensorflow` 和 `bazel-testlogs` 等目录的快捷方式，编译成功的 `dll`、`lib` 和 `include` 在 `bazel-bin/tensorflow` 中，复制到任意安装目录即可。

## 使用 libtensorflow 所遇到的错误

1. 找不到 protobuf 的相关头文件

   解决方法：将编译生成 `bazel-bin/tensorflow` 下的 `include/src` 添加到包含头文件目录中。

2. C2589 “(”:“::”右边的非法标记

   问题原因：函数模板 `max` 与 Visual Studio C++ 中的全局的宏 `max` 冲突，

   解决方法：添加预处理宏 `NOMINMAX` 。

3. 无法解析的外部符号

   ```powershell
   无法解析的外部符号 "class tensorflow::Status __cdecl tensorflow::NewSession(struct tensorflow::SessionOptions const &,class tensorflow::Session * *)" (?NewSession@tensorflow@@YA?AVStatus@1@AEBUSessionOptions@1@PEAPEAVSession@1@@Z)
   无法解析的外部符号 "public: __cdecl tensorflow::SessionOptions::SessionOptions(void)" (??0SessionOptions@tensorflow@@QEAA@XZ)
   ```

   解决方法：修改源码目录下 `tensorflow/tools/def_file_filter/def_file_filter.py.tpl`，搜索 `args.target`，添加下列代码并重新编译 `dll` 及 `lib`。

   ```python
   def_fp.write("\t ?NewSession@tensorflow@@YA?AVStatus@1@AEBUSessionOptions@1@PEAPEAVSession@1@@Z\n")
   ```

   ```python
   def_fp.write("\t ??0SessionOptions@tensorflow@@QEAA@XZ\n")
   ```

   **注：** 第二个错误没有报，所以实际只添加了第一个 `NewSession`。

## 附录

CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.0)
set(PROJECT_NAME project_name)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
add_definitions(-D NOMINMAX)
include_directories(
    path_to_bazel-bin/tensorflow/include
    path_to_bazel-bin/tensorflow/include/src
)
link_directories(
    path_to_bazel-bin/tensorflow
)
link_libraries(
    tensorflow_cc
)

add_executable(${PROJECT_NAME} main.cpp)
```

