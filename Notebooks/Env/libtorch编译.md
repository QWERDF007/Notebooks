# PyTorch C++ 编译

## 平台 & 环境依赖

- Ubuntu 16.04.7
- Intel(R) Xeon(R) Platinum 8180 CPU @ 2.50GHz
- RTX 3090
- cuda 11.2
- cudnn 8.1.1
- NVIDIA driver 460.39
- gcc & g++ 8.4.0
- PyTorch 1.10.0

[gcc&g++安装](gcc&g++安装.md)

[NVIDIA驱动&CUDA&CUDNN安装](NVIDIA驱动&CUDA&CUDNN安装.md)

## PyTorch 编译

下载

```bash
git clone https://github.com/pytorch/pytorch.git
```

切换 `pytorch` 源码目录

```bash
cd pytorch
```

创建一个 conda 环境

```bash
conda create -n ctorch python=3.8
```

激活 conda

```bash
conda activate ctorch
```

安装依赖

```bash
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
```

同步源码的子模块

```bash
git submodule sync
```

```bash
git submodule update --init --recursive
```

设置 CMake 变量

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
```

编译

```bash
python tools/build_libtorch.py
```

复制头文件到编译目录

```bash
cp -r torch/include build
```

编译完成后将编译目录下的头文件目录 `build/include`  和 `build/include/torch/csrc/api/include` 的绝对路径添加到项目 `CMakeLists.txt` 的 `include_directories`。

将编译目录下的 `.so` 文件所在目录 `build/lib` 的绝对路径添加到项目 `CMakeLists.txt` 的 `link_directories`

最后将 `c10 c10_cuda torch torch_cuda torch_cpu "-Wl,--no-as-needed -ltorch_cuda"` 添加到 `CMakeLists.txt` 的 `link_libraries` 后编译项目即可。

**注意：** `"-Wl,--no-as-needed -ltorch_cuda"`，`--no-as-needed` 就是不忽略链接时没有用到的动态库这，句是不要忽略链接时没有用到的 `torch_cuda`，为啥没用到是个 :bug:。

[issue#42018](https://github.com/pytorch/pytorch/issues/42018#issuecomment-664526309) 在 `CMakeLists.txt` 添加 `set(CMAKE_LINK_WHAT_YOU_USE TRUE)` 这个也行。

[issue#36437](https://github.com/pytorch/pytorch/issues/36437#issuecomment-612992717)
