# [安装 Faiss](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss)

## 标准安装

我们支持使用 cmake 从源代码编译 Faiss 以及在有限的平台上通过 conda 安装：Linux (x86 和 ARM)，Mac (x86 和 ARM)，Windows (仅x86)。有关此信息，请参阅 [INSTALL.md](#INSTALL.md)。

### 为什么不支持通过 XXX 安装?

我们不支持更多平台的原因是，确保 Faiss 在受支持的配置中运行需要做大量的工作：为新版本的 Faiss 构建 conda 包总是会出现兼容性问题。 Anaconda 提供了一个足够受控的环境，我们可以确信它将在用户的机器上运行 (pip 则不然)。 此外，平台 (硬件和操作系统) 必须得到我们的 CI 工具（circleCI）的支持。

因此，在添加新的官方支持的平台(硬件和软件)之前，我们非常谨慎。 我们对将其移植到其他平台的成功 (或失败!) 的故事以及相关 PR 非常感兴趣。

## 特殊配置

### 在 Anaconda 安装中编译 python 接口

这个想法是通过 anaconda 安装所有东西，并将 Faiss 与它联系起来。这对于确保 MKL 实现尽可能快很有用。

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate host_env_for_faiss     # an environment that contains python and numpy 

git clone https://github.com/facebookresearch/faiss.git faiss_xx

cd faiss_xx

LD_LIBRARY_PATH= MKLROOT=/private/home/matthijs/anaconda3/envs/host_env_for_faiss/lib CXX=$(which g++) \
$cmake -B build -DBUILD_TESTING=ON -DFAISS_ENABLE_GPU=OFF \
             -DFAISS_OPT_LEVEL=axv2 \
             -DFAISS_ENABLE_C_API=ON \
             -DCMAKE_BUILD_TYPE=Release \
             -DBLA_VENDOR=Intel10_64_dyn .


make -C build -j10 swigfaiss && (cd build/faiss/python ; python3 setup.py build)

(cd tests ; PYTHONPATH=../build/faiss/python/build/lib/ OMP_NUM_THREADS=1 python -m unittest -v discover )
```

### 在 ARM 上编译 Faiss

Amazon c6g.8xlarge 计算机上的 ubuntu 18 映像的命令：

```bash

set -e

sudo apt-get install libatlas-base-dev libatlas3-base
sudo apt-get install clang-8
sudo apt-get install swig

# cmake provided with ubuntu is too old

wget https://github.com/Kitware/CMake/releases/download/v3.19.3/cmake-3.19.3.tar.gz

tar xvzf cmake-3.19.3.tar.gz
cd cmake-3.19.3/
./configure --prefix=/home/matthijs/cmake &&  make -j

cd $HOME

alias cmake=$HOME/cmake/bin/cmake

# clone Faiss

git clone https://github.com/facebookresearch/faiss.git

cd faiss

cmake  -B build -DCMAKE_CXX_COMPILER=clang++-8 -DFAISS_ENABLE_GPU=OFF  -DPython_EXECUTABLE=$(which python3) -DFAISS_OPT_LEVEL=generic -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST\
ING=ON

(cd build/faiss/python/ ; python3 setup.py build)

# run tests

export PYTHONPATH=$PWD/build/faiss/python/build/lib/

python3 -m unittest discover

```

## INSTALL.md

### 通过 conda 安装 Faiss

安装 Faiss 的推荐方法是通过 [conda](https://docs.conda.io/)。稳定版本以及预发布的 nightly 版本会定期推送到 pytorch conda 频道。

仅适用于 CPU 的 `faiss-cpu` conda 软件包目前可在 Linux、OSX 和 Windows 上使用。包含 CPU 和 GPU 索引的 `faiss-gpu` 可在 Linux 系统上使用，适用于 CUDA 11.4。软件包是为 Python 版本 3.8-3.10 构建的。

要安装最新的稳定版本：

```c++
# CPU-only version
$ conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl

# GPU(+CPU) version
$ conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
```

对于 faiss-gpu，cudatoolkit=11.4 需要 nvidia 通道，该通道未在主要的 anaconda 通道中发布。

注意：由于最新 1.7.4 版本中存在错误，需要在适用的情况下单独安装 Intel MKL 2021。在非 Intel 平台上安装时删除 MKL 引用。

Nightly 预发布包可以按如下方式安装。无需单独安装 MKL，必要时会自动安装正确的包作为依赖项：

```c++
# CPU-only version
$ conda install -c pytorch/label/nightly faiss-cpu

# GPU(+CPU) version
$ conda install -c pytorch/label/nightly -c nvidia faiss-gpu=1.7.4
```

安装 GPU Faiss 与 CUDA 11.4 和 Pytorch 的版本组合（截至 2023-06-19）：

```c++
conda create --name faiss_1.7.4 python=3.10
conda activate faiss_1.7.4
conda install faiss-gpu=1.7.4 mkl=2021 pytorch pytorch-cuda numpy -c pytorch -c nvidia
```

### 从 conda-forge 安装

Faiss 还由 [conda-forge](https://conda-forge.org/) 进行打包，conda-forge 是 conda 社区驱动的打包生态系统。打包工作正在与 Faiss 团队合作，以确保高质量的包装构建。

由于 conda-forge 的全面基础设施，**甚至可能会发生 conda-forge 支持某些构建组合，但无法通过 pytorch 通道获得的情况**。要安装，请使用

```c++
# CPU version
$ conda install -c conda-forge faiss-cpu

# GPU version
$ conda install -c conda-forge faiss-gpu
```

您可以使用 `conda list` 来判断您的 conda 包来自哪个渠道。如果您在使用 conda-forge 构建的软件包时遇到问题，请在 conda-forge 软件包“feedstock”上提出[issue](https://github.com/conda-forge/faiss-split-feedstock/issues)。

windows 安装 1.7.4，直接查看 [conda 对应的安装信息](https://anaconda.org/conda-forge/faiss-gpu)

```bash
conda install -c conda-forge faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl cudatoolkit=11.8
```

**注意**：指定 cuda 版本避免与后续安装 torch 不一致

### 从源码构建

- [Windows](<Windows编译faiss.md>)



# 进一步阅读

- 上一章：[Home](<Faiss-Home.md>)
- 下一章：[入门](<Getting started.md>)





<!-- 完成标志, 看不到, 请忽略! -->