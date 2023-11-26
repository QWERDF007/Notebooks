# Windows 编译 Faiss

- faiss 版本：1.7.4
- Visual Studio 16 2019
- win11

## 1. 下载源码

   ```
   git clone https://github.com/facebookresearch/faiss.git
   ```

## 2. 移动至 faiss 目录，打开命令行，创建 build 目录

   ```bash
   mkdir build
   cd build
   ```

## 3. 设置 MKL

下载 MKL，此处使用版本为  2024.0.0.49672

- [官方下载地址](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=window&distributions=offline)

安装完成后将在 `C:\Program Files (x86)\intel` 中找到，将其拷贝复制到其他目录，将其组织成下列结构：

```bash
/path/to/mkl
├─common
├─compiler
├─compiler_ide
├─mkl
│  ├─bin
│  ├─include
│  ├─lib
│  └─share
├─tbb
└─tcm
```


设置环境变量

```bash
set MKLROOT=D:/Software/dev/mkl/mkl
```

注意：将 `MKLROOT` 设置该目录的 `mkl`，FindMKL 会寻找父目录下的 `tbb` 和 `compiler` 目录，用于设置部分 `lib` 的寻找路径

## 4. cmake 配置

   ```bash
   cmake -G "Visual Studio 16 2019" -T host=x64 -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_RAFT=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=Intel10_64ilp -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8" -DCMAKE_CUDA_ARCHITECTURES="89;86" -DCMAKE_INSTALL_PREFIX=../install  ..
   ```

   - `FAISS_ENABLE_GPU`：是否支持 GPU
   - `BUILD_SHARED_LIBS`：是否编译为动态库
   - `FAISS_ENABLE_PYTHON`：是否编译 python
   - `FAISS_ENABLE_RAFT`：使用 RAFT 加速，windows 编译似乎有问题
   - `BLA_VENDOR`：
   - `CMAKE_CUDA_ARCHITECTURES`：设备计算能力，40 系为 `89`，30 系为 `86`
   - `CMAKE_INSTALL_PREFIX`：安装目录

## 5. cmake 编译

   ```bash
   cmake --build . --config Release -j --target install
   ```

   