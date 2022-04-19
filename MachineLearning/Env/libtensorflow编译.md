# TensorFlow C++ 编译

## 平台 & 环境依赖

- Ubuntu 16.04.7
- Intel(R) Xeon(R) Platinum 8180 CPU @ 2.50GHz
- RTX 3090
- cuda 11.2
- cudnn 8.1.1
- NVIDIA driver 460.39
- gcc & g++ 8.4.0
- TensorFlow 2.5.0
- protobuf 3.9.2
- bazel 3.7.2

[gcc&g++安装](gcc&g++安装.md)

[NVIDIA驱动&CUDA&CUDNN安装](NVIDIA驱动&CUDA&CUDNN安装.md)

## TensorFlow 编译

### 安装 bazel

下载 bazel

```bash
https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-installer-linux-x86_64.sh
```

添加执行权限

```bash
chmod +x bazel-3.7.2-installer-linux-x86_64.sh
```

安装 bazel 到指定目录，`prefix` 按需更改

```
./bazel-3.7.2-installer-linux-x86_64.sh --prefix=/home/14T/sources/bazel
```

### 安装 protobuf

下载 protobuf

```bash
wget https://github.com/google/protobuf/releases/download/v3.9.2/protoc-3.9.2-linux-x86_64.zip
```

解压 protobuf 到指定目录

```bash
unzip protoc-3.9.2-linux-x86_64.zip -d google-protobuf
```

复制 protoc 到 `/usr/local/bin`目录

```bash
sudo cp google-protobuf/bin/protoc /usr/loca/bin
```

### 编译 TensorFlow

创建一个 conda 环境，并安装 numpy

```bash
conda create -n ctf python=3.8 numpy
```

```bash
conda activate ctf
```

下载 `tensorflow`

```bash
git clone https://github.com/tensorflow/tensorflow.git
```

进入源码目录

```bash
cd tensorflow
```

切换到 v2.5.0

```
git checkout v2.5.0
```

配置编译选项

```bash
./configure
```

```bash
You have bazel 3.7.2 installed.
Please specify the location of python. [Default is /home/14T/anaconda3/envs/ctf/bin/python3]: 


Found possible Python library paths:
  /home/14T/anaconda3/envs/ctf/lib/python3.8/site-packages
Please input the desired Python library path to use.  Default is [/home/14T/anaconda3/envs/ctf/lib/python3.8/site-packages]

Do you wish to build TensorFlow with ROCm support? [y/N]: n
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with TensorRT support? [y/N]: n
No TensorRT support will be enabled for TensorFlow.

Found CUDA 11.2 in:
    /usr/local/cuda-11.2/targets/x86_64-linux/lib
    /usr/local/cuda-11.2/targets/x86_64-linux/include
Found cuDNN 8 in:
    /usr/local/cuda-11.2/targets/x86_64-linux/lib
    /usr/local/cuda-11.2/targets/x86_64-linux/include


Please specify a list of comma-separated CUDA compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus. Each capability can be specified as "x.y" or "compute_xy" to include both virtual and binary GPU code, or as "sm_xy" to only include the binary code.
Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 3.5,7.0]: 8.6


Do you want to use clang as CUDA compiler? [y/N]: n
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 


Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -Wno-sign-compare]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
	--config=mkl         	# Build with MKL support.
	--config=mkl_aarch64 	# Build with oneDNN and Compute Library for the Arm Architecture (ACL).
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=numa        	# Build with NUMA support.
	--config=dynamic_kernels	# (Experimental) Build kernels into separate shared objects.
	--config=v2          	# Build TensorFlow 2.x instead of 1.x.
Preconfigured Bazel build configs to DISABLE default on features:
	--config=noaws       	# Disable AWS S3 filesystem support.
	--config=nogcp       	# Disable GCP support.
	--config=nohdfs      	# Disable HDFS support.
	--config=nonccl      	# Disable NVIDIA NCCL support.
Configuration finished
```

bazel 编译，`output_user_root` 按需更改

```bash
bazel --output_user_root=/home/14T/bazel_build  build --config=opt --config=monolithic --config=cuda //tensorflow:libtensorflow_cc.so //tensorflow:install_headers
```

编译完成后将生成的头文件所在目录 `bazel-bin/tensorflow/include` 的绝对路径添加到项目的 `CMakeList.txt` 的 `include_directories`

将生成的 `libtensorflow_cc.so` 所在目录 `bazel-bin/tensorflow` 的绝对路径添加到项目的 `CMakeList.txt` 的 `link_directories`

最后将 `tensorflow_cc` 添加到 `CMakeLists.txt` 的 `link_libraries` 编译项目即可
