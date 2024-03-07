## NVIDIA 相关依赖安装

下载 NVIDIA 驱动

```
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/460.39/NVIDIA-Linux-x86_64-460.39.run
```

禁止 nouveau。创建文件 `/etc/modprobe.d/blacklist-nouveau.conf`，并添加下面的内容

```
blacklist nouveau
options nouveau modeset=0
```

重新生成内核 initramfs

```bash
sudo update-initramfs -u
```

安装 NVIDIA 驱动

```bash
sudo sh NVIDIA-Linux-x86_64-460.39.run
```

下载 CUDA 11.2

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
```

安装 CUDA

```bash
sudo sh cuda_11.2.0_460.27.04_linux.run
```

下载 cudnn 8.1.1，需要登陆 NVIDIA 账号

```

```

解压 cudnn 并复制到 cuda 安装目录

```bash
tar -zxvf cudnn-11.2-linux-x64-v8.1.1.33.tgz
```

```bash
sudo cp -r cuda/include/* /usr/local/cuda/include
```

```
sudo cp -r cuda/lib64/* /usr/local/cuda/lib64
```





<!-- 完成标志, 看不到, 请忽略! -->
