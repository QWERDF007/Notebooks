# wsl

win11 自带 wsl，其他安装参考网上资料

## 基本命令

<details><summary><em>[点击展开]</em></summary>
<br>

### 更新 WSL

```powershell
wsl --update
```

### 检查 WSL 状态

```powershell
wsl --status
```

查看有关 WSL 配置的常规信息，例如默认发行版类型、默认发行版和内核版本。

### 以特定用户的身份运行

```powershell
wsl --user USER
```

若要以指定用户身份运行 WSL，请将 `USER` 替换为 WSL 发行版中存在的用户名。

### 关闭

```powershell
wsl --shutdown
```

立即终止所有正在运行的发行版和 WSL 2 轻量级实用工具虚拟机。

### 终止

```powershell
wsl --terminate <Distribution Name>
```

终止指定的发行版或阻止其运行，请将 `<Distribution Name>` 替换为目标发行版的名称。

### 导出分发版

```powershell
wsl --export <Distribution Name> <FileName>
```

将指定分发版的快照导出为新的分发文件。 默认为 tar 格式。

### 导入分发版

```powershell
wsl --import <Distribution Name> <InstallLocation> <FileName>
```

导入指定的 tar 文件作为新的分发版。

</details>


## wsl 安装 Ubuntu


查看发行版本

```powershell
wsl.exe --list --online
```


<details><summary><em>输出</em></summary>
<br>

```
C:\Users\pc>wsl --list --online
以下是可安装的有效分发的列表。
使用 'wsl.exe --install <Distro>' 安装。

NAME                                   FRIENDLY NAME
Ubuntu                                 Ubuntu
Debian                                 Debian GNU/Linux
kali-linux                             Kali Linux Rolling
Ubuntu-18.04                           Ubuntu 18.04 LTS
Ubuntu-20.04                           Ubuntu 20.04 LTS
Ubuntu-22.04                           Ubuntu 22.04 LTS
OracleLinux_7_9                        Oracle Linux 7.9
OracleLinux_8_7                        Oracle Linux 8.7
OracleLinux_9_1                        Oracle Linux 9.1
openSUSE-Leap-15.5                     openSUSE Leap 15.5
SUSE-Linux-Enterprise-Server-15-SP4    SUSE Linux Enterprise Server 15 SP4
SUSE-Linux-Enterprise-15-SP5           SUSE Linux Enterprise 15 SP5
openSUSE-Tumbleweed                    openSUSE Tumbleweed
```

</details>

安装：

```bash
wsl --install --web-download -d Ubuntu-20.04
```

<details><summary><em>参数</em></summary>
<br>

- --web-download：从 Internet 而不是 Microsoft Store 下载分发版。
- --distribution, -d <Distro>：指定分发版。

</details>

安装完后会让输入新的用户名和密码，这里使用 `pc` 和 `123456`

### 迁移至非系统盘

默认 WSL 装的 Linux 子系统在 C 盘

1. 关闭子系统

   ```powershell
   wsl --shutdown
   ```

2. 导出Ubuntu

   ```powershell
   wsl --export Ubuntu-20.04 F:\Ubuntu\ubuntu.tar
   ```

3. 注销子系统

   ```powershell
   wsl --unregister Ubuntu-20.04
   ```

4. 导入子系统并安装到其他位置

   ```powershell
   wsl --import Ubuntu-20.04 F:\Ubuntu\ F:\Ubuntu\ubuntu.tar --version 2
   ```



# docker

wsl 启动 Ubuntu

```
wsl -d Ubuntu-20.04 --user pc
```

## 安装

### 更改镜像源

这一步主要是为了加速 Linux各种包的下载速度。


打开清华镜像源，注意Ubuntu版本选择20.04: https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/

```bash
sudo vim /etc/apt/sources.list
```

<details><summary><em>替换内容</em></summary>
<br>

```
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse

deb http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse
# deb-src http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
# # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
```

</details>

### 更新镜像

```bash
sudo apt-get update
```

```bash
sudo apt-get upgrade
```

### 清理 & 安装依赖

如果安装过旧版本，需要先卸载之前的旧版本：

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```

安装部分软件依赖，默认应该安装过了：

```bash
sudo apt-get install ca-certificates curl gnupg lsb-release
```

添加Docker的官方GPG密钥，好像是为了能够防止软件被篡改。

```bash
sudo mkdir -p /etc/apt/keyrings
```

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

添加Docker软件源，使用国内源加速：

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

安装 docker engine

```bash
sudo apt-get update
```

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

### 安装 docker

```bash
sudo apt-get update
```

```bash
sudo apt-get install ./docker-desktop-<version>-<arch>.deb
```

## 验证

```bash
sudo docker run hello-world
```

<details><summary><em>输出</em></summary>
<br>

```bash
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
c1ec31eb5944: Pull complete
Digest: sha256:6352af1ab4ba4b138648f8ee88e63331aae519946d3b67dae50c313c6fc8200f
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

</details>



## wsl - docker-engine 自启动 (实现 systemctl)

## [开机自启动/关闭](https://docs.docker.com/desktop/install/ubuntu/#launch-docker-desktop)

### 启动

```bash
systemctl --user start docker-desktop
```

### 停止

```bash
systemctl --user stop docker-desktop
```

### 自启动

```bash
systemctl --user enable docker-desktop
```

<details><summary><em>输出</em></summary>
<br>

```
Created symlink /home/pc/.config/systemd/user/docker-desktop.service → /usr/lib/systemd/user/docker-desktop.service.
Created symlink /home/pc/.config/systemd/user/graphical-session.target.wants/docker-desktop.service → /usr/lib/systemd/user/docker-desktop.service.
```

</details>

### 关闭自启动

<details><summary><em>输出</em></summary>
<br>

```
Removed /home/pc/.config/systemd/user/graphical-session.target.wants/docker-desktop.service.
Removed /home/pc/.config/systemd/user/docker-desktop.service.
```

</details>

## 卸载

```bash
sudo apt-get purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras
```




# FAQ

- 下列软件包有未满足的依赖关系： docker-desktop : 依赖: docker-ce-cli 但无法安装它

    安装Docker Desktop先得安装Docker引擎，参考文档：https://docs.docker.com/engine/install/ubuntu/#set-up-the-repository

- 无法从“[raw.githubusercontent.com....](https://link.zhihu.com/?target=https%3A//raw.githubusercontent.com/microsoft/WSL/master/distributions/DistributionInfo.json)”中提取列表分发。无法解析服务器的名称或地址

  - 开代理
  - 设置 DNS 为 `8.8.8.8`

- permission denied while trying to connect to the Docker daemon socket at unix

  https://stackoverflow.com/a/48957722

  ```
  sudo groupadd docker
  sudo usermod -aG docker pc
  newgrp docker
  docker run hello-world
  ```

  



# 参考

- [在Ubuntu 22.04(LTS)上安装Docker](https://www.bilibili.com/read/cv17488009/)
- [2023最新WSL搭建深度学习平台教程（适用于Docker-gpu、tensorflow-gpu、pytorch-gpu)](https://zhuanlan.zhihu.com/p/621142457)
- [玩转 Windows 自带的 Linux 子系统 WSL（图文指南）](https://blog.csdn.net/u011262253/article/details/108759785)