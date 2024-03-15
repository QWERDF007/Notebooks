# wsl

win11 自带 wsl，其他安装参考网上资料

## 基本命令

<details open><summary><em>[点击展开]</em></summary>
<br>

### 更新 WSL

```powershell
wsl --update
```

### 检查 WSL 状态

```powershell
wsl --status
```

</details>




## wsl 安装 Ubuntu

查看发行版本

```powershell
wsl.exe --list --online
```

输出

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

安装：

```bash
wsl --install --web-download -d Ubuntu-20.04
```

- --web-download：从 Internet 而不是 Microsoft Store 下载分发版。
- --distribution, -d <Distro>：指定分发版。

安装完后会让输入新的用户名和密码

## 以用户 `pc` 启动 Ubuntu



# docker

## 安装

如果安装过旧版本，需要先卸载之前的旧版本：

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```

安装部分软件依赖：

```bash
sudo apt-get install ca-certificates curl gnupg lsb-release
```



# FAQ

- 下列软件包有未满足的依赖关系： docker-desktop : 依赖: docker-ce-cli 但无法安装它

    安装Docker Desktop先得安装Docker引擎，参考文档：https://docs.docker.com/engine/install/ubuntu/#set-up-the-repository

- 无法从“[raw.githubusercontent.com....](https://link.zhihu.com/?target=https%3A//raw.githubusercontent.com/microsoft/WSL/master/distributions/DistributionInfo.json)”中提取列表分发。无法解析服务器的名称或地址

  - 开代理
  - 设置 DNS 为 `8.8.8.8`

- 





# 参考

- [在Ubuntu 22.04(LTS)上安装Docker](https://www.bilibili.com/read/cv17488009/)
- [2023最新WSL搭建深度学习平台教程（适用于Docker-gpu、tensorflow-gpu、pytorch-gpu)](https://zhuanlan.zhihu.com/p/621142457)
- [玩转 Windows 自带的 Linux 子系统 WSL（图文指南）](https://blog.csdn.net/u011262253/article/details/108759785)