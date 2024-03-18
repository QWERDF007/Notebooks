# 本地部署Ollama

## 本地 docker 部署

Docker 安装设置参考[安装wsl和docker](<./安装wsl和docker.md>)。安装完成后，Windows 使用 wsl 启动 Ubuntu 子系统，然后根据文档进行操作。Ubuntu 直接根据文档进行操作。

建议按照文档[安装wsl和docker-设置国内镜像源](<./安装wsl和docker.md#设置国内镜像源>)设置 docker 的镜像源加速下载部分镜像。

Ollama docker 官方镜像：https://hub.docker.com/r/ollama/ollama

### CPU only

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### Nvidia GPU

#### Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation).

1. 配置仓库

   ```
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   ```

   ```
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ```

   ```
   sudo apt-get update
   ```

2. 安装 NVIDIA Container Toolkit packages

   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   ```

#### 设置 docker  使用 Nvidia 驱动

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

```
sudo systemctl restart docker
```

### 启动容器 ollama

```bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama --restart always ollama/ollama
```

<details><summary><em>docker run 参数</em></summary>
<br>

- [`run`](https://docs.docker.com/reference/cli/docker/container/run/)：从一个镜像创建和运行一个新的容器
- [`-d, --detach`](https://docs.docker.com/reference/cli/docker/container/run/#detach)：后台运行容器并打印容器 ID
- [`--gpus`](https://docs.docker.com/reference/cli/docker/container/run/#gpus)：添加到容器的 GPU 设备 (`all` 传递所有 GPUs)
- [`-v, --volume`](https://docs.docker.com/reference/cli/docker/container/run/#volume)：绑定挂载卷。(`ollama:/root/.ollama`：创建一个名为 `ollama` 的卷，挂载到容器的 `/root/.ollama` 目录)
- [`-p, --expose`]((https://docs.docker.com/reference/cli/docker/container/run/#publish))：将主机上的端口映射到容器中的端口
- [`--name`](https://docs.docker.com/reference/cli/docker/container/run/#name)：为容器分配名称

- [`--restart`](https://docs.docker.com/reference/cli/docker/container/run/#restart)：容器退出时的重启策略 

  - `on-failure[:max-retries]`： 仅在容器退出状态非零（表示错误）时重启容器。还可以使用 `:max-retries` 可选参数来限制 Docker 守护进程尝试重启的次数。
  - `always` 始终重启容器，无论退出状态如何；
  - `unless-stopped` 除非容器被显式停止或 Docker 守护进程本身停止或重新启动，否则会重启容器。


</details>

### 启动容器  open-webui

```bash
docker run -d -p 11435:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/root/.openwebui --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

<details><summary><em>docker run 参数</em></summary>
<br>

- [`--add-host`](https://docs.docker.com/reference/cli/docker/container/run/#add-host)：Add a custom host-to-IP mapping (host:ip)

  - `host.docker.internal:host-gateway`：自定义的主机名 `host.docker.internal` 关联到宿主机的 IP 地址。这样，在容器内部，你可以使用这个主机名来访问宿主机的资源


</details>



启动完成大概需要 3min 左右可以通过访问 https://127.0.0.1:11435 使用 open-webui。

<!-- 完成标志, 看不到, 请忽略! -->
