# docker 设置跟目录

1. 停止 docker 服务

   ```bash
   sudo systemctl stop docker
   ```

   ```bash
   sudo systemctl stop docker.socket
   ```

   ```bash
   sudo systemctl stop containerd
   ```

2. （可选）创建需要设置的目录 `/data/docker`

   ```bash
   sudo mkdir -p /data/docker
   ```

3. 移动 docker 根目录 `/var/lib/docker` 到新的目录 `/data/docker`

   ```bash
   sudo mv /var/lib/docker /data/docker
   ```

4. 编辑 `sudo vim /etc/docker/daemon.json`，添加下述内容

   ```bash
   {
     "data-root": "/data/docker"
   }
   ```

5. 重启 docker 服务

   ```bash
   sudo systemctl start docker
   ```

6. 验证新的 docker 根目录

   ```bash
   docker info -f '{{ .DockerRootDir}}'
   ```

## 参考

- [1]: https://www.ibm.com/docs/en/z-logdata-analytics/5.1.0?topic=software-relocating-docker-root-directory	"Relocating the Docker root directory"

  

<!-- 完成标志, 看不到, 请忽略! -->
