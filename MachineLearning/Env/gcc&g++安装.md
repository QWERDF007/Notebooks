## gcc & g++ 8.4.0 安装

添加 apt 源并更新：

```bash
sudo apt-get install -y software-properties-common
```

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
```

```bash
sudo apt update
```

安装 gcc

```bash
sudo apt install gcc-8
```

安装 g++

```bash
sudo apt install g++-8
```

生成 gcc & g++ 软连接

```
sudo ln -s /usr/bin/gcc-8 /usr/bin/gcc
```

```bash
sudo ln -s /usr/bin/g++-8 /usr/bin/g++
```

