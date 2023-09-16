# [Installing the library](https://pybind11.readthedocs.io/en/stable/installing.html)

获取 `pybind11` 源代码的方法有几种，它位于 GitHub 上的 `pybind/pybind11`。pybind11 开发人员建议以这里列出的前三种方法中的一种获取 pybind11，即子模块、PyPI 或 conda-forge。

## Include as a submodule

当您在 Git 中处理项目时，可以将 pybind11 存储库用作子模块。从您的 git 仓库，使用:

    git submodule add -b stable ../../pybind/pybind11 extern/pybind11
    git submodule update --init

这假设您正在将依赖项放入 `extern/`，并且您正在使用 GitHub；如果您不使用 GitHub，请使用完整的 https 或 ssh URL 而不是上述相对 URL `../../pybind/pybind11`。某些其他服务器也需要 `.git` 扩展名 (GitHub 不需要)。

从这里，您现在可以包含 `extern/pybind11/include`，或者可以直接从本地文件夹使用 pybind11 提供的各种集成工具(请参阅 [构建系统]())。

Include with PyPI
------------------------

您可以使用 Pip 从 PyPI 下载源代码和 CMake 文件作为 Python 包。只需使用:

    pip install pybind11

这将以标准的 Python 包格式提供 pybind11。如果您想要 pybind11 直接可用于您的环境根目录中，可以使用:

    pip install "pybind11[global]"

如果您正在使用系统 Python 进行安装，这不推荐，因为它会向 `/usr/local/include/pybind11` 和 `/usr/local/share/cmake/pybind11` 添加文件，因此，除非这就是您想要的，否则建议仅将其用于虚拟环境或您的 `pyproject.toml` 文件(请参阅 [构建系统]())。

## Include with conda-forge

您可以通过 [conda-forge](https://github.com/conda-forge/pybind11-feedstock) 的 conda 使用 pybind11:

    conda install -c conda-forge pybind11

## Include with vcpkg

您可以使用 Microsoft [vcpkg](https://github.com/Microsoft/vcpkg/) 依赖关系管理器下载和安装 pybind11:

    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh 
    ./vcpkg integrate install
    vcpkg install pybind11

vcpkg 中的 pybind11 端口由 Microsoft 团队成员和社区贡献者保持最新。如果版本过期，请在 vcpkg 存储库上创建问题或拉取请求。

## Global install with brew

brew包管理器 (macOS 上的 Homebrew，或 Linux 上的 Linuxbrew)有一个 [pybind11 包](https://github.com/Homebrew/homebrew-core/blob/master/Formula/pybind11.rb)。要安装:

    brew install pybind11

## Other options

[这里](https://repology.org/project/python:pybind11/versions) 列出了可以找到 pybind11 的其他位置；这些由各种打包程序和社区维护。

