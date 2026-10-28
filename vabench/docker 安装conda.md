你当前是 **Ubuntu 容器、root 用户**，可以通过安装 **Miniforge** 获得 Conda。它支持创建 Python 3.11 环境，不需要卸载现有 Python 3.12。[官方安装说明](https://github.com/conda-forge/miniforge#install)

**1. 安装下载工具**

在容器内执行：

```bash
apt-get update
apt-get install -y curl ca-certificates bzip2
```

**2. 下载并安装 Conda**

```bash
curl -fL --retry 3 \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" \
  -o /tmp/miniforge.sh

bash /tmp/miniforge.sh -b -p /opt/conda
```

安装目录为 `/opt/conda`。如果这个目录已安装过 Conda，直接使用已有安装，跳过这一步。

**3. 激活 Conda，创建 Python 3.11 环境**

```bash
source /opt/conda/etc/profile.d/conda.sh

conda --version

conda create -n vabench311 python=3.11 pip -y
conda activate vabench311
```

**4. 确认当前 Python**

```bash
python3 --version
which python3
python3 -m pip --version
```

预期版本是 `Python 3.11.x`，路径是：

```text
/opt/conda/envs/vabench311/bin/python3
```

之后就在这个环境中安装 VABench 依赖。

以后重新进入容器，执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate vabench311
```

注意：环境会随当前容器保留；如果删除容器后重新创建，需要通过 Dockerfile 或保存的镜像恢复安装。
