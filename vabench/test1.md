这两个模型分别来自 **字节的 LatentSync 仓库**和 **InsightFace 官方发布页**。在容器中按下面的方式下载即可。

**1. `sfd_face.pth`**

文件位于：[ByteDance/LatentSync-1.5 的 auxiliary 目录](https://huggingface.co/ByteDance/LatentSync-1.5/tree/main/auxiliary)。

```bash
# 替换成你实际的 VABench 权重缓存目录
export VABENCH_CACHE_DIR=/data/vabench-cache

mkdir -p "$VABENCH_CACHE_DIR/checkpoints/auxiliary"

curl -fL --retry 3 \
  https://huggingface.co/ByteDance/LatentSync-1.5/resolve/main/auxiliary/sfd_face.pth \
  -o "$VABENCH_CACHE_DIR/checkpoints/auxiliary/sfd_face.pth"
```

**2. InsightFace 的 `buffalo_l`**

它是包含多个 ONNX 文件的模型包，从 [InsightFace 官方 v0.7 发布页](https://github.com/deepinsight/insightface/releases/tag/v0.7)下载 `buffalo_l.zip`，约 **289 MB**。

```bash
curl -fL --retry 3 \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
  -o /tmp/buffalo_l.zip

mkdir -p "$VABENCH_CACHE_DIR/checkpoints/auxiliary/models/buffalo_l"

python -m zipfile -e \
  /tmp/buffalo_l.zip \
  "$VABENCH_CACHE_DIR/checkpoints/auxiliary/models/buffalo_l"
```

**3. 检查最终位置**

```text
/data/vabench-cache/checkpoints/auxiliary/
├── sfd_face.pth
└── models/
    └── buffalo_l/
        ├── 1k3d68.onnx
        ├── 2d106det.onnx
        ├── det_10g.onnx
        ├── genderage.onnx
        └── w600k_r50.onnx
```

执行：

```bash
ls -lh "$VABENCH_CACHE_DIR/checkpoints/auxiliary/sfd_face.pth"
ls -lh "$VABENCH_CACHE_DIR/checkpoints/auxiliary/models/buffalo_l/"
```

**这些文件放在权重缓存目录中，不放到 VABench 源码的 `third_party` 中。**运行评测前，在同一个终端设置好 `VABENCH_CACHE_DIR`，官方代码就会按该路径查找。
