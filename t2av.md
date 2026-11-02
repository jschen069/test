# VABench + AISBench（`vabench311_new`）精简步骤

> 只保留依赖下载和执行命令。环境已经存在时，从第 5 步开始。

## 1. 创建 Conda 环境

```bash
/root/miniconda3/bin/conda create -n vabench311_new python=3.11 pip -y
```

## 2. 安装系统依赖

```bash
apt-get update
apt-get install -y git ffmpeg libsndfile1 build-essential ca-certificates
```

## 3. 下载并安装 VABench 依赖

```bash
cd /workspace/code/benchmark

/root/miniconda3/envs/vabench311_new/bin/python3 \
  -m ais_bench.benchmark.vabench.environment install \
  --source-root /workspace/code/VABench-main \
  --cache-dir /workspace/vabench-cache \
  --work-dir /workspace/vabench-env-audit \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple
```

## 4. 下载并安装 AISBench 依赖

```bash
cd /workspace/code/benchmark

sed -e 's/^gradio-client$/gradio-client==1.14.0/' \
    -e 's/^openai$/openai==1.109.1/' \
    requirements/runtime.txt \
  | grep -viE '^\s*Pillow\s*==' \
  > /workspace/aisbench_runtime_unified.txt

printf 'anyio==4.10.0\n' >> /workspace/aisbench_runtime_unified.txt

/root/miniconda3/envs/vabench311_new/bin/python3 -m pip install \
  -r /workspace/aisbench_runtime_unified.txt \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple

/root/miniconda3/envs/vabench311_new/bin/python3 -m pip install \
  -e /workspace/code/benchmark \
  --no-deps \
  --no-build-isolation

/root/miniconda3/envs/vabench311_new/bin/python3 -m pip install \
  "fsspec==2025.9.0" "tqdm==4.66.2" \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple
```

## 5. 准备评测数据

已有 `/workspace/prepared_all_official_v2/manifest.json` 时跳过本步。评测配置 `eval_vabench_custom_all.py` 的 `PREPARED_DIR` 当前指向该目录，两边必须一致。

```bash
cd /workspace/code/benchmark

/root/miniconda3/envs/vabench311_new/bin/python3 \
  -m ais_bench.benchmark.vabench.cli prepare \
  --source-root /workspace/code/VABench-main \
  --subset custom \
  --videos /workspace/video \
  --output /workspace/prepared_all_official_v2
```

## 6. 执行评测

`--work-dir` 必须是新目录；已有 `report.json` 时会拒绝运行。

```bash
cd /workspace

WORKDIR=/workspace/vabench_runs/aisbench/custom_all_7_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=6,7 \
/root/miniconda3/envs/vabench311_new/bin/ais_bench \
  /workspace/vabench_runs/aisbench/eval_vabench_custom_all.py \
  -m eval \
  --work-dir "$WORKDIR"

echo "$WORKDIR"
```

## 7. 查看结果

使用第 6 步最后打印的 `WORKDIR`；如果换了 shell，取最新目录：

```bash
WORKDIR=$(ls -dt /workspace/vabench_runs/aisbench/custom_all_7_* | head -1)
REPORT=$(find "$WORKDIR" -path '*/vabench_t2av_artifacts/report.json' | sort | tail -1)

/root/miniconda3/envs/vabench311_new/bin/python3 - "$REPORT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
print(json.dumps(report['total'], ensure_ascii=False, indent=2))
PY
```

官方格式备份位于：

```text
<artifacts>/official_eval_results/<category>/
  <category>_<时间>_eval_results.json
  <category>_<时间>_eval_results.csv
  <category>_<时间>_full_info.json
```

```bash
find "$WORKDIR" -path '*/official_eval_results/*/*_eval_results.csv'
```

`second_clap` 对超过 10 秒的音频使用官方随机 fusion 截断，重复运行会有波动；短于 10 秒的样本（如小猫）稳定。不要按逐样本完全相等验收，必要时多次运行取均值。














# VABench GPU评分环境：Conda Python 3.11

本指南依据用户在H200容器中的最终统一环境记录（`过程/10.txt`），使用外部固定版本VABench，不修改其评分源码。15个指标共用一个评分Conda。AISBench主环境通过`BACKEND_PYTHON`或`--python`调用该Conda，无须把GPU评分依赖加入所有数据集的公共runtime。

> 如果需要把AISBench和VABench装进同一个Conda环境，使用已跑通的[vabench311_new统一环境指南](vabench311_new_env_setup.md)和[精简步骤](vabench311_new_env_setup_steps.md)；本文保留独立评分环境方案及其依赖审计细节。

这里交付依赖声明、安装与环境检查；本地Windows上的测试不代表H200真实评分或官方<1%对齐已通过。记录里的4秒小猫仍可能导致desync返回异常兜底1.0，LSA无脸仍会省略；依赖安装不能解决这两项输入限制。只有完成逐指标结果检查，才能认定有15项有效成绩。

## 1. 依赖文件与安装边界

与`requirements/datasets/geometry3k.txt`一样，将数据集专用包放在可选依赖目录，不加入`setup.py`的通用安装依赖。

| 文件 | 用途 |
| --- | --- |
| `requirements/datasets/vabench.txt` | GPU评分PyPI固定清单，来源为官方requirement及10.txt的兼容性调整；包含Torch和NVIDIA运行库 |
| `requirements/datasets/vabench-build.txt` | 固定pip25.2、setuptools80.9.0、wheel0.45.1及NumPy/Cython，先安装以构建InsightFace/Git包 |
| `requirements/datasets/vabench-git.txt` | 固定ImageBind、PyTorchVideo、CLIP的完整提交，只在普通依赖之后以`--no-deps`安装 |
| `requirements/vabench.txt` | 旧路径兼容入口，转向新的PyPI清单；不是完整安装入口 |

不能只执行一次`pip install -r vabench.txt`就宣称15项依赖已完整：NISQA需要本地元数据补丁，Git包必须单独安装，外部DNSMOS/Synchformer代码和模型缓存也不属于PyPI依赖。

具体调整：

- Torch2.6.0／torchvision0.21.0／torchaudio2.6.0，保留官方CUDA12.4运行库固定版本。不把驱动显示的CUDA13.2当成必须安装的PyTorch CUDA版本。
- 只保留`opencv-python-headless==4.9.0.80`，移除另一套`opencv-python`，与NumPy1.26.4配套。
- NISQA2.0.post2仅把Torch、torchvision、SciPy、scikit-learn、Matplotlib的五个`==`改为原版本的`>=`，其余依赖及所有包代码不变。RECORD重新生成，原始wheel SHA256必须匹配PyPI官方文件。
- 保留Pillow10.2.0等其他NISQA约束；AISBench主runtime固定Pillow11.2.1，不能默认将两份完整依赖清单装在同一个环境。安装工具检测到已安装AISBench时拒绝改写它。
- ImageBind固定`53680b02d7e37b19b124fa37bae4b6c98c38f5be`，PyTorchVideo固定用户最终验证的`2ac6ed361309cb25144636ad82dd8bd6a4970e0c`。后者不能只凭显示版本0.1.5判断已安装正确，检查工具核对`direct_url.json`中的实际提交。
- 用户流程未固定CLIP，本交付将其固定为核查时的官方提交`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`，须在H200回归中一起记录。setuptools固定80.9.0保留`pkg_resources`，而不是以后随`<82`漂移。
- `certifi==2025.10.5`及构建工具固定值是此次新增的可复现性约束，不声称这些具体版本已出现在用户服务器freeze中。安装后保存实际`installed.freeze.txt`用于校准。

## 2. 系统工具与Conda

Linux x86_64、NVIDIA GPU容器，已有驱动可见性与完整模型缓存。下面路径按用户容器示例填写，已有环境可直接复用。

“完整FFmpeg”指提供`ffmpeg`、`ffprobe`并具备所需音视频处理能力的发行包；当前解码预检还使用lavfi生成小视频。编译工具主要为gcc、g++、make及C/C++开发头文件，用于构建InsightFace等源码扩展。无需用户自行编译FFmpeg。

Ubuntu/Debian容器可手动执行以下apt步骤，或在第4节安装命令中增加`--install-system-deps`由工具自动下载并安装这些系统包。自动模式要求容器内root，仅使用现有apt源，不修改源配置，也不下载编译FFmpeg源码。

```bash
apt-get update
apt-get install -y git ffmpeg libsndfile1 build-essential ca-certificates
nvidia-smi
ffmpeg -version
ffprobe -version

source /root/miniconda3/etc/profile.d/conda.sh
conda create -n vabench311 python=3.11 pip -y
conda activate vabench311
python3 --version
python3 -m pip --version
```

若已有`vabench311`，跳过创建步骤。不要卸载系统Python或在评分Conda中执行`pip install -e .`安装完整AISBench；AISBench在其主环境中安装即可。评分工具从源码目录运行，模块入口本身不要求mmengine等主环境依赖。

仅当机器使用自有CA时，沿用现有证书配置；不要把这组示例路径强加给其他服务器：

```bash
export PIP_CERT=/root/ca-bundle.crt
export SSL_CERT_FILE=/root/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/root/ca-bundle.crt
export GIT_SSL_CAINFO=/root/ca-bundle.crt
```

## 3. 源码与权重

已有固定版本源码和缓存时，设置对应实际路径，不重复克隆：

```bash
export VABENCH_ROOT=/workspace/code/VABench-main
export VABENCH_CACHE_DIR=/workspace/vabench-cache
export CUDA_VISIBLE_DEVICES=6,7
```

若重新获取源码：

```bash
git clone https://github.com/tanABCC/VABench.git /workspace/code/VABench-main
git -C /workspace/code/VABench-main checkout 12b9617a900a9eca1691da6b7115ce3d58f53a16
```

模型缓存需事先下载或复制，pip不会下载评分模型权重。官方资源为[固定版本模型缓存](https://huggingface.co/datasets/tanABBCC/VABENCH_CACHE_DIR/tree/aa0c3da973d86f2994e9c8860f28785ce0cc6152)。另外补齐`checkpoints/auxiliary/sfd_face.pth`及InsightFace的`models/buffalo_l`目录；来源见[VABench评测指南](vabench_t2av.md)。

安装工具仅在目标缺失时下载固定微软DNSMOS脚本、链接缓存附带的Synchformer源码，并在发现备用buffalo_l目录时补软链。既有文件不覆盖；它们的哈希及后续评分来源指纹仍须与官方基线核对。Synchformer缓存源码不存在时会在预检中报缺失，不擅自换成不匹配权重的最新版。

**先设置缓存环境变量，再导入官方模块**；部分路径在导入时即确定。

## 4. 预览与完整安装

在AISBench源码根目录，例如`/workspace/code/benchmark`，使用评分Conda的Python执行。`plan`仅打印参数列表，不联网、不安装、不生成输出目录：

```bash
cd /workspace/code/benchmark
python3 -m ais_bench.benchmark.vabench.environment plan \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --work-dir /workspace/vabench-env-audit
```

完整安装（普通依赖解析不绕过，只有固定Git包与已修补NISQA采用`--no-deps`）：

```bash
python3 -m ais_bench.benchmark.vabench.environment install \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --work-dir /workspace/vabench-env-audit \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple
```

省略`--index-url`即可沿用pip配置或默认源。PyPI镜像不会代理GitHub克隆。若失败，查看终端及`install.json`；脚本失败返回非零，不把中断后的缓存文件当成已安装成功。可重新运行，固定Git包与补丁NISQA会重装，主依赖继续由pip按清单处理。

若希望同一条命令连系统工具也自动准备，在Ubuntu/Debian容器内以root执行：

```bash
python3 -m ais_bench.benchmark.vabench.environment install \
  --install-system-deps \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --work-dir /workspace/vabench-env-audit \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple
```

它先执行`apt-get update`与系统包安装，再执行pip阶段。默认不带该参数时不运行apt；`plan --install-system-deps`只预览，不安装。`--index-url`仅用于pip，系统包仍从容器现有apt源下载。检查模式不能触发系统安装。

安装包含数GB的Torch/NVIDIA包，完整模型权重另计。若只需事先下载依赖，可在同一Linux/Python3.11目标环境中使用声明清单：

```bash
python3 -m pip download --no-build-isolation \
  -r requirements/datasets/vabench-build.txt \
  -r requirements/datasets/vabench.txt --dest /workspace/vabench-packages
python3 -m pip wheel --no-deps --no-build-isolation \
  -r requirements/datasets/vabench-git.txt --wheel-dir /workspace/vabench-packages
```

这两步应在构建工具已经安装后执行，下载结果中可能含源码包；将`/workspace/vabench-env-audit/original`和`patched`也保留。当前完整安装工具仍按在线固定提交执行，尚未交付一键离线安装；不把上述目录误称为已验证的离线镜像。

## 5. 环境检查与审计结果

安装自动执行检查，也可对用户已准备好的环境单独运行：

```bash
python3 -m ais_bench.benchmark.vabench.environment check \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --work-dir /workspace/vabench-env-audit \
  --video /workspace/video/kittencurledupmeowing.mp4
```

检查固定PyPI版本、Git提交、NISQA五项元数据补丁、15个官方入口导入、DNSMOS外部脚本、ImageBind/PyTorchVideo兼容导入、微型CUDA与torchvision NMS、decord实际解码、系统工具、关键资源及Qwen分片。检查阶段离线，不调用评分模型推理，不更改源码。未提供`--video`时，用FFmpeg生成极小的临时测试视频；完整FFmpeg须支持lavfi。

`pip check`中只有精确的decord0.6.0平台标签提示，且实际解码通过，才作为已记录例外接受。其他冲突或解码失败仍令检查失败，不一律忽略decord错误。

```text
/workspace/vabench-env-audit/
  install.json                 # 安装状态、清单哈希及外部脚本来源
  installed.freeze.txt         # 实际安装版本及Git引用
  original/nisqa-*.whl         # 原始PyPI wheel，不覆盖
  patched/nisqa-*.whl          # 仅改五项依赖元数据的本地wheel
  patched/nisqa_patch.json     # 原始/补丁哈希、修改字段、代码不变说明
  environment_check.json      # 每项检查、输出日志及总状态
```

`environment_check.json`中的`model_inference_verified=false`表示检查通过也不等于15项评分通过；此标记不会被导入成功或微型CUDA测试改成true。

## 6. AISBench调度该环境

回到已安装AISBench的主环境，编辑`ais_bench/configs/vabench_examples/eval_vabench.py`：

```python3
DEVICE = 'cuda:0'
BACKEND_PYTHON = '/root/miniconda3/envs/vabench311/bin/python3'
NUM_DEVICES = 2
```

还需填写`PREPARED_DIR`、`SOURCE_ROOT`和`CACHE_DIR`。可见物理卡6/7在子进程中是逻辑0/1；具体卡可见性也受LocalRunner调度影响，应保留实际运行日志。

独立CLI等价运行：

```bash
python3 -m ais_bench.benchmark.vabench.cli run \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --prepared /workspace/prepared_mini --device cuda:0 \
  --python /root/miniconda3/envs/vabench311/bin/python3 \
  --options ais_bench/configs/vabench_examples/options.json \
  --output /workspace/vabench_runs/gpu_mini
```

install仅安装依赖，不产生评分；需执行prepare和run。run已展示15项原始均分、固定归一化分和正式/预览/缺分总分，并记录LSA省略缺口；断点补跑和更细的场景归因仍待交付。详见[VABench评测指南](vabench_t2av.md)。H200验收需使用同一批输入与官方基线逐项核对原始结果及状态；均值相对误差<1%，官方均值绝对值≤1e-6时采用绝对差≤1e-6。`second_clap`对超过10秒的音频使用官方随机fusion截断，单次运行不能套用<1%阈值，应多次运行取均值并单独记录波动范围。

依据：[NISQA官方分发元数据](https://pypi.org/pypi/nisqa/2.0.post2/json)、[固定ImageBind依赖](https://github.com/facebookresearch/ImageBind/blob/53680b02d7e37b19b124fa37bae4b6c98c38f5be/requirements.txt)、[固定PyTorchVideo解码接口](https://github.com/facebookresearch/pytorchvideo/blob/2ac6ed361309cb25144636ad82dd8bd6a4970e0c/pytorchvideo/data/encoded_video.py)、[固定CLIP源码](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6)。



# VABench T2AV评测接入与GPU运行指南

本期先在H200/GPU接入外部VABench的15个指标，A3/NPU留后续。旧版接入骨架保留；当前代码已具备原始评分调用与落盘，已补固定归一化和总分展示；细化场景错误归因与补跑仍按需求规划继续实现，不宣称已完成GPU官方对齐。

依赖准备以[统一Conda Python3.11安装与预检指南](vabench_gpu_environment.md)为准；已实际跑通的统一环境记录和精简步骤见[vabench311_new统一环境](vabench311_new_env_setup.md)、[精简步骤](vabench311_new_env_setup_steps.md)。`requirements/vabench.txt`已转向GPU专用清单，不能继续使用早期A3安装步骤将其安装进NPU环境。数据集级声明见`requirements/datasets/vabench.txt`，完整安装须使用环境准备工具处理固定Git包与NISQA元数据补丁。

## 数据与输入

固定上游为`tanABCC/VABench@12b9617a900a9eca1691da6b7115ce3d58f53a16`。源码由`--source-root`指定，保留外部仓库，不将官方评分算法复制到AISBench。

T2AV JSON共848条，其中778条与mapping精确对应，采用这778条并提供固定Mini10；其余70条不自动补名。数据文件指纹固定在`ais_bench/benchmark/vabench/protocol.json`，Mini清单含新闻主播、说唱、小猫、音乐、环境、物理同步、复杂场景和虚拟世界等题目。

生成视频的输入文本为清单中的`entries[].metadata.prompt_en`，视频文件基名为`entries[].idx`。本阶段评测已有带音轨MP4，不发送H3或其他收费生成任务。文件名可以包含空格；shell参数须加引号。

```bash
cd /workspace/code/benchmark
export VABENCH_ROOT=/workspace/code/VABench-main
export VABENCH_CACHE_DIR=/workspace/vabench-cache

python3 -m ais_bench.benchmark.vabench.cli catalog \
  --source-root "$VABENCH_ROOT" --subset mini --output /workspace/mini_prompts.json
python3 -m ais_bench.benchmark.vabench.cli catalog \
  --source-root "$VABENCH_ROOT" --subset standard --output /workspace/t2av_778_prompts.json
```

一个输入根目录下，每个选中idx必须恰好对应一个带音轨MP4，可以平铺或分场景。不要混入同题多个模型的结果以免歧义。工具不为无声视频补静音、不重新编码原视频；抽取同源未裁剪、未主动重采样的PCM16 WAV，各官方指标再执行自己的预处理。

```bash
python3 -m ais_bench.benchmark.vabench.cli prepare \
  --source-root "$VABENCH_ROOT" --subset mini \
  --videos /workspace/generated --output /workspace/prepared_mini
```

完整集改为`--subset standard`并使用新目录。manifest记录原MP4绝对路径和生成WAV路径及哈希，原视频需保留；搬到另一服务器后重新prepare。已完成的准备目录不覆盖，输入被改写会在运行前报错。

目录下所有官方样本可准备为自定义子集。仍限于固定778题：官方指标需要题目对应的JSON提示词和辅助信息，目录中出现778之外的视频会报错并列出，不会静默跳过，也不会缩小分母。

```bash
python3 -m ais_bench.benchmark.vabench.cli prepare \
  --source-root "$VABENCH_ROOT" --subset custom \
  --videos /workspace/generated --output /workspace/prepared_custom
```

报告scope为`custom`，完整时总分标记为“自定义子集总分（非官方）”；不得与Mini10/778验收结果混用。

单条冒烟从778题中精确选题，不改变Mini10或完整集：

```bash
python3 -m ais_bench.benchmark.vabench.cli prepare \
  --source-root "$VABENCH_ROOT" --subset smoke \
  --sample-id 'kitten curled up meowing' \
  --videos /workspace/generated --output /workspace/prepared_kitten
```

约4秒的小猫适合验证媒体和CLAP链路；Synchformer可能因输入长度异常返回兜底1.0，LSA无脸没有有效结果。不能把程序运行到末尾或CSV里的-1认作全部15项有效成绩，也不自动拉伸视频或改变官方算法来补分。

## GPU评分

AISBench主环境可以与VABench评分Conda分离。填写后端Python，确保评分包全部来自该Conda。默认专家串行独立子进程，七个MLLM指标在一个子进程中复用Qwen2.5-Omni；不要求部署HTTP模型服务。

```bash
export CUDA_VISIBLE_DEVICES=6,7
python3 -m ais_bench.benchmark.vabench.cli run \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --prepared /workspace/prepared_mini --device cuda:0 \
  --python /root/miniconda3/envs/vabench311/bin/python3 \
  --options ais_bench/configs/vabench_examples/options.json \
  --output /workspace/vabench_runs/gpu_mini
```

仅做小猫CLAP冒烟时，将prepared换为单条准备目录，加`--dimensions second_clap`，并使用新输出目录。`requested_device`是请求值，实际设备证据见报告观测字段；DNSMOS默认显式使用CPU ONNX，其他指标按配置运行。

使用AISBench主命令时，编辑`ais_bench/configs/vabench_examples/eval_vabench.py`中的三个目录、`BACKEND_PYTHON`、`DEVICE`、`NUM_DEVICES`，再执行：

```bash
ais_bench ais_bench/configs/vabench_examples/eval_vabench.py -m eval
```

可见物理卡6/7在进程中对应逻辑0/1。LocalRunner也会设置设备可见性，应根据实际分配填写预留卡数。单条配置为`eval_vabench_smoke.py`。任务看板的Progress以准备集视频数为分母；VABench按指标组调用官方函数，进度按已完成指标组折算，最后的Qwen组包含7个MLLM指标。

## 参考路径与官方对齐

当前`--mode official`在独立评分进程中执行官方VABench类，注入相同本地资源并修正mapping路径；它是带工程包装的参考路径，不等同于用户原样执行官方`evaluate.py`。两种模式均保留原算法。

```bash
python3 -m ais_bench.benchmark.vabench.cli run \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --prepared /workspace/prepared_mini --device cuda:0 --mode official \
  --python /root/miniconda3/envs/vabench311/bin/python3 \
  --options ais_bench/configs/vabench_examples/options.json \
  --output /workspace/vabench_runs/gpu_reference
python3 -m ais_bench.benchmark.vabench.cli compare \
  --reference /workspace/vabench_runs/gpu_reference/report.json \
  --candidate /workspace/vabench_runs/gpu_mini/report.json \
  --output /workspace/vabench_runs/gpu_alignment.json
```

比较默认要求全部15项、相同样本/状态及输入、源码、权重和参数指纹。逐指标原始均分相对误差严格<1%；官方均分绝对值≤1e-6时要求绝对差≤1e-6。LSA允许相同省略集合但至少有一个有效分，其他指标的执行失败不能凭数值相近通过验收。`second_clap`是官方随机指标，见下节；单次运行不能套用<1%阈值，应多次运行取均值并单独记录范围。Mini用于快速对照，778用于完整集验收。

最终仍需与用户官方CLI基线核对：专家进程隔离可能影响decord全局bridge及Qwen解码后端，调用相同compute函数并不能代替真实对齐证据。`compare`仍读取AISBench report；新增的`official_eval_results`可直接与官方CLI的`*_eval_results.json/csv`逐条对照，但自动导入官方基线及来源溯源仍待后续实现。

### second_clap的随机性

官方`vabench/second_clap.py`调用processor时未指定`truncation`，`ClapFeatureExtractor`默认`fusion`。对超过10秒的音频，`_random_mel_fusion()`使用未播种的`np.random.choice`随机抽取3段音频特征，因此同一输入重复运行分数会波动；短于10秒的音频不触发随机截断，结果稳定。实测7个视频中6个约10.144秒，小猫约4.064秒，后者在官方和多次AISBench运行中均为0.30362。

因此：

- 不得要求`second_clap`逐样本完全一致，也不要用单次运行套用默认<1%均值阈值。
- 与官方对比时应多次运行取均值并记录波动范围，或为该项单独设置绝对容差。
- 该项只占总分1/15，单次波动对总分影响通常小于0.1分。
- 需要AISBench内部可复现时，可在CLAP调用前固定`np.random.seed()`；这只能保证自身重复性，不等同于官方某次随机抽样。


## 固定归一化和总分展示

运行`run`完成后，终端和AISBench主流程汇总表均展示15项原始均分、0–100展示分及覆盖率；JSON和CSV同时落盘。每条成功视频等权，先计算原始均分再归一化，不对各场景均分等权平均，也不先裁剪逐样本展示分再平均。

| 指标 | 固定展示公式（m为原始均分） |
| --- | --- |
| DNSMOS、NISQA、5项third_* | (m−1)/4×100 |
| AudioBox | (m+1.75)/9×100 |
| ViCLIP、CLAP、ImageBind | (m+1)/2×100 |
| Desync | (1−m/2)×100 |
| LSA | m/10×100 |
| 2项QA | m×100 |

展示分限制在0–100，原始值不改写，超出固定展示锚点的样本数及均分标记保存在报告。MOS的1/5、AudioBox的−1.75/7.25、LSA的0/10是展示锚点，不宣称为所有模型预测的硬边界。ImageBind保留官方sim_av主分，QA保持官方positive-polarity比例，不改成简单yes占比。

总分为15项展示分等权平均，不能按可用指标数重新分配权重：

- 缺任一指标有效均分：总分为空，展示“—”。未选择评测的指标也显示“未评测”，不补0。
- 15项都有均分但有失败、缺失、LSA省略或未解决的后端错误：展示“预览总分／不完整”。
- 15项都有均分、输入评分覆盖率100%且没有已记录后端错误：展示当前评测范围的正式总分。

单条冒烟、自定义子集、Mini10和778各自保留scope与样本数，Mini10总分不能称为778标准集总分。“正式”仅指本轮评分完整，不代表官方<1%对齐验收已完成。LSA省略是missing的子集，CSV中missing/omitted不可相加重复计数。分组均分另存scenes.csv。

已有AISBench逐样本报告可离线重汇总，无需重新加载模型：

```bash
python3 -m ais_bench.benchmark.vabench.cli summarize \
  --report /workspace/vabench_runs/gpu_mini/report.json \
  --output /workspace/vabench_runs/gpu_mini_new_summary \
  --prepared /workspace/vabench_runs/gpu_mini_prepared
```

原始报告不会被覆盖，新报告保存来源哈希和`model_inference_rerun=false`。这不是新的GPU验证，也不能恢复原始报告没有保存的模型生成文本。`--prepared`用于恢复官方`prompt_en`，省略时CSV回退到原始记录中的`prompt`字段或样本`idx`。

## 输出与尚待改进

```text
run/
  report.json             # 原始记录、各指标/场景均分、状态及溯源
  metrics.csv             # 15项原始均分、归一化分、计数与覆盖率
  total.csv               # 当前评测范围的15项等权总分及状态
  summary.md              # 可直接阅读的15项与总分表格
  samples.csv             # 逐题状态、评分与请求设备
  scenes.csv              # 场景均分
  official_eval_results/  # 与官方evaluate.py同结构的原始指标备份
    <category>/
      <category>_<time>_eval_results.json
      <category>_<time>_eval_results.csv
      <category>_<time>_full_info.json
  input_manifest.json
  partial_report.json
  job_00/request.json
  job_00/response.json
  job_00/process.log
  job_00/指标/场景/backend.log
```

`official_eval_results`专为与官方结果对照：

- JSON保持官方`{dimension: {"video_results": [...]}}`结构，并保留官方评分函数返回的全部原始记录；即使AISBench因解析失败或后端错误不把该样本计入均分，原始`video_results`仍会写入，便于逐条核对。
- CSV表头为`prompt_name`加当前请求的指标列，按`prompt_en`聚合；某指标没有数值时写入`-1`，与官方`*_eval_results.csv`一致。
- `full_info.json`来自`prepare`生成的`json/<category>.json`，与官方`*_full_info.json`字段一致。
- 该备份只复现官方原始分数和CSV聚合方式，不改变AISBench的归一化、覆盖率、失败剔除和总分协议。

官方函数返回记录保存在raw中，不能声称已完整捕获MLLM全部生成文本。CLAP、Desync、ImageBind日志明确打印`Error processing <path>`时，仅将该样本标记为失败，同场景其他有效分保留；未打印样本名的场景级错误仍可能影响同场景其他样本，部分被官方吞掉的解析错误尚不能完整恢复。已识别的空解析/OOM及非法量表输出会剔除并保留原始值；LSA省略已计入覆盖率缺口。正式总分只表示当前选定样本的15项完整性，不代表GPU官方对齐已通过。

已有partial文件用于追踪，不等于已支持断点补跑；当前重跑须新建输出目录。环境准备报告、实际freeze、完整评分目录和官方原始输出应一起保留，以便复现和对齐。

本地轻量回归（在AISBench源码根目录执行）：

```bash
cd /workspace/code/benchmark

/root/miniconda3/envs/vabench311_new/bin/python3 -m pip install --no-deps \
  "pytest==8.3.5" "pluggy==1.5.0" "iniconfig==2.0.0" \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple

/root/miniconda3/envs/vabench311_new/bin/python3 -m pytest tests/UT/vabench -q
```

预期`70 passed, 4 skipped`。4个skip是因为测试固定查找`/workspace/code/VABench`，官方源码在`/workspace/code/VABench-main`时跳过；如需执行，可建立同名软链。模型在集成测试中由模拟实现替代；真实FFmpeg和子进程契约通过，不表示H200模型评分通过。

参考：[VABench官方源码](https://github.com/tanABCC/VABench)、[官方模型缓存](https://huggingface.co/datasets/tanABBCC/VABENCH_CACHE_DIR)、[LatentSync人脸检测权重](https://huggingface.co/ByteDance/LatentSync-1.5/tree/main/auxiliary)、[InsightFace模型资源](https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo)。

