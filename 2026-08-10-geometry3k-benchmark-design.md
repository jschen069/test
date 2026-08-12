# Geometry3K 基准测试模块设计文档

> **状态**：方案评审
> **作者**：jschen069
> **日期**：2026-08-10
> **范围**：`geometry3k.py` + `geometry3k_gen.py` + 关联模块（postprocessor、vllm 配置）

---

## 1. 背景描述

### 1.1 项目背景

Geometry3K 是一个包含 601 道几何选择题的多模态数据集，每题包含一张几何图形图片和一个文字问题。本模块将 Geometry3K 接入 ais_bench 基准测试框架，支持对 VLM（视觉语言模型）进行几何推理能力的自动化评测。

评测流程为：模型接收"图片 + 文字问题 + 指令模板"作为输入，输出带 `<think>` 推理过程和 `\boxed{}` 最终答案的文本，系统自动提取答案并与标准答案比对。

### 1.2 目标

| 目标 | 说明 |
|------|------|
| 数据集接入 | 将 Geometry3K 数据集以标准 `BaseDataset` 子类形式注册到框架 |
| 多模态推理 | 支持图文混合输入，通过 `MMPromptTemplate` 正确传递给 VLM |
| 自动评分 | 实现与 veRL 一致的评分逻辑：答案正确性（mathruler sympy）+ 格式合规性 + 加权综合 |
| 可配置性 | 评分权重、数据路径、推理参数等均可通过配置灵活调整 |
| 生产可用 | 日志策略不干扰正常评测输出，调试模式下可完整追踪 |

### 1.3 范围

- **核心模块**：`geometry3k.py`（数据集加载 + 评分器，~387 行）
- **配置文件**：`geometry3k_gen.py`（reader / inferencer / evaluator 编排，~51 行）
- **关联变更**：`model_postprocessors.py`（`keep_reasoning_content`）、vllm 模型配置

### 1.4 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    ais_bench benchmark 框架                       │
│                                                                   │
│  ┌──────────────────────┐    ┌──────────────────────────┐       │
│  │  geometry3k_gen.py   │    │    geometry3k.py           │       │
│  │  (配置层)             │    │    (核心模块)              │       │
│  │                      │    │                            │       │
│  │  • reader_cfg        │───▶│  Geometry3KDataset.load()  │       │
│  │  • infer_cfg         │    │    ├─ _resolve_parquet_path│       │
│  │    - MMPromptTemplate│    │    ├─ _save_image          │       │
│  │    - ZeroRetriever   │    │    └─ get_content_str      │       │
│  │    - GenInferencer   │    │                            │       │
│  │  • eval_cfg          │───▶│  Geometry3KEvaluator       │       │
│  │                      │    │    ├─ _extract_boxed_content│      │
│  │                      │    │    ├─ _grade_answer         │       │
│  │                      │    │    ├─ format_reward         │       │
│  │                      │    │    └─ _compute_score        │       │
│  └──────────────────────┘    └──────────┬─────────────────┘       │
│                                         │                         │
│                              依赖关系    │                         │
│                              ┌──────────┼──────────┐              │
│                              │          │          │              │
│                              ▼          ▼          ▼              │
│                        mathruler   datasets   PIL/Pillow          │
│                        (判题)     (HF loader)  (图片解码)          │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  关联模块                                                 │    │
│  │  model_postprocessors.py          vllm_api_general_chat.py │    │
│  │  ┌─────────────────────┐          ┌──────────────────────┐ │    │
│  │  │ keep_reasoning_content│◀───────│ pred_postprocessor   │ │    │
│  │  │ (透传保留think标签)  │          │ enable_thinking=True │ │    │
│  │  └─────────────────────┘          └──────────────────────┘ │    │
│  └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.5 完整数据流

```
parquet 文件
    │
    ▼
Geometry3KDataset.load()
    ├── _resolve_parquet_path() → 三级路径回退定位 parquet 文件
    ├── load_dataset("parquet", ...) → 读入内存
    ├── 逐条处理:
    │   ├── _save_image() → 图片解码落盘为 PNG
    │   ├── 拼接 prompt = problem + GEOMETRY3K_INSTRUCTION
    │   └── get_content_str(msgs) → 多模态消息渲染
    └── 输出 Dataset {content, question, image, answer, index}
            │
            ▼
    DatasetReader (reader_cfg: input_columns=["question", "image"])
            │
            ▼
    MMPromptTemplate (infer_cfg: 图片 file:// + {question} → VLM 请求)
            │
            ▼
    VLM 推理 (vllm, enable_thinking=True)
            │
            ▼
    keep_reasoning_content (postprocessor: 透传，保留完整模型输出)
            │
            ▼
    Geometry3KEvaluator.score()
        ├── _extract_boxed_content() → mathruler 提取 \boxed{} 答案
        ├── _grade_answer() → mathruler sympy 数学等价判题
        ├── format_reward() → 格式合规检查（<think> + \boxed{}）
        └── 输出 {combined_score, details}
```

---

## 2. 方案设计

### 2.1 加载文件

数据集以 parquet 格式在本地分发，存放于 `ais_bench/datasets/geo3k/test-00000-of-00001.parquet`。`Geometry3KDataset` 通过 `@LOAD_DATASET.register_module()` 注册到框架工厂，作为 `BaseDataset` 的子类，以 `@staticmethod load()` 实现数据加载。

#### 2.1.1 三级路径回退

`_resolve_parquet_path(path, split)` 实现三级路径解析，保证"指定路径→框架路径→默认路径"的递进式兜底：

```
_resolve_parquet_path(path, split)
    │
    └── 1. path 是相对路径 / 空
        ├── 3a. 尝试 get_data_path(path, local_mode=True)
        └── 3b. 回退到 source-relative: ../../datasets/geometry3k/
            └── 优先查找 data/ 子目录下的 parquet
            └── 再查找目录本身的 parquet
```

设计要点：

- **`data/` 子目录优先**：数据集的标准布局是 `geometry3k/data/test-*.parquet`，与框架中其他数据集保持一致。
- **`sorted(glob(...))[0]`**：parquet 可能分片（如 `test-00000-of-00001.parquet`），glob 自动适配分片命名，`sorted` 保证确定性。
- **失败即抛 `FileNotFoundError`**：找不到文件是配置错误，尽早暴露而非静默降级。

路径配置经历三次演进：

| 阶段 | path 值 | 加载方式 |
|------|---------|---------|
| v1 | `"ais_bench/datasets/geometry3k/data/test-00000-of-00001.parquet"` | 本地 parquet 完整路径 |

#### 2.1.2 数据集注册与加载

```python
@LOAD_DATASET.register_module()
class Geometry3KDataset(BaseDataset):
    @staticmethod
    def load(path=None, split="test", instruction=None):
        parquet_file = _resolve_parquet_path(path, split)
        dataset = load_dataset("parquet", data_files={split: parquet_file}, split=split)
        # ... 逐条处理 ...
        return Dataset.from_list(records)
```

**为什么用 `@staticmethod load()`？** `BaseDataset.__init__` 调用 `self.load(**kwargs)`，Geometry3KDataset 不需要实例状态（路径、split 都是参数传入），`@staticmethod` 避免无意依赖 `self`。

**返回的 Dataset 字段：**

| 字段 | 内容 | 用途 |
|------|------|------|
| `content` | `get_content_str(msgs)` 渲染结果 | 框架内部使用 |
| `question` | `problem + instruction` 完整 prompt | `MMPromptTemplate` 的 `{question}` 占位符 |
| `image` | 本地 PNG 文件路径 | `MMPromptTemplate` 的 `{image}` 占位符 |
| `answer` | 标准答案 | Evaluator 比对的 ground truth |
| `index` | 序号 | 追踪和调试 |

### 2.2 落盘图片

Geometry3K 的每道题包含一张几何图形，以图片列的形式存储在 parquet 中。加载时需要将图片数据解码并保存到本地磁盘，因为下游 `MMPromptTemplate` 使用 `file://` 协议传递图片路径。

#### 2.2.1 三种输入格式统一处理

`datasets` 库从 parquet 加载图片列时，返回类型取决于 parquet 的写入方式。`_save_image()` 统一处理三种情况：

```python
_save_image(image_obj, image_dir, index)
    ├── PIL.Image.Image  → convert("RGB").save("{index}.png")
    ├── dict{"bytes": ...} → BytesIO → PIL → convert("RGB").save
    └── str               → 已经是路径，直接返回
```

- **PIL.Image.Image**：parquet 存的是编码后的二进制 blob → datasets 自动解码为 PIL Image
- **dict{"bytes": ...}**：parquet 存的是原始 bytes → 需手动 `BytesIO` → PIL 解码
- **str**：已经是本地文件路径 → 直接返回

三种情况都要兼容，否则换一个 parquet 源就可能崩溃。

#### 2.2.2 图片存储策略

- **格式**：统一转为 PNG。PNG 无损压缩，适合几何线条图（JPEG 可能模糊细线）。
- **色彩模式**：统一 `RGB`，避免灰度图导致的通道不一致。
- **命名**：`{index}.png`，与数据集序号一一对应，方便人工抽查。
- **存储位置**：`{parquet_dir}/geometry3k_images/{index}.png`，与数据同目录，方便管理和批量清理。

### 2.3 构造 Prompt

#### 2.3.1 指令模板

与 veRL 的 `geo3k.py` 预处理保持一致，指令模板要求模型将推理过程放入 `<think>` 标签，最终答案放入 `\boxed{}`：

```python
GEOMETRY3K_INSTRUCTION = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in \\boxed{}."
)
```

指令模板定义在数据集侧（而非配置侧）：与 veRL 保持一致，且允许在 `load()` 调用时通过 `instruction` 参数覆盖。

#### 2.3.2 多模态消息组装

每条样本在 `load()` 中组装为多模态消息列表：

```python
full_prompt = f"{problem} {inst}"
msgs = [
    {"type": "image_url", "image_url": image_path},
    {"type": "text", "text": full_prompt},
]
content = get_content_str(msgs)
```

`question` 和 `content` 是两个独立字段：
- `question`：纯文本 prompt，传给 `MMPromptTemplate` 的 `{question}` 占位符
- `content`：`get_content_str()` 渲染后的内部格式，供框架使用

两者分离，模板层和渲染层各自独立。

#### 2.3.3 推理配置

在 `geometry3k_gen.py` 中配置推理三件套：

```python
reader_cfg = dict(input_columns=["question", "image"], output_column="answer")
infer_cfg = dict(
    prompt_template=dict(type=MMPromptTemplate, ...),
    retriever=dict(type=ZeroRetriever),    # 零样本，无需 few-shot 检索
    inferencer=dict(type=GenInferencer),   # 生成式推理
)
eval_cfg = dict(evaluator=dict(type=Geometry3KEvaluator))
```

#### 2.3.4 后处理器适配

框架原有 `extract_non_reasoning_content` 后处理器会剥离 `<think>` 标签，但 Geometry3K 的 `format_reward` 需要检查 `<think>` 是否存在。为此新增 `keep_reasoning_content` 透传后处理器：

```python
@TEXT_POSTPROCESSORS.register_module('keep-reasoning-content')
def keep_reasoning_content(text: str | list) -> str:
    """Pass-through：保留完整模型输出，不做任何处理"""
    if isinstance(text, list):
        return [_keep_single(item) for item in text]
    return _keep_single(text)
```

通过模型配置中的 `pred_postprocessor=dict(type=keep_reasoning_content)` 一行切换，不影响其他数据集继续使用 `extract_non_reasoning_content`。

#### 2.3.5 vllm 模型推理参数

```python
generation_kwargs=dict(
    enable_thinking=True,    # 核心：让 vllm 输出 reasoning tokens（<think> 标签）
    temperature=0,           # 评测场景确定性输出，保证可复现
    top_k=-1, top_p=1.0,    # 不做采样截断，配合 temperature=0 实现 greedy decoding
    ignore_eos=True,        # 防止模型过早输出 EOS 截断回答
    repetition_penalty=1.0,  # 不做重复惩罚
    logprobs=0,              # 不返回 logprobs，减少响应体积
)
```

### 2.4 打分

#### 2.4.1 评分器概述

`Geometry3KEvaluator` 继承 `BaseEvaluator`，实现三层评分体系：

```
预测文本
    │
    ├── _extract_boxed_content() → mathruler 提取 \boxed{} 内容
    │       │
    │       └── _grade_answer(extracted, ground_truth) → mathruler sympy 判题 → accuracy ∈ {0, 1}
    │
    └── format_reward() → 检查 <think> + \boxed{} 格式 → format_score ∈ {0, 1}
            │
            └── combined = (1-w)*accuracy + w*format_score
```

#### 2.4.2 答案提取与判题：mathruler 集成

初始版（commit 7ce44ad）手写了一套简化判题逻辑（手动正则提取 `\boxed{}` + 四级字符串匹配），后替换为 `mathruler.grader`：

```python
from mathruler.grader import extract_boxed_content, grade_answer

def _extract_boxed_content(pred_str: str) -> str:
    """Wrapper，日志注入点。返回 "None" 字符串（与 verl 一致）"""
    result = extract_boxed_content(pred_str)

def _grade_answer(given_answer: str, ground_truth: str) -> bool:
    """Wrapper，日志注入点。sympy 数学等价检查。"""
    result = grade_answer(given_answer, ground_truth)
```

**为什么保留 wrapper 而不是直接调用？**
1. **日志注入**：mathruler 是外部库，wrapper 是插入 debug 日志的唯一切入点
2. **隔离外部依赖**：mathruler 接口变化只需改 wrapper 一处
3. **行为一致性**：对 `"None"` 返回值做显式日志记录

**mathruler 相比手写实现的优势：**

| 场景 | 手写实现 | mathruler |
|------|---------|-----------|
| 数学等价判断 | 字符串匹配 | sympy 代数等价（如 `x+1` = `1+x`） |
| 分数比较 | 无 | 支持 `\frac{1}{2}` = `0.5` |
| 元组/区间 | 无 | 支持 `(3, 5)`、`[3, 5]` 等 |
| 严格整数匹配 | 无 | 区分 `3` 和 `3.0` |

#### 2.4.3 格式分

`format_reward()` 用正则检查模型输出是否同时包含 `<think>...</think>` 和 `\boxed{...}`，且 `<think>` 必须在 `\boxed{}` 之前：

```python
def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0
```

格式分的设计目的是验证模型是否遵循了"先思考再回答"的指令——这在上游 RL 训练中是重要的 reward 信号。

#### 2.4.4 综合评分公式

```python
class Geometry3KEvaluator(BaseEvaluator):
    def __init__(self, format_weight: float = 0.0):
        self.format_weight = format_weight

    def _compute_score(self, pred_str, ground_truth):
        acc = 1.0 if _grade_answer(extracted, ground_truth) else 0.0
        fmt = format_reward(pred_str)
        combined = (1.0 - self.format_weight) * acc + self.format_weight * fmt
```

**为什么默认 `format_weight=0.0`？** 格式分是对输出格式的评价，不是对答案正确性的评价。纯评测场景中格式分不应影响最终分数（`combined = 1.0*acc`）。当需要引导模型学习输出格式时（如 RL 训练），通过配置 `format_weight` 启用。

**`score()` 与 `_compute_score()` 分离：** `score()` 负责遍历和聚合，`_compute_score()` 负责单样本评分。分离后单样本评分可独立测试；如果未来需要并行评分，只需改 `score()` 的遍历方式；子类可覆盖 `_compute_score()` 实现不同评分策略。

#### 2.4.5 返回值设计

```python
result = {
    "combined_score": final_combined,  # 唯一顶层指标
    "details": [                       # 逐样本详情
        {"pred": ..., "answer": ..., "extracted_answer": ...,
         "accuracy": ..., "format_score": ..., "combined_score": ...},
    ],
}
```

- **只保留 `combined_score` 作为顶层指标**：`accuracy` 和 `format_score` 可从 `details` 聚合，顶层保留是冗余
- **`details` 包含全部原始信息**：支持下游自行分析每个样本的具体表现
- **参考值为 dict 时自动提取**：`ref.get("answer", str(ref))` 兼容 `str` 和 `{"answer": ...}` 两种输入

### 2.5 日志策略

以 601 条样本的 test split 为例，原 INFO 级别日志量约 12000 行，严重干扰终端输出。日志策略的核心决策：

| 决策 | 内容 |
|------|------|
| **两层分层** | 正常路径全部 `DEBUG`（默认静默），异常路径 `WARNING`（始终可见），`INFO` 不使用 |
| **多行合并** | 一个逻辑事件（如一次评分）对应一次 `logger.debug()`，内部用 `\n` 排版，保证原子性和可 grep |
| **图片信息预采集** | 循环内先构建 `img_info` 字符串，循环末尾一次性 log，减少日志调用次数 |
| **WARNING 保留** | 真正的异常（如未知图片类型）用 `WARNING` 始终输出，不被静默 |

```
生产环境 (LOG_LEVEL=INFO) → 模块零输出，WARNING 仍可见，异常直接抛出
调试模式 (LOG_LEVEL=DEBUG) → 全量路径追踪、每样本评分详情、图片类型/尺寸
```

---

## 3. 使用说明

### 3.1 前置条件

1. **数据集文件**：确保 `ais_bench/datasets/geo3k/test-00000-of-00001.parquet` 存在
2. **vllm 推理服务**：已在 `localhost:8005` 启动，且模型支持 `enable_thinking=True`

### 3.2 命令行执行

通过 ais_bench CLI 指定数据集和模型配置即可运行评测：

```bash
    ais_bench \
    --datasets geometry3k_gen \
    --models vllm_api_general_chat \
```

`--datasets geometry3k_gen` 对应配置文件 `configs/datasets/geometry3k/geometry3k_gen.py` 中的 `geometry3k_datasets` 列表，框架自动发现并加载。

`--mode all` 表示依次执行推理和评测两个阶段。也可以分开执行：

```bash
# 仅推理（生成 predictions）
ais_bench --datasets geometry3k_gen --models vllm_api_general_chat --mode infer

# 仅评测（对已有 predictions 评分）
ais_bench --datasets geometry3k_gen --models vllm_api_general_chat --mode eval
```

### 3.3 自定义数据路径

如果 parquet 文件不在默认位置，可以在配置中指定路径：

```python
geometry3k_datasets = [dict(
    ...
    path="/your/custom/path/to/geo3k",   # 目录：自动搜索 {split}-*.parquet
    # 或者
    path="/your/custom/path/to/geo3k/test-00000-of-00001.parquet",  # 直接指定文件
)]
```

### 3.4 调整评分权重

通过配置修改 `format_weight`：

```python
eval_cfg = dict(
    evaluator=dict(
        type=Geometry3KEvaluator,
        format_weight=0.1,   # 格式分占 10%
    )
)
```

- `format_weight=0.0`（默认）：综合分 = 答案正确率，适合纯能力评测
- `format_weight=0.1`：综合分 = 0.9×正确率 + 0.1×格式分，适合需要关注格式合规的场景

### 3.5 切换后处理器

不同的后处理器影响评分器能看到的输出内容：

```python
# 需要 <think> 标签的数据集（如 Geometry3K）
pred_postprocessor=dict(type=keep_reasoning_content)

# 不需要 <think> 标签的数据集
pred_postprocessor=dict(type=extract_non_reasoning_content)
```

### 3.6 开启调试日志

默认（INFO 级别）下模块零日志输出。需要排查问题时，设置环境变量开启 DEBUG：

```bash
LOG_LEVEL=DEBUG python -m ais_bench --datasets geometry3k_gen --models vllm_api_general_chat
```

### 3.7 输出结果

评测完成后，在 `outputs/` 目录下生成预测文件和评分结果。返回结果展示：

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- | -----|
| geometry3k | e3713f | accuracy | gen | 36.00 |

---

## 4. 测试设计

测试文件位于 `tests/UT/datasets/test_geometry3k.py`，共 6 个 TestCase 类，覆盖模块的所有对外接口。

### 4.1 测试结构总览

| TestCase | 覆盖函数 | 用例数 | 测试重点 |
|----------|---------|--------|---------|
| `TestExtractBoxedContent` | `_extract_boxed_content()` | 5 | 正常提取、无 boxed、多个 boxed、空 boxed、含 think 标签 |
| `TestGradeAnswer` | `_grade_answer()` | 6 | 精确匹配、不匹配、数值等价、分数、LaTeX、单位文本 |
| `TestFormatReward` | `format_reward()` | 6 | 格式正确、缺 think、缺 boxed、两者皆无、think 在 boxed 之后、尾部多余文本 |
| `TestSaveImage` | `_save_image()` | 3 | PIL Image、dict+bytes、字符串路径 |
| `TestResolveParquetPath` | `_resolve_parquet_path()` | 3 | 绝对文件路径、目录+分片匹配、无文件抛异常 |
| `TestGeometry3KDataset` | `Geometry3KDataset.load()` | 3 | 最小加载、自定义 instruction、默认 split |
| `TestGeometry3KEvaluator` | `Geometry3KEvaluator` | 9 | 满分、答错、无格式、format_weight=0、特殊 token 剥离、长度不匹配、全对、全错、混合、details 字段、参考为 dict |

### 4.2 关键测试用例详解

#### 4.2.1 答案提取测试 (`TestExtractBoxedContent`)

```python
def test_no_boxed(self):
    result = _extract_boxed_content("No boxed content here")
    self.assertEqual(result, "None")   # mathruler 返回字符串 "None"

def test_multiple_boxed_returns_last(self):
    result = _extract_boxed_content("\\boxed{1} and \\boxed{2}")
    self.assertEqual(result, "2")      # 验证取最后一个 boxed

def test_empty_boxed(self):
    result = _extract_boxed_content("\\boxed{}")
    self.assertEqual(result, "None")   # 空 boxed 视为无有效答案
```

**设计要点**：验证 mathruler 的边界行为——无 boxed 返回 `"None"` 字符串（不是 Python `None`），多个 boxed 取最后一个，空 boxed 视为无效。这些是 verl 中确定的行为，测试确保 wrapper 与 verl 一致。

#### 4.2.2 判题测试 (`TestGradeAnswer`)

```python
def test_numeric_equivalence(self):
    self.assertTrue(_grade_answer("3.14000", "3.14"))

def test_fraction_match(self):
    self.assertTrue(_grade_answer("\\frac{1}{2}", "0.5"))

def test_latex_superscript(self):
    self.assertTrue(_grade_answer("90^{\\circ}", "90"))

def test_unit_text(self):
    self.assertTrue(_grade_answer("\\text{cm}", "cm"))
```

**设计要点**：覆盖 Geometry3K 中最常见的答案差异模式——数值精度、LaTeX 分数、度数符号、单位文本。这些是手写版判题容易出错、而 mathruler 能正确处理的场景。

#### 4.2.3 格式分测试 (`TestFormatReward`)

```python
def test_think_after_boxed(self):
    """format_reward uses re.fullmatch: <think> must appear before \\boxed{}."""
    self.assertEqual(format_reward("\\boxed{42} <think>too late</think>"), 0.0)

def test_partial_match_trailing_text(self):
    """re.fullmatch rejects text after the \\boxed{}."""
    self.assertEqual(format_reward("<think>x</think> \\boxed{42} extra"), 0.0)
```

**设计要点**：`re.fullmatch` 要求字符串**完全**匹配模式——think 必须在 boxed 之前（顺序约束），boxed 之后不能有额外文字（严格格式约束）。这两个测试精确验证了格式要求的边界。

#### 4.2.4 图片保存测试 (`TestSaveImage`)

```python
class TestSaveImage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_pil_image(self):
        img = PILImage.new("RGB", (10, 10), color="red")
        path = _save_image(img, self.tmpdir, 0)
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith("0.png"))

    def test_save_bytes_dict(self):
        img = PILImage.new("RGB", (10, 10), color="blue")
        buf = BytesIO()
        img.save(buf, format="PNG")
        path = _save_image({"bytes": buf.getvalue()}, self.tmpdir, 1)
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith("1.png"))

    def test_save_string_path(self):
        path = _save_image("/tmp/existing.png", self.tmpdir, 2)
        self.assertEqual(path, "/tmp/existing.png")  # 原样返回
```

**设计要点**：三种输入格式各测一种——PIL Image 验证直接保存路径、dict+bytes 验证 BytesIO 解码、字符串验证透传。用 `setUp`/`tearDown` 管理临时目录，避免测试污染文件系统。

#### 4.2.5 路径解析测试 (`TestResolveParquetPath`)

```python
def test_absolute_directory_with_split_pattern(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "data", "test-0000.parquet")
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        Path(test_file).touch()
        result = _resolve_parquet_path(tmpdir, "test")
        self.assertTrue(result.endswith("test-0000.parquet"))

def test_no_parquet_files_raises(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            _resolve_parquet_path(tmpdir, "test")
```

**设计要点**：验证目录模式下自动搜索 `data/` 子目录、以及空目录时正确抛出异常。测试文件仅 `touch()` 创建空文件（不关心内容），最小化测试开销。

#### 4.2.6 Evaluator 综合测试 (`TestGeometry3KEvaluator`)

```python
def test_compute_score_perfect(self):
    pred = "<think>1+1=2</think> \\boxed{2}"
    result = self.evaluator._compute_score(pred, "2")
    self.assertEqual(result["accuracy"], 1.0)
    self.assertEqual(result["format_score"], 1.0)
    self.assertEqual(result["combined_score"], 1.0)

def test_score_returns_details(self):
    predictions = ["<think>x</think> \\boxed{5}"]
    references = ["5"]
    result = self.evaluator.score(predictions, references)
    self.assertIn("details", result)
    self.assertIn("combined_score", result)
    self.assertNotIn("accuracy", result)      # 验证顶层不含 accuracy
    self.assertNotIn("format_score", result)  # 验证顶层不含 format_score

def test_compute_score_format_weight_zero(self):
    eva = Geometry3KEvaluator(format_weight=0.0)
    pred = "answer is 2"
    result = eva._compute_score(pred, "2")
    self.assertEqual(result["combined_score"], result["accuracy"])  # 格式分不影响

def test_score_reference_is_dict(self):
    predictions = ["<think>x</think> \\boxed{7}"]
    references = [{"answer": "7"}]     # 参考值为 dict 格式
    result = self.evaluator.score(predictions, references)
    self.assertAlmostEqual(result["combined_score"], 100.0)
```

**设计要点**：

- **满分测试**：验证 perfect 输入产生全 1.0 的分数
- **返回值结构测试**：验证顶层只有 `combined_score` + `details`，不含被移除的 `accuracy`/`format_score`
- **format_weight=0 测试**：验证默认配置下格式分不参与综合分计算
- **参考值为 dict 测试**：验证 `ref.get("answer", str(ref))` 的兼容逻辑

### 4.3 运行测试

```bash
cd tests
python -m pytest UT/datasets/test_geometry3k.py -v
```

### 4.4 测试覆盖情况

| 覆盖维度 | 状态 | 说明 |
|---------|------|------|
| 正常路径 | 全覆盖 | 答案正确、格式正确、路径解析成功、图片保存成功 |
| 边界条件 | 全覆盖 | 空 boxed、无 boxed、多 boxed、think 在 boxed 后、尾部多余文本 |
| 错误路径 | 已覆盖 | 文件不存在抛异常、预测/参考长度不等 |
| 输入格式兼容 | 全覆盖 | ref 为 str/dict、image 为 PIL/dict/str |
| 可配置参数 | 已覆盖 | format_weight=0.0 和 0.1 两种场景 |
| 聚合逻辑 | 已覆盖 | 全对、全错、混合三种场景 |
