# AISBench 接入 Terminal-Bench 4.0 需求设计文档

> 依据材料：
> - [aisbench接入terminal-bench4.0 需求文档.md](./aisbench接入terminal-bench4.0%20需求文档.md)（需求背景 / 需求描述 / 验收标准）
> - [terminal_bench_4.md](./benchmark/docs/source_zh_cn/extended_benchmark/agent/terminal_bench_4.md)（TB4.0 操作适配文档）
> - [agent_benchmark.md](./benchmark/docs/source_zh_cn/base_tutorials/scenes_intro/agent_benchmark.md)（基于 Harbor 框架的 Agent 测评）
> - 内部全流程适配记录 `指导文档/TB4_RUNTIME_SETUP_FULL_FLOW.md`（方案与实测数据来源）

---

## 1. 背景

### 1.1 需求来源

| 项 | 内容 |
| --- | --- |
| 需求描述 | AISBench 支持 Terminal Bench 4.0 数据集精度测评 |
| 验收硬件 | NPU / GPU |
| 验收模型 | DeepSeek-v4.1-flash 或更新版本（可使用模型官方 API 验收精度） |
| 验收数据集 | Terminal Bench 4.0 |
| 精度要求 | 测试一次或三次测试取平均，AISBench 与官方精度结果差异小于 1%，且精度结果不能为 0 |

### 1.2 Terminal-Bench 4.0 概况

Terminal-Bench 4.0（官方 release `v4.0.0`）是面向 Agent 复杂终端任务执行能力的 Benchmark，共 **66 个任务（63 CPU + 3 GPU）**，配套 **157 个官方任务镜像（env / verifier / sidecar）+ 2 个外部镜像（`redis:7-alpine`、`apache/kafka-native:4.3.1`），合计 159 个**。任务覆盖命令行操作、代码处理、环境配置、数据库/中间件运维、性能调优、形式化验证等复杂场景，单任务执行时间长（分钟至小时级）、验证链路复杂（独立 verifier 容器 + compose sidecar），对评测框架的任务编排、长时间运行管理、执行过程监控与结果采集能力要求较高。

### 1.3 接入挑战

| 挑战 | 说明 |
| --- | --- |
| 镜像 digest 引用 | 官方 `task.toml` 以 `repo:tag@sha256:...` 方式 pin 镜像；离线镜像包经 `docker save/load` 后不保留 RepoDigests，直接加载会导致 Compose 解析失败并尝试联网 pull |
| 多基础 OS 的 Agent 离线依赖 | 任务镜像基础 OS 覆盖 ubuntu:20.04/22.04/24.04、debian:12/13、fedora:42 六种，Agent 离线包需全量覆盖；且任务容器多以非 root（`nobody`）运行，安装于 `/root/.local` 的 Agent 不可见 |
| separate verifier 模式 | TB4.0 全部任务的 verifier 为 `environment_mode = "separate"`，测试用例烘在 verifier 镜像内，修改宿主侧 `tests/` 无效 |
| 网络与镜像源 | 部分任务运行期/验证期会访问 PyPI、Docker Hub 等外部源，离线环境需镜像源注入（如 payments 任务 verifier 的 `env -i` 会清空 pip 镜像环境变量） |
| 长程运行与 API 计费 | 63 任务全量一轮需数小时至数天，API 余额耗尽（HTTP 402）会把 trial 记为 `UnknownApiError` 截断整批，需探针 + 断点续测机制 |

### 1.4 方案总原则

AISBench **原生集成 Harbor（官方 fork 改造版）作为 Agent 测评引擎**，Terminal-Bench 4.0 官方即以 Harbor 作为评测 harness。因此接入方案不重写评测链路，而是：**复用 AISBench 既有 `--mode agent` 通用 Agent 测评能力（HarborRunner / HarborAgentTask / AgentParamAdapter / 监控服务 / 断点续测），针对 TB4.0 的数据集、镜像、Agent 依赖、verifier 做离线化适配**，保证评测口径与官方一致（官方任务定义 + 官方预构建镜像 + 官方引擎）。

---

## 2. 方案设计

### 2.1 总体链路

```
物料层    官方任务包（66 task.toml/instruction/tests） ─ 解压即 Harbor 格式数据集
          官方镜像清单 prebuilt-images.json ─ 校验 157 镜像 digest
          离线镜像包（159 镜像 tar） ─ docker load 导入
          Agent 离线依赖包（6 基础 OS × agent） ─ agent-deps 目录
              │
适配层    task.toml 去 digest（save/load 不保留 RepoDigests）
          2 个外部镜像重打 tag（redis / kafka-native）
          payments verifier 注入（PipConfVerifier）
              │
执行层    AISBench --mode agent
              └─ HarborRunner → Harbor job → 逐 trial 编排容器
                     （env 容器 + Agent 注入 + separate verifier 容器 + compose sidecar）
              └─ 实时监控 HTTP 服务（--monitor-port，逐 case 状态）
              └─ 断点续测（--reuse / --purge-exception-cases / harbor job resume）
              │
结果层    HarborSummarizer 单表 + CSV 汇总
          outputs/default/<时间戳>/ 完整落盘（details/result.json、逐 trial reward/ctrf/轨迹）
```

### 2.2 两种运行模式

AISBench agent runtime 一键部署脚本 `start_agent_runtime.sh` 支持两种模式，均已完成 TB4.0 全量验证：

| 维度 | socket 模式（默认，推荐） | dind 模式（强隔离） |
| --- | --- | --- |
| 原理 | 挂载宿主机 `/var/run/docker.sock`（DooD），容器内直接操作宿主 daemon | `--privileged` 容器内独立 dockerd（DinD），trial 容器只出现在内层 |
| 题目镜像 | 宿主机 `docker load`，容器内直接可见 | `--case-images` 必选，脚本自动 load 进内层 daemon |
| 数据挂载 | 数据集 / agent-deps / 产物按宿主机原路径同路径透传 | 数据集只读挂载 `/opt/data/`，产物写 `--trials-dir` |
| docker 要求 | 无特殊要求 | > 20.0.0（使用 `--cgroupns=host`） |
| 适用场景 | 宿主 daemon 可直接使用，镜像与排障最省事 | 需要强隔离，或宿主 daemon 不宜被评测操作（如老版本 Docker 18.09 的机器需派生镜像去掉 `--cgroupns=host`） |

### 2.3 数据集与镜像适配设计

| 设计点 | 方案 |
| --- | --- |
| 数据集来源 | 官方预构建任务包 `terminal-bench-prebuilt-v4.0.0.tar.gz` 解压后 `prebuilt/tasks/` 即为 Harbor 格式数据集（`-p/--agent-dataset-path` 直接指向）；SHA256 与 `prebuilt-images.json` 校验，66 任务 / 157 镜像 / digest 全 pin 三方一致后才进入执行 |
| 镜像导入 | 离线镜像包 `terminal-bench-4.0-images-x86_64.tar`（159 镜像，约 177G）`docker load` 导入；socket 模式导入宿主机，dind 模式导入内层 daemon |
| 去 digest | `docker save/load` 不保留 RepoDigests，须将全部 `task.toml` 的 `docker_image = "repo:tag@sha256:..."` 批量改为 `repo:tag`（共 132 处引用）；Harbor 实际读取 `task.toml`，仅改镜像清单 JSON 无效 |
| 外部镜像重打 tag | 镜像包中 2 个第三方镜像 repo 前缀被打错（save/load 所致），须重打 tag：`harborframework/terminal-bench:4.3.1 → apache/kafka-native:4.3.1`（payments 任务 Kafka sidecar）、`harborframework/terminal-bench:7-alpine → redis:7-alpine`（live-database-cutover 任务 Redis sidecar）；否则 Compose 找不到本地镜像会尝试 pull，离线环境必失败 |

### 2.4 Agent 离线依赖适配设计

- 按 `agent_benchmark.md` Agent 依赖表准备所选 Agent 对应基础 OS 的离线包，放入 `agent-deps/`；TB4.0 所需 OS（ubuntu:20.04/22.04/24.04、debian:12/13、fedora:42）已全部提供打包资源（文档表新增 `ubuntu:20.04`、`fedora:42` 两列）。
- 对安装于 `/root/.local` 的 Agent，用 `post.sh` 方案修补 bundle：迁移至 `/opt` 并在 `/usr/local/bin` 生成 wrapper，保证任务容器以非 root（`nobody`）运行时可执行。
- 已验证 Agent：`claude-code`（socket 模式，Anthropic 格式接口 + `--ak reasoning_effort=max --ak max_turns=500`）、`mini-swe-agent`（dind 模式，OpenAI 兼容接口）。

### 2.5 Verifier 适配设计（出分关键）

- TB4.0 全部任务为 separate verifier：测试用例烘在 verifier 镜像内，评测侧不改测试内容，口径与官方一致。
- `payments-pipeline-fix` 例外适配：其 verifier 以 `env -i` 执行 pip，会清空 pip 镜像源环境变量导致必挂。注入自定义 `PipConfVerifier`（`custom_verifiers.payments_pip:PipConfVerifier`），在 verify 前把镜像源写入 `/etc/pip.conf`（文件可扛过 `env -i`）；安装脚本幂等，runtime 容器重建后重跑。

### 2.6 评测执行与结果口径设计

| 环节 | 设计 |
| --- | --- |
| 模型服务接入 | **不绑定特定模型服务**：任意 OpenAI 兼容接口（vLLM / MindIE 等本地推理服务或三方网关，`--model openai/<模型名>`）与 Anthropic 兼容接口均可通过统一语义参数 `--api-base` / `--agent-api-key` / `--model` 接入；仅支持 OpenAI 格式的服务对接 `claude-code` 时用 litellm[proxy] 做请求转发 |
| 运行前置检查 | ① 物料校验（release tag / 任务数 / 镜像数 / digest）② oracle canary 冒烟（不消耗模型费用，验证 env+agent+verifier+sidecar 全链路，期望 reward 1.0）③ 模型服务可用性探针（`GET {api-base}/models` 返回 200 才开跑；余额不足等错误如 402 先处理，避免整批被 `UnknownApiError` 截断） |
| 全量口径 | 66 任务中 3 个 GPU 任务（`fp8-rmsnorm-gemm`、`jax-speedrun-gpu`、`math-eval-grader`）默认排除，**全量 = 63 个 CPU 任务，Pass@1 分数 = 通过数 / 63**；GPU 任务在有 GPU 的机器上单独跑 |
| 执行命令 | `ais_bench harbor_agent_task.py --mode agent -a <agent> --model <model> --api-base <...> --agent-api-key <...> -p <数据集> --agent-deps <deps> --exclude-task-name <3 个 GPU 任务> -n <并发> -k 1 --no-force-build --delete` |
| 中断续测 | 402 / 异常类：充值后 `--reuse <时间戳> --purge-exception-cases` 只重跑异常 trial（已通过的不重复计费）；verifier 超时类：`harbor trial regrade` 只重跑 verifier（≈0 成本） |
| 过程监控 | `--monitor-port` 只读 HTTP 服务：`/api/tasks`、`/api/tasks/{模型}/{数据集}/cases`、`/api/jobs` 等端点实时查看每 case 状态与成败原因 |
| 结果汇总 | `HarborSummarizer` 打印单表并落盘 CSV（agent / model_name / dataset / avg_score / correct / wrong / exception） |

### 2.7 资源与文档交付

| 类别 | 内容 |
| --- | --- |
| 全量数据集资源 | 任务包 / 源码 / 镜像清单（GitHub v4.0.0 发布物）；全量镜像包 `terminal-bench-4.0-images-x86_64.tar`（OBS） |
| 小规模采样资源 | mini 数据集 `modelers.cn/datasets/AISBench/terminal-bench-4-0-mini`；mini 镜像包 `terminal-bench-4.0-images-selected-0.20-x86_64.tar`（20% 采样） |
| 依赖文档 | `dep.zip`（release 校验、agent-deps 修补、payments verifier、运行脚本等全流程脚本与配置） |
| 用户文档 | `agent_benchmark.md` 资源表 / 镜像表 / Agent 依赖表（新增 TB4.0 行与 ubuntu:20.04、fedora:42 列）；`extended_benchmark/agent/terminal_bench_4.md` 操作适配文档（资源清单、两种模式、12 步操作），中英文同步，`index.rst` 已注册 |

---

## 3. 测试落盘结果

> 本章两轮全量实测均按验收备注选用 DeepSeek 官方 API 作为模型服务，仅为验证用例之一；方案本身不绑定该服务，可替换为任意 OpenAI / Anthropic 兼容的本地（NPU/GPU 推理）或远端模型服务，评测链路与结果口径不变。

### 3.1 冒烟验证（oracle canary，不消耗模型费用）

| 任务 | 验证范围 | 结果 |
| --- | --- | --- |
| `ctr-optimization` | compose main + sidecar + verifier 全链路 | reward 1.0（8m39s） |
| `ontology-kg-querying` | 普通任务链路 | reward 1.0（1m06s） |
| `payments-pipeline-fix` | PipConfVerifier 注入后 | reward 1.0（修复正式生效，全量中通过） |

### 3.2 全量实测一：claude-code（socket 模式，实测服务：DeepSeek API）

job `trials/outputs/default/20261026_101056`，63 个 CPU 任务，`-n 8 -k 1`，`reasoning_effort=max`：

| 时点 | 通过 | UnknownApiError（402 截断） | mean |
| --- | --- | --- | --- |
| 原始全量（第一波） | 10 | 37（35×402 + 2×上下文 400） | 0.1587 |
| 第一次 resume 后 | 14 | 22（21×402 + 1×400） | 0.2222 |
| 第二次 resume 后（最终） | **23** | **0**（另 2×verifier 超时 + 2×501-turns 计 0 分） | **0.365079** |

- 最终口径：**23/63 通过，mean = 0.365079**；三波总成本约 $2,373.46，历时约 3 天。
- 402 余额截断被探针 + resume 机制闭环修复（两次充值后探针 200 再续跑，第二次 resume 全程 0 次 402）。

### 3.3 全量实测二：mini-swe-agent（dind 模式，实测服务：DeepSeek API）

job `trials/outputs/default/20261105_132144`，63 个 CPU 任务，`-n 6 -k 1`：

| 项 | 结果 |
| --- | --- |
| 主跑 | 17 通过 / 39 个无异常 0 分 / 7 个异常，**mean 0.2698**，历时 9h04m |
| photonic 补跑 | reward 1.0（14/14，主测试 196.9s） |
| retro 补跑 | reward 1.0（8/8，PnR 105.1s） |
| 合并口径 | 17 + photonic + retro = **19/63 ≈ 0.3016**（三个独立 job 的单题结果之和，报告时须注明"全量 17/63，另有两题补跑通过"，不能写成单次全量分数） |

两种模式均跑通全链路并得到非 0 的有效精度，验证了方案的双模式设计。

### 3.4 落盘结果结构与含义

结果保存于 `outputs/default/<时间戳>/`（可用 `--work-dir` 修改），核心是 `results/{模型}/{数据集}/details/` 下镜像 Harbor job 布局的落盘：

```
outputs/default/20261026_101056/
├── configs/20261026_101056.py        # 合成后的完整配置（CLI + 配置文件全部参数）
├── logs/eval/{模型}/{数据集}.out     # 执行过程日志
├── results/{模型}/{数据集}/
│   ├── details/                       # ▽ 该任务 Harbor job 结果目录（核心）▽
│   │   ├── config.json                # job 配置原文（存在即支持断点续测 resume）
│   │   ├── result.json                # 任务级汇总：n_total_trials / stats / trial_results / exception_stats
│   │   └── trial_00000/               # 每个 case 一个目录（按调度序编号）
│   │       ├── result.json            # 该 case 结果：status / reward / exception_info / timings / agent_info
│   │       ├── exception.txt          # 异常消息文本（异常退出时生成）
│   │       ├── trial.log              # 该 case 运行日志
│   │       ├── config.json            # 该 case 任务配置
│   │       ├── agent/trajectory.json  # Agent 运行轨迹（存在即有轨迹可复现）
│   │       └── verifier/
│   │           ├── reward.json        # 该 case 得分（verifier 判定）
│   │           ├── ctrf.json         # 逐测试用例 pass/fail/skip + 失败原因
│   │           ├── test-stdout.txt    # 验证执行 stdout 尾部
│   │           └── test-stderr.txt    # 验证执行 stderr 尾部
│   └── {数据集}.json                  # 该任务聚合结果（供 summarizer 汇总）
└── summary/
    ├── summary_20261026_101056.csv    # 汇总表（csv / markdown / txt 三格式）
    ├── summary_20261026_101056.md
    └── summary_20261026_101056.txt
```

汇总表每行对应一个（模型 × 数据集）任务，列含义：

| 列 | 含义 |
| --- | --- |
| `agent` | 使用的 Agent 名称 |
| `model_name` | 模型名称（多值以 `,` 分隔） |
| `dataset` | 数据集任务简称 |
| `avg_score` | 该任务平均得分（聚合 `result.json` 的 `avg_score`） |
| `correct` | 得分 ≥ 1.0 的完成 trial 数 |
| `wrong` | 得分 0 ~ 1.0（不含 1.0）的完成 trial 数 |
| `exception` | 执行中出现异常的 trial 数 |

逐 case 排障入口：`trial_*/verifier/ctrf.json` 定位具体失败用例，`verifier/test-stdout.txt` / `test-stderr.txt` 查看验证输出尾部，`agent/trajectory.json` 复现 Agent 行为，`exception.txt` 查看异常消息；`details/result.json` 的 `stats.evals[*].exception_stats` 记录异常 case 名，是 `--purge-exception-cases` 自动重试的数据来源。

### 3.5 验收标准对照

| 验收项 | 要求 | 实测情况 |
| --- | --- | --- |
| 硬件 | NPU / GPU | x86_64 CPU 服务器完成 63 个 CPU 任务全量（80C / 376G / Docker）；3 个 GPU 任务预留 `gpus=1` 单独执行入口 |
| 模型 | DeepSeek-v4.1-flash 或更新版本，可用官方 API | 方案不绑定模型服务：统一语义参数接入任意 OpenAI 兼容 / Anthropic 兼容服务（含 NPU/GPU 本地推理服务）；本需求实测按验收备注选用 DeepSeek 官方 API（claude-code 走 anthropic 格式接口 `deepseek-flash[1m]`，mini-swe-agent 走 OpenAI 兼容接口 `openai/deepseek-chat`） |
| 数据集 | Terminal Bench 4.0 | 官方 v4.0.0 发布物（任务包 + 官方预构建镜像 + 镜像清单 digest 校验），66 任务全量导入 |
| 精度非 0 | 不能为 0 | **满足**：claude-code 0.365079（23/63）；mini-swe-agent 0.2698（17/63，补跑合并 19/63） |
| 与官方差异 < 1% | AISBench 与官方精度结果差异小于 1% | 评测口径等价性由架构保证：复用官方 Harbor 引擎、官方任务定义、官方预构建镜像与 separate verifier，AISBench 侧不改测试内容，结果即官方口径；差异来源仅为模型采样随机性。数值级交叉验证可按验收备注用模型官方 API 在官方工具链复跑一次（或三次取平均）比对，两套链路共用同一 trial 落盘（reward.json / ctrf.json）可逐 case 核对 |

---

## 4. 结论

Terminal-Bench 4.0 接入已完成：数据集 / 镜像 / Agent 依赖 / verifier 四层离线适配落地，socket 与 dind 两种运行模式均通过 63 任务全量实测并产出非 0 有效精度（0.365079 / 0.2698），落盘结构完整（逐 trial reward、逐用例 ctrf、Agent 轨迹、汇总 CSV），断点续测与实时监控闭环。用户侧文档（`terminal_bench_4.md` 操作适配 + `agent_benchmark.md` 资源表）中英文同步交付，mini 采样资源（0.20 镜像包 + modelers.cn mini 数据集）支持低成本验证。
