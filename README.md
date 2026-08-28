<!-- generated-by: gsd-doc-writer -->
# AgileAgent 灵动智能体

AgileAgent 是面向 IR/SAR 图像的可审计 4+2 目标检测与类别增量学习系统，提供 x86/CUDA 训练、Web/CLI 推理和可直接物化的 Ascend310B v2 三模型发布包。

## 当前正式版本

系统识别六个全局类别：

| 阶段 | 全局 ID | 类别 | 模型 owner |
| --- | --- | --- | --- |
| Base | `0–3` | soldier、small_aircraft、warship、tank | 四类 Base YOLO26s |
| Round 1 | `4` | patrol_boat | 增量专家 |
| Round 2 | `5` | armored_vehicle | 增量专家 |

两个运行平台共享类别 ID 和 owner 契约，并采用各自冻结的推理参数：

| 平台 | 正式模型组合 | 运行资产 |
| --- | --- | --- |
| x86/CUDA | 四类 Base YOLO26s、二类 Incremental YOLO26s、Scene-SensorNet | [`models/production/incremental_detection/`](models/production/incremental_detection/) 与 [`models/context/`](models/context/) |
| Ascend310B v2 | Base、Incremental、Scene-SensorNet 三个独立 OM | [`20260824-4plus2-yolo26-runtime-calibration-v1`](models/ascend310b/full-score/20260824-4plus2-yolo26-runtime-calibration-v1/README.md) |

Scene-SensorNet 输出 IR/SAR 传感器概率和 air、forest、sea、urban 闭集场景概率。x86/CUDA 使用场景概率调整六类逐类有效阈值；Ascend310B v2 先并发执行 Base 与 Scene，并根据场景和 Base 检测证据决定是否执行 Incremental OM。在线门控由图像内容、场景概率和检测结果共同驱动。

## 冻结指标

### x86/CUDA production

一号结果是 75 张 Base lock 与 14 张 Increment lock 组成的独立 mixed lock；二号结果覆盖 Base/Increment 的 train、dev、lock 全部 890 张标注图像，包含训练图像，只作拟合与错误诊断。

| 结果 | Base mAP50 | New-mAP50 | KRR | Full-mAP50 | TP / FP / precision | 口径 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 一号 | `0.845782` | `0.750368` | `0.997179` | `0.800605` | `335 / 108 / 0.756208` | 独立 mixed lock，正式结果 |
| 二号 | `0.888703` | `0.737477` | `1.004975` | `0.827444` | `3533 / 986 / 0.781810` | 全部 890 张，诊断结果 |

Scene-SensorNet 在 mixed lock 上的 sensor / scene / joint accuracy 为 `0.988764 / 0.831461 / 0.820225`。双口径及逐类错误结果位于 [`all_images_diagnostics.md`](models/production/incremental_detection/evidence/all_images_diagnostics.md)；原始冻结运行点保存在 [`metrics.json`](models/production/incremental_detection/metrics.json)，production 代际由 [`models/generations.json`](models/generations.json) 登记。包括硬门禁、逐类 P/R/AP、误激活、场景准确率和 310B FPS 的统一账本见 [`docs/current-metrics.md`](docs/current-metrics.md)。

### Ascend310B v2 release

当前板端运行时 release 为 `20260824-4plus2-yolo26-replica-pool-v1`，沿用 `20260824-4plus2-yolo26-runtime-calibration-v1` 的冻结三模型资产。它使用 `608×736` AIPP 输入，两个 YOLO26s 检测 OM 均输出 `[1,300,6]`，并通过三个同构实例并行处理批次；模型、阈值、场景门控和融合规则均未修改：

| 指标 | 结果 |
| --- | ---: |
| Base mAP50 | `0.816663` |
| New-mAP50 | `0.611461` |
| Full-mAP50 | `0.722005` |
| KRR | `1.000000` |
| 新类误激活 | `17/75 = 0.226667` |
| 公共 `8501` 全流程 aggregate FPS（两次独立复测） | `31.9616 / 32.6565` |
| 旧 engine-only mixed 20 图中位 FPS（仅诊断） | `38.2175` |
| 旧 engine-only 纯增量 140 图中位 FPS（仅诊断） | `37.3997` |

上表是当前真实 OM lock 与正式公共入口结果。正式 FPS 按两次 `30 + 3×20` 复测的“60 帧 ÷ 三轮全流程总墙钟耗时”计算，计入 PNG 解码、Scene、决策、Base/Incremental 检测、后处理和同 stem 六列结果写出；两次均通过 30 FPS 门禁。旧的 `38.x FPS` 只保留为 engine-only 历史诊断，不能再作为官方成绩。新旧 runtime 的冻结输出完全一致，误激活较上一代 `35/75` 降至 `17/75`。仓库内模型资产仍由 [`runtime-calibration-v1`](models/ascend310b/full-score/20260824-4plus2-yolo26-runtime-calibration-v1/README.md) 提供，历史 engine-only 证据见 [`reports/ascend310b/20260824-replica-pool-v1/`](reports/ascend310b/20260824-replica-pool-v1/README.md)，完整状态见 [`docs/current-metrics.md`](docs/current-metrics.md)。

## 安装

x86/CUDA 工作台要求 x86-64 Linux 或 WSL、可用的 NVIDIA 驱动，以及 Python `>=3.10,<3.13`。引导脚本会创建或复用符合版本要求的环境，准备 CUDA PyTorch、Ultralytics、Web 工作台与开发依赖，并登记 `agile-agent` 命令。

```bash
git clone https://github.com/Sakauma/AgileAgent.git
cd AgileAgent
chmod +x scripts/bootstrap_x86.sh scripts/start_agent.sh
./scripts/bootstrap_x86.sh
```

已有符合版本要求的 Python 环境时，可以通过 `AGILE_AGENT_PYTHON=/path/to/python` 指定解释器。

`scripts/start_agent.sh` 是统一启动入口：它通过系统架构自动选择运行栈。`x86_64/AMD64` 使用 `configs/agent_pipeline.yaml`、CUDA 与 `.pt` 模型；`aarch64/ARM64` 使用 `configs/agent_pipeline_ascend310b.yaml`、PyACL 与 `.om` 模型。ARM 默认复用 `/usr/local/miniconda3/envs/agileagent` 和现有 CANN 环境，不安装或替换驱动、CANN、CUDA 或 PyTorch。可用顶层 `--config` 或 `AGILE_AGENT_CONFIG` 显式覆盖自动选择。

## 快速开始

1. 在 SSH 或本地终端进入 CLI 主界面：

   ```bash
   ./scripts/start_agent.sh --cli
   ```

   完成引导安装后也可以直接运行 `agile-agent`。不带子命令时会自动进入视觉识别终端，首屏提供单图识别、目录批量识别、最近结果、运行状态和模型信息。

2. CLI 会自动识别 `x86 · CUDA` 或 `ARM · Ascend`，并自动保存每次识别的标注图、JSON、CSV、预测 TXT 和终端摘要。模型、阈值、场景门控与增量专家均绑定当前 production，不需要人工设置。

3. 需要浏览器界面时再启动 Web 工作台：

   ```bash
   ./scripts/start_agent.sh
   ```

   浏览器打开 `http://127.0.0.1:8501`。完整 CLI 指南见 [`docs/CLI.md`](docs/CLI.md)。

## 使用示例

查看当前 production 代际、模型和运行状态：

```bash
agile-agent status --format json --refresh
```

对单张图像执行 Scene-SensorNet、Base 和 Incremental 组合推理。默认输出易读的终端表格，并在 `runs/cli_detections/` 下自动创建独立结果目录：

```bash
agile-agent detect --source /path/to/image.png
```

目录批量识别和机器可读输出：

```bash
agile-agent detect --source /path/to/images --recursive
agile-agent detect --source /path/to/image.png \
  --output /path/to/empty-result-dir --format json
```

检测命令优先复用本机已验证的正式服务；服务未运行时才加载本地引擎。两条路径使用相同 production 代际、场景门控和冻结阈值。

Web 服务启动后，可通过 HTTP 检查健康状态并提交图像：

```bash
curl -fsS http://127.0.0.1:8501/api/health
curl -fsS -F "file=@/path/to/image.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

## 4+2 数据与增量协议

固定划分位于 [`splits/strict_4plus2/`](splits/strict_4plus2/README.md)：

| 数据 | train | dev | lock | 总计 |
| --- | ---: | ---: | ---: | ---: |
| 四类 Base | 600 | 75 | 75 | 750 |
| 二类 Increment | 112 | 14 | 14 | 140 |
| Round 1 patrol_boat | 56 | 7 | 7 | 70 |
| Round 2 armored_vehicle | 56 | 7 | 7 | 70 |

[`configs/incremental_round_registry_4plus2.yaml`](configs/incremental_round_registry_4plus2.yaml) 定义 patrol_boat → armored_vehicle 的两轮注入顺序、局部到全局类别映射和父子代际。每轮增量训练使用该轮 Increment train/dev，Base 和已经学习的专家保持冻结；系统级校准负责 Scene-SensorNet、场景先验和门控参数；冻结后的 joint evaluation 在累计类别 lock 上输出 New-mAP50、KRR 与 Full-mAP50。

当前 x86 production 通道绑定 `incremental_detection_generation_4plus2`，增量成员为二类 YOLO26s 专家。两轮源码流程会为每个新增类别生成独立候选、冻结评分和代际登记，再由完整轮次证据驱动 production 晋级。

主要入口：

| 阶段 | 工具 |
| --- | --- |
| 数据划分复核 | `tools/03_split_r2_4plus2.py`、`tools/11_prepare_incremental_round_splits.py` |
| Base 训练与选模 | `tools/04_train_base_4plus2.py`、`tools/05_select_base_4plus2.py` |
| 逐轮增量训练与选模 | `tools/06_train_incremental_4plus2.py`、`tools/07_select_incremental_4plus2.py` |
| 累计评测、登记与汇总 | `tools/08_evaluate_4plus2.py`、`tools/13_register_incremental_round_candidate.py`、`tools/12_summarize_incremental_rounds.py` |
| Scene-SensorNet 与系统校准 | `tools/60_train_scene_sensor.py`、`tools/61_select_scene_sensor_4plus2.py`、`tools/09_optimize_scene_aware_4plus2.py` |
| production 晋级 | `tools/10_promote_scene_aware_4plus2.py` |
| 一号 lock 与二号全量诊断 | `tools/14_evaluate_all_images_4plus2.py` |

训练默认采用 `1280` 输入、最多 `500` epoch 和 `50` epoch 无改善早停。数据边界、冻结规则和逐轮证据格式见 [`docs/compliant-incremental-learning.md`](docs/compliant-incremental-learning.md)。

### 现场 4+2+n 一键增量

若现场继续提供真正的新类别 ZIP，可在 CUDA 节点先执行只读预检，再用同一入口自动完成导入、类别注册、训练、累计 lock、FPS、候选部署和代际切换：

```bash
agile-agent incremental onsite --bundle /path/to/onsite_increment.zip --plan-only
agile-agent incremental onsite --bundle /path/to/onsite_increment.zip --target x86
```

本轮新类自动分配到全局 `6...`，Base、现有二类专家和历史专家保持冻结。入口同时接受标准 `images/labels + data.yaml` 与赛题现有“图像/标签平铺 + classes.txt”ZIP。CUDA 节点负责新检测专家训练；Ascend310B 部署编排连续完成 ONNX/OM 构建、隔离候选、累计精度、30 FPS、原子提升和回滚。系统以全部精度与 FPS 门禁作为 production 切换条件。完整数据格式、命令、门禁、状态文件和板端流程见 [`docs/onsite-4plus2plusn.md`](docs/onsite-4plus2plusn.md)。

## Ascend310B 板端增量训练与离线演示

[`extras/ascend_edge_incremental/`](extras/ascend_edge_incremental/) 提供隔离的板端轻量增量训练能力。它按类别注册表逐轮读取 Increment train/dev，冻结现有 Base、Incremental 与 Scene-SensorNet，只在 `npu:0` 更新每个新增类别 8 个参数的无 MatMul 置信度 Adapter；随后独立执行 mixed dev 强度选择、mixed lock 联合评分、ONNX/OM 导出和 ACL 延迟验证。

该功能使用独立 Conda 前缀和运行目录，将训练产物、候选配置与当前 production 完整隔离。2026-08-25 底层流水线实测两轮训练搜索耗时约 9 分 16 秒，NPU 探测、训练和 ONNX/OM 导出合计约 12 分 12 秒；mixed lock 的 Base mAP50 / New-mAP50 / KRR / Full-mAP50 为 `0.816663 / 0.649306 / 1.000000 / 0.736421`，Adapter OM 的保守串行投影为 `37.9639 FPS`。

现场演示当前 `4→4+2` 时，在已预装离线训练环境的 310B 上只需一条命令：

```bash
./scripts/run_ascend310b_incremental_demo.sh /path/to/datasets_r2_inc_train
```

该入口自动完成数据对齐、两轮 NPU 训练、dev/lock、ONNX/OM、隔离演示部署，并在 Adapter 接入运行时后复测完整图像链路 FPS。脚本启用强制离线配置，训练数据范围固定为 Increment train/dev，Base 图像只参与冻结后的联合评分；验收结果写入独立演示通道，当前满分 production 保持可随时启动。完整现场手册见 [`docs/ascend-310b-offline-incremental-demo.md`](docs/ascend-310b-offline-incremental-demo.md)；底层环境与实测证据见 [`docs/ascend-310b-edge-incremental-training.md`](docs/ascend-310b-edge-incremental-training.md)。

2026-08-26 板端整链验收已通过：Base mAP50 `0.816663`、New-mAP50 `0.624935`、KRR `1.000000`、Full-mAP50 `0.726497`；当时记录的 Adapter `38.6995 FPS` 是未包含正式结果落盘的 legacy engine-only 诊断，不能替代 2026-08-28 启用的新官方全流程口径。热态一键流程耗时 `16分47秒`，首次冷态建议预留 30 分钟。

## Ascend310B v2 部署

板端需要已配置的 CANN 运行时和 `/usr/local/miniconda3/envs/agileagent` 环境。仓库中的正式包已经包含三个 OM、配置、源 checkpoint、ONNX、AIPP、构建 provenance、冻结预测和验收报告。

在仓库根目录物化并启动 release：

```bash
./scripts/materialize_ascend310b_full_score_release.sh

RELEASE=/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
```

服务健康响应包含 `backend: ascend_acl`、`model_layout: independent_yolo26_e2e_v1`、`context_mode: model` 和 `validated: true`。评分、性能复验、服务安装和路由操作见 [`docs/ascend-310b-deployment.md`](docs/ascend-310b-deployment.md)。

## 配置与验证

默认使用 `auto` 配置选择：x86 加载 [`configs/agent_pipeline.yaml`](configs/agent_pipeline.yaml)，ARM 加载 [`configs/agent_pipeline_ascend310b.yaml`](configs/agent_pipeline_ascend310b.yaml)。常用检查命令：

```bash
agile-agent config validate
agile-agent config get inference.backend
agile-agent doctor
python scripts/verify_release.py
python scripts/smoke_models.py --load-only
```

`config validate` 输出同时包含检测到的机器架构、后端、模型格式和配置选择来源。需要固定平台配置时使用 `agile-agent --config PATH ...`；显式配置不会被自动替换。

完整测试命令和设备要求见 [`docs/TESTING.md`](docs/TESTING.md)。

## 项目结构

```text
fair_agent/      Python 核心包、推理后端、业务模块、CLI 与 Web 服务
configs/         x86/Ascend 配置、增量策略和两轮类别注册表
models/          production 权重、代际注册表、冻结证据与 Ascend v2 release
splits/          strict 4+2 固定数据清单
tools/           数据处理、训练、选模、校准、评分和发布工具
scripts/         环境引导、服务启动、OM 构建与板端验收脚本
tests/           单元、集成和发布契约测试
docs/            架构、配置、开发、测试、协议和部署文档
extras/          与 production 隔离的可选能力和实验入口
```

## 文档

| 文档 | 内容 |
| --- | --- |
| [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md) | 环境准备与首次运行 |
| [`docs/CLI.md`](docs/CLI.md) | SSH 交互主界面、单图/批量识别、结果保存与取回 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 三模型架构、组件边界与数据流 |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | x86/CUDA 与 Ascend 配置 |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | 开发流程和代码规范 |
| [`docs/TESTING.md`](docs/TESTING.md) | 测试范围、命令和设备要求 |
| [`docs/API.md`](docs/API.md) | Web API、增量工作台接口和运行时控制接口 |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | x86、WSL 与 Ascend310B 部署总览 |
| [`docs/current-metrics.md`](docs/current-metrics.md) | x86、Ascend 与板端增量演示指标总账 |
| [`docs/functional-models.md`](docs/functional-models.md) | Scene、Base、Incremental 三模型职责与协作策略 |
| [`docs/incremental-workbench.md`](docs/incremental-workbench.md) | 数据上传、审计、训练、生命周期与代际操作 |
| [`docs/compliant-incremental-learning.md`](docs/compliant-incremental-learning.md) | 两轮增量数据与评测契约 |
| [`docs/onsite-4plus2plusn.md`](docs/onsite-4plus2plusn.md) | 现场新类别一键训练、候选部署、验收与回滚 |
| [`docs/ascend-310b-offline-incremental-demo.md`](docs/ascend-310b-offline-incremental-demo.md) | 310B 断网一键 `4→4+2` 训练、演示部署与验收 |
| [`docs/ascend-310b-edge-incremental-training.md`](docs/ascend-310b-edge-incremental-training.md) | 板端轻量增量训练环境、操作、指标与边界 |
| [`docs/ascend-310b-full-score-method.md`](docs/ascend-310b-full-score-method.md) | Ascend310B v2 模型转换、内容门控与评分方法 |
| [`docs/ascend-310b-current-status.md`](docs/ascend-310b-current-status.md) | 当前 release 状态和证据索引 |
| [`docs/ascend-310b-deployment.md`](docs/ascend-310b-deployment.md) | 板端物化、启动、验收和路由操作 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 开发、测试、文档与提交协作规范 |

## 许可证

本项目采用 [MIT License](LICENSE) 开源。
