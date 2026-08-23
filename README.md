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
| Ascend310B v2 | Base、Incremental、Scene-SensorNet 三个独立 OM | [`20260823-4plus2-yolo26-content-gate-v2`](models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/README.md) |

Scene-SensorNet 输出 IR/SAR 传感器概率和 air、forest、sea、urban 闭集场景概率。x86/CUDA 使用场景概率调整六类逐类有效阈值；Ascend310B v2 先并发执行 Base 与 Scene，并根据场景和 Base 检测证据决定是否执行 Incremental OM。在线门控由图像内容、场景概率和检测结果共同驱动。

## 冻结指标

### x86/CUDA production

评测集由 75 张 Base lock 和 14 张 Increment lock 组成。以下结果来自冻结的 production 权重与场景软阈值运行点：

| 指标 | 结果 |
| --- | ---: |
| Base mAP50 | `0.856067` |
| New-mAP50 | `0.773368` |
| Full-mAP50 | `0.794994` |
| KRR | `0.973126` |
| Scene-SensorNet sensor / scene / joint accuracy | `0.988764 / 0.831461 / 0.820225` |

完整结果位于 [`metrics.json`](models/production/incremental_detection/metrics.json)，production 代际由 [`models/generations.json`](models/generations.json) 登记。

### Ascend310B v2 release

正式 release `20260823-4plus2-yolo26-content-gate-v2` 使用 `608×736` AIPP 输入，两个 YOLO26s 检测 OM 均输出 `[1,300,6]`：

| 指标 | 结果 |
| --- | ---: |
| Base mAP50 | `0.825671` |
| New-mAP50 | `0.618859` |
| Full-mAP50 | `0.724927` |
| KRR | `1.000000` |
| 公共 `8501` 两次 batch 中位 FPS | `39.5726 / 39.5883` |

冻结评分见 [`validation/score.json`](models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/validation/score.json)，模型构建、完整性和板端性能证据随 release 一并保存。

## 安装

x86/CUDA 工作台要求 x86-64 Linux 或 WSL、可用的 NVIDIA 驱动，以及 Python `>=3.10,<3.13`。引导脚本会创建或复用符合版本要求的环境，准备 CUDA PyTorch、Ultralytics、Web 工作台与开发依赖，并登记 `agile-agent` 命令。

```bash
git clone https://github.com/Sakauma/AgileAgent.git
cd AgileAgent
chmod +x scripts/bootstrap_x86.sh scripts/start_agent.sh
./scripts/bootstrap_x86.sh
```

已有符合版本要求的 Python 环境时，可以通过 `AGILE_AGENT_PYTHON=/path/to/python` 指定解释器。

## 快速开始

1. 启动 Web 工作台：

   ```bash
   ./scripts/start_agent.sh
   ```

2. 在浏览器打开 `http://127.0.0.1:8501`。

3. 无浏览器环境可进入终端工作台：

   ```bash
   ./scripts/start_agent.sh --cli
   ```

## 使用示例

查看当前 production 代际、模型和运行状态：

```bash
agile-agent status --format json --refresh
```

对单张图像执行 Scene-SensorNet、Base 和 Incremental 组合推理；命令输出包含场景概率、检测框、类别 owner 和模型执行轨迹的 JSON：

```bash
agile-agent detect \
  --source /path/to/image.png \
  --confidence 0.10 \
  --profile incremental-detection
```

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

训练默认采用 `1280` 输入、最多 `500` epoch 和 `50` epoch 无改善早停。数据边界、冻结规则和逐轮证据格式见 [`docs/compliant-incremental-learning.md`](docs/compliant-incremental-learning.md)。

## Ascend310B v2 部署

板端需要已配置的 CANN 运行时和 `/usr/local/miniconda3/envs/agileagent` 环境。仓库中的正式包已经包含三个 OM、配置、源 checkpoint、ONNX、AIPP、构建 provenance、冻结预测和验收报告。

在仓库根目录物化并启动 release：

```bash
./scripts/materialize_ascend310b_full_score_release.sh

RELEASE=/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
```

服务健康响应包含 `backend: ascend_acl`、`model_layout: independent_yolo26_e2e_v1`、`context_mode: model` 和 `validated: true`。评分、性能复验、服务安装和路由操作见 [`docs/ascend-310b-deployment.md`](docs/ascend-310b-deployment.md)。

## 配置与验证

默认 x86 配置为 [`configs/agent_pipeline.yaml`](configs/agent_pipeline.yaml)，Ascend 正式配置为 [`configs/agent_pipeline_ascend310b.yaml`](configs/agent_pipeline_ascend310b.yaml)。常用检查命令：

```bash
agile-agent config validate --config configs/agent_pipeline.yaml
agile-agent doctor
python scripts/verify_release.py
python scripts/smoke_models.py --load-only
```

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
```

## 文档

| 文档 | 内容 |
| --- | --- |
| [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md) | 环境准备与首次运行 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 三模型架构、组件边界与数据流 |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | x86/CUDA 与 Ascend 配置 |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | 开发流程和代码规范 |
| [`docs/TESTING.md`](docs/TESTING.md) | 测试范围、命令和设备要求 |
| [`docs/compliant-incremental-learning.md`](docs/compliant-incremental-learning.md) | 两轮增量数据与评测契约 |
| [`docs/ascend-310b-full-score-method.md`](docs/ascend-310b-full-score-method.md) | Ascend310B v2 模型转换、内容门控与评分方法 |
| [`docs/ascend-310b-current-status.md`](docs/ascend-310b-current-status.md) | 当前 release 状态和证据索引 |
| [`docs/ascend-310b-deployment.md`](docs/ascend-310b-deployment.md) | 板端物化、启动、验收和路由操作 |

## 许可证

本仓库当前未声明软件许可证。
