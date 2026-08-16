# AgileAgent 灵动智能体

AgileAgent 面向多模态目标检测与小样本增量学习，提供 Web 工作台、CLI、模型代际管理、数据审计和 Ascend 310B 推理链路。当前 production 由冻结基础检测器、增量检测器和 Scene-SensorNet 组成，每张无标签图像都经过完整模型组合与框级融合。

## 已实现能力

| 能力 | 当前实现 |
| --- | --- |
| 多模态目标检测 | IR/SAR 图像统一进入 YOLO 检测链路，输出全局类别 ID、边界框和置信度。 |
| 类别增量学习 | 上传五列 YOLO 数据后自动完成审计、拆分、训练、校准、复核、代际登记和 production 切换。 |
| 场景与传感器认知 | Scene-SensorNet 输出 IR/SAR 与 air/forest/sea/urban 概率，并作为逐类软阈值证据。 |
| 模型代际管理 | `models/generations.json` 记录父子代际、类别所有权、权重身份、阈值和评测指标。 |
| Web 与 CLI | Web 支持检测、批量检测和增量工作台；CLI 支持状态、检测、配置、日志、实验与代际操作。 |
| Ascend 310B | PyACL/AscendCL 后端加载三个 OM，完成预处理、三模型编排、后处理、融合和 API 响应。 |
| 审计证据 | 数据血缘、任务状态、模型哈希、预测记录、指标和运行事件形成可追踪证据链。 |

## Production 模型组合

| 成员 | 职责 | 当前绑定 |
| --- | --- | --- |
| Scene-SensorNet | 传感器与已知场景认知 | `models/context/scene_sensor_net.pt` |
| 三类基础检测器 | soldier、small_aircraft、tank | `models/production/incremental_detection/three_class_base_detector.pt` |
| 增量检测器 | warship，全局类别 ID 2 | `models/production/incremental_detection/incremental_detector.pt` |

在线推理按 production 代际解析模型成员。基础检测器与增量检测器共同处理每张图像，Scene-SensorNet 提供软上下文，最终结果经过类别所有权映射、逐类阈值、冲突仲裁和 class-aware NMS。

## 赛题评选标准

赛题总分为 100 分，由性能指标评分 60 分和主观设计方案评分 40 分组成。性能指标在华为昇腾 310B 系列嵌入式计算平台上，依据智能体系统在给定数据集上的实际运行结果评分。

### 性能指标评分：60 分

#### 1. 基础目标检测识别性能：30 分

基础测试集采用基础类别样本，以平均精度均值 mAP 评估多类别目标检测识别性能。

| mAP | 分值 |
| ---: | ---: |
| ≥ 0.80 | 30 |
| ≥ 0.70 | 25 |
| ≥ 0.65 | 20 |
| ≥ 0.60 | 15 |
| ≥ 0.50 | 10 |
| ≥ 0.40 | 5 |
| < 0.40 | 0 |

#### 2. 增量学习性能：20 分

测试分为基础学习阶段和多轮增量学习阶段。基础阶段使用基础类别训练集完成模型训练；赛题鼓励在端侧完成增量学习，增量阶段使用当轮增量数据集，每轮注入若干新类别，每个新类别提供少量标注样本。评估集覆盖截至当轮全部已学习类别，包括旧类别与新类别。

新类别识别精度 New-mAP 占 10 分，衡量模型对新增类别的学习效果：

| New-mAP | 分值 |
| ---: | ---: |
| ≥ 0.60 | 10 |
| ≥ 0.50 | 7 |
| ≥ 0.40 | 4 |
| < 0.40 | 0 |

旧类别知识保持率 KRR 占 10 分，计算公式为 `KRR = mAP_old_after / mAP_old_before`，衡量增量学习前后旧类别检测精度的保持程度：

| KRR | 分值 |
| ---: | ---: |
| ≥ 0.95 | 10 |
| ≥ 0.90 | 7 |
| ≥ 0.80 | 4 |
| < 0.80 | 0 |

#### 3. 端侧推理效率：10 分

端侧推理效率以华为昇腾 310B 平台完整处理单帧多模态数据的端到端帧率 FPS 评分。

| FPS | 分值 |
| ---: | ---: |
| ≥ 30 | 10 |
| ≥ 20 | 7 |
| ≥ 10 | 4 |
| < 10 | 0 |

### 主观设计方案评分：40 分

#### 1. 作品符合性：10 分

- 国内外发展现状调研分析全面，研究思路合理，技术路线符合赛题要求：5 分。
- 智能体系统架构设计：5 分。系统包含至少 3 个不同功能模型，并清晰说明模型分工、协同机制和信息传递路径。

#### 2. 作品完整性：5 分

源代码和可执行程序在华为昇腾 310B 平台稳定运行，并提供运行视频、总体方案设计报告、数据说明文档和程序说明（用户手册），各项文档完整、规范。

#### 3. 增量学习效率：10 分

考核新类别小样本注入后的快速学习能力与模型更新效率，包括训练迭代轮次、收敛速度、从样本注入到形成有效识别能力的响应周期，以及端侧有限算力下快速学习策略的设计。总体方案设计报告记录增量学习的时间开销、收敛速度和实测分析。

#### 4. 场景理解准确性：5 分

考核场景认知模型输出的准确性与合理性，包括地面、天空等场景类型，天时天候、噪声与干扰判断，以及语言表述的流畅度和合理性。总体方案设计报告展示场景认知模型的输出结果与分析。

#### 5. 创新性：10 分

- 增量学习策略、抗遗忘机制和小样本快速学习创新：5 分。
- 多模态信息融合、端侧轻量化部署和推理优化创新：5 分。

## 已验证指标

固定 3+1 基础数据实验使用 573 张训练源池、88 张开发源池和 89 张混合测试集：

| 指标 | 仓库 production |
| --- | ---: |
| Base mAP50 | `0.814142` |
| New-mAP50 | `0.638688` |
| KRR | `1.000000` |
| 新类 precision | `0.924528` |
| 旧类图误激活率 | `0.014286` |

Ascend 310B 正式 release 已完成三模型 OM 推理与 89 图复核：

| 指标 | 板端正式 release |
| --- | ---: |
| Base mAP50 | `0.819407` |
| New-mAP50 | `0.728761` |
| KRR | `1.000000` |
| 新类 precision | `0.933333` |
| 误激活率 | `0.014286` |
| 89 图平均引擎耗时 | `57.849 ms/图` |
| 89 图平均墙钟耗时 | `71.491 ms/图` |

AIPP staging 已完成 1,068 次真实 multipart PNG 请求，服务端均值为 `51.203 ms`，P95 为 `63.9 ms`，吞吐为 `19.53 FPS`。

Ascend 310B 的隔离满分候选进一步使用共享骨干双逻辑头、固定中性上下文和 batch fast path，在四项机器评分中得到 Base mAP50 `0.804901`、New-mAP50 `0.605033`、KRR `1.0`，20 图 batch 两次复核中位为 `30.066/30.080 FPS`。该候选尚未替换正式 `8501`；复现和新数据集阈值选择见 [`docs/ascend-310b-full-score-method.md`](docs/ascend-310b-full-score-method.md)。

## 快速开始

在 WSL/Linux 仓库根目录执行：

```bash
chmod +x scripts/bootstrap_x86.sh scripts/start_agent.sh
./scripts/bootstrap_x86.sh
./scripts/start_agent.sh
```

引导脚本准备 Python 3.10–3.12、CUDA PyTorch、项目依赖和命令入口，并执行发布校验与模型冒烟。启动脚本读取 `.agent-python` 中登记的解释器并启动 Web 工作台。

浏览器入口：

```text
http://127.0.0.1:8501
```

CLI 模式：

```bash
./scripts/start_agent.sh --cli
```

## 常用 CLI

```bash
agile-agent doctor
agile-agent status --format json --refresh
agile-agent detect --source path/to/image.png --confidence 0.50
agile-agent decide --source path/to/image.png
agile-agent logs --limit 100
```

配置管理：

```bash
agile-agent config validate --config configs/agent_pipeline.yaml
agile-agent config show --config configs/agent_pipeline.yaml --effective
agile-agent config get routing.conflict_iou
agile-agent config set routing.conflict_iou 0.50
agile-agent --set inference.confidence_default=0.60 serve
```

## 增量学习工作台

增量数据包采用图像与同 stem 五列 YOLO 标签：

```text
class x_center y_center width height
```

推荐目录：

```text
images/train/
images/val/
labels/train/
labels/val/
data.yaml
```

完整生命周期：

```bash
agile-agent incremental audit --batch /path/to/new_batch.zip
agile-agent incremental run --batch BATCH_ID
agile-agent incremental status --run-id TRAIN_JOB_ID
```

数据管理命令：

```bash
agile-agent incremental-data upload --archive /path/to/new_batch.zip --name 新批次
agile-agent incremental-data list
agile-agent incremental-data show --batch-id BATCH_ID
agile-agent incremental-data rename --batch-id BATCH_ID --class-names 新类别名称
agile-agent incremental-data inject --batch-id BATCH_ID
agile-agent incremental-data train --batch-id BATCH_ID
agile-agent incremental-data jobs --batch-id BATCH_ID
```

工作台保存源包、拆分清单、类别注册表、训练快照、权重、阈值、指标、任务日志和代际记录。训练数据范围固定为当前批次 train/dev，lock 在权重与阈值冻结后进入复核。

## 固定 3+1 数据协议

`splits/` 保存固定源池和模型清单：

| 清单 | 图片数 | 用途 |
| --- | ---: | --- |
| `pool_train.txt` | 573 | 训练源池 |
| `pool_dev.txt` | 88 | 开发源池 |
| `mixed_test.txt` | 89 | 四类混合测试集 |
| `strict_3plus1/base_train.txt` | 441 | 三类基础训练 |
| `strict_3plus1/base_dev.txt` | 70 | 三类基础开发 |
| `strict_3plus1/increment_train.txt` | 132 | 舰船增量训练 |
| `strict_3plus1/increment_dev.txt` | 18 | 舰船增量校准 |
| `strict_3plus1/base_test.txt` | 70 | 基础指标评分子集 |

重新生成：

```bash
python tools/02_split_dataset.py \
  --increment-class warship \
  --output-dir reports/splits_check
```

## 可复现实验

```bash
agile-agent experiment validate --config configs/incremental/warship_3plus1.yaml
agile-agent experiment run --config configs/incremental/warship_3plus1.yaml
agile-agent experiment reproduce \
  --manifest runs/experiments/warship_3plus1/RUN_ID/run_manifest.json
```

实验 manifest 记录配置、数据清单、模型身份、阈值、环境和指标，支持同一实验条件的复核。

## Ascend 310B

板端配置位于 `configs/agent_pipeline_ascend310b.yaml`，正式服务使用命名环境 `agileagent` 与 PyACL/AscendCL：

```bash
./scripts/start_agent_ascend310b.sh
curl -fsS http://127.0.0.1:8501/api/health
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

正式板端模型产物位于 release 目录的 `om/`，配置记录三个回滚 OM 的路径与 SHA256。满分候选由一个共享双逻辑头 OM 和一个已加载但在 fixed-neutral 正常路径不执行前向推理的 context 回滚 OM 组成，候选始终使用 `8502`。

## 验证

```bash
python -m pytest -q
python scripts/verify_release.py
python scripts/smoke_models.py
```

发布校验状态为 `passed`；当前改动的实际回归结果以提交时的测试输出为准。

## 文档

| 文档 | 内容 |
| --- | --- |
| [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md) | 环境准备与首次启动 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 系统组件与数据流 |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | schema 3 配置与命令 |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | 开发流程与代码规范 |
| [`docs/TESTING.md`](docs/TESTING.md) | 测试范围与运行方式 |
| [`docs/ascend-310b-deployment.md`](docs/ascend-310b-deployment.md) | Ascend 310B 部署实现 |
| [`docs/ascend-310b-full-score-method.md`](docs/ascend-310b-full-score-method.md) | 满分候选结构、新数据集复现与阈值选优 |
| [`docs/ascend-310b-current-status.md`](docs/ascend-310b-current-status.md) | 当前状态与历史证据索引 |
| [`docs/ascend-310b-ssh-environment.md`](docs/ascend-310b-ssh-environment.md) | 板端环境与服务操作 |

## 项目结构

```text
fair_agent/          Python 核心包、后端、业务模块与 Web 服务
configs/             主配置、模型配置与实验配置
models/              发布模型、代际注册表与指标元数据
scripts/             环境准备、启动和发布校验
tools/               数据、实验、导出与板端验收工具
tests/               单元测试与集成回归测试
splits/              固定 573/88/89 源池与 3+1 清单
native_ascend/       Ascend C ABI 契约夹具
demo_artifacts/      脱敏演示状态
```
