# AgileAgent 灵动智能体

AgileAgent 面向多模态目标检测与小样本增量学习，提供 Web 工作台、CLI、模型代际管理、数据审计和 Ascend 310B 推理链路。production 保留冻结基础检测、增量检测和 Scene/Sensor 三个逻辑职责；x86 使用独立 PyTorch 模型组合，Ascend 正式主线使用三个独立 OM、真实 Scene-SensorNet 与双证据内容执行门控。

## 已实现能力

| 能力 | 当前实现 |
| --- | --- |
| 多模态目标检测 | IR/SAR 图像统一进入 YOLO 检测链路，输出全局类别 ID、边界框和置信度。 |
| 类别增量学习 | 上传五列 YOLO 数据后自动完成审计、拆分、训练、校准、复核、代际登记和 production 切换。 |
| 场景与传感器认知 | Scene-SensorNet 对已知场景 air/forest/sea/urban 与 IR/SAR 输出概率；当前 4+2 production 用场景概率对新旧六类执行逐类软阈值门控。 |
| 模型代际管理 | `models/generations.json` 记录父子代际、类别所有权、权重身份、阈值和评测指标。 |
| Web 与 CLI | Web 支持检测、批量检测和增量工作台；CLI 支持状态、检测、配置、日志、实验与代际操作。 |
| Ascend 310B | 4+2 YOLO26s 三-OM release 已在公共 `8501` 正式部署并完成两轮 FPS 复验；旧 listener 保留为即时回滚，`8502` 专用于后续候选。 |
| 审计证据 | 数据血缘、任务状态、模型哈希、预测记录、指标和运行事件形成可追踪证据链。 |

## Production 模型组合

当前 x86/CUDA 与 Ascend310B1 production 都实现 strict 4+2 的三个逻辑职责。x86 使用下列 `.pt`；Ascend 使用正式包内对应的 Base、Incremental 和 Scene 三个 OM。

| 成员 | 职责 | 当前绑定 |
| --- | --- | --- |
| Scene-SensorNet | 传感器与已知场景认知 | `models/context/scene_sensor_net.pt` |
| 四类 Base 检测器 | soldier、small_aircraft、warship、tank；全局类 0–3 | `models/production/incremental_detection/four_class_base_detector.pt` |
| 二类增量专家 | patrol_boat、armored_vehicle；全局类 4–5 | `models/production/incremental_detection/incremental_detector.pt` |

在线推理按 production 代际解析模型成员，类别 owner 固定。x86 继续执行六类场景软阈值门控；Ascend 为速度采用双证据执行门控：Base 与 Scene 先并发，只有 `air >= 0.5` 且 Base 同时检出 `small_aircraft` 时才跳过二类专家。两端线上都不读取文件名或真值标签。

当前 Ascend 正式模型包位于 [`models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/`](models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/README.md)。旧 2026-08-16 共享双头 3+1 包保持不可变，只作历史参考。

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

本工程统一将 `incremental_learning` 限定为“仅用当轮 Increment train/dev 训练新类检测器、完成新类映射及新类专属学习”，Base 检测器权重保持冻结。Scene-SensorNet 训练和六类场景门控搜索单列为 `system_calibration`：允许使用 Base/Increment train/dev 与 mixed dev，context lock 只做冻结功能模型复核，且全程不更新任何检测器权重。参数冻结后的六类 mixed lock/test 评分称为 `joint_evaluation`，只评估、不训练也不选参。完整契约见 [`docs/compliant-incremental-learning.md`](docs/compliant-incremental-learning.md)。

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

strict 4+2 冻结评估使用 75 张 Base lock 与 14 张 Increment lock，共 89 张 mixed lock。赛题检测门禁仍由 mAP50 及其保持率 KRR 决定；precision、FP 和误激活率是非阻断诊断。本次候选只在 mixed dev 上以 precision 为优化目标并受 mAP50/KRR 下限约束，参数冻结后才一次性复核 mixed lock：

| 指标 | 仓库 production |
| --- | ---: |
| Base lock mAP50 | `0.856067` |
| New-mAP50 | `0.773368` |
| patrol_boat AP50 | `0.691000` |
| armored_vehicle AP50 | `0.855735` |
| KRR | `0.973126` |
| Scene lock sensor / scene / joint | `0.988764 / 0.831461 / 0.820225` |

当前逐类基础阈值为 `0=.21, 1=.14, 2=.36, 3=.05, 4=.57, 5=.82`，最大场景惩罚为 `0=.15, 1=.88, 2=.26, 3=.19, 4=.65, 5=0`。有效阈值按 `基础阈值 + 最大惩罚 × (1 - 场景亲和度)` 计算并封顶为 `1.0`。小型飞机对应 air，舰船与巡逻艇对应 sea，人员、坦克和装甲车辆主要对应 forest/urban；这些对应关系来自训练正样本上的模型概率，而不是人工在线规则。

在 mixed lock 上，二类增量专家共有 `69 TP / 10 FP`，合并 precision 为 `0.873418`；patrol_boat 为 `29 TP / 1 FP / 0.966667 precision`、误激活率 `1/82 = 0.012195`，armored_vehicle 为 `40 TP / 9 FP / 0.816327 precision`、误激活率 `7/82 = 0.085366`。六类融合共有 `342 TP / 170 FP`，整体 precision 为 `0.667969`；89 张图中有 14 张至少发生一次六类误激活，旧运行点为 72 张。这里的 FP 是按类别、单图一对一匹配且 IoU `>=0.50` 后得到的错误检测框，包含重复框、定位不足、错类和负样本图上的检测；它不是错误图像数。逐类明细与复算口径见 [`models/production/incremental_detection/evidence/operating_point_diagnostics.md`](models/production/incremental_detection/evidence/operating_point_diagnostics.md)，dev 选择过程见 [`scene_aware_dev_search.md`](models/production/incremental_detection/evidence/scene_aware_dev_search.md)。

当前 4+2 Ascend310B1 正式 release 的四项硬指标全部进入满分档：

| 指标 | 板端正式结果 |
| --- | ---: |
| Base mAP50 | `0.825671` |
| New-mAP50 | `0.618859` |
| KRR | `1.000000` |
| Full-mAP50 | `0.724927` |
| 候选 20 图 batch 中位 FPS | `39.3468 / 39.4244` |
| 公共 `8501` 中位 FPS | `39.5726 / 39.5883` |

公共 `8501` 两轮逐轮结果为 `39.5726/39.5804/39.3933` 与 `39.5883/39.5023/39.6668 FPS`。诊断 precision 为 `0.677551`、recall 为 `0.661111`、误激活率为 `0.466667`，即 75 张不含新增类的图像中 35 张至少误激活一个新增类；这些诊断不阻断四项赛题硬指标。完整证据见 [`docs/ascend-310b-current-status.md`](docs/ascend-310b-current-status.md)。

## Ascend 310B 比赛满分方案

当前 production release 为 `20260823-4plus2-yolo26-content-gate-v2`，使用四类 Base YOLO26s、二类 Incremental YOLO26s 和真实 Scene-SensorNet 三个 OM。检测输入为 `608×736`，输出为 E2E `[1,300,6]`；Base 与 Scene 先并发，双证据门控决定是否执行二类专家。

零训练物化：

```bash
git clone https://github.com/Sakauma/AgileAgent.git
cd AgileAgent
./scripts/materialize_ascend310b_full_score_release.sh
```

新板直启：

```bash
RELEASE=/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
```

已有回滚 listener 的板使用固定拓扑：

| 职责 | 地址 | 说明 |
| --- | --- | --- |
| 公共入口 | `127.0.0.1:8501` | 精确路由到主实例 |
| 正式主实例 | `127.0.0.1:18501` | `independent_yolo26_e2e_v1`、`context_mode:model` |
| 回滚 listener | 物理 `127.0.0.1:8501` | 正常被路由旁路，删除规则即回滚 |
| 后续候选 | `127.0.0.1:8502` | 正式状态下空闲 |

当前方法入口：

| 阶段 | 入口 |
| --- | --- |
| Base/Incremental/Scene 训练与冻结评测 | `tools/04`–`tools/13`、`tools/60`–`tools/61` |
| 两个 YOLO26 E2E OM 构建 | `scripts/build_ascend_yolo26_e2e_oms.sh` |
| 4+2 候选物化 | `tools/112_materialize_ascend_yolo26_candidate.py` |
| 板端四项评分 | `scripts/run_ascend310b_score_gate.sh` |
| 正式 release 物化 | `tools/111_promote_ascend_full_score_release.py` |
| 原子提升与回滚 | `scripts/install_ascend310b_primary_services.sh`、`scripts/manage_ascend310b_primary_route.sh` |

只有 Base mAP50 `>=0.80`、New-mAP50 `>=0.60`、KRR `>=0.95` 且三轮 20 图 batch 中位 FPS `>=30` 才能提升。正式提升后还必须从公共 `8501` 复跑性能。完整转换、门控、评分和回滚说明见 [`docs/ascend-310b-full-score-method.md`](docs/ascend-310b-full-score-method.md) 与 [`docs/ascend-310b-deployment.md`](docs/ascend-310b-deployment.md)。
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

## 固定 4+2 数据协议

`splits/strict_4plus2/` 是当前正式清单；旧 3+1 清单仅作兼容副本，归档正本位于 `splits/archive/2026-08-21_strict_3plus1/`。

| 数据 | train | dev | lock | train+dev | all |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 四类 Base | 600 | 75 | 75 | 675 | 750 |
| R2 二类增量 | 112 | 14 | 14 | 126 | 140 |
| Round 1 patrol_boat | 56 | 7 | 7 | — | 70 |
| Round 2 armored_vehicle | 56 | 7 | 7 | — | 70 |

R2 原始标签同时包含旧类与新类。正式源码通过 `configs/incremental_round_registry_4plus2.yaml` 注册两轮不同新类别：先注入 patrol_boat，再注入 armored_vehicle；每轮训练视图只保留当轮新类并映射为该专家的局部类别，不改写原始标签。随机拆分按比赛得分优先，不做相邻帧隔离。完整协议见 [`splits/strict_4plus2/README.md`](splits/strict_4plus2/README.md)。

## 4+2 可复现实验

Base 训练与选模入口为 `tools/04`–`tools/05`。`tools/11_prepare_incremental_round_splits.py` 固化逐轮清单，`tools/06`–`tools/07` 按 `--round-id` 训练和选择当轮专家，`tools/08_evaluate_4plus2.py` 按类别注册表累计执行父代/子代评测。每轮通过后，`tools/13_register_incremental_round_candidate.py` 将权重与证据复制进受版本控制的候选目录并登记 `models/generations.json`，但不切换 production；两轮都登记后，`tools/12_summarize_incremental_rounds.py` 联合校验 New-mAP50、KRR、Full-mAP50、父子链和模型身份。Scene-SensorNet 使用 `tools/60`–`tools/61`；场景门控的 dev 搜索与冻结 lock 复核由 `tools/09_optimize_scene_aware_4plus2.py` 完成，属于独立的 `system_calibration`/`joint_evaluation`。最终仅 `tools/10_promote_scene_aware_4plus2.py` 能在完整两轮证据存在时切换 production。正式参数为 1280 输入、500 epoch、50 轮无改善早停。

上述工具属于同一离线流水线，但协议阶段不同：`tools/04`–`tools/05` 是 `base_learning`；`tools/06`–`tools/07` 每轮只使用该轮 Increment train/dev，Base 与历史专家冻结，属于 `incremental_learning`；`tools/13` 只登记已冻结产物，不训练；`tools/60`–`tools/61` 与 `tools/09 --mode dev` 的场景功能模型训练和门控搜索是 `system_calibration`，不属于增量学习；`tools/08` 的 lock 评分及 `tools/09 --mode lock` 是冻结后的 `joint_evaluation`。

严格源码交付顺序固定为：

```text
11 固化轮次清单
  -> 06/07 Round 1 训练与 Increment dev 选模
  -> 08 Round 1 累计冻结评测 -> 13 登记 Round 1 candidate
  -> 06/07 Round 2 训练与 Increment dev 选模
  -> 08 Round 2 累计冻结评测 -> 13 登记 Round 2 candidate
  -> 12 汇总并核验完整两轮证据
  -> 09 system_calibration dev 搜索与 joint_evaluation lock 复核
  -> 10 切换 generations.json 的 production，并退役联合二类基线
```

当前胜出组合：

- Base：YOLO26s，seed `8675309`，dev mAP50 `0.913454`，best epoch `24`，共运行 `74` 轮；
- Increment：YOLO26s，seed `20260821`，dev mAP50 `0.983917`，best epoch `209`，共运行 `259` 轮；
- Scene-SensorNet：seed `20260821`，best epoch `81`。

上述胜出 Increment 是现有“一次联合训练两个新类”的 production 性能基线，不作为两轮顺序注入证据。严格两轮源码流程不会在新候选完成训练与逐轮验收前覆盖该 production。

场景识别是 air/forest/sea/urban 四个已知类的闭集识别。Base 四类先验只从 Base train 正样本学习，新增二类先验只从 Increment train 正样本学习；因此场景判断会同时影响旧类和新类的有效阈值，但不会改变类别所有权或决定某个模型是否执行。

2026-08-22 已在 4090 服务器的 `sam_hq2_dinov3` 环境使用独立 GPU 完成海面和陆地两张真实图的 CUDA 编排冒烟。两次推理均加载 Scene-SensorNet、四类 Base 和二类增量专家，production 代际为 `incremental_detection_generation_4plus2`，并分别实际激活全局类 4 与全局类 5。

## Ascend 310B

板端配置位于 `configs/agent_pipeline_ascend310b.yaml`，正式服务使用 `/usr/local/miniconda3/envs/agileagent` 与 PyACL/AscendCL：

```bash
./scripts/start_agent_ascend310b.sh
curl -fsS http://127.0.0.1:8501/api/health
curl -fsS -F "file=@sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

当前模型包与板端 release 分别为：

```text
models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/
/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2
```

包内包含 Base、Incremental、Scene 三个 OM、source checkpoint、ONNX、AIPP、ATC 日志、构建 provenance、冻结预测、score、候选 benchmark 和两轮公共 `8501` benchmark。正式公共入口经精确路由进入内部 `18501`；`8502` 保留给候选。健康响应应为 `independent_yolo26_e2e_v1`、`context_mode:model`、`validated:true`。
## 验证

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_release.py
.venv/bin/python scripts/smoke_models.py
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
splits/              当前 strict 4+2 清单与可恢复的旧 3+1 归档
native_ascend/       Ascend C ABI 契约夹具
demo_artifacts/      脱敏演示状态
```
