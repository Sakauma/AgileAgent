<!-- generated-by: gsd-doc-writer -->
# 增量学习数据与评测契约

本文件定义赛题报告、配置和发布证据共同使用的正式术语。`incremental_learning` 只指新类别检测器的训练及新类专属学习；Scene-SensorNet 训练、场景先验学习、六类门控搜索和联合评分不属于增量学习。

## 四阶段统一口径

| 阶段 | 正式标识 | 数据范围 | 是否更新检测器权重 |
| --- | --- | --- | --- |
| 基础学习 | `base_learning` | Base train/dev | 只更新 Base 检测器 |
| 增量学习 | `incremental_learning` | 当轮 Increment train/dev | 只更新 Increment 检测器；Base 冻结 |
| 系统级校准 | `system_calibration` | Base/Increment train/dev、mixed dev；context lock 仅作冻结功能模型复核 | 否，Base 与 Increment 均冻结 |
| 联合评估 | `joint_evaluation` | 覆盖截至当轮全部旧类与新类的 lock/test | 否；禁止训练和选参 |

这四个阶段是统计口径，不要求必须对应四个独立进程。工作台可以连续编排训练、校准、复核和晋级，但不能把整个编排生命周期都称为“增量学习”。

## 增量学习阶段

增量学习按类别注册表逐轮执行，只包含：

- 使用当轮新增类别样本训练当轮增量检测专家；
- 将专家局部类别映射到全局新增类别 ID；
- 学习只服务于新增类别的专属产物。

该阶段只允许读取当轮 Increment train/dev 图像与标签、通用预训练初始化、冻结父代元数据和必要配置。禁止 Base 图像/标签、历史增量轮次样本回放和旧特征缓存；Base 与历史专家权重始终冻结。类别 ID、轮次、局部到全局映射以及父子代际只能来自 `configs/incremental_round_registry_4plus2.yaml`，训练与评估代码不再固定写死 `4/5`。

Scene-SensorNet、六类场景先验、门控阈值、场景惩罚、融合参数以及 mixed lock 评分均明确排除在 `incremental_learning` 之外。

## 系统级校准

系统级校准是三模型协同所需的独立步骤，可以使用基础与增量数据，但不得反向更新任何检测器权重：

- Scene-SensorNet 使用 Base 与 Increment train 训练，并只按对应 dev 结果选模；配置中的 context lock 只做功能模型冻结复核；
- Base 类场景先验只使用 Base train 正样本；
- 新增类场景先验只使用 Increment train 正样本；
- 六类基础阈值、最大场景惩罚和融合策略只在 mixed dev 上搜索；
- 所有参数在 mixed lock 联合评估前冻结。

因此，系统级校准读取 Base train/dev 不构成增量训练阶段的数据违规：它不会训练 Base 或 Increment 检测器，只校准独立场景功能模型及冻结推理策略。

## 联合评估

每轮联合评估覆盖截至该轮的全部已学习类别。模型组合和参数先冻结，再读取累计 lock 标签评分；该阶段不允许梯度、不允许训练，也不允许根据 lock/test 结果继续选参。

| 指标 | 数据范围 | 作用 |
| --- | --- | --- |
| Base mAP50 | 固定基础评分子集 | 基础能力 |
| New-mAP50 | 当轮新注入类别 | 本轮新知识学习 |
| KRR | 当轮开始前已学习的全部类别 | 父代知识保持 |
| Full-mAP50 | 截至当轮全部已学习类别 | 累计综合性能 |
| precision / FP / 误激活率 | 冻结运行点的预测 | 非阻断部署诊断 |

赛题硬门禁只使用 Base mAP50、New-mAP50 和 KRR；precision、FP 和误激活率继续记录，但不改变比赛门禁结论。

## 两轮顺序注入

正式源码协议登记两个不同的类别增量轮次：

| 轮次 | 父代 | 本轮新类 | 累计类别 |
| --- | --- | --- | --- |
| `round_01_patrol_boat` | `base_detection_generation_4plus2` | patrol_boat | Base `0–3` + `4` |
| `round_02_armored_vehicle` | Round 1 子代 | armored_vehicle | Base `0–3` + `4/5` |

`tools/11_prepare_incremental_round_splits.py` 在训练前从固定 Increment 总清单生成每轮不可静默覆盖的 train/dev/lock 清单。`tools/06`、`tools/07` 每次只接收一个 `--round-id`；`tools/08` 接收截至当前轮的专家权重，在读 lock 标签前冻结全部预测，并输出 `lineage` 与 `round_metrics`。

每轮评测通过后必须执行 `tools/13_register_incremental_round_candidate.py`。该工具同时核对选模权重、累计评测权重、历史专家权重和父代 owner，将当轮权重及 selection/metrics/calibration 复制到 `models/candidates/incremental_detection/<generation_id>/`，再把模型和子代登记为 `registered_candidate`。它只更新 `candidate` 通道，绝不更新 `production`。

两轮都登记后，`tools/12_summarize_incremental_rounds.py` 同时校验指标文件与 `models/generations.json` 中的父子链、不同新增类别、零旧样本、模型身份和三项逐轮指标。`tools/10_promote_scene_aware_4plus2.py` 必须接收该汇总生成的 `round_evidence.json`；缺少任一轮登记或证据时拒绝晋级。晋级成功后，Round 2 子代成为 production，两个单类专家分别拥有类 `4/5`，旧联合二类代际改为 `retired_baseline`。

## 当前 4+2 Production 基线

当前 production 由四类冻结 Base 检测器、二类增量专家和 Scene-SensorNet 组成。Base 固定负责全局类 `0–3`，增量专家固定负责全局类 `4–5`。两个检测器对每张图都执行；场景概率只软调节逐类有效阈值，不改变类别 owner，也不做硬路由。

当前 production 是已经通过六类指标的联合二类专家正式版本。两轮顺序类别注入作为独立的合规证据链运行；候选完成逐轮训练、冻结评估、登记和晋级后，再按正式发布流程替换 production。

## 机器可读规则

统一口径固化在以下文件：

```text
configs/incremental_detection_policy.yaml
configs/incremental_round_registry_4plus2.yaml
configs/scene_sensor_model_4plus2.yaml
models/manifest.json
models/production/incremental_detection/calibration.json
models/production/incremental_detection/metrics.json
models/candidates/incremental_detection/<generation_id>/registration.json
models/production/incremental_detection/evidence/sequential_round_evidence.json
```

新产物使用 `phase`、`counted_as_incremental_learning`、`detector_weights_updated`、`round_id`、`parent_generation_id` 和 `generation_id` 显式声明阶段与代际。联合二类 4+2 production 记录当前部署性能；两轮顺序注入产物记录逐类增量合规证据。
