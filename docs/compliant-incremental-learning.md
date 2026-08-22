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

增量学习只包含：

- 使用当轮新增类别样本训练二类增量检测专家；
- 将专家局部类别映射到全局新增类别 ID；
- 学习只服务于新增类别的专属产物。

该阶段只允许读取当轮 Increment train/dev 图像与标签、冻结 Base 权重和必要配置。禁止旧图、旧标签、旧样本回放和旧特征缓存；Base 检测器权重始终冻结。当前 4+2 专家训练视图只保留全局类 `4/5`，并映射为局部类 `0/1`。

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

联合评估在 mixed lock/test 上覆盖全部六个已学习类别。模型组合和参数先冻结，再读取标签评分；该阶段不允许梯度、不允许训练，也不允许根据 lock/test 结果继续选参。

| 指标 | 数据范围 | 作用 |
| --- | --- | --- |
| Base mAP50 | 固定基础评分子集 | 基础能力 |
| New-mAP50 | 完整混合评分集中的新增类别 | 新知识学习 |
| KRR | 完整混合评分集中的基础类别 | 旧知识保持 |
| Full-mAP50 | 完整混合评分集全部六类 | 综合诊断 |
| precision / FP / 误激活率 | 冻结运行点的预测 | 非阻断部署诊断 |

赛题硬门禁只使用 Base mAP50、New-mAP50 和 KRR；precision、FP 和误激活率继续记录，但不改变比赛门禁结论。

## 当前 4+2 Production

当前 production 由四类冻结 Base 检测器、二类增量专家和 Scene-SensorNet 组成。Base 固定负责全局类 `0–3`，增量专家固定负责全局类 `4–5`。两个检测器对每张图都执行；场景概率只软调节逐类有效阈值，不改变类别 owner，也不做硬路由。

## 机器可读规则

统一口径固化在以下文件：

```text
configs/incremental_detection_policy.yaml
configs/scene_sensor_model_4plus2.yaml
models/manifest.json
models/production/incremental_detection/calibration.json
models/production/incremental_detection/metrics.json
```

新产物使用 `phase`、`counted_as_incremental_learning` 和 `detector_weights_updated` 显式声明所属阶段。旧版 3+1 证据仅作为归档兼容，不作为当前 4+2 协议口径。
