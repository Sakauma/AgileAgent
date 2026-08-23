<!-- generated-by: gsd-doc-writer -->
# 三个功能模型

AgileAgent 的当前 x86/CUDA production 使用环境认知、四类基础目标检测和二类增量目标检测形成 strict 4+2 推理链路。

## 模型职责

| 模型 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| Scene-SensorNet | 已知场景与传感器认知 | RGB 图像 | IR/SAR、air/forest/sea/urban 概率 |
| 四类 Base 检测器 | 基础目标检测 | RGB 图像 | soldier、small_aircraft、warship、tank 框（全局类 0–3） |
| 二类增量专家 | 新增类别检测 | RGB 图像 | patrol_boat、armored_vehicle 框（局部类 0/1 映射为全局类 4/5） |

Scene-SensorNet 为双头 CNN，权重位于 `models/context/scene_sensor_net.pt`。Base 与增量专家权重位于 `models/production/incremental_detection/`。`configs/functional_models.yaml` 登记三个功能及其资产，`models/generations.json` 登记 production 成员、类别所有权、逐类阈值和场景先验。

## 协同流程

```text
图像
  -> Scene-SensorNet 已知场景概率
  -> Base 与 Increment 对每张图执行
  -> 固定 owner 与全局类别映射
  -> 六类逐类训练先验亲和度
  -> 基础阈值 + 场景软惩罚
  -> class-aware NMS
  -> 检测结果与审计轨迹
```

场景识别是 air/forest/sea/urban 四个已知类的闭集识别，不是开放集场景发现。训练正样本上的概率形成逐类先验：small_aircraft 偏向 air，warship 与 patrol_boat 偏向 sea，soldier、tank 与 armored_vehicle 主要偏向 forest/urban。Base 先验只使用 Base train，新增类先验只使用 Increment train。

线上仅使用 Scene-SensorNet 概率计算亲和度，不读取文件名或真值标签。有效阈值为 `min(1, 基础阈值 + 最大惩罚 × (1 - 亲和度))`。场景结果因此会同时影响旧类和新类，但不会改变类别 owner，也不会跳过任一检测器。

## 当前 CUDA production 运行点

| 类别 | 基础阈值 | 最大场景惩罚 | lock AP50 | precision | FP | 误激活率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| soldier | `0.21` | `0.15` | `0.629499` | `0.575000` | `34` | `4/48 = 0.083333` |
| small_aircraft | `0.14` | `0.88` | `0.918265` | `0.890411` | `8` | `1/72 = 0.013889` |
| warship | `0.36` | `0.26` | `0.929170` | `0.712871` | `29` | `0/65 = 0` |
| tank | `0.05` | `0.19` | `0.746292` | `0.502793` | `89` | `3/43 = 0.069767` |
| patrol_boat | `0.57` | `0.65` | `0.691000` | `0.966667` | `1` | `1/82 = 0.012195` |
| armored_vehicle | `0.82` | `0.00` | `0.855735` | `0.816327` | `9` | `7/82 = 0.085366` |

| 汇总指标 | 数值 |
| --- | ---: |
| Base mAP50 | `0.856067` |
| New-mAP50 | `0.773368` |
| KRR | `0.973126` |
| 六类 TP / FP / precision | `342 / 170 / 0.667969` |
| 新类 TP / FP / precision | `69 / 10 / 0.873418` |
| 六类误激活图像 | `14 / 89` |
| Scene sensor / scene / joint accuracy | `0.988764 / 0.831461 / 0.820225` |

候选参数只由 mixed dev 选择，随后冻结并一次性复核 mixed lock。赛题硬门禁使用 Base mAP50、New-mAP50 与 KRR；precision、FP 和误激活率是非阻断诊断。完整口径见 `models/production/incremental_detection/evidence/operating_point_diagnostics.md`。

## Ascend310B v2

板端正式发布使用四类 Base YOLO26s E2E OM、二类 Specialist YOLO26s E2E OM 和 Scene-SensorNet OM。两个检测 OM 的输入均为 `1×3×608×736`，AIPP 输入为 `1×608×736×3`，输出为 `[1,300,6]`。内容门控仅在 air 概率达到 `0.5` 且 Base 检出 small_aircraft 时跳过 Specialist，在线输入只包括场景概率和 Base 检测。

| 指标 | 数值 |
| --- | ---: |
| Base mAP50 | `0.825671` |
| New-mAP50 | `0.618859` |
| KRR | `1.000000` |
| Full-mAP50 | `0.724927` |
| 公共 8501 两轮 batch 中位 FPS | `39.5726 / 39.5883` |

完整模型身份、评分产物和部署入口见 `docs/ascend-310b-current-status.md`。
