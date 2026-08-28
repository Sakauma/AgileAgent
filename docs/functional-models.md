<!-- generated-by: gsd-doc-writer -->
# 三个功能模型

AgileAgent 的当前 production 使用环境认知、四类基础目标检测和二类增量目标检测形成 strict 4+2 推理链路；x86/CUDA 运行 PT 权重，Ascend310B 运行同职责的三-OM release。板端离线增量演示还可以在冻结三模型之后接入每类 8 参数的置信度 Adapter。

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
  -> 全类别跨类别高度重叠抑制
  -> 检测结果与审计轨迹
```

场景识别是 air/forest/sea/urban 四个已知类的闭集识别，不是开放集场景发现。训练正样本上的概率形成逐类先验：small_aircraft 偏向 air，warship 与 patrol_boat 偏向 sea，soldier、tank 与 armored_vehicle 主要偏向 forest/urban。Base 先验只使用 Base train，新增类先验只使用 Increment train。

线上仅使用 Scene-SensorNet 概率计算亲和度，不读取文件名或真值标签。有效阈值为 `min(1, 基础阈值 + 最大惩罚 × (1 - 亲和度))`。场景结果因此会同时影响旧类和新类，但不会改变类别 owner，也不会跳过任一检测器。固定 owner 合并后，对所有类别执行同一重叠规则：跨类别框 `IoU >= 0.50` 或小框覆盖率 `>= 0.95` 时只保留最高置信度框；规则不读取数据划分、文件名或标签。

## 当前 CUDA production 运行点

### 一号结果：独立 mixed lock

| 类别 | 基础阈值 | 最大场景惩罚 | lock AP50 | precision | FP | 误激活率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| soldier | `0.21` | `0.15` | `0.632134` | `0.582278` | `33` | `3/48 = 0.062500` |
| small_aircraft | `0.14` | `0.88` | `0.918265` | `0.890411` | `8` | `1/72 = 0.013889` |
| warship | `0.36` | `0.26` | `0.948502` | `0.878049` | `10` | `0/65 = 0` |
| tank | `0.05` | `0.19` | `0.803992` | `0.634328` | `49` | `2/43 = 0.046512` |
| patrol_boat | `0.57` | `0.65` | `0.645000` | `1.000000` | `0` | `0/82 = 0` |
| armored_vehicle | `0.82` | `0.00` | `0.855735` | `0.833333` | `8` | `6/82 = 0.073171` |

| 汇总指标 | 数值 |
| --- | ---: |
| Base mAP50 | `0.845782` |
| New-mAP50 | `0.750368` |
| KRR | `0.997179` |
| Full-mAP50 | `0.800605` |
| 六类 TP / FP / precision | `335 / 108 / 0.756208` |
| 新类 TP / FP / precision | `67 / 8 / 0.893333` |
| 六类误激活图像 | `11 / 89` |
| Scene sensor / scene / joint accuracy | `0.988764 / 0.831461 / 0.820225` |

模型阈值和权重保持不变，只增加数据来源无关的正式后处理。固定 89 张 lock 预测回放中，Full-mAP50 从 `0.794994` 提升到 `0.800605`，FP 从 `170` 降到 `108`，overall precision 从 `0.667969` 提升到 `0.756208`；Base mAP50、New-mAP50 与 KRR 仍满足赛题硬门禁。全部 890 张现有标注图像中没有真实跨类别标注框触发该规则。

### 二号结果：全部标注图像诊断

二号结果在 Base `600/75/75` 与 Increment `112/14/14` 的 train/dev/lock 上统一回放，共 890 张图像。它包含训练图像，只用于检查冻结系统在全部已知数据上的拟合、漏检和误检，不是独立测试集成绩，也不参与选模或阈值选择。

| 汇总指标 | 正式后处理 |
| --- | ---: |
| Base mAP50 | `0.888703` |
| New-mAP50 | `0.737477` |
| KRR | `1.004975` |
| Full-mAP50 | `0.827444` |
| 六类 TP / FP / precision | `3533 / 986 / 0.781810` |
| 新类 TP / FP / precision | `669 / 55 / 0.924033` |
| 六类误激活图像 | `88 / 890 = 0.098876` |
| 跨类别重复框抑制数 | `660` |

相对同一批图像的抑制前结果，Full-mAP50 为 `0.818983 -> 0.827444`，FP 为 `1581 -> 986`，precision 为 `0.694729 -> 0.781810`；同时 TP 为 `3598 -> 3533`，因此该层后处理的收益与召回代价均保留在诊断证据中。完整逐类结果及本机 lock 复现差异见 `models/production/incremental_detection/evidence/all_images_diagnostics.md`。

## Ascend310B v2

板端正式发布使用四类 Base YOLO26s E2E OM、二类 Specialist YOLO26s E2E OM 和 Scene-SensorNet OM。两个检测 OM 的输入均为 `1×3×608×736`，AIPP 输入为 `1×608×736×3`，输出为 `[1,300,6]`。内容门控仅在 air 概率达到 `0.5` 且 Base 检出 small_aircraft 时跳过 Specialist，在线输入只包括场景概率和 Base 检测。

| 指标 | 当前 immutable release 实测 |
| --- | ---: |
| Base mAP50 | `0.816663` |
| New-mAP50 | `0.611461` |
| KRR | `1.000000` |
| Full-mAP50 | `0.722005` |
| 新类误激活 | `17/75 = 0.226667` |
| 公共 8501 全流程 aggregate FPS（两次复测） | `31.961599 / 32.656507` |

上表是当前 `20260824-4plus2-yolo26-runtime-calibration-v1` 的真实 OM lock 和部署后板端结果。候选与 release-local 复验精度完全一致，四项满分门禁均通过。

### 板端离线增量 Adapter

`scripts/run_ascend310b_incremental_demo.sh` 从当前 Increment 数据目录重建 `4→4+1→4+2` 轮次，在 `npu:0` 训练两组 8 参数 Adapter。它使用冻结检测器候选的置信度、框面积、场景概率和传感器概率形成 8 维输入，在正式 score calibration 之前更新新增类置信度；Base、Incremental 检测器和 Scene-SensorNet 权重保持冻结。

2026-08-26 `board-full-check-v6` 已完成训练、mixed lock、ONNX/OM、ACL 数值对齐、隔离部署和启用 Adapter 后的完整图像链路复测：

| 指标 | 隔离演示实测 |
| --- | ---: |
| Base mAP50 | `0.816663` |
| New-mAP50 | `0.624935` |
| KRR | `1.000000` |
| Full-mAP50 | `0.726497` |
| 新类误激活 | `17/75` |
| Adapter OM 最大绝对误差 | `5.96e-08` |
| 旧 engine-only 图像链路中位 FPS | `38.6995`（非当前官方口径） |

通过门禁的 Adapter 由独立 `agent_pipeline_ascend310b_demo.yaml` 激活，结果中的 `agent.decision.edge_incremental_adapter` 记录运行身份；原 production 与父代 release 保持可直接启动。

完整硬门禁、逐类精度、误激活、场景模型、正式 release FPS 与板端增量结果见 `docs/current-metrics.md`；模型身份、评分产物和部署入口见 `docs/ascend-310b-current-status.md`。
