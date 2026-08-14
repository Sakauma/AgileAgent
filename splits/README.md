# 严格 3+1 类别增量数据划分

该目录只描述一套固定的 3+1 模拟实验，不做交叉验证，也不轮换新增类别。
当前实例暂时把 `warship` 作为模拟新增类别；正式官方增量数据到达后，当前四类全部属于基础类。

## 检测模型数据边界

| 阶段 | 清单 | 图片数 | 可见目标类别 |
|---|---|---:|---|
| 三类基础训练 | `splits/strict_3plus1/base_train.txt` | 441 | soldier, small_aircraft, tank |
| 三类基础验证 | `splits/strict_3plus1/base_dev.txt` | 70 | soldier, small_aircraft, tank |
| 单类增量训练 | `splits/strict_3plus1/increment_train.txt` | 132 | warship |
| 单类增量验证 | `splits/strict_3plus1/increment_dev.txt` | 18 | warship |
| 基础测试 | `splits/strict_3plus1/base_test.txt` | 70 | soldier, small_aircraft, tank |
| 最终混合测试 | `splits/strict_3plus1/mixed_test.txt` | 89 | 全部四类 |

混合测试集由 70 张旧类图和 19 张新增类图组成；不要求同一张图同时含旧类和新类。
`splits/strict_3plus1/base_test.txt` 只定义基础指标的评分子集，不得用于图片级模型路由。
冻结基础检测器和增量专家都必须先对完整混合测试集的每张图执行无标签推理并冻结预测，随后评分器才读取 base_test 清单和标签。
本目录只为前三项官方精度提供数据视图；第四项端到端FPS不由数据划分产生，必须在 Ascend 310B 上对完整单帧链路独立测量。赛题原文使用 `mAP`；仓库当前评分实现为 `mAP@0.5`（`mAP50`），最终以官方评分程序的 IoU 口径为准。

| 官方指标 | 分值 | 完整评分档位 | 本目录的作用 |
| --- | ---: | --- | --- |
| 基础目标检测 mAP | 30 | `≥0.80:30`；`≥0.70:25`；`≥0.65:20`；`≥0.60:15`；`≥0.50:10`；`≥0.40:5`；`<0.40:0` | `splits/strict_3plus1/base_test.txt` 定义基础类评分子集。 |
| New-mAP | 10 | `≥0.60:10`；`≥0.50:7`；`≥0.40:4`；`<0.40:0` | 在完整 `splits/strict_3plus1/mixed_test.txt` 预测上只评新增类别。 |
| KRR | 10 | `≥0.95:10`；`≥0.90:7`；`≥0.80:4`；`<0.80:0` | 在完整 `splits/strict_3plus1/mixed_test.txt` 上计算旧类增量后/前 mAP 比值。 |
| Ascend 310B 端到端 FPS | 10 | `≥30:10`；`≥20:7`；`≥10:4`；`<10:0` | 不使用 split 做图片级路由；由板端完整链路独立计分。 |

仓库发布门禁采用前三项的满分线：基础 mAP50 `≥0.80`、New-mAP50 `≥0.60`、KRR `≥0.95`。`base_dev` 只用于选权重，四类总体 mAP50 只作诊断；precision 与误激活率是内部质量门禁，不增加官方分数。
不得依据测试标签、文件名、数据集身份或场景类别决定是否运行某个类别 owner。

`splits/pool_train.txt` 与 `splits/pool_dev.txt` 只是生成上述模型专用清单的源池，不能直接作为三类基础检测器的训练数据。

## 已知场景识别

场景模型使用 `splits/strict_3plus1/scene_train.txt`、`splits/strict_3plus1/scene_dev.txt` 和 `splits/strict_3plus1/scene_test.txt`（573/88/89），覆盖 air、forest、sea、urban 全部已知场景。场景训练只能读取场景/传感器标签，不得读取目标类别标签、共享检测器特征或建立场景到目标类别的硬绑定。

## 750 张全量覆盖

源池为 573/88/89，三者互斥且恰好覆盖全部 750 张图。上一版的 51 张边界隔离图已全部并入训练源池，活动划分不再强制连续帧边界间距；3+1 类别隔离和测试标签封存约束保持不变。上一版严格时序划分已归档到 `archive/splits_strict_temporal_3plus1_405_117/`，旧随机逐帧划分已归档到 `archive/splits_legacy_random_560_95_95/`。

可用其他可独立拆分的类别重新生成模板实例：

```bash
python tools/02_split_dataset.py --protocol strict-3plus1 --increment-class warship --output-dir reports/splits_check
```
