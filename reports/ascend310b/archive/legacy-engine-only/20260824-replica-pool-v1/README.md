# Ascend310B 三实例推理池 legacy engine-only 档案

本目录是 `20260824-4plus2-yolo26-replica-pool-v1` 的历史快照。精度报告仍可用于复核冻结输出；四份性能报告均为 schema v7，只统计 engine/service 段或未覆盖正式结果写出，因此不得作为当前赛题 FPS 或 release 晋级证据。

当前正式成绩与 schema v8 全流程证据见 [`../../../20260829-full-score-recheck-v1/`](../../../20260829-full-score-recheck-v1/README.md)。不可变 release 内的旧 schema 文件保持原样，以继续通过历史包完整性与兼容验证。

## 赛题门禁

| 指标 | 实测 | 门槛 | 结果 |
| --- | ---: | ---: | --- |
| Base mAP50 | `0.816663` | `>=0.80` | PASS |
| New-mAP50 | `0.611461` | `>=0.60` | PASS |
| KRR | `1.000000` | `>=0.95` | PASS |
| Full-mAP50 | `0.722005` | 诊断项 | - |
| Lock precision / recall | `0.729167 / 0.612698` | 诊断项 | - |
| 新类误激活率 | `17/75 = 0.226667` | 诊断项 | - |

冻结预测与旧正式版本完全一致。原始评分见 [score.json](score.json)。

## 历史性能（仅诊断）

这些 FPS 从服务进入推理引擎开始，到 Scene/Base/Specialist OM、内容门控、置信度校准、融合和 NMS 完成为止；未覆盖当前要求的全流程总墙钟与正式六列 TXT 写出。

| 运行点 | 图像/轮 | 第 1 轮 | 第 2 轮 | 第 3 轮 | 中位 FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 候选 `8502` | 20 | `37.36` | `37.51` | `35.04` | `37.36` |
| 候选独立复跑 | 20 | `37.46` | `37.97` | `38.50` | `37.97` |
| 正式 `8501` mixed lock | 20 | `35.38` | `38.62` | `38.22` | `38.22` |
| 正式 `8501` 纯增量全集 | 140 | `38.53` | `36.05` | `37.40` | `37.40` |

历史 CLI 也使用了不同计时边界。所有这些数值只解释当时的运行时优化，不参与当前 `>=30 FPS` 判定。

该目录记录 2026-08-24 快照，归档操作未改写任何 JSON 内容。

原始性能报告：

- [候选三轮（schema v7）](benchmark-candidate.json)
- [候选独立复跑（schema v7）](benchmark-candidate-repeat.json)
- [公共 mixed lock 三轮（schema v7）](benchmark-formal-mixed-20x3.json)
- [公共纯增量 140 图三轮（schema v7）](benchmark-formal-incremental-140x3.json)
