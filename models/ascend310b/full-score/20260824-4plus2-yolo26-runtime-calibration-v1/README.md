# 2026-08-24 Ascend310B 4+2 运行时校准发布包

这是当前 Ascend310B1 正式 4+2 release 的可版本化副本，对应板端目录：

```text
/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1
```

该 release 在 Atlas 200I DK A2 / Ascend310B1、CANN `7.0.RC1` 上完成 mixed dev 约束选参、lock 隔离评分、原子提升以及公共 `8501` 部署后复验。模型权重与上一代一致；本次只更新受约束的置信度校准、逐类阈值、场景软惩罚与重叠仲裁。

## 正式结果

| 指标 | mixed dev | lock / 部署后 | 赛题门槛 |
| --- | ---: | ---: | ---: |
| Base mAP50 | `0.823083` | `0.816663` | `>=0.80` |
| New-mAP50 | `0.705836` | `0.611461` | `>=0.60` |
| KRR | `1.000000` | `1.000000` | `>=0.95` |
| Full-mAP50 | `0.770734` | `0.722005` | 诊断项 |
| 新类误激活 | `4/75 = 0.053333` | `17/75 = 0.226667` | 诊断项 |
| 新类 precision | `0.877622` | `0.729167` | 诊断项 |

lock 误激活由上一代的 `35/75 = 0.466667` 降至 `17/75 = 0.226667`，降幅为 `51.43%`。六类 lock AP50 依次为 `0.490069 / 0.939302 / 0.888246 / 0.791493 / 0.476365 / 0.746557`。

20 图 batch 性能：

- lock 候选：`38.18 / 38.40 / 38.39 FPS`，中位 `38.39 FPS`；
- mixed dev 独立复跑：`38.97 / 39.14 / 39.16 FPS`，中位 `39.14 FPS`；
- 公共 `8501` 部署后：`38.58 / 38.67 / 38.66 FPS`，中位 `38.66 FPS`。

四项满分门禁 Base、New、KRR 和 FPS 均通过。

## 冻结运行时策略

- 六类阈值：`0=.075, 1=.05, 2=.05, 3=.05, 4=.20, 5=.50`；
- Base logit 校准：temperature `1.5`、bias `0`；
- Specialist logit 校准：temperature `1.0`、bias `-0.5`；
- 类 4 已知场景软惩罚 `0.05`，类 5 为 `0`；
- old/new conflict IoU `0.5`，Specialist margin `0.15`；
- 全类别重叠抑制 IoU `0.9`、smaller-box coverage `0.95`；
- incremental-over-base margin `0`。

参数只由 `mixed_dev` 选择，lock 仅用于一次冻结验收。首次 lock 不通过后没有重新调参；只修复 Base 输出来源标识使校准在真实 OM 路径生效，然后用同一候选复验。

## 包内容

```text
configs/       validated:true 的正式 Agent 配置
om/            Base、Incremental、Scene-SensorNet 三个 OM
provenance/    源 checkpoint、ONNX、AIPP、ATC 日志与构建清单
validation/    dev/lock 结果、选参证据及部署后冻结复验
release.json   正式 release 摘要
SHA256SUMS     大型发布资产物化时的完整性清单
```

仓库不包含竞赛原始图像或标签。零训练物化与原子提升流程见 [`docs/ascend-310b-deployment.md`](../../../../docs/ascend-310b-deployment.md)。
