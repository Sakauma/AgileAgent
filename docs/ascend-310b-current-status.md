<!-- generated-by: gsd-doc-writer -->
# Ascend 310B 当前状态

截至 2026-08-24，4+2 独立 YOLO26s 三-OM 方案已完成 mixed dev 约束校准、lock 冻结验收、三实例推理池 release 物化、公共 `8501` 原子提升和纯增量最坏分布复验。Base mAP50、New-mAP50、KRR 和 FPS 四项赛题门禁均进入满分档。

## 正式指标

| 指标 | mixed dev | lock / 部署后 | 满分门槛 | 结果 |
| --- | ---: | ---: | ---: | --- |
| Base mAP50 | `0.823083` | **`0.816663`** | `>=0.80` | 通过 |
| New-mAP50 | `0.705836` | **`0.611461`** | `>=0.60` | 通过 |
| KRR | `1.000000` | **`1.000000`** | `>=0.95` | 通过 |
| Full-mAP50 | `0.770734` | `0.722005` | 诊断项 | - |
| 新类误激活 | `4/75 = 0.053333` | `17/75 = 0.226667` | 诊断项 | - |

性能均使用 loopback HTTP multipart PNG、`30` 次预热，并按完整图像推理耗时计算：

| 运行点 | 第 1 轮 | 第 2 轮 | 第 3 轮 | 中位 FPS | 门禁 |
| --- | ---: | ---: | ---: | ---: | --- |
| 候选 `8502`，20 图 | `37.3571` | `37.5103` | `35.0359` | `37.3571` | PASS |
| 候选独立复跑，20 图 | `37.4577` | `37.9696` | `38.5046` | `37.9696` | PASS |
| 公共 `8501` mixed，20 图 | `35.3751` | `38.6201` | `38.2175` | `38.2175` | PASS |
| 公共 `8501` 纯增量，140 图 | `38.5337` | `36.0538` | `37.3997` | `37.3997` | PASS |

## 当前正式身份

| 项目 | 当前值 |
| --- | --- |
| Release | `/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-replica-pool-v1` |
| 仓库模型包 | `models/ascend310b/full-score/20260824-4plus2-yolo26-runtime-calibration-v1/` |
| 公共入口 | `127.0.0.1:8501` |
| 主实例 | `127.0.0.1:18501` |
| 回滚实例 | `20260824-4plus2-yolo26-runtime-calibration-v1`，物理监听 `8501` |
| 候选端口 | `127.0.0.1:8502`，当前不监听 |
| 路由 | `route=primary public=8501 target=18501` |
| 布局 | `independent_yolo26_e2e_v1` |
| Context | `model`，真实 Scene-SensorNet |
| 推理池 | `3` 个同构实例，批次均衡分片并按输入顺序合并 |
| 类别 | `soldier / small_aircraft / warship / tank / patrol_boat / armored_vehicle` |

`agileagent-ascend310b-main.service`、`agileagent-ascend310b-rollback.service` 和 `agileagent-ascend310b-route.service` 均为 `active`。物理 `8501` 回滚 listener 与 `18501` 主实例同时存在；公共请求由唯一精确 loopback NAT 规则进入主实例，删除该规则即回到旧 release。

健康响应已确认：

```json
{
  "status": "ready",
  "validated": true,
  "validation_candidate": false,
  "model_layout": "independent_yolo26_e2e_v1",
  "context_mode": "model",
  "inference_replicas": 3,
  "generation_id": "incremental_detection_generation_4plus2"
}
```

## 冻结的运行时策略

- 六类阈值：`0=.075, 1=.05, 2=.05, 3=.05, 4=.20, 5=.50`；
- Base 校准：temperature `1.5`、bias `0`；
- Specialist 校准：temperature `1.0`、bias `-0.5`；
- 类 4 已知场景软惩罚 `0.05`，类 5 为 `0`；
- conflict IoU `0.5`，Specialist margin `0.15`；
- 全类别重叠抑制 IoU `0.9`、smaller-box coverage `0.95`；
- incremental-over-base margin `0`。

候选共搜索 `5,476` 组，其中 `4,467` 组满足 mixed dev 三项精度约束。选参进程不打开 lock；lock 仅用于冻结验收。

## 部署诊断

| 诊断项 | 上一代 | 当前 lock | 变化 |
| --- | ---: | ---: | ---: |
| 新类误激活 | `35/75 = 0.466667` | `17/75 = 0.226667` | `-51.43%` |
| 新类 lock precision | `0.677551` | `0.729167` | `+0.051616` |
| 新类 lock recall | `0.661111` | `0.612698` | `-0.048413` |
| patrol_boat AP50 | `0.406635` | `0.476365` | `+0.069730` |
| armored_vehicle AP50 | `0.831083` | `0.746557` | `-0.084526` |

precision 和误激活率仍未达工程建议线，因此继续作为风险诊断；它们不是当前赛题的四项满分门禁。

## 活动入口

- [当前指标总账](current-metrics.md)
- [满分方法与复现手册](ascend-310b-full-score-method.md)
- [部署、切换和回滚](ascend-310b-deployment.md)
- [板端连接与运行环境](ascend-310b-ssh-environment.md)
- [正式模型包](../models/ascend310b/full-score/20260824-4plus2-yolo26-runtime-calibration-v1/README.md)
- [三实例推理池验收证据](../reports/ascend310b/20260824-replica-pool-v1/README.md)
