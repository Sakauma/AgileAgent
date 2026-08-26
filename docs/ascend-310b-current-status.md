<!-- generated-by: gsd-doc-writer -->
# Ascend 310B 当前状态

截至 2026-08-26，4+2 独立 YOLO26s 三-OM 方案已完成 mixed dev 约束校准、lock 冻结验收、三实例推理池 release 物化、公共 `8501` 原子提升和纯增量最坏分布复验；断网 `4→4+1→4+2` 板端演示也已完成真实 NPU 训练、OM、隔离部署和完整运行时 FPS 验收。正式 release 与增量演示的 Base mAP50、New-mAP50、KRR 和 FPS 四项赛题门禁均进入满分档。

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
| 板端增量隔离演示，20 图 | `39.05` | `38.70` | `37.92` | `38.6995` | PASS |

## 板端离线增量演示

`scripts/run_ascend310b_incremental_demo.sh` 已在强制断网环境下完成 `board-full-check-v6`。命令从 Increment 数据目录自动对齐两轮注册表，在 `npu:0` 更新每类 8 参数 Adapter，随后完成 mixed lock、ONNX/OM、ACL 数值核对、隔离配置部署和真实完整图像链路复测。

| 指标 | 实测 |
| --- | ---: |
| Base mAP50 | `0.816663` |
| New-mAP50 | `0.624935` |
| KRR | `1.000000` |
| Full-mAP50 | `0.726497` |
| 新类误激活 | `17/75` |
| Adapter OM 最大绝对误差 | `5.96e-08` |
| 完整图像链路中位 FPS | `38.6995` |
| NPU 训练搜索 / 完整热态命令 | `155.17 / 1007.07 秒` |

演示候选以 `accepted` 状态写入独立运行目录，`base_images_used_for_training=0`、`old_raw_image_count=0`，production 模型和配置身份保持原样。

## 当前正式身份

| 项目 | 当前值 |
| --- | --- |
| Release | `/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-replica-pool-v1` |
| 仓库模型包 | `models/ascend310b/full-score/20260824-4plus2-yolo26-runtime-calibration-v1/` |
| 公共入口 | `127.0.0.1:8501` |
| 主实例 | `127.0.0.1:18501` |
| 回滚实例 | `20260824-4plus2-yolo26-runtime-calibration-v1`，物理监听 `8501` |
| 候选端口 | `127.0.0.1:8502`，候选验收时使用 |
| 已验收路由 | `route=primary public=8501 target=18501` |
| 布局 | `independent_yolo26_e2e_v1` |
| Context | `model`，真实 Scene-SensorNet |
| 推理池 | `3` 个同构实例，批次均衡分片并按输入顺序合并 |
| 类别 | `soldier / small_aircraft / warship / tank / patrol_boat / armored_vehicle` |

三项 systemd unit 与精确 loopback 路由均已完成 active 状态验收。2026-08-26 全部任务收尾后，板端三个 Agent 服务已停止，设备处于静默待机；再次启动这些 unit 即恢复已验收的 `8501 → 18501` 正式拓扑。物理 `8501` 回滚 listener 与 `18501` 主实例在运行态同时存在，移除唯一精确路由即可回到上一 release。

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

precision、recall 和误激活率继续作为工程优化与错误分析信号；当前赛题的四项满分门禁均已通过。

## 活动入口

- [当前指标总账](current-metrics.md)
- [满分方法与复现手册](ascend-310b-full-score-method.md)
- [部署、切换和回滚](ascend-310b-deployment.md)
- [板端连接与运行环境](ascend-310b-ssh-environment.md)
- [断网一键增量学习演示](ascend-310b-offline-incremental-demo.md)
- [板端轻量增量训练实现](ascend-310b-edge-incremental-training.md)
- [正式模型包](../models/ascend310b/full-score/20260824-4plus2-yolo26-runtime-calibration-v1/README.md)
- [三实例推理池验收证据](../reports/ascend310b/20260824-replica-pool-v1/README.md)
