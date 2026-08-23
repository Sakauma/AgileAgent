<!-- generated-by: gsd-doc-writer -->
# Ascend 310B 当前状态

截至 2026-08-23，4+2 独立 YOLO26s 三-OM 方案已经完成 ATC 转换、候选四项评分、独立复跑、不可变 release 物化、公共 8501 正式提升和两轮部署后 FPS 复验。当前四项赛题硬门槛全部进入满分档。

## 正式指标

| 指标 | 实测 | 满分门槛 | 结果 |
| --- | ---: | ---: | --- |
| Base mAP50 | **0.8256706047** | ≥0.80 | 通过 |
| New-mAP50 | **0.6188591828** | ≥0.60 | 通过 |
| KRR | **1.0000000000** | ≥0.95 | 通过 |
| 候选 20 图 batch 中位 FPS | **39.3468 / 39.4244** | ≥30 | 两次通过 |
| 公共 8501 中位 FPS | **39.5726 / 39.5883** | ≥30 | 两次通过 |

Full-mAP50 为 **0.7249274787**。旧类 mAP50 增量前后均为 **0.7779616266**，旧类冻结预测完全等价。

部署后公共 8501 两轮“30 次预热 + 3×20 图”的逐轮结果为：

- 第一轮：39.5726 / 39.5804 / 39.3933 FPS；
- 第二轮：39.5883 / 39.5023 / 39.6668 FPS。

## 当前正式身份

| 项目 | 当前值 |
| --- | --- |
| Release | /home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2 |
| 仓库模型包 | models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/ |
| 公共入口 | 127.0.0.1:8501 |
| 主实例 | 127.0.0.1:18501 |
| 候选端口 | 127.0.0.1:8502，当前不监听 |
| 布局 | independent_yolo26_e2e_v1 |
| Context | model，真实 Scene-SensorNet |
| 类别 | soldier / small_aircraft / warship / tank / patrol_boat / armored_vehicle |

三个 systemd unit 均为 active：

- agileagent-ascend310b-main.service；
- agileagent-ascend310b-rollback.service；
- agileagent-ascend310b-route.service。

物理 8501 回滚 listener 与 18501 主实例同时存在。公共请求由带固定 comment 的精确 loopback NAT 进入主实例；删除该唯一规则即可恢复回滚 listener。

健康响应必须同时包含：

~~~json
{
  "status": "ready",
  "validated": true,
  "validation_candidate": false,
  "model_layout": "independent_yolo26_e2e_v1",
  "context_mode": "model",
  "generation_id": "incremental_detection_generation_4plus2"
}
~~~

## 正式运行结构

- Base：四类 YOLO26s E2E OM，owner 为 frozen_base_model；
- Specialist：二类 YOLO26s E2E OM，owner 为 incremental_model；
- Context：Scene-SensorNet OM；
- 两个检测 OM 输入均为 1×3×608×736，AIPP 为 1×608×736×3；
- 两个检测 OM 输出均为 [1,300,6]；
- 六类活动阈值与计分请求阈值均为 0.10；
- Base 与 Scene 先并发执行；只有 air ≥ 0.5 且 Base 检出全局类 1 时跳过 Specialist；
- 门控在线输入只包括 Scene 概率和 Base 检测，不读取标签或文件名。

## 非阻断诊断

| 诊断项 | 当前值 |
| --- | ---: |
| lock precision | 0.677551 |
| lock recall | 0.661111 |
| 误激活率 | 0.466667 |
| patrol_boat AP50 | 0.406635 |
| armored_vehicle AP50 | 0.831083 |

误激活率按图像计算：75 张不含新增类的图像中有 35 张至少激活了一个新增类。这些结果保留为部署风险提示，但按当前赛题评分契约不淘汰 Base mAP50、New-mAP50、KRR 与 FPS 已满分的 release。

## 活动入口

- [满分方法与复现手册](ascend-310b-full-score-method.md)
- [部署、切换和回滚](ascend-310b-deployment.md)
- [板端连接与运行环境](ascend-310b-ssh-environment.md)
- [正式模型包](../models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/README.md)
