# Ascend 310B 当前状态索引

2026-08-16 的共享双逻辑头方案已经在赛题四项机器评分上重复进入满分档，并已物化为正式 release 后原子提升为板端主线。

- 活动方法和新数据集复现入口：[Ascend 310B 满分方法与复现手册](ascend-310b-full-score-method.md)
- 零训练正式模型包：[20260816-full-score-1493b04](../models/ascend310b/full-score/20260816-full-score-1493b04/README.md)
- 新板部署与既有板双实例回滚：[Ascend 310B 部署实现](ascend-310b-deployment.md)
- 当前轻量证据：[2026-08-16 满分证据](archive/ascend310b/2026-08-16-full-score-evidence.json)
- P0–P11 完整板端记录：[历史执行记录](archive/ascend310b/p0-p11-execution-record.md)

当前部署边界：

- 公共 `127.0.0.1:8501` 经精确 loopback NAT 返回 `validated: true` 的 `shared_backbone_dual_head_v1` 主线；
- 正式主实例实际监听 `18501`，release 为 `/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04`；
- 原三 OM service 继续真实监听 `8501`，正常被路由规则旁路；删除该规则即可即时回滚；
- `8502` 当前空闲，专用于下一轮数据集或结构候选；
- x86 本机 `configs/agent_pipeline.yaml` 继续使用其所在主机的 `8501`，与板端端口互不冲突。

发布后公共 `8501` 的 `30 + 3×20` 端到端复核为 `30.234/30.243/30.294 FPS`，中位 `30.243 FPS`，报告 SHA256 `bb011d96b62f627d36388f4237017570afd4195e6327522162ab6a0fab15b4e5`。正式配置、release manifest 和 validation summary SHA256 分别为 `39f6472094b3e7f61950a903a0ff914d1e620c557d9b9b747151fd9a502be490`、`ffca93c54aa600a268acc31cdee82e14a040f6313427a180c3597e07db5fc2dd`、`62234e2aba8921c07b8c8e0d66c87f912ffba8b00d8b43245a908524c3a56891`。

仓库已包含上述正式 release 的两个 OM、实际 source checkpoint、ONNX、AIPP、ATC 日志、完整 provenance、冻结预测和原始 validation 报告。已布置好 CANN/Python 环境的 310B 可通过 `scripts/materialize_ascend310b_full_score_release.sh` 直接物化并验证，无需训练或 ATC。仓库不包含竞赛原始图像/标签：仅查看证据和启动服务不需要数据集；重新测 FPS 需要 20 张契约 PNG，重新计算精度需要合法取得的 89 图及标签。
