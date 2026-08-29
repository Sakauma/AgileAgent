# Ascend310B 当前满分复核（2026-08-29）

本目录记录 Ascend310B1 公共 `8501` 的最新四项评分复核。板端运行 `e9351cf`，三实例健康响应为 `validated=true`、`validation_candidate=false`、`inference_replicas=3`，模型布局为 `independent_yolo26_e2e_v1`。

| 指标 | 实测 | 满分门槛 | 结果 |
| --- | ---: | ---: | --- |
| Base mAP50 | `0.8166630282` | `>=0.80` | PASS |
| New-mAP50 | `0.6114608956` | `>=0.60` | PASS |
| KRR | `1.0000000000` | `>=0.95` | PASS |
| 公共 `8501` 全流程 aggregate FPS | `33.8973263713` | `>=30` | PASS |

FPS 报告为 schema v8：30 次预热、三轮各 20 图，共 60 帧；三轮全流程耗时合计 `1770.051105 ms`，逐轮 FPS 为 `34.491763 / 33.613921 / 33.601533`。计时覆盖 loopback HTTP 解析、图像解码、Scene、决策、Base/Incremental 检测、后处理、响应解析及 60 个正式六列 TXT 写出。`includes_result_persistence=true`、`formal_results_valid=true`。

仓库保留精度报告、预测摘要和性能报告；完整 89 图冻结预测及 60 个正式 TXT 位于板端同名目录。验证完成后三个 Agent systemd unit 与原子路由均已恢复为 `inactive`。

文件校验：

- `score.json`: `26fa4fe01dbb5118645394a3b254684333ed2d399f5e20f7d02bf33e20b2e610`
- `benchmark-public-8501.json`: `4fe0c4ebf31a8df9f643d78d4b0c4ca56215b636d572d5909a1c3338ec493cdb`
- `predictions-summary.json`: `23e9e4c1de1d3237310c6650cc900641e57bf6766936311fb7c801f21157a56a`
- 板端 `predictions.jsonl`: `28fa07e077885ddc1a1b155f562bb2e75073e9996cba3fa686c76776353102f9`
