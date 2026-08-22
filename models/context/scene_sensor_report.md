# Scene-SensorNet 训练报告

该模型属于 system_calibration 功能模型，不计入竞赛口径的 incremental_learning，且不会更新任何检测器权重。

- 生成时间：`2026-08-22T05:22:22`
- 参数量：`192350`
- 权重 SHA256：`aa99a0e58db1f80b0b6e0cc0e5049b6dc39e88f4349d2e7c362b1e2a25b83a7e`
- 增量场景：`[]`
- 旧场景行最大漂移：`0.0`
- 验收：`True`

| split | images | sensor accuracy | scene accuracy | joint accuracy |
|---|---:|---:|---:|---:|
| dev | 89 | 1.0000 | 0.8539 | 0.8539 |
| lock | 89 | 0.9888 | 0.8315 | 0.8202 |
