# strict 4+2 production 证据

本目录绑定当前 4+2 Base 与二类增量专家的 dev 选模、训练参数和冻结预测：

- `base_selection.*` / `incremental_selection.*`：只按 dev mAP50 选模，未读取 lock；
- `base_training/` / `incremental_training/`：胜出模型的 `args.yaml` 与 `results.csv`；
- `frozen/`：Base/专家在 mixed dev 和 mixed lock 上的预测，lock 标签在预测冻结后才进入评分。

最终评分、逐类阈值和非阻断诊断见上级目录的 `metrics.json` 与 `calibration.json`。

最终运行点的 TP、FP、precision、recall、逐类误激活率和错误图像去重统计见 `operating_point_diagnostics.md`。其中框级 FP 与图像级误激活是两个不同口径，不可直接互换。
