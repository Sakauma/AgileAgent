# strict 4+2 production 证据

本目录绑定当前 4+2 Base 与二类增量专家的 dev 选模、训练参数和冻结预测：

- `base_selection.*` / `incremental_selection.*`：只按 dev mAP50 选模，未读取 lock；
- `base_training/` / `incremental_training/`：胜出模型的 `args.yaml` 与 `results.csv`；
- `frozen/`：Base/专家在 mixed dev 和 mixed lock 上的预测，lock 标签在预测冻结后才进入评分。

最终评分、逐类阈值和非阻断诊断见上级目录的 `metrics.json` 与 `calibration.json`。
