# 多批次小样本压力测试

该矩阵使用三个独立随机种子重新划分现有舰船样本。同一场景内的 train、dev、lock 和各轮完全互斥；不同场景是独立重复实验，不共享权重、注册表或 production 状态。

| 场景 | 轮数 | 训练图像 | 目的 |
|---|---:|---|---|
| balanced_micro_8r | 8 | 首轮24张，后续8–9张/轮 | 验证极小批次连续更新 |
| sensor_shift_6r | 6 | 10–24张/轮 | 验证IR/SAR分布交替偏移 |
| diminishing_7r | 7 | 30、20、15、12、9、6、3张 | 定位可靠学习的样本下限 |

每轮必须同时满足基础 mAP50、New-mAP50、KRR 和旧数据零交集满分档硬门禁。任意一轮未通过时立即停止当前场景，保留上一代 production；其他场景仍继续执行，用于区分随机波动和系统性下限。

```bash
python tools/82_run_multibatch_stress_matrix.py \
  --config configs/incremental/multibatch_stress_matrix.yaml
```

汇总写入 `reports/multibatch_validation/matrices/<run_id>/`，各场景的完整证据仍保存在独立的 `runs/multibatch_validation/<scenario_run_id>/` 和 `reports/multibatch_validation/<scenario_run_id>/`。该测试不修改正式模型注册表或默认 Web production。

## RTX 4060 压力测试结果

`local-4060-stress-matrix-20260715-v1` 首先发现了两个机制缺陷：新专家可以删除旧类 owner 预测，且基础 mAP50 被错误地在累计增量 lock 上重算。两个缺陷分别导致 KRR 和基础 mAP50 硬门禁失败，候选代际均被正确回滚。

修复后的 `local-4060-stress-matrix-20260715-v2` 使旧类 owner 成为不可被新专家删除的结构性不变量，并将基础 mAP50 固定到根代际和原始 base lock。三组共 21 轮全部形成连续晋升链：

| 场景 | 连续晋升 | 最低基础mAP50 | 最低New-mAP50 | 最低KRR | 最高误激活率 |
|---|---:|---:|---:|---:|---:|
| balanced_micro_8r | 8/8 | 0.8169 | 0.7656 | 0.9735 | 0.6351 |
| sensor_shift_6r | 6/6 | 0.8179 | 0.8550 | 0.9513 | 0.6622 |
| diminishing_7r | 7/7 | 0.8169 | 0.9750 | 0.9908 | 0.6351 |

这证明了当前数据和三个随机种子下的机制可行性，包括最后一轮 3-shot 更新；它不代表任意三张未知数据都能训练出合格模型。误激活率和组合 mAP50 仍作为质量告警，不冒充赛题硬门禁。
