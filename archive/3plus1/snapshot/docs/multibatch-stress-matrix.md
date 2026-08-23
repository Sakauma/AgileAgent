# 多批次小样本压力测试

压力矩阵使用三个独立随机种子重新划分舰船样本，并验证连续增量学习、传感器分布变化和样本量递减。

## 场景

| 场景 | 轮数 | 训练图像 |
| --- | ---: | --- |
| `balanced_micro_8r` | 8 | 首轮 24 张，后续 8–9 张/轮 |
| `sensor_shift_6r` | 6 | 10–24 张/轮 |
| `diminishing_7r` | 7 | 30、20、15、12、9、6、3 张 |

运行命令：

```bash
python tools/82_run_multibatch_stress_matrix.py \
  --config configs/incremental/multibatch_stress_matrix.yaml
```

## RTX 4060 结果

`local-4060-stress-matrix-20260715-v2` 完成三组共 21 轮连续晋升：

| 场景 | 连续晋升 | 最低 Base mAP50 | 最低 New-mAP50 | 最低 KRR | 最高误激活率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `balanced_micro_8r` | 8/8 | `0.8169` | `0.7656` | `0.9735` | `0.6351` |
| `sensor_shift_6r` | 6/6 | `0.8179` | `0.8550` | `0.9513` | `0.6622` |
| `diminishing_7r` | 7/7 | `0.8169` | `0.9750` | `0.9908` | `0.6351` |

结果覆盖最后一轮 3-shot 更新，并确认模型代际、类别所有权、基础指标和连续晋升状态保持一致。

汇总写入 `reports/multibatch_validation/matrices/<run_id>/`，逐场景资产写入 `runs/multibatch_validation/` 与 `reports/multibatch_validation/`。
