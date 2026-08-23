# 四批次小样本验证

该实验使用 126 张舰船图像模拟一次类别增量和三次目标增量。四轮数据互斥，首轮使用 `40/4/4` 张 train/dev/lock，后三轮各使用 `18/4/4` 张。

## 运行

```bash
python tools/81_validate_multibatch_incremental.py \
  --config configs/incremental/multibatch_small_sample.yaml
```

续跑：

```bash
python tools/81_validate_multibatch_incremental.py \
  --config configs/incremental/multibatch_small_sample.yaml \
  --run-id RUN_ID --resume
```

## RTX 4060 结果

| 轮次 | 模式 | New-mAP50 | KRR | 组合 mAP50 | 推理 ms/图 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 类别增量 | `0.8950` | `1.0000` | `0.8053` | `70.98` |
| 2 | 目标增量 | `0.8850` | `1.0000` | `0.8045` | `57.03` |
| 3 | 目标增量 | `0.7350` | `0.9998` | `0.7965` | `55.69` |
| 4 | 目标增量 | `0.8079` | `0.9982` | `0.7976` | `66.93` |

四批次均完成数据审计、训练、阈值校准和 lock 复核，Base mAP50、New-mAP50 与 KRR 均达到赛题满分档，历史权重 SHA256 保持稳定。

重新判分：

```bash
python tools/81_validate_multibatch_incremental.py \
  --config configs/incremental/multibatch_small_sample.yaml \
  --run-id RUN_ID --reassess
```

产物写入 `runs/multibatch_validation/<run_id>/` 与 `reports/multibatch_validation/<run_id>/`。
