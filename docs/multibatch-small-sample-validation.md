# 四批次小样本验证

该实验使用现有舰船数据模拟一次类别增量和三次目标增量。126 张图像在四轮间严格互斥；首轮使用 `40/4/4` 张 train/dev/lock，后三轮各使用 `18/4/4` 张。训练、校准和调参只能读取当前批次 train/dev，累计 lock 在权重与阈值冻结后解封。

## 运行

```bash
python tools/81_validate_multibatch_incremental.py \
  --config configs/incremental/multibatch_small_sample.yaml
```

中断后可对同一 `run_id` 续跑：

```bash
python tools/81_validate_multibatch_incremental.py \
  --config configs/incremental/multibatch_small_sample.yaml \
  --run-id RUN_ID --resume
```

产物分别写入 `runs/multibatch_validation/<run_id>/` 和 `reports/multibatch_validation/<run_id>/`，不会修改正式模型注册表。

## 本机结果

RTX 4060 Laptop、PyTorch `2.5.1+cu124`、Ultralytics `8.4.92` 的四轮验证结果：

| 轮次 | 模式 | New-mAP50 | KRR | 组合mAP50 | 推理ms/图 | 结论 |
|---:|---|---:|---:|---:|---:|---|
| 1 | 类别增量 | 0.8950 | 1.0000 | 0.8053 | 70.98 | 晋升 |
| 2 | 目标增量 | 0.8850 | 1.0000 | 0.8045 | 57.03 | 晋升 |
| 3 | 目标增量 | 0.7350 | 0.9998 | 0.7965 | 55.69 | 满分档通过，累计质量告警 |
| 4 | 目标增量 | 0.8079 | 0.9982 | 0.7976 | 66.93 | 满分档通过，累计质量告警 |

四批次均完成审计、训练和复核，基础 mAP50、New-mAP50 与 KRR 均达到赛题满分档，所有历史权重 SHA256 均无漂移。原始执行错误地把“累计组合 mAP50 >= 0.80”设为硬门禁，因此第 3、4 轮历史状态仍为 `REJECTED`，且第 4 轮从第 2 轮 production 初始化。新版守护器已将组合 mAP50 改为内部告警；该结果可以证明四轮指标达标，但在重新运行前不能表述为四轮连续晋升。

已有结果可在不启动训练的情况下重新判分：

```bash
python tools/81_validate_multibatch_incremental.py \
  --config configs/incremental/multibatch_small_sample.yaml \
  --run-id RUN_ID --reassess
```

重新判分只生成 `guardian_reassessment.json` 和 `guardian_reassessment.md`，不会追溯修改历史注册表或 production。
