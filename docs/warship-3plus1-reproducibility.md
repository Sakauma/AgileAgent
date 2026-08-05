# 舰船 3+1 可复现实验指南

## 实验边界

本协议把舰船作为模拟新增类别。“基础检测代际”只学习人员、小型飞行器和坦克，基础 train/dev 排除所有含舰船图像；增量 train/dev 只包含舰船。准确表述是“基础模型未接触赛题舰船图像和标签”，不宣称通用预训练从未包含船舶语义。

当前完整增量批次包含 126 张 train 图像和 532 个舰船框，另有 22 张 dev 图像和 89 个舰船框；增量 train 图像数约为基础 train 的 29.03%。这些数量、传感器/场景分布和比例仍由审计脚本每次重新统计，`expected_counts` 只作断言。原始数据和标签保持只读。

## 运行入口

先执行只读校验：

```bash
agile-agent experiment validate --config configs/incremental/warship_3plus1.yaml
```

确认输出中的 `lock_sealed=true`，且旧图、旧标签、旧特征缓存数量均为 0 后，才可启动：

```bash
agile-agent experiment run --config configs/incremental/warship_3plus1.yaml
```

复现必须引用父实验 manifest；数据、标签或 split 任一指纹变化都会拒绝：

```bash
agile-agent experiment reproduce --manifest runs/experiments/warship_3plus1/<run_id>/run_manifest.json
```

## 审计产物

每个 run 使用不可覆盖目录，保存配置快照、环境与 Git 状态、`dataset_snapshot.json`、`events.jsonl`、完整 argv/PID/退出码、训练日志、逐图预测、阈值曲线、指标、权重哈希、`run_manifest.json` 及其 SHA256 旁车文件。数据快照逐文件记录图像/标签 SHA256、尺寸、类别、目标数、sensor 和 scene。lock 在基础权重、增量权重和 dev 阈值冻结前只记录 split 哈希与 stem，不解析标签。

状态严格按 `CREATED → DATA_AUDITED → PARTITIONED → BASE_TRAINED → BASE_FROZEN → INCREMENT_TRAINED → INCREMENT_FROZEN → THRESHOLD_CALIBRATED → LOCK_UNSEALED → EVALUATED → ACCEPTED/REJECTED` 推进。只有完成注册门禁后才写入新代际。

## 验收门槛

基础旧类 mAP50、New-mAP50 和 KRR 分别不得低于 0.80、0.60 和 0.95，以赛题满分档作为内部上线门限。基础权重必须零漂移，增量阶段旧图、旧标签和旧缓存必须为 0。四类组合 mAP50、precision 和误激活率作为内部风险观察指标，不单独否决候选。阈值只用增量 dev 在 0.01-0.99 扫描；冻结后 lock 只评一次。

通用增量检测器的舰船类别绑定只依据增量 dev 将阈值冻结为0.63。冻结后，基础检测器与增量检测器对完整 95 张 lock 图像共同执行无标签推理，标签只在预测结束后用于评分。复核得到基础增量前/后 mAP50 `0.87172/0.87278`、New-mAP50 `0.81485`、KRR `1.00121`、组合 mAP50 `0.81613`；74 张舰船负样本中误激活 0 张。赛题硬门禁均通过，当前代际以存在时延告警的状态进入 production。“基础检测代际”仍是回滚点，四类统一模型始终是 `benchmark_only`。

本次 RTX 4060 Laptop、Ultralytics CUDA 组合复核的模型平均推理观察值为 `520.28 ms/图`，未达到内部 x86 时延目标。该结果只用于标记后续 TensorRT 优化任务，不作为候选拒绝条件，也不代表 Ascend 310B 的板端 FPS。Ascend 310B、OM 转换和板端 FPS 继续标记为外部硬件阻塞。

## 当前实现范围

通用 YAML、数据快照和代际 schema 可表达多轮及多个新增类别。训练适配器 v1 目前只执行单轮 3+1；多轮执行器完成并经过同等测试前，不得宣称系统已完成多轮在线更新。
