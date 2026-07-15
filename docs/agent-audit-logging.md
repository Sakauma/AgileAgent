# Agent 全流程审计日志

## 覆盖范围

全局日志覆盖 Agent 的完整行为链，同时保留各阶段原始 manifest 和 `events.jsonl`：

| 阶段 | 统一事件 | 关键证据 |
| --- | --- | --- |
| 数据审计 | `incremental.data_audit.completed/failed` | 数据包哈希、样本/目标/类别统计、旧样本访问计数 |
| 数据视图生成 | `incremental.data_view.generated` | `batch.yaml`、`dataset.yaml` 及其 SHA256、train/val 数量 |
| 增量训练 | `incremental.training.started/completed/failed` | argv、GPU、任务号、返回码、耗时、训练日志和候选权重证据 |
| dev 阈值校准 | `incremental.dev_calibration.completed` | dev-only 来源、阈值、precision、校准文件及 SHA256 |
| lock 复核 | `incremental.lock.unsealed`、`incremental.lock_recheck.completed` | lock split 哈希、解封时间、mAP50、KRR、precision 和误激活率 |
| generation 注册 | `generation.registered` | 父代际、类别所有者、模型与阈值、注册表修改前后哈希 |
| production 切换 | `generation.production_switch.started/completed/failed` | 切换前后代际、复核 manifest、注册表修改前后哈希 |
| 回滚 | `generation.rollback.started/completed/failed` | 回滚源和目标、注册表修改前后哈希 |
| 推理与路由 | `inference.single.completed`、`inference.batch.completed` | 场景认知、使用模型、专家激活/跳过原因、融合计数和分段耗时 |
| Agent 决策 | `agent.decision.completed`、`agent.pipeline.started/completed` | 黑板上下文、推荐动作、执行步骤、终止原因和运行 manifest |
| 配置变更 | `config.changed` | 操作键、配置修改前后哈希和自动备份 |

训练完成但尚未校准时额外写入 `incremental.dev_calibration.pending`。任何门禁拒绝、异常、超时和用户停止均单独记录，不会只留下成功事件。

## 关联标识

- `trace_id`：一次 Web、CLI 或代际操作。
- `batch_id`：Web 上传的小样本批次。
- `job_id`：后台训练任务。
- `experiment_id` 与 `run_id`：可复现实验及其不可覆盖运行。
- `protocol_id`：当前增量轮次或类别协议。
- `generation_id`：候选或活动模型代际。

这些标识允许从一次数据上传追踪到训练候选，也允许从 production 代际反查校准、lock 复核和注册证据。

## 查询示例

```bash
agile-agent logs --batch-id BATCH_ID
agile-agent logs --job-id JOB_ID
agile-agent logs --experiment-id warship_3plus1 --run-id RUN_ID
agile-agent logs --protocol-id round_01
agile-agent logs --generation-id incremental_detection_generation
agile-agent logs --level error
```

Web 仅返回指定批次的脱敏时间线。完整事件详情、哈希、内部命令和错误上下文只通过本地 CLI 或 `reports/agent_logs/agent-*.jsonl` 查看。

## 完整性原则

全局日志不是实验真值的替代品。指标以冻结 manifest 和逐图预测为准，模型身份以权重 SHA256 为准，production 以代际注册表为准。日志负责将这些证据按时间和关联 ID 串联；注册和 production 切换仍采用原子写入与修改前备份。
