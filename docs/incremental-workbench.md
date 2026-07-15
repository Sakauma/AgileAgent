# 增量学习工作台

## 目标与边界

增量学习工作台把外部小样本批次转换为可追踪的本地资产，并串联上传、审计、注入、GPU训练和候选输出。增量训练只读取本次上传的数据，不读取基础训练图像、旧类标签或旧特征缓存。训练完成仅表示产生候选权重，不会自动修改当前 production。

## 数据包约定

上传 ZIP 中应包含同 stem 的图像和 YOLO 标签。支持 PNG、JPEG、BMP 和 TIFF；标准标签为 `class x_center y_center width height`，也兼容缺少类别列的 `x_center y_center width height` 单类标注，坐标位于 `[0,1]`。建议按 `images/train`、`images/val`、`labels/train`、`labels/val` 组织，并提供：

```yaml
names:
  0: new_class_name
```

Agent 会拒绝路径穿越、符号链接、解压容量超限、重复路径、重复图像 stem、缺失标签、非法类别和越界框。没有验证集时按内容哈希确定性划分。赛题增量数据与基础训练集采用相同的五列 YOLO 标签，生产配置仍兼容 `bbox_only`：检测到四列无类别标签时，批次不会被拒绝，而是按单一待确认类别导入，并在审计结果、Web 后台和结构化日志中写入警告。若数据集提供 `names`，类别语义直接继承；若只有数字 ID，则按已有类别数量依次命名为“类别N”。每项绑定同时保存源 ID、连续训练 ID 和稳定全局 ID。Web 批次详情或 `incremental-data rename` 可在人工确认后补充真实名称，重命名不会改变这三个 ID。

每个批次维护当前 `class_registry.yaml` 和不可覆盖的 `class_registry_history/revision-*.yaml`。训练任务启动时，将当前注册表和 `dataset.yaml` 冻结到任务快照目录；训练进程只读取该快照。训练完成后继续修改显示名称只会产生新的注册表修订，历史训练快照、标签映射和权重保持不变。

## 状态流转

```text
AUDITED -> INJECTED -> TRAINING -> TRAINED_CANDIDATE
    |          |           |
 REJECTED    阻塞       FAILED / CANCELLED
```

- `AUDITED`：文件已持久保存，格式和数据边界通过。
- `INJECTED`：已生成只含当前批次的训练视图、`dataset.yaml` 和内部 `batch.yaml`。
- `TRAINING`：后台 GPU 子进程正在执行，标准输出持续写入任务日志。
- `TRAINED_CANDIDATE`：候选权重已生成，仍需校准、完整测试和代际上线审核。

## 目录与证据

每个批次保存到 `data/incremental_batches/<batch_id>/`：

```text
source.zip                 原始上传包
extracted/                 安全解压内容
prepared/                  独立训练视图
batch.yaml                 Agent内部批次定义
batch_manifest.json        数据、状态和训练摘要
class_registry.yaml        当前类别名称与稳定ID绑定
class_registry_history/    类别名称修订历史
jobs/<job_id>.json         任务状态与命令
jobs/<job_id>.log          完整训练输出
jobs/snapshots/<job_id>/   该次训练的数据与类别注册表快照
training/<job_id>/         Ultralytics训练产物
```

全局事件写入 `reports/agent_logs/agent-*.jsonl`。日志用 `trace_id`、`batch_id` 和 `job_id` 串联请求、审计与训练阶段，敏感字段自动脱敏。

## 参数配置

持久参数位于 `configs/agent_pipeline.yaml`：

```yaml
logging:
  root: reports/agent_logs
  max_file_bytes: 10485760
  retained_files: 14
  request_bodies: false

incremental_workbench:
  root: data/incremental_batches
  max_archive_bytes: 2147483648
  max_extracted_bytes: 5368709120
  max_extracted_files: 20000
  max_image_pixels: 25000000
  allowed_image_extensions: [.png, .jpg, .jpeg, .bmp, .tif, .tiff]
  require_labels: true
  validation_fraction: 0.20
  minimum_images: 2
  preview_limit: 12
  job_log_tail_lines: 300
  poll_interval_ms: 2000
  training:
    device: "0"
    imgsz: 640
    batch: 32
    epochs: 80
    patience: 20
    optimizer: AdamW
    lr0: 0.001
    deterministic: true
    amp: true
```

可用 `agile-agent --set incremental_workbench.training.device=1 serve` 做单次覆盖，也可用 `agile-agent config set incremental_workbench.training.epochs 60` 原子写回。修改持久配置后需重启 Agent。

## 上线规则

训练任务仅调用受控 Python 模块，不执行用户提供的 shell。候选模型必须另外完成增量 dev 阈值校准、冻结权重哈希、完整 lock 评测、New-mAP50/KRR/组合 mAP50 和误激活率门禁，最后通过 `generation promote` 原子上线；失败时保留原 production 和完整日志。
