# 增量学习工作台

## 目标与边界

增量学习工作台把外部小样本批次转换为可追踪的本地资产，并串联上传、审计、注入、GPU训练和候选输出。增量训练只读取本次上传的数据，不读取基础训练图像、旧类标签或旧特征缓存。训练完成仅表示产生候选权重，不会自动修改当前 production。

## 数据包约定

上传 ZIP 中应包含同 stem 的图像和 YOLO 标签。支持 PNG、JPEG、BMP 和 TIFF；每行标签必须为 `class x_center y_center width height`，坐标位于 `[0,1]`。建议按 `images/train`、`images/val`、`labels/train`、`labels/val` 组织，并提供：

```yaml
names:
  0: new_class_name
```

Agent 会拒绝路径穿越、符号链接、解压容量超限、重复路径、重复图像 stem、缺失标签、非法类别和越界框。没有验证集时按内容哈希确定性划分。没有类别名称时批次可保存和浏览，但不能注入训练，需重新上传并提供名称。

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
jobs/<job_id>.json         任务状态与命令
jobs/<job_id>.log          完整训练输出
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
