# 增量学习工作台

## 目标与边界

增量学习工作台把外部小样本批次转换为可追踪的本地资产，并串联上传、数据血缘审计、自动封存、GPU训练、逐类校准、独立复核、代际注册和受控上线。增量训练只读取本次上传的train/dev，不读取历史原始图像、标签、缓存或封存lock。

## 数据包约定

上传 ZIP 中应包含同 stem 的图像和 YOLO 标签。支持 PNG、JPEG、BMP 和 TIFF；标准标签为 `class x_center y_center width height`，也兼容缺少类别列的 `x_center y_center width height` 单类标注，坐标位于 `[0,1]`。建议按 `images/train`、`images/val`、`labels/train`、`labels/val` 组织，并提供：

```yaml
names:
  0: new_class_name
```

Agent 会拒绝路径穿越、符号链接、解压容量超限、重复路径、重复图像 stem、缺失标签、非法类别和越界框。上传文件还会与冻结基础代际及历次已上线批次的 stem、图像 SHA256、标签 SHA256 和缓存 SHA256 求交；缺少基础指纹、发现历史数据交集或训练可达目录出现来源不明缓存时禁止训练。注入和训练启动前都会重新审计，避免审计后替换文件。没有显式 lock 时按种子 `20260705` 和类别组合自动封存20%，随后再生成 dev；每个类别必须同时覆盖 train、dev 和 lock。生产配置仍兼容 `bbox_only`：检测到四列无类别标签时，批次按单一待确认类别导入并记录警告。若数据集提供 `names`，类别语义直接继承；若只有数字 ID，则依次命名为“类别N”。类别重命名不会改变源 ID、训练 ID 或全局 ID。

每个批次维护当前 `class_registry.yaml` 和不可覆盖的 `class_registry_history/revision-*.yaml`。训练任务启动时，将当前注册表和 `dataset.yaml` 冻结到任务快照目录；训练进程只读取该快照。训练完成后继续修改显示名称只会产生新的注册表修订，历史训练快照、标签映射和权重保持不变。

## 状态流转

```text
AUDITED -> INJECTED -> TRAINING -> TRAINED_CANDIDATE
-> CALIBRATING -> CALIBRATED -> REGISTERED_CANDIDATE
-> QUANTIZING -> QUANTIZED
-> LOCK_RECHECKING -> ACCEPTED / REJECTED
-> SHADOW_LOADING -> PROMOTED / ROLLED_BACK
```

- `AUDITED`：文件已持久保存，格式和数据边界通过。
- `INJECTED`：已生成只含当前批次 train/dev 的训练视图、封存 lock 和内部 `batch.yaml`。
- `TRAINING`：后台 GPU 子进程正在执行，标准输出持续写入任务日志。
- `TRAINED_CANDIDATE`：候选权重已生成并冻结。
- `CALIBRATING / CALIBRATED`：只使用增量 dev 扫描每个类别的激活阈值。
- `REGISTERED_CANDIDATE / LOCK_RECHECKING`：注册动态类别所有权并在冻结 lock 上复核。
- `QUANTIZING / QUANTIZED`：当设备配置启用 INT8 时，只使用本轮增量 train/dev 自动完成专家 PTQ，engine 与校准指纹登记到候选代际；FP16/CUDA 配置跳过此阶段。
- `ACCEPTED / REJECTED`：全部上线门禁通过或候选被拒绝。
- `SHADOW_LOADING / PROMOTED`：候选完成预热并原子切换运行时；失败时保留原 production。
- `ROLLED_BACK`：候选加载、运行时切换或已上线血缘冻结失败，注册表与运行时均恢复父代际。

## 目录与证据

每个批次保存到 `data/incremental_batches/<batch_id>/`：

```text
source.zip                 原始上传包
extracted/                 安全解压内容
prepared/                  独立训练视图
sealed_lock/               训练不可达的封存样本与lock manifest
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
  lock_fraction: 0.20
  split_seed: 20260705
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

训练任务仅调用受控 Python 模块，不执行用户提供的 shell。候选会自动完成增量 dev 逐类阈值校准、可选 INT8 PTQ、冻结权重与 engine 哈希、完整 lock 评测以及 shadow smoke。数据隔离和资产完整性有效，且赛题基础 mAP50、New-mAP50、KRR、组合 mAP50 与 FPS 达标后自动 promotion；precision、误激活率、P95 和一致性差值仅作为风险诊断。失败候选保留原 production 和完整日志。INT8 校准不读取封存 lock，也不允许使用历史旧类原始样本校准新增专家。

CLI 的 `agile-agent incremental run --batch BATCH_ID` 在前台等待完整生命周期结束；`status --run-id` 可读取同一任务、批次状态和审计证据。Web 使用相同实现，但以后台任务方式运行。
