<!-- generated-by: gsd-doc-writer -->
# 增量学习工作台

增量学习工作台将外部小样本批次转换为可追踪的数据、训练和模型代际资产。

## 数据包

上传 ZIP 使用图像与同 stem 五列 YOLO 标签：

```text
class x_center y_center width height
```

推荐结构：

```text
images/train/
images/val/
labels/train/
labels/val/
data.yaml
```

工作台校验路径、压缩包容量、图像解码、图像与标签映射、类别 ID 和边界框范围。数据血缘摘要只在批次入库或缓存生成时自动记录一次，用于判断资产是否被替换；日常训练与复核不重复执行全数据集哈希扫描。模型权重在候选登记和 production 发布时仍保留必要的身份校验。

## 生命周期

```text
AUDITED
  -> INJECTED
  -> TRAINING
  -> TRAINED_CANDIDATE
  -> CALIBRATING
  -> CALIBRATED
  -> REGISTERED_CANDIDATE
  -> DIAGNOSING
  -> LOCK_RECHECKING
  -> SHADOW_LOADING
  -> PROMOTED
```

生命周期完成固定拆分、训练快照、GPU 训练、逐类阈值校准、混淆图、候选登记、lock 复核、shadow 预热和 production 切换。

现场真正新类别使用候选优先的一键总控：

```text
PREFLIGHT_PASSED
  -> AUDITED
  -> CLASSES_REGISTERED
  -> DATA_READY
  -> TRAINING_STARTED
  -> CANDIDATE_ACCEPTED
  -> FPS_GATE_PASSED / ASCEND_GATES_PASSED
  -> PROMOTED
```

该总控会临时禁止普通生命周期提前自动晋级；累计 lock 与候选 FPS 或板端完整门禁全部通过后才切换 production。若本机正式服务正在运行，则在服务内部原子热切换；否则更新注册表供下一进程自动加载。失败时保留父代际或执行已登记回滚。详见 [`onsite-4plus2plusn.md`](onsite-4plus2plusn.md)。

正式 4+2 类别增量使用 `configs/incremental_round_registry_4plus2.yaml` 将同一 R2 数据包登记为两个不同类别轮次。Round 1 只训练 patrol_boat 专家；Round 2 的父代是 Round 1 冻结子代，只训练 armored_vehicle 专家。每轮用 `tools/13_register_incremental_round_candidate.py` 登记为 candidate，两轮由 `tools/12_summarize_incremental_rounds.py` 汇总；登记阶段不会切换 production。Scene-SensorNet 和场景门控属于独立 `system_calibration`，不进入增量训练数据计数。

## 批次目录

```text
data/incremental_batches/<batch_id>/
├── source.zip
├── extracted/
├── prepared/
├── sealed_lock/
├── batch.yaml
├── batch_manifest.json
├── class_registry.yaml
├── class_registry_history/
├── jobs/
└── training/
```

## CLI

```bash
agile-agent incremental-data upload --archive /path/to/new_batch.zip --name 新批次
agile-agent incremental-data list
agile-agent incremental-data show --batch-id BATCH_ID
agile-agent incremental-data rename --batch-id BATCH_ID --class-names 新类别名称
agile-agent incremental run --batch BATCH_ID
agile-agent incremental status --run-id TRAIN_JOB_ID
agile-agent incremental onsite --bundle /path/to/new_classes.zip --plan-only
agile-agent incremental onsite --bundle /path/to/new_classes.zip --target x86
agile-agent incremental onsite-status --run-id ONSITE_RUN_ID
```

## 配置

主要参数位于 `configs/agent_pipeline.yaml` 的 `incremental_workbench` 章节，包括上传容量、文件数量、图像像素、拆分比例、随机种子、训练设备、输入尺寸、batch、epochs、optimizer 和学习率。

## 审计证据

批次 manifest 记录数据摘要、类别映射、拆分清单、训练任务、指标和代际结果。全局日志使用 `trace_id`、`batch_id` 与 `job_id` 连接 Web 请求和后台任务。
