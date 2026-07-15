# Clean-Room 类别增量重训指南

## 目标与边界

本实验使用舰船数据模拟第一批在线增量注入。三类教师检测器和上下文模型只能读取不含舰船的基础阶段数据；四类学生检测器只能读取126张舰船增量 train 和22张增量 dev。四类 YOLO11s 仅作为 benchmark，禁止作为初始化权重或 Web 增量证据。最终部署只有一个四类学生检测器，不使用独立舰船模型。

三类教师的120轮训练属于离线基础阶段，不计入增量更新时间。在线更新固定最多30轮、早停 patience 8，目标是在4090上15分钟内完成；报告必须单独记录该阶段耗时。

当前修复实验编号为 `clean-ci-v2-warship-r02`。所有目录均带该编号，已存在时脚本会拒绝覆盖。重新实验必须同时修改两份 YAML 中的 `experiment.run_id`。

## 环境检查

在 4090 服务器仓库根目录进入 `irsar-yolo` 环境：

```bash
conda activate irsar-yolo
python tools/70_run_strict_3plus1.py --config configs/clean_class_incremental_v2.yaml --check-only
```

预检只读取环境、权重、划分和标签，不创建数据视图，也不会启动训练。输出中的 `ready` 必须为 `true`。本次运行检测器使用 GPU 1；上下文模型后续使用 GPU 2。若可用设备发生变化，应先修改两份 YAML 中的设备编号并重新预检。

`fair_agent.cli doctor` 是完整 Agent 发布诊断，会校验 Web、CLI、模型注册表和全部冻结资产；训练服务器的精简工作副本不使用该命令作为训练前置条件。

## 第一步：重训检测模型

全部参数位于 `configs/clean_class_incremental_v2.yaml`，执行：

```bash
python tools/70_run_strict_3plus1.py --config configs/clean_class_incremental_v2.yaml
```

脚本依次完成：

1. 校验 train/dev/lock 无重复 stem，并固定源划分 SHA256。
2. 构建434张基础 train、73张基础 dev，只保留 soldier、small_aircraft、tank。
3. 从官方 `yolo11s.pt` 重训三类教师检测器。
4. 创建四类学生检测头，将教师本地类别 `0、1、2` 映射到全局类别 `0、1、3`。
5. 复制并冻结共享骨干、框回归分支和旧类分类通道；全局类别 `2` 显式随机重置后才开放训练，不能继承 COCO 的 boat 等类别通道。
6. 使用126张舰船 train 和22张舰船 dev 更新学生的新类通道。
7. 仅使用增量 dev 扫描新类置信度阈值。
8. 权重和阈值确定后读取95张 lock-val，对单个四类学生执行 bootstrap 和门禁判定。

训练保持 `batch=32`。评测取证固定使用 `evaluation_batch=1、rect=true`，并由记录型 Ultralytics Validator 在同一次 `val` 中同时生成官方 mAP50 和逐框预测，避免 `predict` 与 `val` 的预处理、多标签 NMS 不一致。逐框证据写入报告目录下的 `predictions/*.jsonl`，自定义 AP50 必须与同次 Validator 结果相差不超过 `0.005`。

严禁把 `models/base/yolo11s_ir_sar_imgsz640.pt` 或 `models/incremental/` 下的权重写入该 YAML。

## 第二步：重训上下文模型

必须等待第一步完全结束，再执行：

```bash
python tools/60_train_scene_sensor.py --config configs/clean_context_warship_v2.yaml --check-only
python tools/60_train_scene_sensor.py --config configs/clean_context_warship_v2.yaml
```

第一条命令必须输出 `ready: true`。该模型先在基础阶段视图上训练已有场景，再仅使用舰船增量 train/dev 更新新增的 `sea` 场景输出行；旧场景行、传感器头和特征提取器保持冻结。lock 仅用于最终验收，不参与模型选择、早停或调参。

不要使用 `--force`。需要重跑时创建新的 run ID，禁止覆盖旧证据。

## 第三步：验收

检测报告位置：

```text
reports/clean_class_incremental_v2/<run_id>/warship-incremental/
```

上下文报告为同目录下的 `context_report.md`。检测模型必须同时满足：

| 指标 | 门槛 |
|---|---:|
| New-mAP50 | >= 0.60 |
| KRR | >= 0.95 |
| dev precision | >= 0.90 |
| lock precision | >= 0.70 |
| lock recall | >= 0.75 |
| lock 图像误激活率 | <= 0.15 |
| 四类组合 mAP50 | >= 0.80（内部观察目标，不单独否决） |
| 自定义评测器误差 | <= 0.005 |

上下文模型必须满足 sensor/scene/joint accuracy `>= 0.90/0.70/0.65`。基础 mAP50、New-mAP50、KRR、数据合规或资产完整性失败时不得切换 Web；组合 mAP50 和部署诊断未达目标时记录告警。

## 上线前检查

通过后仍先保留在 `models/experiments/clean_class_incremental_v2_staging/`。核对以下事实后才能更新活动配置：

- 三类教师没有舰船输出通道，映射为全局类别 `0、1、3`。
- 四类学生只有全局类别 `2` 的分类通道允许更新，旧通道最大漂移不超过 `1e-6`。
- 最终 `student_4class.pt` 单独输出四类边界框、置信度和类别，不依赖第二个检测器。
- 三份新权重的 SHA256 与冻结 manifest 一致。
- Web、CLI 和报告引用同一个 run ID，不再加载全量四类模型。

完成活动资产切换后运行：

```bash
pytest -q
python scripts/verify_release.py
python scripts/smoke_models.py
```

只有三项全部通过，新的 clean-room 模型才可作为评委端类别增量证据。
