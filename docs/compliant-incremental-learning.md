# 类别增量学习合规方案

## 规则解释

本赛题的增量任务按**类别增量目标检测**实现。设基础类别为 `C_old`，本轮新增类别为 `C_new`，二者必须非空且互不重叠。增量结束后，系统应同时输出 `C_old ∪ C_new` 的检测结果。

增量学习阶段的训练、验证、早停、阈值选择和超参数调整只能读取本轮增量数据集。以下输入明确禁止：

- 旧类原始图像和标签；
- 旧样本 replay 或经过增强的旧样本；
- 从旧数据离线缓存的特征或裁剪目标。

允许使用基础阶段已经冻结的模型权重和类别原型。教师蒸馏只能在增量图像上运行，不能重新读取旧数据生成伪标签。

## 主线实现

当前采用 `冻结旧类检测器 + 新类 specialist`：

1. 基础检测器完全冻结，继续负责 `C_old`。
2. 将新增类别映射为 specialist 类别 `0`。
3. specialist 的训练集和验证集都只能来自增量数据集。
4. 训练结束并冻结权重后，才进入独立评测阶段。
5. 推理时将 specialist 类别 `0` 映射回全局类别 ID，并与旧类预测组合。

这种结构使 KRR 不依赖旧样本回放。教师伪标签蒸馏仅作为消融方案，其教师推理输入同样限定为增量图像。

## 阶段隔离

学习 YAML 只包含 `train` 和 `val`，不写入旧类测试集。旧类与新类测试视图只允许在增量权重冻结后读取，且不得参与梯度、早停、模型选择或阈值调优。

机器可读规则位于 `configs/class_incremental_policy.yaml`。数据构建和训练入口会同时校验：

- 训练集严格来自授权增量训练划分；
- 验证集严格来自授权增量验证划分；
- 图像主名与 SHA256 内容同时匹配，防止旧图改名混入；
- 训练集与验证集没有重叠；
- `C_old` 与 `C_new` 互不重叠；
- `old_raw_image_count = 0`。

任一检查失败时，训练拒绝启动。

## 运行顺序

```bash
python tools/27_build_new_class_specialist_dataset.py \
  --config configs/incremental_no_old_distill_yolo11s.yaml

python tools/24_run_no_old_distill.py \
  --config configs/incremental_no_old_distill_yolo11s.yaml \
  --protocol p01_new_small_aircraft --device 0

python tools/25_aggregate_no_old_distill.py \
  --config configs/incremental_no_old_distill_yolo11s.yaml
```

训练参数保留在 YAML，命令行只选择协议和 GPU。

## 验收标准

```text
task_type = class_incremental_object_detection
learning_data_scope = incremental_dataset_only
learning_scope_verified = true
old_raw_image_count = 0
New-mAP50 >= 0.60
KRR >= 0.95
```

当前合规实验中 p02-p04 达到满分阈值；p01 New-mAP50 为 `0.55860`，按官方分档可获得部分分，但仍标记为未达到内部满分门槛。
