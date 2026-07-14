# 增量目标检测合规方案

## 规则解释

本赛题任务统一建模为**增量目标检测**，以类别增量为主，同时保留目标增量能力：

- `class_incremental`：新增类别集合 `C_new` 与基础类别 `C_old` 互不重叠，增量后输出 `C_old ∪ C_new`。
- `target_incremental`：新增目标样本可以属于已有类别，不强制类别集合互斥，用于补充新实例、新外观、新场景或新传感器条件下的目标。

两种模式的实现方式可以不同，但数据使用边界相同。

增量学习阶段的训练、验证、早停、阈值选择和超参数调整只能读取本轮增量数据集。以下输入明确禁止：

- 旧类原始图像和标签；
- 旧样本 replay 或经过增强的旧样本；
- 从旧数据离线缓存的特征或裁剪目标。

允许使用基础阶段已经冻结的模型权重和类别原型。教师蒸馏只能在增量图像上运行，不能重新读取旧数据生成伪标签。

## 当前主线实现

当前可复现主线采用类别增量模式下的 `冻结旧类检测器 + 新类 specialist`。舰船 3+1 协议中，三类基础检测器未接触赛题舰船图像和标签，舰船专家只读取舰船增量 train/dev：

1. 基础检测器完全冻结，继续负责 `C_old`。
2. 将新增类别映射为 specialist 类别 `0`。
3. specialist 的训练集和验证集都只能来自增量数据集。
4. 训练结束并冻结权重后，才进入独立评测阶段。
5. 推理时将 specialist 类别 `0` 映射回全局类别 ID，并与旧类预测组合。

这种结构使 KRR 不依赖旧样本回放。教师伪标签蒸馏仅作为消融方案，其教师推理输入同样限定为增量图像。

## 阶段隔离

学习 YAML 只包含 `train` 和 `val`，不写入旧类测试集。旧类与新类测试视图只允许在增量权重冻结后读取，且不得参与梯度、早停、模型选择或阈值调优。

机器可读规则位于 `configs/incremental_detection_policy.yaml`。数据构建和训练入口会同时校验：

- 训练集严格来自授权增量训练划分；
- 验证集严格来自授权增量验证划分；
- 图像主名与 SHA256 内容同时匹配，防止旧图改名混入；
- 训练集与验证集没有重叠；
- 类别增量时 `C_old` 与 `C_new` 互不重叠；目标增量时允许增量目标属于已有类别；
- `old_raw_image_count = 0`。

任一检查失败时，训练拒绝启动。

## 运行顺序

```bash
agile-agent experiment validate --config configs/incremental/warship_3plus1.yaml
agile-agent experiment run --config configs/incremental/warship_3plus1.yaml
agile-agent experiment reproduce --manifest runs/experiments/warship_3plus1/<run_id>/run_manifest.json
```

训练参数保留在 YAML，命令行只选择协议和 GPU。

## 验收标准

```text
task_type = incremental_object_detection
incremental_mode in {class_incremental, target_incremental}
learning_data_scope = incremental_dataset_only
learning_scope_verified = true
old_raw_image_count = 0
New-mAP50 >= 0.60
KRR >= 0.95
```

历史 p01-p04 权重的目标类别已被当时的四类基础模型见过，因此只作为目标增量/专项增强演练，不作为严格类别增量证据。strict-p02 舰船折达到核心指标，但尚未通过部署 precision 与误激活门禁，production 仍为三类 `generation-0`。完整说明见 `docs/warship-3plus1-reproducibility.md`。
