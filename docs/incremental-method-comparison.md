# DuET-YOLO11s 与 YOLO-IOD-lite 复现指南

## 实验目标

本实验在同一舰船 3+1 增量批次上比较两种无旧样本类别增量方法。两者复用完全相同的三类基础权重、126张增量 train、22张增量 dev 和95张冻结 lock-val。增量训练阶段禁止读取旧图像、旧标签、旧特征缓存和 replay。

这两项实现是面向 YOLO11s 与本赛题约束的适配复现，不宣称逐行等同原论文：

- `DuET-YOLO11s` 基于 ICCV 2025 DuET：训练单类当前任务模型，加入旧教师定位/特征蒸馏和方向一致性，随后合并基础任务与当前任务的共享参数向量。旧类分类行来自基础模型，新类行来自当前任务模型。
- `YOLO-IOD-lite` 基于 AAAI 2026 YOLO-IOD：先训练当前阶段教师，再使用旧教师和当前教师非对称蒸馏四类学生；按当前任务向量幅度选择25%的关键输出通道更新。由于 p02 图像不存在旧类共现，CPR 伪标签在本实验中明确禁用。

## 配置与资源

唯一配置为 `configs/incremental_method_comparison.yaml`。默认使用：

| 方法 | GPU | 增量阶段 |
|---|---:|---|
| DuET-YOLO11s | 1 | 当前任务模型30轮 + 任务向量合并 |
| YOLO-IOD-lite | 2 | 当前教师30轮 + 四类学生30轮 |

两种方法使用 `imgsz=640`、`batch=32`、AdamW 和种子 `20260714`。GPU 3 保持空闲，不参与本次比较。共享基础权重必须存在于配置声明的位置，脚本会核对两份结果中的基础权重 SHA256。

## 运行

在4090服务器仓库根目录及 `irsar-yolo` 环境中执行：

```bash
python tools/71_compare_incremental_methods.py --check-only
python tools/71_compare_incremental_methods.py
```

第一条命令必须输出 `ready: true`。正式运行会并行启动两个方法，并在训练完成后自动生成统一比较报告。已有 `run_id` 不会被覆盖；重新实验必须修改 YAML 中的 `experiment.run_id`。

训练已经完成但比较报告缺失时，可以只汇总：

```bash
python tools/71_compare_incremental_methods.py --report-only
```

## 输出与判定

逐方法结果位于：

```text
reports/incremental_method_comparison/<run_id>/<protocol>/
runs/detect/incremental_method_comparison/<run_id>/<protocol>/
```

统一输出为 `comparison.json`、`comparison.csv` 和 `comparison.md`。只有同时满足以下门禁的方法才可成为候选：

- New-mAP50 不低于 `0.60`；
- KRR 不低于 `0.95`；
- 四类 mAP50 不低于 `0.80`；
- dev 校准 precision 不低于 `0.90`；
- lock precision/recall 不低于 `0.70/0.75`；
- 图像误激活率不高于 `0.15`；
- 旧类分类行漂移不高于 `1e-6`。

如果两种方法都未通过，报告必须写为“无 verified winner”，不得选一个较高结果替换 Web 模型。lock-val 只做本次冻结验收，不得用于修改本 run 的损失权重、任务向量系数或训练轮数。
