# 已归档：旧随机逐帧划分

本目录保存随机种子 `20260705` 生成的历史划分清单。它不再是活动训练或测试入口，只用于复现已发布的旧模型和指标。当前唯一活动划分位于 `splits/`。

清单只包含相对于仓库根目录的图像路径，不包含竞赛图像或标签。

| 划分 | 图像数 | 用途 |
| --- | ---: | --- |
| `train.txt` | 560 | 基础训练与增量协议的数据来源 |
| `dev_val.txt` | 95 | 开发验证、阈值校准 |
| `lock_val.txt` | 95 | 冻结权重和阈值后的最终内部复核 |
| `lock_val_base_3plus1.txt` | 74 | 舰船 3+1 中不含新增类别的旧类复核子集 |
| `lock_val_increment_3plus1.txt` | 21 | 舰船 3+1 中包含新增类别的复核子集 |

`*_ir.txt` 和 `*_sar.txt` 是对应主划分的传感器子集。三个主划分互不重叠，共覆盖 750 张图像。
两个 `3plus1` 文件只是 `lock_val.txt` 的互斥子集，合并后仍为原来的 95 张图像，不改变 train/dev/lock 边界。组合系统会对两个子集的全部图像执行同一无标签推理流程，类别标签仅在预测完成后用于指标计算。

原始数据应放在仓库根目录的 `datasets_r1_base_train/`。如确需复现历史划分，先运行：

```bash
python tools/00_check_dataset.py
python tools/01_build_metadata.py
python tools/02_split_dataset.py --protocol legacy --output-dir reports/legacy_splits_check
```

生成到临时目录后与本归档比较；不得把它写回 `splits/`，也不得据此报告当前严格 3+1 的性能。
