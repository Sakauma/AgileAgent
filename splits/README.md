<!-- generated-by: gsd-doc-writer -->
# strict 4+2 数据划分

`active.json` 指向 `strict_4plus2/manifest.json`。正式划分覆盖四类 Base 与两轮单类增量训练，按 `sensor | scene` 分层后在帧级随机分配。

| 来源 | train | dev | lock | train+dev | all |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 四类 Base | 600 | 75 | 75 | 675 | 750 |
| R2 二类增量 | 112 | 14 | 14 | 126 | 140 |

## 正式清单

| 阶段 | 训练 | 开发 | 冻结评估 |
| --- | --- | --- | --- |
| Base | `base_train.txt` | `base_dev.txt` | `base_lock.txt` |
| Increment 总视图 | `increment_train.txt` | `increment_dev.txt` | `increment_lock.txt` |
| Round 1：patrol_boat | `round_01_patrol_boat_train.txt` | `round_01_patrol_boat_dev.txt` | `round_01_patrol_boat_lock.txt` |
| Round 2：armored_vehicle | `round_02_armored_vehicle_train.txt` | `round_02_armored_vehicle_dev.txt` | `round_02_armored_vehicle_lock.txt` |

`mixed_dev.txt` 用于系统校准，参数冻结后在 `mixed_lock.txt` 上联合复核。两份 mixed 清单均为 75 张 Base 图与 14 张 Increment 图，共 89 张。

## 评分口径

| 指标 | 评分范围 | 发布门槛 |
| --- | --- | ---: |
| Base mAP50 | `base_lock.txt` 中的四个基础类别 | `0.80` |
| New-mAP50 | `mixed_lock.txt` 中截至当前轮的新增类别 | `0.60` |
| KRR | `mixed_lock.txt` 中增量前后已学习类别 mAP50 的比值 | `0.95` |

每轮训练视图只保留注册表声明的新增类，并映射为专家局部类别。类别、轮次和父子代际由 `configs/incremental_round_registry_4plus2.yaml` 管理。

## 生成与复核

从仓库根目录运行：

```bash
python tools/03_split_r2_4plus2.py --verify-only
python tools/11_prepare_incremental_round_splits.py \
  --data-root /path/to/tiaozhanbei_4plus2_dataset_20260821
```

随机种子、分层配额、类别分布、传感器分布和相邻帧覆盖率见 `strict_4plus2/manifest.json`，训练说明见 `strict_4plus2/README.md`。
