<!-- generated-by: gsd-doc-writer -->
# 4+2 训练数据包

本数据包用于在 Linux/4090 服务器上开展后续训练，包含完整 R1 四类数据、
完整 R2 增量数据、正式 4+2 划分 TXT 及官方数据说明。

## 解压

```bash
tar -xzf tiaozhanbei_4plus2_dataset_20260821.tar.gz
cd tiaozhanbei_4plus2_dataset_20260821
```

## 目录

```text
datasets_r1_base_train/       # 750 张 R1 图像及四类标签
datasets_r2_inc_train/        # 140 张 R2 图像及原始六类标签
splits/strict_4plus2/         # 正式划分 TXT
official_docs/                # R1/R2 官方数据说明
README_4090.md
```

## 类别

| 全局 ID | 类别 | owner |
| ---: | --- | --- |
| 0 | soldier | Base |
| 1 | small_aircraft | Base |
| 2 | warship | Base |
| 3 | tank | Base |
| 4 | patrol_boat | Increment |
| 5 | armored_vehicle | Increment |

## 划分

| 数据 | train | dev | lock | train+dev | all |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 四类 Base | 600 | 75 | 75 | 675 | 750 |
| R2 二类增量 | 112 | 14 | 14 | 126 | 140 |

首轮开发使用 `base_train.txt` / `base_dev.txt` 和
`increment_train.txt` / `increment_dev.txt`，预测冻结后再打开对应 lock。
超参确定后的比赛复训可使用 `*_train_plus_dev.txt`。

## R2 标签注意事项

R2 保留赛题方发布的原始标签：海面图像同时包含全局类 2/4，
陆地图像可同时包含全局类 3/5。二类增量头不能直接用这些原始 ID 训练。
后续 4+2 训练程序必须在派生目录中：

- 只保留全局类 4 和 5；
- 将全局 `4 -> 局部 0`；
- 将全局 `5 -> 局部 1`；
- 不改写本包中的原始标签。
