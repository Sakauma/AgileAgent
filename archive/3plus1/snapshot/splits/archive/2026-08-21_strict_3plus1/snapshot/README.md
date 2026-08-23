# 固定 3+1 类别增量数据划分

本目录固定一套覆盖 750 张基础数据的 3+1 模拟实验。当前实例以 `warship` 为新增类别，
基础类别为 `soldier`, `small_aircraft`, `tank`。

## 源池分配

图像按 `sensor | dataset_round | scene` 组成序列，并按帧号排序。每个序列末段形成测试窗口，
其前一段形成开发窗口；评估窗口边界 4 帧范围内的样本归入训练源池。最终源池规模为：

| 源池 | 图片数 | IR | SAR | 用途 |
| --- | ---: | ---: | ---: | --- |
| `pool_train.txt` | 573 | 405 | 168 | 检测与场景训练清单的来源 |
| `pool_dev.txt` | 88 | 68 | 20 | 检测与场景开发清单的来源 |
| `mixed_test.txt` | 89 | 67 | 22 | 固定混合测试集 |

三个源池互斥并完整覆盖 750 张图。`manifest.json` 记录分配规则、逐序列窗口边界、类别分布和传感器分布。

## 检测模型清单

| 阶段 | 清单 | 图片数 | 可见目标类别 |
| --- | --- | ---: | --- |
| 三类基础训练 | `strict_3plus1/base_train.txt` | 441 | soldier, small_aircraft, tank |
| 三类基础验证 | `strict_3plus1/base_dev.txt` | 70 | soldier, small_aircraft, tank |
| 单类增量训练 | `strict_3plus1/increment_train.txt` | 132 | warship |
| 单类增量验证 | `strict_3plus1/increment_dev.txt` | 18 | warship |
| 基础测试 | `strict_3plus1/base_test.txt` | 70 | soldier, small_aircraft, tank |
| 最终混合测试 | `strict_3plus1/mixed_test.txt` | 89 | 全部四类 |

混合测试集由 70 张旧类图和 19 张新增类图组成。基础检测器与增量检测器先对完整 89 张混合测试集执行无标签推理并冻结预测，评分器随后读取固定标签与评分清单：

| 指标 | 评分范围 | 发布门槛 |
| --- | --- | ---: |
| 基础 mAP50 | `base_test.txt` 中的基础类别 | `0.80` |
| New-mAP50 | 完整 `mixed_test.txt` 中的新增类别 | `0.60` |
| KRR | 完整 `mixed_test.txt` 中增量前后的基础类别 mAP50 比值 | `0.95` |

`base_train.txt`、`base_dev.txt`、`increment_train.txt` 和 `increment_dev.txt` 是检测训练入口。
`pool_train.txt` 与 `pool_dev.txt` 生成这些类别隔离清单。在线路由使用当前 production 代际、
无标签图像内容和场景软证据；评分器在预测冻结后读取评分标签与评分子集。

## 已知场景识别

场景模型使用 `strict_3plus1/scene_train.txt`、`scene_dev.txt` 和 `scene_test.txt`，规模为 573/88/89，覆盖 air、forest、sea、urban 四个已知场景。
场景模型训练输入由图像、传感器标签和场景标签组成。

## 重新生成

使用可独立拆分的四类数据和 metadata 生成同一协议：

```bash
python tools/02_split_dataset.py \
  --increment-class warship \
  --output-dir reports/splits_check
```
