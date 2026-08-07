# 严格 3+1 类别增量数据划分

该目录只描述一套固定的 3+1 模拟实验，不做交叉验证，也不轮换新增类别。
当前实例暂时把 `warship` 作为模拟新增类别；正式官方增量数据到达后，当前四类全部属于基础类。

## 检测模型数据边界

| 阶段 | 清单 | 图片数 | 可见目标类别 |
|---|---|---:|---|
| 三类基础训练 | `strict_3plus1/base_train.txt` | 441 | soldier, small_aircraft, tank |
| 三类基础验证 | `strict_3plus1/base_dev.txt` | 70 | soldier, small_aircraft, tank |
| 单类增量训练 | `strict_3plus1/increment_train.txt` | 132 | warship |
| 单类增量验证 | `strict_3plus1/increment_dev.txt` | 18 | warship |
| 最终混合测试 | `strict_3plus1/mixed_test.txt` | 89 | 全部四类 |

混合测试集由 70 张旧类图和 19 张新增类图组成；不要求同一张图同时含旧类和新类。
活动目录不发布旧/新增类别成员清单，单张测试图身份在预测冻结前保持未知。
冻结基础检测器和增量专家都必须先对完整混合测试集的每张图执行无标签推理并冻结预测，再解封标签评分。
正式门槛固定为基础测试代理 mAP50 >= 0.80、New-mAP50 >= 0.60、KRR >= 0.95；base_dev 只用于选权重，四类总体 mAP50 只作诊断。
不得依据测试标签、文件名、数据集身份或场景类别决定是否运行某个类别 owner。

`pool_train.txt` 与 `pool_dev.txt` 只是生成上述模型专用清单的源池，不能直接作为三类基础检测器的训练数据。

## 已知场景识别

场景模型使用 `scene_train/dev/test.txt`（573/88/89），覆盖 air、forest、sea、urban 全部已知场景。场景训练只能读取场景/传感器标签，不得读取目标类别标签、共享检测器特征或建立场景到目标类别的硬绑定。

## 750 张全量覆盖

源池为 573/88/89，三者互斥且恰好覆盖全部 750 张图。上一版的 51 张边界隔离图已全部并入训练源池，活动划分不再强制连续帧边界间距；3+1 类别隔离和测试标签封存约束保持不变。上一版严格时序划分已归档到 `archive/splits_strict_temporal_3plus1_405_117/`，旧随机逐帧划分已归档到 `archive/splits_legacy_random_560_95_95/`。

可用其他可独立拆分的类别重新生成模板实例：

```bash
python tools/02_split_dataset.py --protocol strict-3plus1 --increment-class warship --output-dir reports/splits_check
```
