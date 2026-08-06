# 严格 3+1 类别增量数据划分

该目录只描述一套固定的 3+1 模拟实验，不做交叉验证，也不轮换新增类别。
当前实例暂时把 `warship` 作为模拟新增类别；正式官方增量数据到达后，当前四类全部属于基础类。

## 检测模型数据边界

| 阶段 | 清单 | 图片数 | 可见目标类别 |
|---|---|---:|---|
| 三类基础训练 | `strict_3plus1/base_train.txt` | 405 | soldier, small_aircraft, tank |
| 三类基础验证 | `strict_3plus1/base_dev.txt` | 70 | soldier, small_aircraft, tank |
| 单类增量训练 | `strict_3plus1/increment_train.txt` | 117 | warship |
| 单类增量验证 | `strict_3plus1/increment_dev.txt` | 18 | warship |
| 最终混合测试 | `strict_3plus1/mixed_test.txt` | 89 | 全部四类 |

混合测试集由 70 张旧类图和 19 张新增类图组成；不要求同一张图同时含旧类和新类。
父代和增量后的候选代都必须对完整混合测试集推理，再分别计算 Old-mAP、New-mAP 和 KRR。

`pool_train.txt` 与 `pool_dev.txt` 只是生成上述模型专用清单的源池，不能直接作为三类基础检测器的训练数据。

## 已知场景识别

场景模型使用 `scene_train/dev/test.txt`（522/88/89），覆盖 air、forest、sea、urban 全部已知场景。场景训练只能读取场景/传感器标签，不得读取目标类别标签、共享检测器特征或建立场景到目标类别的硬绑定。

## 连续帧隔离

源池为 522/88/89，另有 51 张边界隔离图。任意训练、开发和测试序列边界帧距均大于 4。旧随机逐帧划分已归档到 `archive/splits_legacy_random_560_95_95/`，只用于历史复现。

可用其他可独立拆分的类别重新生成模板实例：

```bash
python tools/02_split_dataset.py --protocol strict-3plus1 --increment-class warship --output-dir reports/splits_check
```
