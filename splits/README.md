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
活动目录不发布两组图像的成员清单，预测冻结前不能获知单张测试图属于哪一组。
冻结基础检测器和增量专家都必须对完整混合测试集的每张图推理。正式三项门槛固定为：不含新增类别的基础测试代理 `base_test_map50 >= 0.80`、完整混合测试预测中只按新增类别计算的 `new_map50 >= 0.60`，以及同一完整混合测试上的 `KRR >= 0.95`。`base_dev` 只用于选权重，四类总体 mAP50、precision 与误激活率只作诊断。

89 张图必须先以无标签清单同时送入两个类别 owner，原始预测与融合预测保存并哈希后才能读取标签。解封后，70 张不含新增类别的图像用于当前 `base_test_map50` 代理；评分前不得根据标签、文件名、数据集身份或场景类别决定运行哪个模型。

当前候选训练模板在基础阶段生成三类检测器，并按赛题 mAP50 保存最佳权重；增量阶段只训练单类专家。推理时冻结基础检测器永久拥有旧类，增量专家拥有新类，两者在每张图上并行运行并做框级融合。旧类预测不再执行第二次 NMS，避免 Agent 后处理无意降低 KRR；全流程不使用场景到目标类别的硬绑定。

`pool_train.txt` 与 `pool_dev.txt` 只是生成上述模型专用清单的源池，不能直接作为三类基础检测器的训练数据。

## 已知场景识别

场景模型使用 `scene_train/dev/test.txt`（522/88/89），覆盖 air、forest、sea、urban 全部已知场景。场景训练只能读取场景/传感器标签，不得读取目标类别标签、共享检测器特征或建立场景到目标类别的硬绑定。

## 连续帧隔离

源池为 522/88/89，另有 51 张边界隔离图。任意训练、开发和测试序列边界帧距均大于 4。旧随机逐帧划分已归档到 `archive/splits_legacy_random_560_95_95/`，只用于历史复现。

可用其他可独立拆分的类别重新生成模板实例：

```bash
python tools/02_split_dataset.py --protocol strict-3plus1 --increment-class warship --output-dir reports/splits_check
```
