<!-- generated-by: gsd-doc-writer -->
# 正式 4+2 比赛优先数据划分

该目录是 `4 旧类 + 2 新类` 正式工作的固定数据清单。划分以比赛得分为优先，
按 `sensor | scene` 分层后在帧级随机分配，不做时间隔离，并显式优选评估帧旁边存在训练帧的随机种子。

| 数据 | train | dev | lock | train+dev | all |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 四类 Base | 600 | 75 | 75 | 675 | 750 |
| R2 二类增量 | 112 | 14 | 14 | 126 | 140 |
| Round 1 patrol_boat | 56 | 7 | 7 | — | 70 |
| Round 2 armored_vehicle | 56 | 7 | 7 | — | 70 |

- 首轮开发：使用 `*_train.txt` 训练、`*_dev.txt` 选参、`*_lock.txt` 冻结评分。
- 比赛复训：超参和阈值冻结后，可使用 `*_train_plus_dev.txt` 重训，lock 仍保留。
- `*_all.txt` 仅用于明确决定放弃本地 lock 独立性后的全量重训；使用它后不再声称本地 lock 是独立评测。当前 production 未使用该视图，全量重训只由明确的比赛提交策略触发。
- R2 总清单引用原始六类标签。`round_01_patrol_boat_*` 与 `round_02_armored_vehicle_*` 在训练前按类别注册表从同一总清单固化，三种 split 均互斥且合并后严格等于对应 R2 总清单。
- 每轮专家训练视图只保留该轮 `new_class_ids` 并按注册表映射为局部类别；不得读取 Base 或上一轮增量图像，也不得改写原始标签。
- 轮次、类别映射和父子代际的单一来源是 `configs/incremental_round_registry_4plus2.yaml`，可用 `tools/11_prepare_incremental_round_splits.py` 复核清单内容。
- 每轮通过 `tools/08` 冻结评测后，必须用 `tools/13` 登记该轮专家；`tools/12` 只接受已登记的完整两轮父子链。登记 candidate 不会切换当前 production。
- 该工具只生成划分，不启动训练。

具体随机种子、各层配额、类别/传感器/场景分布和相邻帧覆盖率见 `manifest.json`。
