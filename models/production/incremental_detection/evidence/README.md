# strict 4+2 production 证据

本目录绑定当前 4+2 Base 与联合二类增量专家的 dev 选模、训练参数和冻结预测，记录已通过六类指标的 production 性能。两轮顺序类别注入由 `configs/incremental_round_registry_4plus2.yaml` 和 `tools/06`–`tools/13` 驱动，其逐轮训练、复核、候选登记和证据汇总保存在独立证据链中。

联合二类专家权重训练只使用 Increment train/dev，属于 `incremental_learning`；Scene-SensorNet 与六类场景门控是 `system_calibration`，可用 Base/Increment train/dev 与 mixed dev，但不更新任何检测器权重；冻结候选的 mixed lock 复核是 `joint_evaluation`，不训练也不选参。顺序增量证据包含两个不同新类的分轮训练、父子代际与逐轮 New-mAP50/KRR/Full-mAP50。

- `base_selection.*` / `incremental_selection.*`：只按 dev mAP50 选模，未读取 lock；
- `base_training/` / `incremental_training/`：胜出模型的 `args.yaml` 与 `results.csv`；
- `frozen/`：Base/专家在 mixed dev 和 mixed lock 上的预测，lock 标签在预测冻结后才进入评分。
- `scene_aware_dev_search.*`：`system_calibration` 证据，使用 Scene-SensorNet 实际概率完成六类逐类 dev 搜索，未读取 lock 标签；
- `scene_aware_candidate.json`：dev 选择并冻结的 `guarded_precision` 候选；
- `scene_aware_lock_recheck.json`：`joint_evaluation` 证据，记录冻结候选的一次性 mixed lock 复核。
- `all_images_diagnostics.json/.md`：使用当前正式后处理并列记录一号 mixed lock 与二号全部 890 张标注图像；二号包含 train，只是诊断结果。
- `ascend310b_cross_class_replay.json`：对当前 310B release 的 89 张冻结板端预测执行新的全类别重叠抑制回放；它证明精度门禁仍通过，不包含修改后的板端 FPS。

严格两轮候选晋级后，本目录会新增 `rounds/<round_id>/` 与 `sequential_round_evidence.json/.md`，`incremental_selection.*` 会替换为两轮聚合记录；`models/generations.json` 成为两个单类专家的唯一运行来源，当前联合二类代际改为 `retired_baseline`。

最终评分、逐类阈值和非阻断诊断见上级目录的 `metrics.json` 与 `calibration.json`。

场景识别是 air/forest/sea/urban 四个已知类的闭集识别。`../base_context_prior.json` 只由 Base train 正样本学习，`../incremental_context_prior.json` 只由 Increment train 正样本学习。线上只读取 Scene-SensorNet 概率；场景亲和度同时软调节 Base `0–3` 和 Increment `4–5` 的有效阈值，但不读取文件名或标签、不改变 owner，也不做硬路由。

当前冻结阈值为 `0=.21, 1=.14, 2=.36, 3=.05, 4=.57, 5=.82`，最大场景惩罚为 `0=.15, 1=.88, 2=.26, 3=.19, 4=.65, 5=0`。全类别重叠抑制前的 lock 基线仍保存在 `operating_point_diagnostics.md`；正式推理结果使用下列双口径：

| 结果 | 图像范围 | Full-mAP50 | TP / FP / precision | 六类误激活图像 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 一号 | mixed lock，75 Base + 14 Increment | `0.800605` | `335 / 108 / 0.756208` | `11 / 89` | 独立测试口径、正式结果 |
| 二号 | train/dev/lock 全部 890 张 | `0.827444` | `3533 / 986 / 0.781810` | `88 / 890` | 含训练图像、仅作诊断 |

一号结果来自版本库中冻结的 lock 预测回放。本机全量推理中的 lock 复现值为 `0.800599`，TP/FP 相同，微小数值差异不会覆盖正式一号结果。二号结果不参与模型选择、阈值搜索或正式门禁。

最终后处理的逐类 TP、FP、precision、recall 与误激活率见 `all_images_diagnostics.md`。其中框级 FP 与图像级误激活是两个不同口径，不可直接互换。
