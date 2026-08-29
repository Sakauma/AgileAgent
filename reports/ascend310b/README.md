# Ascend310B 评分证据索引

当前正式证据是 [`20260829-full-score-recheck-v1/`](20260829-full-score-recheck-v1/README.md)：Base mAP50、New-mAP50、KRR 与公共 `8501` schema v8 全流程 FPS 四项门禁均通过。

旧 engine-only 报告已统一移入 [`archive/legacy-engine-only/`](archive/legacy-engine-only/README.md)。不可变 release 中随包冻结的 schema v5/v6/v7 文件不做改写，发布验证器继续兼容这些历史包；新候选与当前成绩只接受 schema v8。
