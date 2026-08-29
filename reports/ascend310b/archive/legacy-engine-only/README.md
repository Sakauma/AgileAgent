# Ascend310B legacy engine-only 档案

本目录只保存旧计时契约的原始报告。归档内容可以用于性能演进分析和不可变 release 兼容复核，但不满足当前官方 FPS 契约：

- 门禁必须使用 schema v8；
- FPS 必须按全部处理帧数除以全部轮次的全流程总墙钟耗时计算；
- 计时必须覆盖解码、Scene、决策、Base/Incremental 检测、后处理和正式六列结果写出；
- `includes_result_persistence` 与 `formal_results_valid` 必须均为 `true`。

当前满分复核证据见 [`../../20260829-full-score-recheck-v1/`](../../20260829-full-score-recheck-v1/README.md)。
