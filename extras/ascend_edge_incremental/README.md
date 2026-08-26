# Ascend310B 板端轻量增量训练

本目录提供 Ascend310B 端侧增量功能：冻结现有 Base、Incremental 和 Scene-SensorNet，训练每个注册新增类别 8 个参数的无 MatMul 置信度残差 Adapter，并完成 dev 强度选择、lock 评分、ONNX/OM 导出、ACL 数值验证、隔离演示部署和完整运行时 FPS 复测。

主要入口：

- `scripts/run_ascend310b_incremental_demo.sh`：现场主入口，仅给当前 Increment 数据目录即可断网执行 `4→4+2` 训练、隔离部署和真实运行时验收；
- `bootstrap_env.sh`：从 production Python 离线克隆独立 Conda 环境，再安装经过验证的 PyTorch/torch_npu aarch64 wheel；
- `run_pipeline.sh`：执行完整注册表驱动流水线；
- `workflow.py plan`：只输出逐阶段命令，不创建文件或运行训练；
- `protocol.py`：校验轮次、类别、数据范围、父子代际和输出隔离；
- `train.py`、`calibrate.py`、`evaluate.py`：分别对应增量学习、系统校准和联合评估；
- `export_onnx.py`、`benchmark_om.py`：导出静态 Adapter OM 并验证数值与延迟。
- `promote_demo.py`、`benchmark_demo_runtime.py`：写入隔离演示配置，并在 Adapter 真正接入完整图像链路后重测 FPS。

所有运行产物写入独立 `runs/ascend_edge_incremental_demo/<run_id>/`，训练固定使用 `npu:0`；production 模型、配置和固定 splits 作为只读父代。通过门禁的 Adapter 由独立演示配置加载，运行数据、checkpoint、ONNX 和 OM 保留在板端运行目录。

2026-08-26 `board-full-check-v6` 已完成断网整链验收：Base/New/KRR/Full mAP50 为 `0.816663 / 0.624935 / 1.000000 / 0.726497`，Adapter OM 最大绝对误差 `5.96e-08`，完整图像链路中位 `38.6995 FPS`，热态完整命令 `1007.07 秒`。候选状态为 `accepted`，训练审计中 Base 与旧类原始图像计数均为 `0`。

现场一键操作见 [`docs/ascend-310b-offline-incremental-demo.md`](../../docs/ascend-310b-offline-incremental-demo.md)；底层前提、命令、产物和实测结果见 [`docs/ascend-310b-edge-incremental-training.md`](../../docs/ascend-310b-edge-incremental-training.md)。
