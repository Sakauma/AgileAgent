# Ascend310B 板端轻量增量训练

本目录提供隔离的可选功能：在 Ascend310B 上冻结现有 Base、Incremental 和 Scene-SensorNet，仅训练每个新增类别 8 个参数的无 MatMul 置信度残差 Adapter，并完成 dev 强度选择、lock 评分、ONNX/OM 导出和 ACL 延迟验证。

主要入口：

- `scripts/run_ascend310b_incremental_demo.sh`：现场主入口，仅给当前 Increment 数据目录即可断网执行 `4→4+2` 训练、隔离部署和真实运行时验收；
- `bootstrap_env.sh`：从 production Python 离线克隆独立 Conda 环境，再安装经过验证的 PyTorch/torch_npu aarch64 wheel；
- `run_pipeline.sh`：执行完整注册表驱动流水线；
- `workflow.py plan`：只输出逐阶段命令，不创建文件或运行训练；
- `protocol.py`：校验轮次、类别、数据范围、父子代际和输出隔离；
- `train.py`、`calibrate.py`、`evaluate.py`：分别对应增量学习、系统校准和联合评估；
- `export_onnx.py`、`benchmark_om.py`：导出静态 Adapter OM 并验证数值与延迟。
- `promote_demo.py`、`benchmark_demo_runtime.py`：写入隔离演示配置，并在 Adapter 真正接入完整图像链路后重测 FPS。

功能不会写入 `models/production/`、`models/ascend310b/`、`configs/` 或 `splits/`，也不允许 CPU fallback。运行产物、数据、checkpoint、ONNX 和 OM 不属于源码提交范围。

现场一键操作见 [`docs/ascend-310b-offline-incremental-demo.md`](../../docs/ascend-310b-offline-incremental-demo.md)；底层前提、命令、产物和实测结果见 [`docs/ascend-310b-edge-incremental-training.md`](../../docs/ascend-310b-edge-incremental-training.md)。
