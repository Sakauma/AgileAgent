# TensorRT 部署指南

TensorRT 是可选的本地加速后端。AgileAgent 默认使用 CUDA 版 PyTorch 加载训练权重，无需 TensorRT 即可启动。

## 准备环境

在目标 NVIDIA GPU 设备上安装可选依赖：

```bash
AGENT_PYTHON=/path/to/env/bin/python
"$AGENT_PYTHON" -m pip install -e ".[workbench,inference,tensorrt,export]"
```

创建本地配置：

```bash
PROFILE="configs/agent_pipeline.local.yaml"
cp configs/agent_pipeline.yaml "$PROFILE"
```

修改 `$PROFILE`：

- `inference.backend`: 导出与验收阶段保持 `ultralytics_cuda`
- `runtime.default_device`: 使用的 GPU 编号
- `tensorrt_backend.export.device`: 导出使用的 GPU 编号
- `tensorrt_backend.expected_version`: 当前 TensorRT 版本
- `tensorrt_backend.expected_compute_capability`: 当前 GPU 计算能力
- `tensorrt_backend.validated`: `false`
- `tensorrt_backend.precision`: `fp16` 或 `int8`
- `tensorrt_backend.dynamic`: 同时启用动态 batch 与矩形空间尺寸
- `tensorrt_backend.minimum_spatial_size`: 动态空间 profile 的最小边长，必须是 32 的倍数
- `tensorrt_backend.engines.*.opt_height/opt_width`: 当前模型校准和 tactic 优化使用的代表性矩形尺寸
- `tensorrt_backend.mixed_precision`: 使用 glob 选择必须保留 FP16 的敏感层，并设置精度约束与最小命中层数

engine 路径应位于 `runs/engines/`；首次导出时对应 `sha256` 保持为 `null`。

## 导出与校验

```bash
./scripts/export_tensorrt_engines.sh "$PROFILE"
```

脚本依次完成环境核对、三份 engine 导出、真实 SHA256 回填和只读完整性校验。YAML 中的 `min_batch_size / opt_batch_size / batch_size` 会作为真实 TensorRT optimization profile 写入 engine，默认值为 `1 / 8 / 20`。脚本不会安装依赖、修改训练权重或将部署产物加入 Git。

选择 INT8 时可直接运行完整自动流程：

```bash
"$AGENT_PYTHON" -m fair_agent.cli --config "$PROFILE" \
  tensorrt calibrate --activate
```

Agent 会按固定种子选择最多 `int8_calibration.max_images` 张代表图像，保证每个模型所有类别达到最小覆盖，生成校准 manifest 和缓存后再构建 engine。基础检测器使用基础 train；当前增量检测器只使用对应增量类别样本；场景模型使用跨传感器与场景的 train 样本。校准过程禁止访问 lock。

混合精度或 INT8 engine 构建完成后，Agent 还会仅使用 `int8_calibration.threshold_split` 中的严格增量 dev 样本重新校准专家激活阈值。阈值冻结后才允许读取 lock 进行部署复核。

## 上线验收

使用专用门禁命令在同一 lock 上复算基础 mAP50、New-mAP50、KRR、组合 mAP50，并继续记录 CUDA/TensorRT 差值、API 平均延迟、P95、动态批量吞吐和8请求并发：

```bash
"$AGENT_PYTHON" -m fair_agent.cli --config "$PROFILE" \
  tensorrt validate --activate
```

`--activate` 的指标硬门禁只采用赛题口径：基础 mAP50、New-mAP50、KRR、组合 mAP50 和 FPS；数据隔离、阈值生成、资产哈希等完整性检查仍属于上线前置条件。CUDA/TensorRT 差值、lock precision、误激活率、API 平均延迟、P95 和并发结果继续写入报告，但仅作为部署风险告警，不否决候选。当前检测器推荐将模块 `0-1` 量化为 INT8，并通过 `mixed_precision.fp16_layer_patterns` 强制模块 `2-23` 保持 FP16。通过后运行：

```bash
"$AGENT_PYTHON" -m fair_agent.cli --config "$PROFILE" doctor
"$AGENT_PYTHON" -m fair_agent.cli --config "$PROFILE" serve
```

设备配置、engine、性能报告和运行缓存均为本地部署产物，不纳入版本控制。
