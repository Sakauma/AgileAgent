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

- `inference.backend`: `tensorrt_engine`
- `runtime.default_device`: 使用的 GPU 编号
- `tensorrt_backend.export.device`: 导出使用的 GPU 编号
- `tensorrt_backend.expected_version`: 当前 TensorRT 版本
- `tensorrt_backend.expected_compute_capability`: 当前 GPU 计算能力
- `tensorrt_backend.validated`: `false`

engine 路径应位于 `runs/engines/`；首次导出时对应 `sha256` 保持为 `null`。

## 导出与校验

```bash
./scripts/export_tensorrt_engines.sh "$PROFILE"
```

脚本依次完成环境核对、三份 engine 导出、真实 SHA256 回填和只读完整性校验。它不会安装依赖、修改训练权重或将部署产物加入 Git。

## 上线验收

先使用进程级临时配置执行环境、精度和性能验收：

```bash
"$AGENT_PYTHON" -m fair_agent.cli --config "$PROFILE" \
  --set tensorrt_backend.validated=true doctor
"$AGENT_PYTHON" -m fair_agent.cli --config "$PROFILE" \
  --set tensorrt_backend.validated=true \
  generation recheck --candidate incremental_detection_generation
"$AGENT_PYTHON" -m fair_agent.cli --config "$PROFILE" \
  --set tensorrt_backend.validated=true benchmark-api
```

三项验收均通过后，将 `$PROFILE` 中的 `tensorrt_backend.validated` 设为 `true`，再次运行 `doctor`，随后启动服务：

```bash
"$AGENT_PYTHON" -m fair_agent.cli --config "$PROFILE" doctor
"$AGENT_PYTHON" -m fair_agent.cli --config "$PROFILE" serve
```

设备配置、engine、性能报告和运行缓存均为本地部署产物，不纳入版本控制。
