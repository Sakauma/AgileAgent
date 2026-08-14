# 测试指南

AgileAgent 使用 Pytest 覆盖配置、数据、增量生命周期、模型代际、Web API、推理融合和 Ascend 契约。

## 完整回归

```bash
python -m pytest -q
```

当前完整回归确认：

```text
214 passed
```

## 发布校验

```bash
python scripts/verify_release.py
```

发布校验核对：

- schema 3 主配置；
- 发布模型路径与 SHA256；
- `models/manifest.json`；
- `models/generations.json`；
- 三个功能模型与证据；
- production 类别所有权和指标；
- 增量检测策略。

当前发布校验状态为 `passed`。

## 模型冒烟

```bash
python scripts/smoke_models.py
```

该脚本加载 Scene-SensorNet、三类基础检测器和增量检测器，并执行 production 模型组合冒烟。

## 测试范围

| 范围 | 主要测试 |
| --- | --- |
| 配置与 CLI | `test_configuration_runtime.py` |
| 数据与固定划分 | `test_strict_3plus1_splits.py`、`test_strict_incremental.py` |
| 增量工作台 | `test_incremental_workbench.py`、`test_incremental_lifecycle_v2.py` |
| 模型代际 | `test_runtime_maturity.py`、`test_incremental_lifecycle_v2.py` |
| 推理与融合 | `test_web_inference.py`、`test_incremental_rejection.py` |
| Web API 与 UI | `test_agent_workbench.py`、`test_web_ui_flow.py` |
| Ascend | `test_ascend_acl.py`、`test_ascend_preflight.py` |
| 多批次增量 | `test_incremental_guardian.py`、`test_incremental_experiment.py` |

## 聚焦运行

单文件：

```bash
python -m pytest -q tests/test_web_inference.py
```

单节点：

```bash
python -m pytest -q \
  tests/test_configuration_runtime.py::test_default_config_loads
```

关键字过滤：

```bash
python -m pytest -q -k "generation or incremental"
```

Ascend 契约：

```bash
python -m pytest -q \
  tests/test_ascend_acl.py \
  tests/test_runtime_maturity.py
```

## 测试夹具

测试使用 `tmp_path` 创建临时配置、注册表、批次目录和模型占位文件；使用 Pillow 生成小型 PNG；使用 ZIP 构造增量上传包；使用假引擎验证 Web 与代际切换。所有测试状态保留在测试临时目录中。

## CI

GitHub Actions 在 Python 3.10 与 3.12 上安装 `.[dev,workbench]` 并运行完整 Pytest。发布前本地执行完整回归、发布校验和模型冒烟，形成与 CI 一致的验证链。
