# 测试指南

AgileAgent 使用 Pytest 覆盖配置、数据、增量生命周期、模型代际、Web API、推理融合和 Ascend 契约。

## 完整回归

```bash
.venv/bin/python -m pytest -q
```

测试数量会随功能增长，提交记录应写明本次实际通过数量，不在本文维护容易过期的固定计数。

## 发布校验

```bash
.venv/bin/python scripts/verify_release.py
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
.venv/bin/python scripts/smoke_models.py
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
| Ascend 满分方法 | `test_ascend_full_score_workflow.py`、`test_shared_dual_head.py`、`test_ascend_release.py` |
| 多批次增量 | `test_incremental_guardian.py`、`test_incremental_experiment.py` |

## 聚焦运行

单文件：

```bash
.venv/bin/python -m pytest -q tests/test_web_inference.py
```

单节点：

```bash
.venv/bin/python -m pytest -q \
  tests/test_configuration_runtime.py::test_default_config_loads
```

关键字过滤：

```bash
.venv/bin/python -m pytest -q -k "generation or incremental"
```

Ascend 契约：

```bash
.venv/bin/python -m pytest -q \
  tests/test_ascend_acl.py \
  tests/test_runtime_maturity.py
```

满分方法和候选选择：

```bash
.venv/bin/python -m pytest -q \
  tests/test_ascend_full_score_workflow.py \
  tests/test_shared_dual_head.py \
  tests/test_configuration_runtime.py \
  tests/test_ascend_release.py \
  tests/test_strict_incremental.py
bash -n scripts/build_ascend_dual_head_om.sh
bash -n scripts/run_ascend310b_score_gate.sh
```

### 满分工作流覆盖范围

| 契约 | 覆盖内容 |
| --- | --- |
| 方法配置 | schema、参考指标、无绝对板端路径、训练/导出/运行时/评分协议 |
| 训练 | residual adapter、Base/BN/EMA 零漂移、best/last 授权、增量数据隔离 |
| 导出和构建 | 双输出 shape、owner/class map、CANN/AIPP/ATC 与资产 SHA256 |
| 候选生成 | `8501` 保护、强制 `8502`、哈希不匹配拒绝、`validated: false` |
| 评分 | score schema v2、benchmark schema v5、三项精度和三轮 batch FPS |
| 选择器 | 四项门禁、诊断项不阻断、确定性排名、无满分候选分支 |
| 回滚 | raw dual head、context 回滚资产、异常清理和正式服务健康状态 |

评分报告的阻断边界：

- `competition_gates` 只包含 Base mAP50、New-mAP50 和 KRR；
- benchmark 只用三轮 20 图 batch 的中位 FPS 作为性能门禁；
- `diagnostic_warnings` 可以包含逐框/业务 JSON、precision、误激活率、Scene/Sensor 或单请求时延问题，但不得改变 `full_score`；
- `predictions_frozen_before_labels`、增量数据隔离、Base 零漂移和资产哈希属于有效性前置条件，失败时结果无效。

### 本机与板端测试边界

| 环境 | 执行内容 | 禁止事项 |
| --- | --- | --- |
| WSL 现有 `.venv` | Pytest、Ruff、配置/文档/哈希校验、训练与 ONNX 导出 | 不安装依赖，不下载 CPU PyTorch |
| Ascend 候选 `8502` | 探针、无标签预测冻结、三项精度评分、30 次预热、三轮 20 图 batch | 不运行 Web pytest，不停止或替换 `8501` |
| 正式 `8501` | score gate 开始和结束时检查 `/api/health` 为 `ready` | 整理和候选验收阶段不切换发布资产 |

板端输入必须是评分目录根层的 `640×512`、8-bit RGB/RGBA PNG，stem 唯一。任一输入契约、CANN 版本、候选端口或 manifest 哈希不一致时，score gate 在正式测量前停止并保留已有报告。

## 测试夹具

测试使用 `tmp_path` 创建临时配置、注册表、批次目录和模型占位文件；使用 Pillow 生成小型 PNG；使用 ZIP 构造增量上传包；使用假引擎验证 Web 与代际切换。所有测试状态保留在测试临时目录中。

## CI

GitHub Actions 在 Python 3.10 与 3.12 上安装 `.[dev,workbench]` 并运行完整 Pytest。发布前本地执行完整回归、发布校验和模型冒烟，形成与 CI 一致的验证链。
