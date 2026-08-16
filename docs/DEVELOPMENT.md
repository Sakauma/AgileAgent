# 开发指南

## 环境准备

在 WSL/Linux 仓库根目录执行：

```bash
./scripts/bootstrap_x86.sh
```

脚本准备 Python、CUDA PyTorch、开发依赖、Web 依赖和模型推理依赖，并以 editable 模式注册当前仓库。解释器路径保存在 `.agent-python`。

手动安装方式：

```bash
.venv/bin/python -m pip install -c constraints-agent.txt -e ".[dev,workbench,inference]"
```

## 日常开发循环

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_release.py
./scripts/start_agent.sh
```

模型与推理变更同时执行：

```bash
.venv/bin/python scripts/smoke_models.py
```

Ascend 相关变更使用：

```bash
.venv/bin/python -m pytest -q tests/test_ascend_acl.py tests/test_runtime_maturity.py
.venv/bin/python tools/91_smoke_ascend_contract.py
```

已有 `.venv` 的满分方法维护任务不得重新安装依赖或下载 CPU PyTorch；直接使用 `.venv/bin/python`。只有初始化一台全新的开发机时才运行 bootstrap 或手动安装命令。

## 代码组织

| 路径 | 责任 |
| --- | --- |
| `fair_agent/core/` | 配置、哈希、日志和通用基础设施 |
| `fair_agent/backends/` | 推理后端适配器 |
| `fair_agent/modules/` | 数据、训练、评测、代际与部署流程 |
| `fair_agent/policies/` | 决策与路由策略 |
| `fair_agent/executors/` | 受控动作执行器 |
| `fair_agent/web/` | FastAPI 服务与前端 |
| `configs/` | 运行、模型与实验配置 |
| `scripts/` | 环境、启动和发布脚本 |
| `tools/` | 编号工具入口 |
| `tests/` | 单元测试与集成回归 |

## 编码风格

- Python 使用四空格缩进和类型注解；
- 模块、函数、变量和 YAML 键使用 `snake_case`；
- 类使用 `PascalCase`；
- 常量使用 `UPPER_SNAKE_CASE`；
- 导入按标准库、第三方库和项目模块分组；
- CLI 与 Web 文案使用简体中文；
- 可配置值进入 YAML；
- 文件修改保持相邻代码风格。

## 测试写法

- 测试文件命名为 `tests/test_<能力>.py`；
- 测试函数命名为 `test_<预期行为>()`；
- 文件系统测试使用 `tmp_path`；
- 环境与外部依赖使用 `monkeypatch`；
- 数据测试使用小型合成 PNG、ZIP 和临时注册表；
- 新增分支同时覆盖成功路径、校验路径和状态转换。

## 配置与资产变更

配置变更同步更新 schema 校验、两套主配置、测试和配置文档。模型变更同步更新 manifest、generation registry、SHA256、指标证据和发布校验。

## Ascend 满分方法维护

`configs/ascend310b/full_score_method.yaml` 是满分流程的单一方法源。修改共享双头训练、导出、运行时或评分逻辑时，至少同步检查：

| 变更范围 | 必须同步的实现/证据 |
| --- | --- |
| 训练参数或冻结范围 | `tools/107_train_shared_dual_head.py`、training report schema、best/last 零漂移测试 |
| 输入、输出或 logical head | `tools/108_export_ascend_dual_head.py`、build 脚本、Ascend release 校验、owner/class map 测试 |
| CANN/ATC/AIPP | `scripts/build_ascend_dual_head_om.sh`、build manifest 哈希契约；CANN 仍固定 `7.0.RC1` |
| 候选配置 | `tools/109_materialize_ascend_full_score_candidate.py`、`8501` 保护和 `validated: false` 测试 |
| 比赛门禁或排序 | `tools/94_score_ascend_agent.py`、`tools/97_benchmark_ascend_api.py`、`tools/110_select_ascend_full_score_candidate.py` |
| 板端评分协议 | `scripts/run_ascend310b_score_gate.sh`、score/benchmark schema 和正式服务健康检查 |

开发约束：

- 不把阈值写入 OM/build manifest 身份；old/new 阈值由 logical head 的 Host 参数承载。
- 不把板端绝对路径、密码、候选产物或数据集写入方法 YAML 或 Git。
- 不在候选流程停止、替换或复用未知的 `8501`/`8502` 进程。
- 不以逐框/业务 JSON、precision、误激活率或单请求 P95/P99 否决四项比赛指标已满分的候选。
- 数据隔离、预测先冻结、共享参数零漂移和资产哈希仍是结果有效性的前置条件。
- P7/P10 历史实验保存在本地 ignored archive；活动代码只维护共享双头满分路径和正式三 OM 回滚路径。

修改后优先运行：

```bash
.venv/bin/python -m pytest -q \
  tests/test_ascend_full_score_workflow.py \
  tests/test_shared_dual_head.py \
  tests/test_configuration_runtime.py \
  tests/test_ascend_release.py \
  tests/test_strict_incremental.py
.venv/bin/python -m ruff check \
  tools/107_train_shared_dual_head.py \
  tools/108_export_ascend_dual_head.py \
  tools/109_materialize_ascend_full_score_candidate.py \
  tools/110_select_ascend_full_score_candidate.py
bash -n scripts/build_ascend_dual_head_om.sh
bash -n scripts/run_ascend310b_score_gate.sh
```

板端只运行探针、预测冻结、三项精度评分和 `30 + 3×20` batch 验收，不运行 Web pytest。完整操作顺序见 [`ascend-310b-full-score-method.md`](ascend-310b-full-score-method.md)。

## 提交

提交主题使用简短英文祈使句，例如：

```text
Simplify runtime configuration
Document current Ascend runtime
```

每个提交围绕一个明确目标组织，并在正文中记录关键行为变化与验证命令。
