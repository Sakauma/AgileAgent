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

配置变更同步更新 schema 校验、两套主配置、测试和配置文档。模型变更同步更新 manifest、generation registry、SHA256、指标证据和发布校验。Ascend 正式资产还必须同步更新 `models/ascend310b/` 包、包内 `SHA256SUMS`、`models/SHA256SUMS.txt`、`models/manifest.json` 和 `tests/test_ascend_packaged_release.py`；不得只更新板端 release。

当前 `20260823-4plus2-yolo26-content-gate-v2` 是 production 不可变发布包；`20260816-full-score-1493b04` 是历史包。普通功能开发不得覆盖任一已发布包；新模型先生成新的 release ID 和目录，完成评分与包验证后再更新 manifest 的 production 指针。`.om/.onnx/.pt` 通常被忽略，只有正式包路径允许精确白名单，提交前必须用 `git check-ignore -v` 和 `git ls-files` 确认二进制确实进入版本控制。

## Ascend 满分方法维护

部署或复核当前模型使用 `scripts/materialize_ascend310b_full_score_release.sh`，不重新训练或运行 ATC。形成新 release 时，`configs/ascend310b/full_score_method.yaml` 是单一方法源。

| 变更范围 | 必须同步的实现或证据 |
| --- | --- |
| Base/Incremental 训练与类别轮次 | `tools/04`–`tools/13`、类别注册表和数据隔离证据 |
| Scene 与系统校准 | `tools/60`–`tools/61`、`tools/09`–`tools/10` |
| YOLO26 E2E 输入/输出 | ONNX、AIPP、`scripts/build_ascend_yolo26_e2e_oms.sh`、build manifest |
| 候选配置与内容门控 | `tools/112_materialize_ascend_yolo26_candidate.py`、generation registry、配置校验 |
| 比赛门禁与排序 | `tools/94_score_ascend_agent.py`、`tools/97_benchmark_ascend_api.py`、`tools/110_select_ascend_full_score_candidate.py` |
| 正式发布与回滚 | `tools/111_promote_ascend_full_score_release.py`、systemd 安装器、路由脚本 |
| 预构建包 | `models/manifest.json`、全局/包内清单、package test 和文档 |

开发约束：

- Base 固定负责全局类 0–3，Specialist 固定负责 4–5；
- 阈值是 Host 配置，不写入 OM 身份；
- 双证据门控只能读取 Scene 概率与 Base 检测，不读取标签或文件名；
- 方法 YAML 不写板端绝对候选路径、密码或数据集；
- 候选只能使用 `8502`，不得停止或复用未知进程；
- 正式主实例使用 `18501`，公共入口保持 `8501`；
- 数据隔离、预测先冻结、Base 冻结和资产身份是结果有效性的前置条件；
- precision、误激活率和单请求时延是诊断，不替代四项比赛门禁；
- 旧共享双头工具继续作为历史兼容代码，不得误写成当前 production。

按变更范围选择验证。完整回归可运行 pytest；仅做发布归档时至少执行语法、JSON/YAML、release verifier、包清单和凭据扫描。板端只运行探针、预测冻结、精度评分与 `30 + 3×20` batch，不运行 Web pytest。
## 提交

提交主题使用简短英文祈使句，例如：

```text
Simplify runtime configuration
Document current Ascend runtime
```
每个提交围绕一个明确目标组织，并在正文中记录关键行为变化与验证命令。
