# 开发指南

## 环境准备

在 WSL/Linux 仓库根目录执行：

```bash
./scripts/bootstrap_x86.sh
```

脚本准备 Python、CUDA PyTorch、开发依赖、Web 依赖和模型推理依赖，并以 editable 模式注册当前仓库。解释器路径保存在 `.agent-python`。

手动安装方式：

```bash
python -m pip install -c constraints-agent.txt -e ".[dev,workbench,inference]"
```

## 日常开发循环

```bash
python -m pytest -q
python scripts/verify_release.py
./scripts/start_agent.sh
```

模型与推理变更同时执行：

```bash
python scripts/smoke_models.py
```

Ascend 相关变更使用：

```bash
python -m pytest -q tests/test_ascend_acl.py tests/test_runtime_maturity.py
python tools/91_smoke_ascend_contract.py
```

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

## 提交

提交主题使用简短英文祈使句，例如：

```text
Simplify runtime configuration
Document current Ascend runtime
```

每个提交围绕一个明确目标组织，并在正文中记录关键行为变化与验证命令。
