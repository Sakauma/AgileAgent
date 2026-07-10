# Repository Guidelines

## 项目结构与模块组织

`fair_agent/` 是智能体主包：`core/` 管理配置、黑板和审计，`modules/` 解析实验证据，`policies/` 生成决策，`executors/` 执行低风险动作，`ui/` 提供 Streamlit 工作台。运行参数集中在 `configs/`，维护脚本位于 `scripts/`，数据处理和提交推理入口位于 `tools/`，自动化测试位于 `tests/`。冻结权重存放在 `models/`，脱敏演示证据存放在 `demo_artifacts/`。`reports/`、`runs/`、数据集和凭据均为本地私有产物，不得提交。

## 构建、测试与开发命令

仅支持 WSL/Linux 和 NVIDIA GPU。首次配置运行：

```bash
./scripts/bootstrap_x86.sh
```

环境就绪后，日常使用 `./scripts/start_agent.sh` 一键启动；该脚本不得安装依赖或修改环境。常用验收命令：

```bash
pytest -q
python scripts/verify_release.py
python scripts/smoke_models.py
```

前两项验证逻辑、配置和资产哈希；最后一项在 GPU 0 上加载六份权重、验证三种不同功能，并执行 `imgsz=640`、`batch=32` 推理。

## 编码风格与命名规范

使用 Python 3.10-3.12、4 空格缩进和类型标注。函数及变量采用 `snake_case`，类采用 `PascalCase`，配置键保持小写。新增运行参数优先写入 YAML，不把长参数串硬编码进 shell。面向用户的文档、CLI 提示和 UI 文案使用简体中文。输出必须写入配置允许的目录，并使用独立 `run_id`，不得覆盖固定报告。

## 测试与校验规范

测试文件命名为 `tests/test_*.py`，测试函数命名为 `test_*`。修改策略、路径门禁、黑板新鲜度或提交逻辑时必须补充回归测试。提交前至少运行完整 Pytest 和静态发布验收；涉及模型、依赖或推理配置时还需运行 GPU 冒烟测试。

## 提交与 Pull Request 规范

提交信息使用简短祈使句，例如 `Fix pipeline audit loop`。每个提交聚焦一个完整变更。Pull Request 应说明目的、行为变化、已执行的验收命令和剩余阻塞项；UI 变化附截图，模型变化附指标、配置及 SHA256。禁止提交竞赛原始数据、标签、SSH 凭据和未经授权的派生产物。
