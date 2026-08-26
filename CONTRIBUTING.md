# 参与 AgileAgent 开发

AgileAgent 的贡献流程围绕可复验的配置、数据协议、模型代际和硬件证据展开。代码、配置、模型或文档更新都应保持六类全局 ID、类别 owner、增量轮次和 production 身份链一致。

## 开发环境

支持 Python 3.10–3.12。在 WSL/Linux 仓库根目录执行：

```bash
chmod +x scripts/bootstrap_x86.sh scripts/start_agent.sh
./scripts/bootstrap_x86.sh
python -m pip install -c constraints-agent.txt -e ".[dev,workbench,inference]"
```

`bootstrap_x86.sh` 会优先复用当前激活的 Conda/venv Python，补齐缺少的项目依赖，并将解释器位置写入本地 `.agent-python`。运行 Web 或 CLI：

```bash
./scripts/start_agent.sh
./scripts/start_agent.sh --cli
```

## 修改原则

- 配置字段同步更新 `fair_agent/core/config.py`、示例配置、契约测试和 `docs/CONFIGURATION.md`。
- 类别与轮次从 `configs/incremental_round_registry_4plus2.yaml` 和 generation registry 读取，保持全局类别 ID 与 owner 稳定。
- 涉及 lock 的流程先冻结候选和预测，再读取 lock 标签；训练/选择只读取对应阶段的 train/dev。
- production 更新通过候选、manifest、门禁和晋级工具完成；板端演示候选写入独立运行目录并保留父代。
- 硬件性能同时记录设备、CANN/CUDA 环境、输入列表、预热、重复轮次、计时字段和聚合方式。
- 文档保留历史实验与证据，同时将当前入口、配置、指标和发布身份放在显著位置。

## 提交前验证

默认回归：

```bash
python -m pytest -q
python scripts/verify_release.py
python tools/03_split_r2_4plus2.py --verify-only
git diff --check
```

Shell 与本次 Python 改动的静态检查：

```bash
bash -n scripts/*.sh extras/ascend_edge_incremental/*.sh
git diff --name-only --diff-filter=ACMR -- '*.py' | xargs -r ruff check
```

修改推理模型、融合、阈值或发布配置时，还需按 [`docs/TESTING.md`](docs/TESTING.md) 完成 x86/CUDA lock 验收与 Ascend310B score gate。修改板端 Adapter 或一键演示时，执行 [`docs/ascend-310b-offline-incremental-demo.md`](docs/ascend-310b-offline-incremental-demo.md) 中的真实 NPU、OM、ACL 和 FPS 验收。

## 文档与提交

公开入口从 [`README.md`](README.md) 开始；架构、配置、API、部署、测试和现场流程分别维护在 `docs/`。提交信息使用简洁的祈使句并聚焦一个交付主题。提交前检查暂存区，确保训练缓存、数据集、凭据和本地运行产物保持在忽略范围内。

本项目采用 [MIT License](LICENSE)。提交贡献即表示相关内容可以按该许可证发布。
