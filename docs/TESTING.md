<!-- generated-by: gsd-doc-writer -->
# 测试指南

## 测试框架与准备

项目使用 [pytest](https://docs.pytest.org/) 运行单元测试、集成回归和 Web 流程测试。项目元数据要求 `pytest>=8.0`，仓库的 [`constraints-agent.txt`](../constraints-agent.txt) 将可复现环境固定为 `pytest==9.0.2`；支持的 Python 版本为 `3.10` 至 `3.12`。pytest 配置位于 [`pyproject.toml`](../pyproject.toml)，其中 `testpaths = ["tests"]`。

测试会通过相对路径读取 `configs/`、`models/` 和 `scripts/`，因此所有命令都应从仓库根目录执行。复现当前 GitHub Actions 声明的最小环境可按以下方式安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints-agent.txt -e ".[dev,workbench]"
```

[`requirements-agent-dev.txt`](../requirements-agent-dev.txt) 是较早的汇总入口：它通过 [`requirements-agent.txt`](../requirements-agent.txt) 带入 Ultralytics 等推理依赖，但没有项目 `dev` extra 中的 `httpx2`，因此不再把它描述为完整测试环境。Web 测试使用 Starlette `TestClient`、Pillow 和 workbench 依赖；仓库没有全局 `conftest.py` 或额外 pytest 启动钩子。

当前还有一个依赖声明缺口：[`tests/test_strict_incremental.py`](../tests/test_strict_incremental.py) 在收集阶段直接导入 `torch`，而 CI 安装的 `.[dev,workbench]` 没有显式声明 PyTorch。GitHub runner 或已有环境中碰巧存在 `torch` 不能替代依赖声明；本地 x86/CUDA 环境可使用 [`scripts/bootstrap_x86.sh`](../scripts/bootstrap_x86.sh) 建立完整环境，CPU/CI 干净环境则需显式安装兼容的测试版 PyTorch。工程侧后续应在测试依赖中明确声明兼容 PyTorch，或把相关用例改成有原因的可选跳过。

部分用例会根据本地能力跳过：

- [`tests/test_agent_workbench.py`](../tests/test_agent_workbench.py) 中两项发布资产测试仅在 `models/production/incremental_detection/three_class_base_detector.pt` 存在时运行。
- Ascend 板前测试按用例使用 `pytest.importorskip()` 检查 OpenCV、NumPy、TorchVision、Ultralytics 或 ONNX。
- pytest 套件本身不执行真实 GPU 或 Ascend 设备推理；GPU 模型加载由 `python scripts/smoke_models.py --load-only` 单独验证。

## 运行测试

据本轮 Python 3.10 运行记录，测试套件成功收集 `221` 个用例；按当前 310B 部署范围执行了 Ascend 后端、Web 契约与静态发布检查相关的 `33` 项，结果为 `33 passed, 1 warning`。原始 stdout/JUnit 报告尚未纳入仓库，因此该次数值属于本轮审查记录；其余用例包含 x86/CUDA、训练和完整工作台能力，本轮未要求全部执行。

运行完整测试套件：

```bash
pytest -q
```

运行单个测试文件：

```bash
pytest -q tests/test_web_inference.py
```

运行单个测试函数：

```bash
pytest -q tests/test_web_ui_flow.py::test_health_and_static_product_contract
```

按名称筛选一组测试：

```bash
pytest -q -k incremental
```

发布配置和资产哈希检查不是 pytest 用例；它在 CI 中紧随测试套件运行，也可手动执行：

```bash
python scripts/verify_release.py
```

仓库没有配置 pytest watch 模式或专用的 `test:watch` 命令。需要反复运行某一范围时，直接使用文件、节点 ID 或 `-k` 过滤器。

## 测试范围

测试文件采用扁平目录组织，并按被测能力分组：

| 范围 | 主要测试文件 |
| --- | --- |
| Agent、配置与运行时 | `test_agent_workbench.py`、`test_configuration_runtime.py`、`test_functional_models.py`、`test_runtime_maturity.py` |
| 增量学习与代际生命周期 | `test_incremental_experiment.py`、`test_incremental_guardian.py`、`test_incremental_lifecycle_v2.py`、`test_incremental_methods.py`、`test_incremental_rejection.py`、`test_incremental_workbench.py`、`test_strict_incremental.py` |
| 在线推理与 Web | `test_unlabeled_inference.py`、`test_web_inference.py`、`test_web_ui_flow.py`、`test_submission_safety.py` |
| Ascend 合约与板前处理 | `test_ascend_acl.py`、`test_ascend_preflight.py` |
| 数据划分 | `test_archived_legacy_splits.py`、`test_strict_3plus1_splits.py` |

## 编写新测试

- 将测试放在 `tests/` 下，文件命名为 `test_<能力>.py`，测试函数命名为 `test_<预期行为>()`。当前套件不使用测试类。
- 优先使用 pytest 内置夹具：`tmp_path` 隔离文件写入，`monkeypatch` 替换外部或硬件边界，`capsys` 验证 CLI 输出。
- 仅由单个文件使用的构造器和假对象应留在该测试文件中。现有模式包括 `_fixture()`、`_png()`、`settings()`、`make_store()` 以及 Web 测试中的 `FakeEngine`；当前没有共享测试辅助模块。
- Web 路径使用 `starlette.testclient.TestClient` 调用由 `create_app()` 构造的应用，并以假推理引擎代替真实设备执行。
- 对可选库使用 `pytest.importorskip()`；对仓库外或未随检出提供的发布资产使用带明确原因的 `pytest.mark.skipif()`。不要让普通 pytest 回归依赖 GPU、Ascend 设备、外部服务或未提交的数据集。
- 涉及文件系统、安全门禁或数据隔离的测试应使用临时目录和合成的小型 PNG/ZIP 数据，避免写入活动的 `runs/`、`reports/` 或模型注册表。

## 覆盖率要求

仓库未配置覆盖率阈值（No coverage threshold configured）。没有发现 `.coveragerc`、`coverage.toml`、pytest-cov 参数或 CI `--cov` 步骤。

| 类型 | 阈值 |
| --- | --- |
| Lines | 未配置 |
| Branches | 未配置 |
| Functions | 未配置 |
| Statements | 未配置 |

因此，当前合入门禁依据测试结果和发布校验结果，而不是覆盖率百分比。

## CI 集成

GitHub Actions 工作流 [`tests.yml`](../.github/workflows/tests.yml) 名为 `tests`，在每次 `push` 和 `pull_request` 时运行；未配置分支过滤。`pytest` job 使用 `ubuntu-latest`，并分别在 Python `3.10` 和 `3.12` 上执行同一流程：

1. 使用 `actions/checkout@v4` 检出代码。
2. 使用 `actions/setup-python@v5` 安装矩阵中的 Python 版本。
3. 安装项目：

   ```bash
   pip install -c constraints-agent.txt -e ".[dev,workbench]"
   ```

4. 运行测试：

   ```bash
   pytest -q
   ```

5. 校验发布配置、模型资产和证据：

   ```bash
   python scripts/verify_release.py
   ```

任一 Python 版本上的 pytest 或发布校验失败都会使该矩阵任务失败。
