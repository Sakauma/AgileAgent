<!-- generated-by: gsd-doc-writer -->
# 开发指南

## 本地开发环境

### 前置条件

完整工作台面向 x86-64 Linux 或 WSL 2，要求 Python `>=3.10,<3.13` 和可用的 NVIDIA GPU 驱动（`nvidia-smi`）。经验上建议为 Python 环境、CUDA 依赖、模型和运行产物预留约 10 GB；仓库当前不自动检查磁盘空间，也没有固定安装体积基准。训练、模型加载、GPU 冒烟和板前代理测试需要 CUDA 版 PyTorch；常规单元测试可以使用与 CI 相同的无推理依赖环境，但完整套件目前仍有一项未显式声明的 `torch` 收集依赖，详见 [测试指南](TESTING.md)。

### Fork 与克隆

先在 GitHub 上 Fork `Sakauma/AgileAgent`，再克隆个人 Fork，并保留已验证的上游地址：

```bash
git clone git@github.com:<your-github-account>/AgileAgent.git
cd AgileAgent
git remote add upstream https://github.com/Sakauma/AgileAgent.git
```

若只需要从上游只读检出，可以直接执行：

```bash
git clone https://github.com/Sakauma/AgileAgent.git
cd AgileAgent
```

### 完整 GPU 开发环境

仓库提供的引导脚本会选择或创建兼容的 Python 环境，补齐 CUDA PyTorch、`workbench`、`inference` 和 `dev` 依赖，以 editable 模式注册当前检出，并执行环境检查和三模型加载冒烟：

```bash
chmod +x scripts/bootstrap_x86.sh scripts/start_agent.sh
./scripts/bootstrap_x86.sh
```

如需复用已有 Python 3.10–3.12 环境，可显式指定解释器：

```bash
AGILE_AGENT_PYTHON=/path/to/env/bin/python ./scripts/bootstrap_x86.sh
```

引导成功后，所选解释器路径写入 Git 忽略的 `.agent-python`；日常启动脚本会优先读取该文件。引导脚本默认安装 PyTorch `2.5.1+cu124` 和 TorchVision `0.20.1+cu124`，但检测到兼容的现有 CUDA 环境时会直接复用。

### 仅测试环境

无需运行模型时，可复用 [`.github/workflows/tests.yml`](../.github/workflows/tests.yml) 中的 CI 安装方式：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-agent.txt -e ".[dev,workbench]"
pytest -q
```

该环境不安装 `inference` extra，也不适用于 `doctor`、GPU 冒烟、训练或实际检测。当前 [`tests/test_strict_incremental.py`](../tests/test_strict_incremental.py) 会在收集阶段直接导入 `torch`，所以真正的干净环境仍需补齐兼容 PyTorch；这是现有测试依赖声明的待修复缺口，不应依赖 runner 中偶然存在的包。

### 本地配置

项目不使用 `.env` 文件。默认运行配置是 [`configs/agent_pipeline.yaml`](../configs/agent_pipeline.yaml)，可以先校验并查看有效值：

```bash
agile-agent config validate --config configs/agent_pipeline.yaml
agile-agent config show --config configs/agent_pipeline.yaml --effective
```

使用全局 `--config` 选择另一份完整配置，使用可重复的 `--set key=value` 做当前进程覆盖；不要为局部修改复制一份不完整 YAML。环境变量、默认值和设备配置的完整说明见 [CONFIGURATION.md](CONFIGURATION.md)。

## 构建与常用命令

Python 源码通过 `setuptools.build_meta` 构建，并以 editable 安装用于开发。仓库没有 `package.json`、Makefile、tox 或 nox 命令别名，也没有必须先执行的独立前端构建步骤。

| 命令 | 说明 |
| --- | --- |
| `./scripts/bootstrap_x86.sh` | 创建或复用完整 x86/CUDA 开发环境，安装项目 extras，注册 `agile-agent` 并执行环境与模型加载检查。 |
| `./scripts/start_agent.sh` | 执行环境门禁，刷新状态与决策，然后启动 Web 工作台。 |
| `./scripts/start_agent.sh --cli` | 在同一已配置环境中启动交互式 CLI 工作台。 |
| `agile-agent doctor` | 检查解释器、依赖、GPU、推理后端、模型身份、哈希和核心资产。完整 GPU 环境中使用。 |
| `pytest -q` | 运行 [`tests/`](../tests/) 下的完整 Pytest 测试套件。 |
| `python scripts/verify_release.py` | 校验默认配置、受保护资产、模型权重和公开证据；失败时返回非零退出码。 |
| `python scripts/smoke_models.py --load-only` | 校验模型哈希并把基础、增量和场景模型加载到 CUDA，不启动 Web 服务。 |
| `python scripts/smoke_models.py` | 在加载检查之外执行合成推理、批量推理、场景锁定集检查和 Web 编排冒烟。 |
| `cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release` | 配置历史 x86 TensorRT/CUDA 原生后端；需要 CUDA、TensorRT 和 OpenCV 开发包。 |
| `cmake --build build/native --config Release -j` | 构建已配置的 x86 原生后端。该后端不用于 Ascend 310B 正式链路。 |
| `cmake -S native_ascend -B build/native_ascend_stub -DCMAKE_BUILD_TYPE=Release` | 配置不依赖 CANN 的 Ascend C ABI contract stub。 |
| `cmake --build build/native_ascend_stub --config Release -j` | 构建 Ascend contract stub。 |
| `python tools/91_smoke_ascend_contract.py build/native_ascend_stub/libagile_agent_ascend_contract_stub.so` | 验证 stub 的 ABI、Not Ready 状态和禁止 CPU 回退契约。 |

默认开发验收顺序为：

```bash
pytest -q
python scripts/verify_release.py
python scripts/smoke_models.py --load-only
```

前两项也是 GitHub Actions 的核心检查；第三项需要 NVIDIA GPU 和完整推理依赖。修改 Ascend 后端时，应另外按照 [Ascend 310B 部署文档](ascend-310b-deployment.md) 运行对应的契约、golden、精度和性能门禁。

## 代码风格

仓库当前没有 Ruff、Black、Flake8、isort、mypy、EditorConfig 或 pre-commit 配置，也没有 `lint`、`format` 命令；[测试工作流](../.github/workflows/tests.yml) 只执行 Pytest 和发布校验，不执行自动格式检查。因此目前没有可声明为强制标准的格式化器或静态检查器。

修改代码时应与相邻模块保持一致：Python 文件使用 4 空格缩进，公共数据边界优先保留类型标注，路径操作优先使用 `pathlib.Path`，命令行入口使用 `argparse` 并以非零退出码表示失败。测试文件位于 [`tests/`](../tests/)，文件名采用 `test_*.py`；Pytest 的测试根目录由 [`pyproject.toml`](../pyproject.toml) 固定为 `tests`。

提交前至少运行：

```bash
pytest -q
python scripts/verify_release.py
```

若变更涉及模型、GPU 推理或后端，再运行相应的模型冒烟和设备验收命令。引入格式化或静态检查工具时，应同时提交其配置和 CI 步骤，避免只依赖个人编辑器设置。

## 分支约定

仓库的主分支是 `main`。当前没有 `CONTRIBUTING.md`、Pull Request 模板或其他文件规定分支命名格式，因此不存在可验证的 `feat/*`、`fix/*` 等强制约定。创建分支前从最新的 `upstream/main` 开始，并使用能直接说明工作范围的短名称；若维护者另有要求，以维护者约定为准。

```bash
git fetch upstream
git switch main
git merge --ff-only upstream/main
git switch -c <descriptive-branch-name>
```

提交信息也没有仓库级格式约束。使用简短的祈使句主题，并将配置、模型资产或兼容性影响写入正文，便于审核者判断验证范围。

## Pull Request 流程

仓库没有现成的 PR 模板或书面审核清单。以下流程与当前 CI 和项目资产门禁保持一致：

- 将变更限制在单一目的内，并在 PR 描述中说明行为变化、涉及的配置或模型资产以及兼容性影响。
- 在本地运行 `pytest -q` 和 `python scripts/verify_release.py`；附上实际执行的命令与结果。涉及 GPU、模型或后端时，同时写明硬件/运行时环境和对应冒烟或设备门禁结果。
- 将文档、配置、测试和校验哈希与实现一起更新；不要提交 `.venv/`、`.agent-python`、`build/`、`runs/` 或 `reports/` 等 Git 忽略的本地产物。
- 向 `main` 提交 PR。`.github/workflows/tests.yml` 会在 Pull Request 上分别使用 Python 3.10 和 3.12 安装 `.[dev,workbench]`，然后运行 `pytest -q` 与 `python scripts/verify_release.py`。
- CI 通过后再请求审核，并在评审期间保持分支可重放到最新 `main`；若修改验收证据，解释证据如何生成以及为何仍满足现有门禁。
