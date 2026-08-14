<!-- generated-by: gsd-doc-writer -->
# 快速开始

本指南用于在 x86-64 Linux 或 WSL 2 上配置灵动Agent，并启动本地 Web 或终端工作台。默认启动使用 NVIDIA CUDA 推理；Ascend 310B 的板端发布流程不在本文范围内。

## 先决条件

- x86-64 Linux，或启用了 Linux 与 GPU 支持的 WSL 2。仓库的自动配置脚本会拒绝其他操作系统和处理器架构。
- NVIDIA GPU、可用的 NVIDIA 驱动，以及能够正常执行的 `nvidia-smi`。
- `Python >= 3.10, < 3.13`。解释器必须带有 `pip`；自动创建虚拟环境时还需要 `venv`/`ensurepip`，也可以预先安装 `uv`。
- Git 和 Bash。
- 经验上建议预留至少 10 GB 可用磁盘空间，用于 Python 环境、CUDA 依赖、模型和运行产物；仓库当前没有自动磁盘空间检查或可复现的安装体积基准。

配置脚本默认安装经过验证的 PyTorch `2.5.1+cu124` 与 TorchVision `0.20.1+cu124`，但也会复用通过兼容性检查的现有 CUDA 环境。默认检测和 Web 工作台不要求下载竞赛原始训练数据。

## 安装步骤

1. 克隆仓库并进入项目目录：

   ```bash
   git clone https://github.com/Sakauma/AgileAgent.git
   cd AgileAgent
   ```

2. 赋予配置与启动脚本执行权限：

   ```bash
   chmod +x scripts/bootstrap_x86.sh scripts/start_agent.sh
   ```

3. 创建或复用兼容的 Python 环境，并安装项目及开发、Web、推理依赖：

   ```bash
   ./scripts/bootstrap_x86.sh
   ```

   脚本会检查 CUDA、PyTorch、TorchVision 和项目依赖，注册当前仓库的 `agile-agent` 命令，运行 `doctor` 与模型加载冒烟测试，并将最终解释器路径写入 `.agent-python`。

   如需复用已有环境，请显式传入带 `pip` 的 Python 3.10–3.12 解释器：

   ```bash
   AGILE_AGENT_PYTHON=/path/to/environment/bin/python ./scripts/bootstrap_x86.sh
   ```

## 首次运行

从项目根目录启动 Web 工作台：

```bash
./scripts/start_agent.sh
```

启动脚本会先执行静默环境诊断，再刷新运行状态并启动服务。看到 `正在启动灵动Agent工作台：http://127.0.0.1:8501` 后，在浏览器打开 [http://127.0.0.1:8501](http://127.0.0.1:8501)。按 `Ctrl+C` 停止服务。

在无浏览器环境中，可以改为启动终端工作台：

```bash
./scripts/start_agent.sh --cli
```

## 常见设置问题

### 配置脚本提示不支持当前平台或找不到 `nvidia-smi`

`scripts/bootstrap_x86.sh` 只支持 x86-64 Linux/WSL，并要求 NVIDIA 驱动可见。先确认以下命令都成功，再重新运行配置脚本：

```bash
uname -s
uname -m
nvidia-smi
```

预期系统为 `Linux`、架构为 `x86_64`。在 Windows 上应从已启用 NVIDIA GPU 支持的 WSL 2 发行版中运行这些命令。

### Python 版本不受支持，或现有环境缺少 `pip`

项目仅接受 Python 3.10、3.11 或 3.12。准备一个带 `pip` 的兼容解释器后，通过 `AGILE_AGENT_PYTHON` 重新配置：

```bash
AGILE_AGENT_PYTHON=/path/to/python ./scripts/bootstrap_x86.sh
```

如果脚本报告现有 `.venv` 不完整，应先保留或移走该环境，再让配置脚本创建新的 `.venv`；不要继续使用未通过兼容性检查的环境。

### 启动脚本提示找不到已配置的 Python

`.agent-python` 记录的解释器可能已被移动或删除。用当前有效的解释器重新运行配置脚本，它会复核依赖并刷新该记录：

```bash
AGILE_AGENT_PYTHON=/path/to/python ./scripts/bootstrap_x86.sh
./scripts/start_agent.sh
```

### 数据集相关命令提示缺少样本

默认 Web/CLI 检测使用仓库内发布权重，不依赖原始竞赛数据；`benchmark-api`、固定 3+1 实验和指标复核则需要授权数据。将数据放入 `datasets_r1_base_train/` 后，先执行：

```bash
python tools/00_check_dataset.py
python tools/01_build_metadata.py
agile-agent experiment validate --config configs/incremental/warship_3plus1.yaml
```

## 后续步骤

- 阅读 [DEVELOPMENT.md](DEVELOPMENT.md)，了解本地开发命令、代码风格与协作流程。
- 阅读 [TESTING.md](TESTING.md)，了解完整测试、单文件测试和 CI 检查方式。
- 阅读 [CONFIGURATION.md](CONFIGURATION.md)，了解 YAML 配置、环境变量和设备级覆盖。
- 如需部署到 Ascend 310B，请从 [Ascend 310B 稳定加速设计](ascend-310b-deployment.md) 开始。
