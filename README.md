# AgileAgent

面向 IR/SAR 目标检测竞赛的可审计快速学习智能体。系统包含场景/传感器认知模型、统一 YOLO11s 检测模型和增量目标检测模型库三种不同功能模型，以及面向检测用户的 Web 产品和面向运维人员的 CLI。系统统一运行在 x86-64 架构的 WSL/Linux 环境中；训练数据、标签和竞赛提交结果不随仓库分发。

## 首次配置环境

要求：x86-64 电脑、NVIDIA GPU、可用的显卡驱动、Python 3.10-3.12、Git，以及约 5 GB 可用空间。仓库为私有仓库，克隆前需获得 GitHub 访问权限。配置脚本按 `python3.12`、`python3.11`、`python3.10` 顺序选择兼容解释器；安装了 `uv` 时由它可靠地创建环境，并仅在本机没有兼容解释器时获取 Python 3.12。脚本不会误用 WSL 中常见的 Python 3.8，也可通过 `PYTHON_BIN` 显式指定。

### WSL / Linux

```bash
git clone git@github.com:Sakauma/AgileAgent.git
cd AgileAgent
chmod +x scripts/bootstrap_x86.sh
./scripts/bootstrap_x86.sh
```

首次配置脚本只负责创建或复用 `.venv`、安装 CUDA 版 PyTorch 和智能体依赖、运行环境诊断，并在 GPU 0 上校验六份权重及三种功能。配置完成后脚本退出，不启动工作台。

配置脚本优先复用显式指定的 `AGILE_AGENT_PYTHON`、项目 `.venv` 或当前激活的 Conda/venv。现有环境只要满足 Python 3.10-3.12、`torch>=2.0`、CUDA 可用、`torch.version.cuda` 有效且 torchvision CUDA NMS 可执行，就会跳过 PyTorch 安装；其余依赖达到 `pyproject.toml` 的最低兼容版本且 `pip check` 无冲突时也会整体跳过安装。`2.5.1+cu124` 和 `constraints-agent.txt` 仅作为新环境的已验证默认组合与严格复现选项。

例如直接复用已有的 `egor` 环境：

```bash
AGILE_AGENT_PYTHON=/home/sakauma/data/miniconda3/envs/egor/bin/python ./scripts/bootstrap_x86.sh
```

通过全部验收后，脚本会把选中的解释器路径写入本地 `.agent-python`；后续直接运行 `./scripts/start_agent.sh` 即可，无需再次传入环境变量。若兼容环境不存在，脚本才会创建 `.venv` 并安装默认组合。若显卡驱动需要其他 CUDA 软件包版本，可按 PyTorch 官方安装页设置 `PYTORCH_INDEX_URL`、`TORCH_VERSION` 和 `TORCHVISION_VERSION`。`doctor` 会列出 CUDA状态、GPU数量和显卡名称；默认GPU不可用时脚本会停止。

需要严格复现已验收依赖组合时，可手动安装：

```bash
python -m pip install "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c constraints-agent.txt -e ".[workbench,inference,dev]"
python -m fair_agent.cli doctor
```

## 发布验收

无需 GPU 的静态验收会验证配置、全部模型哈希、推理参数、脱敏证据和启动脚本：

```bash
python scripts/verify_release.py
```

GPU 冒烟验收读取推理配置和功能模型注册表，验证 Scene-SensorNet、基础检测器及四个增量权重，并执行 `imgsz=640、batch=32` 的基础模型批量推理：

```bash
python scripts/smoke_models.py
```

## 日常一键启动

环境配置完成后，日常启动不再安装或修改任何依赖：

```bash
./scripts/start_agent.sh
```

启动脚本仅依次执行环境门禁、刷新黑板、生成默认决策和启动 Starlette/Uvicorn Web 服务。服务强制监听本机回环地址，浏览器访问 `http://127.0.0.1:8501`，在终端按 `Ctrl+C` 停止服务。

服务器、SSH 会话或无浏览器环境使用终端模式：

```bash
./scripts/start_agent.sh --cli
```

Web 前端面向评委和检测使用者，提供单图检测、批量检测、自动 IR/SAR 与场景识别、标注图预览以及 JSON/ZIP 导出。单图页可切换“标准识别”和“增量演示”：增量模式会激活已通过 New-mAP/KRR 门禁的新类 specialist，并与冻结旧类检测器组合输出；场景认知结果用于解释和分析，不限制增量模型激活。单GPU采用公平队列串行推理；单文件限制20MB、单批限制20张和200MB，当前会话最多保留20条轻量历史。权重路径、哈希和部署门禁等运维细节不会显示在 Web 中。

CLI 前端面向开发与运维人员，包含总览、模型、数据、增量和部署五个页面，并可刷新黑板、生成策略、创建 Dry-run 和执行经过确认的低风险动作。

只输出一次终端摘要：

```bash
agile-agent console --once
```

面向外部脚本或未来 Ascend 310B 服务进程，可获取机器可读状态：

```bash
agile-agent status --format json --refresh
```

当私有实验报告存在时，工作台使用 `live` 证据；全新克隆时自动使用 `demo_artifacts/` 中不含原始图像、标注和真实文件名的脱敏证据。首页会明确显示当前证据模式。

## 手动调试命令

先生成本机黑板和策略结果，再启动网页工作台：

```bash
python -m fair_agent.cli refresh
python -m fair_agent.cli decide --sensor sar --scene urban --class-focus soldier
python -m fair_agent.cli context-predict --source data/images/example.png
python -m fair_agent.cli decide --source data/images/example.png --class-focus soldier
python -m fair_agent.cli pipeline --mode dryrun
python -m fair_agent.cli serve
```

通常无需手动执行上述命令；它们用于单独调试某个阶段。`doctor` 在 GPU、模型、SHA256 或核心依赖异常时返回非零。全新克隆不包含私有数据分析报告，但 Web 检测和 CLI 脱敏状态均可正常使用；正式提交保持阻塞状态（`blocked`），直到官方测试目录和格式得到确认。

## 本地 GPU 推理

将有权限使用的图片放入 `data/images/`，无需在命令行堆叠参数：

```bash
python tools/42_predict_submission.py --config configs/local_infer_gpu.yaml
```

配置集中在 `configs/local_infer_gpu.yaml`，默认使用 GPU 0、`batch=32` 和 `imgsz=640`。结果写入独立的 `runs/submission/<run_id>/`，包含 YOLO 标签、带置信度标签、CSV、JSON、运行清单和压缩包，不会覆盖已有结果。多卡机器仍默认使用 GPU 0；如需切换显卡，只修改 YAML 中的 `predict.device`。

## 模型清单

| 模型 | 功能 | 核心指标 | 验收 |
| --- | --- | --- | --- |
| `models/context/scene_sensor_net.pt` | IR/SAR 与 air/forest/sea/urban 认知 | lock sensor 0.98947 / scene 0.76842 | 通过 |
| `models/base/yolo11s_ir_sar_imgsz640.pt` | IR/SAR 四类统一检测 | lock-all 0.91202 | 已冻结 |
| `p01_new_small_aircraft_best.pt` | 新类 small_aircraft | 0.55860 / 1.0 | 未通过 |
| `p02_new_warship_best.pt` | 新类 warship | 0.83539 / 1.0 | 通过 |
| `p03_new_tank_best.pt` | 新类 tank | 0.74989 / 1.0 | 通过 |
| `p04_new_soldier_best.pt` | 新类 soldier | 0.76914 / 1.0 | 通过 |

检测与增量指标均为 mAP50。p01 权重仅用于复现与后续改进，未达到 `New-mAP50 >= 0.60` 门槛。三模型输入输出契约和协同链路见 `configs/functional_models.yaml` 与 `docs/functional-models.md`；完整路径、状态和摘要见 `models/manifest.json`，文件哈希见 `models/SHA256SUMS.txt`。

## 常用命令

```bash
python -m fair_agent.cli doctor
python -m fair_agent.cli refresh
python -m fair_agent.cli status --refresh
python -m fair_agent.cli console
python -m fair_agent.cli decide --sensor sar --class-focus soldier
python -m fair_agent.cli pipeline --mode execute
pytest -q
```

`pipeline --mode execute` 只执行 YAML 允许列表中的低风险动作。训练、正式推理、打包和提交始终需要人工触发并保留审计记录。智能体主配置位于 `configs/agent_pipeline.yaml`。

## 数据与增量训练

数据文件名遵循 `{sensor}_r1_base_{scene}_{id}`，图像和 YOLO 标签同名配对。获得授权数据后，可依次运行 `tools/00_check_dataset.py`、`01_build_metadata.py` 和 `02_split_dataset.py`。

增量任务统一按增量目标检测处理，以类别增量为主，同时支持目标增量。训练、验证、早停和调参只能读取增量数据集，禁止旧样本 replay；旧类测试数据只允许在增量权重冻结后的评分阶段使用。机器规则位于 `configs/incremental_detection_policy.yaml`，完整流程见 `docs/compliant-incremental-learning.md`。

Scene-SensorNet 的训练参数全部位于 `configs/scene_sensor_model.yaml`，训练入口为 `python tools/60_train_scene_sensor.py`。固定权重已经随仓库发布，日常启动不会重新训练。

## 已知限制

- x86 版本默认使用 NVIDIA GPU；GPU 推理速度不代表 Ascend 310B 端侧 FPS。
- 仓库不包含竞赛数据、标签、PDF、SSH 凭据、预测结果或训练运行产物。
- 官方隐藏测试目录和提交格式尚未确认，因此正式提交门禁默认关闭。
- Ascend 310B 转换与推理接口仍待板卡就绪后补充。

Web 与 CLI 的操作说明见 `docs/agent-operations.md`。
