# AgileAgent

面向 IR/SAR 目标检测竞赛的可审计快速学习智能体。仓库包含统一 YOLO11s 检测器、四个合规增量专用模型、命令行工具和 Streamlit 工作台。x86-64 架构的 Windows、WSL 与 Linux 均可直接运行；训练数据、标签和竞赛提交结果不随仓库分发。

## 快速部署

要求：x86-64 电脑、NVIDIA GPU、可用的显卡驱动、Python 3.10-3.12、Git，以及约 5 GB 可用空间。仓库为私有仓库，克隆前需获得 GitHub 访问权限。

### WSL / Linux

```bash
git clone git@github.com:Sakauma/AgileAgent.git
cd AgileAgent
chmod +x scripts/bootstrap_x86.sh
./scripts/bootstrap_x86.sh
```

### Windows PowerShell

```powershell
git clone git@github.com:Sakauma/AgileAgent.git
Set-Location AgileAgent
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_x86.ps1
```

脚本会创建 `.venv`，默认从 PyTorch 的 `cu128` 软件源安装 CUDA 版 PyTorch 和智能体依赖，随后运行环境诊断，并在 GPU 0 上加载五个模型完成合成图冒烟测试。已有合适的 CUDA 版 PyTorch 环境时，也可手动执行：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[workbench,inference,dev]"
python -m fair_agent.cli doctor
python scripts/smoke_models.py
```

若显卡驱动需要其他 CUDA 软件包版本，可按 PyTorch 官方安装页替换软件源。WSL/Linux 设置 `PYTORCH_INDEX_URL`，Windows PowerShell 使用 `-TorchIndexUrl` 参数。`doctor` 会列出 CUDA 状态、GPU 数量和显卡名称；默认 GPU 不可用时返回非零。

## 启动智能体

先生成本机黑板和策略结果，再启动网页工作台：

```bash
python -m fair_agent.cli refresh
python -m fair_agent.cli decide --sensor sar --scene urban --class-focus soldier
python -m fair_agent.cli pipeline --mode dryrun
python -m fair_agent.cli serve
```

浏览器访问 `http://localhost:8501`。`doctor` 在模型缺失、SHA256 错误或核心依赖缺失时返回非零。全新克隆不包含私有数据分析报告，因此数据页可能为空，正式提交也会保持阻塞状态（`blocked`）；这不影响模型校验、策略框架和工作台启动。

## 本地 GPU 推理

将有权限使用的图片放入 `data/images/`，无需在命令行堆叠参数：

```bash
python tools/42_predict_submission.py --config configs/local_infer_gpu.yaml
```

配置集中在 `configs/local_infer_gpu.yaml`，默认使用 GPU 0、`batch=32` 和 `imgsz=640`。结果写入独立的 `runs/submission/<run_id>/`，包含 YOLO 标签、带置信度标签、CSV、JSON、运行清单和压缩包，不会覆盖已有结果。多卡机器仍默认使用 GPU 0；如需切换显卡，只修改 YAML 中的 `predict.device`。

## 模型清单

| 模型 | 用途 | mAP50 / KRR | 验收 |
| --- | --- | --- | --- |
| `models/base/yolo11s_ir_sar_imgsz640.pt` | IR/SAR 四类统一检测 | lock-all 0.91202 | 已冻结 |
| `p01_new_small_aircraft_best.pt` | 新类 small_aircraft | 0.55860 / 1.0 | 未通过 |
| `p02_new_warship_best.pt` | 新类 warship | 0.83539 / 1.0 | 通过 |
| `p03_new_tank_best.pt` | 新类 tank | 0.74989 / 1.0 | 通过 |
| `p04_new_soldier_best.pt` | 新类 soldier | 0.76914 / 1.0 | 通过 |

所有指标均为 mAP50。p01 权重仅用于复现与后续改进，未达到 `New-mAP50 >= 0.60` 门槛。完整路径、状态和摘要见 `models/manifest.json`，文件哈希见 `models/SHA256SUMS.txt`。

## 常用命令

```bash
python -m fair_agent.cli doctor
python -m fair_agent.cli refresh
python -m fair_agent.cli decide --sensor sar --class-focus soldier
python -m fair_agent.cli pipeline --mode execute
pytest -q
```

`pipeline --mode execute` 只执行 YAML 允许列表中的低风险动作。训练、正式推理、打包和提交始终需要人工触发并保留审计记录。智能体主配置位于 `configs/agent_pipeline.yaml`。

## 数据与增量训练

数据文件名遵循 `{sensor}_r1_base_{scene}_{id}`，图像和 YOLO 标签同名配对。获得授权数据后，可依次运行 `tools/00_check_dataset.py`、`01_build_metadata.py` 和 `02_split_dataset.py`。合规增量方案不使用旧类原始样本，具体协议与审计约束见 `docs/compliant-incremental-learning.md`。

## 已知限制

- x86 版本默认使用 NVIDIA GPU；GPU 推理速度不代表 Ascend 310B 端侧 FPS。
- CPU 不是默认部署路径。仅在明确需要兼容性验证时，将 YAML 的 `predict.device` 改为 `cpu`，并同时减小 `batch`。
- 仓库不包含竞赛数据、标签、PDF、SSH 凭据、预测结果或训练运行产物。
- 官方隐藏测试目录和提交格式尚未确认，因此正式提交门禁默认关闭。
- Ascend 310B 转换与推理接口仍待板卡就绪后补充。
