# AgileAgent

面向 IR/SAR 目标检测竞赛的可审计快速学习 Agent。仓库包含统一 YOLO11s 检测器、四个合规增量 specialist、命令行工具和 Streamlit 工作台。x86 Windows、WSL 与 Linux 均可直接运行；训练数据、标签和竞赛提交结果不随仓库分发。

## 快速部署

要求：x86-64 电脑、Python 3.10-3.12、Git，以及约 3 GB 可用空间。仓库为私有仓库，克隆前需获得 GitHub 访问权限。

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

脚本会创建 `.venv`、安装 CPU 版 PyTorch 和 Agent 依赖、运行环境诊断，并加载五个模型完成合成图 CPU smoke test。已有合适 PyTorch 环境时，也可手动执行：

```bash
python -m pip install -e ".[workbench,inference,dev]"
python -m fair_agent.cli doctor
python scripts/smoke_models.py
```

## 启动 Agent

先生成本机黑板和策略结果，再启动 Web 工作台：

```bash
python -m fair_agent.cli refresh
python -m fair_agent.cli decide --sensor sar --scene urban --class-focus soldier
python -m fair_agent.cli pipeline --mode dryrun
python -m fair_agent.cli serve
```

浏览器访问 `http://localhost:8501`。`doctor` 在模型缺失、SHA256 错误或核心依赖缺失时返回非零。干净 clone 没有私有数据分析报告，因此数据页可能为空，正式提交也会保持 `blocked`；这不影响模型校验、策略框架和工作台启动。

## 本地 CPU 推理

将有权限使用的图片放入 `data/images/`，无需在命令行堆叠参数：

```bash
python tools/42_predict_submission.py --config configs/local_infer_cpu.yaml
```

配置集中在 `configs/local_infer_cpu.yaml`。结果写入独立的 `runs/submission/<run_id>/`，包含 YOLO 标签、带置信度标签、CSV、JSON、manifest 和 zip，不会覆盖已有结果。使用 NVIDIA GPU 时复制该 YAML，将 `predict.device` 改为 `0`，并按 PyTorch 官方说明安装对应 CUDA wheel。

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

`pipeline --mode execute` 只执行 YAML allowlist 中的低风险动作。训练、正式推理、打包和提交始终需要人工触发并保留审计记录。Agent 主配置位于 `configs/agent_pipeline.yaml`。

## 数据与增量训练

数据文件名遵循 `{sensor}_r1_base_{scene}_{id}`，图像和 YOLO 标签同名配对。获得授权数据后，可依次运行 `tools/00_check_dataset.py`、`01_build_metadata.py` 和 `02_split_dataset.py`。合规增量方案不使用旧类原始样本，具体协议与审计约束见 `docs/compliant-incremental-learning.md`。

## 已知限制

- x86 CPU 版本用于展示、开发和功能验证，不代表端侧 FPS。
- 仓库不包含竞赛数据、标签、PDF、SSH 凭据、预测结果或训练 runs。
- 官方隐藏测试目录和提交格式尚未确认，因此正式提交门禁默认关闭。
- Ascend 310B 转换与推理接口仍待板卡就绪后补充。
