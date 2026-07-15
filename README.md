# 灵动Agent

面向 IR/SAR 目标检测竞赛的可审计快速学习智能体。系统登记场景/传感器认知、冻结旧类检测和增量类别专家三种不同功能模型，并提供面向检测用户的 Web 产品和面向运维人员的 CLI。当前 production 为 `generation-1-recheck-v2`：三类基础检测器负责人员、小型飞行器和坦克，舰船由已通过部署复核的类别增量专家负责。四类统一 YOLO11s 仅作上限基准，不参与默认推理、融合或回滚。系统统一运行在 x86-64 架构的 WSL/Linux 环境中；训练数据、标签和竞赛提交结果不随仓库分发。

## 快速开始

### 支持环境

- x86-64 WSL 2 或 Linux、NVIDIA GPU、可用的 NVIDIA 驱动和 Git。
- Python `3.10-3.12`；首次安装建议预留至少 `10 GB` 可用空间。
- 发布包内的 TensorRT engine 使用 TensorRT `10.8.0.43` 构建，并已在 RTX 4060 Laptop（SM `8.9`）验证。其他 TensorRT 版本或 GPU 架构不属于开箱即用范围，Agent 不会自动回退到 CPU 或 PyTorch。
- 仓库为私有仓库，克隆前需获得 GitHub 访问权限。

先在 WSL/Linux 终端确认 GPU 可见：

```bash
nvidia-smi
```

通过 HTTPS 克隆仓库：

```bash
git clone https://github.com/Sakauma/AgileAgent.git
cd AgileAgent
```

已配置 GitHub SSH 密钥时，也可使用：

```bash
git clone git@github.com:Sakauma/AgileAgent.git
cd AgileAgent
```

首次克隆只需执行一次环境配置：

```bash
chmod +x scripts/bootstrap_x86.sh scripts/start_agent.sh
./scripts/bootstrap_x86.sh
```

配置成功后启动 Web 工作台：

```bash
./scripts/start_agent.sh
```

浏览器访问 `http://127.0.0.1:8501`。终端按 `Ctrl+C` 停止服务；以后启动只需重新运行 `./scripts/start_agent.sh`，不会安装或修改依赖。

## 首次配置说明

`bootstrap_x86.sh` 会创建或复用 Python 环境、补齐 CUDA 版 PyTorch、TensorRT 和 Agent 依赖，然后在 GPU 0 上执行环境诊断、权重校验、TensorRT engine 校验和三种功能模型冒烟测试。配置完成后脚本退出，不会启动 Web 服务。

脚本按以下顺序选择环境：显式指定的 `AGILE_AGENT_PYTHON`、项目 `.venv`、当前激活的 venv/Conda，最后才创建新的 `.venv`。现有环境满足 Python 3.10-3.12、`torch>=2.0`、CUDA 可用、`torch.version.cuda` 有效且 torchvision CUDA NMS 可执行时，会跳过 PyTorch 安装；其余依赖达到 `pyproject.toml` 的最低版本且 `pip check` 无冲突时也会跳过安装。

复用已有兼容环境时，传入其 Python 绝对路径：

```bash
AGILE_AGENT_PYTHON=/path/to/compatible/env/bin/python ./scripts/bootstrap_x86.sh
```

没有兼容环境时，脚本按 `python3.12`、`python3.11`、`python3.10` 顺序选择解释器；安装了 `uv` 时可由 `uv` 创建环境。也可通过 `PYTHON_BIN` 指定解释器。脚本不会使用 Python 3.8。

配置通过后，所选解释器会写入本地 `.agent-python`。`start_agent.sh` 会自动读取它；无需激活环境。执行本文后续的 `python` 或 `agile-agent` 手动命令前，请在每个新终端运行：

```bash
AGENT_PYTHON="$(cat .agent-python)"
source "$(dirname "$AGENT_PYTHON")/activate"
```

需要严格复现已验收依赖组合时，可手动安装：

```bash
python -m pip install "torch==2.5.1+cu124" "torchvision==0.20.1+cu124" --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c constraints-agent.txt -e ".[workbench,inference,export,dev]"
python -m fair_agent.cli doctor
```

已验证组合为：Python `3.10.19`、PyTorch `2.5.1+cu124`、TorchVision `0.20.1+cu124`、Ultralytics `8.4.92`、TensorRT `10.8.0.43`、ONNX `1.17.0`、ONNX Runtime GPU `1.23.2`，部署显卡为 RTX 4060 Laptop（SM `8.9`）。环境名称和安装路径不作要求。

## 发布验收

无需 GPU 的静态验收会验证配置、全部模型哈希、推理参数、脱敏证据和启动脚本：

```bash
python scripts/verify_release.py
```

GPU 冒烟验收读取推理配置和功能模型注册表，验证 Scene-SensorNet、基础检测器、增量专家和 TensorRT engine，并调用完整 Agent 自动路由与融合链路；本地存在 lock-val 时会自动抽取 IR/SAR 样本：

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

Web 前端面向评委和检测使用者，提供单图检测、批量检测、增量学习工作台、自动 IR/SAR 与场景识别、标注图预览以及 JSON/ZIP 导出。用户只需上传图像，Agent 会自动解析当前活动代际，执行其中所有类别所有者并融合结果，不要求用户选择模型。未通过部署门禁的候选专家不会进入 Web。批量任务完成后可在网页逐张选择和查看标注结果，也可下载完整结果包；会话记录可直接跳转到当前页面最近10份完整结果。单GPU请求采用公平队列，单次请求内的上下文、基础检测和类别专家可按 YAML 并行调度。默认单文件限制20MB、单批限制20张和200MB、当前会话最多保留20条轻量历史，实际值均由主 YAML 与只读公开配置接口提供。权重路径、哈希和部署门禁等运维细节不会显示在 Web 中。

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
| `strict_p02_base_v1` | 三类冻结基础检测 | 单模型旧类 mAP50 0.82738 | `generation-1-recheck-v2`旧类所有者 |
| `strict_p02_warship_recheck_v2` | 舰船类别增量专家 | 组合系统 New-mAP50 0.79500 / KRR 1.0 | production，precision 1.0 / 误激活率 0.0 |
| `models/base/yolo11s_ir_sar_imgsz640.pt` | IR/SAR 四类统一检测 | lock-all 0.91202 | `benchmark_only` |
| `p01_new_small_aircraft_best.pt` | small_aircraft 专项增强演练 | 0.55860 / 1.0 | 未通过、禁用 |
| `p02_new_warship_best.pt` | warship 目标增量演练 | 0.83539 / 1.0 | 通过 |
| `p03_new_tank_best.pt` | tank 目标增量演练 | 0.74989 / 1.0 | 通过 |
| `p04_new_soldier_best.pt` | soldier 目标增量演练 | 0.76914 / 1.0 | 通过 |
| `strict-p02` 实验档 | warship 严格 3+1 类别增量 | 0.90903 / 1.0 | 可审计候选、CLI 可用 |

检测与增量指标均为 mAP50。现有 p01-p04 的目标类别已经包含在统一模型中，因此只作为目标增量/专项增强演练。严格 3+1 双折使用三类基础模型模拟未知新类；舰船折经 `generation-1-recheck-v2` 以增量 dev 固定阈值 0.63 后，在完整 lock-val 上通过全部精度和误激活门禁，飞行器折 New-mAP50 为 0.54062，未注册。三模型输入输出契约和协同链路见 `configs/functional_models.yaml` 与 `docs/functional-models.md`；代际注册表位于 `models/generations.json`，严格实验档独立位于 `models/experiments/strict_3plus1/`。

## 常用命令

```bash
python -m fair_agent.cli doctor
python -m fair_agent.cli refresh
python -m fair_agent.cli status --refresh
python -m fair_agent.cli console
python -m fair_agent.cli detect --source path/to/image.png --confidence 0.50
python -m fair_agent.cli decide --sensor sar --class-focus soldier
python -m fair_agent.cli pipeline --mode execute
python -m fair_agent.cli benchmark-api
pytest -q
```

`pipeline --mode execute` 只执行 YAML 允许列表中的低风险动作。训练、正式推理、打包和提交始终需要人工触发并保留审计记录。智能体主配置位于 `configs/agent_pipeline.yaml`。

`benchmark-api` 会在 8501 未运行时按当前有效配置启动临时服务，测试结束后自动关闭；若已有健康服务则直接测试该实例。未达到性能门槛时命令返回非零，同时仍会将完整报告写入 `reports/api_performance/<run_id>/benchmark.json`。

## 增量学习工作台

Web 顶部的“增量学习”入口支持上传、浏览、注入和训练增量数据。上传包必须是 ZIP，内部采用 YOLO 标签；建议同时包含带 `names` 字段的 `data.yaml`：

```text
new_batch.zip
├── data.yaml
├── images/train/*.png
├── images/val/*.png
├── labels/train/*.txt
└── labels/val/*.txt
```

若没有显式验证集，Agent 会确定性划分 train/val；若没有 `data.yaml`，可在上传页填写逗号分隔的类别名称。系统会安全解压并检查图像可读性、YOLO五列格式、坐标范围、重复 stem、图像标签对应关系和类别分布，再依据当前 production 类别自动判断 `class_incremental` 或 `target_incremental`。上传数据保存在 `data/incremental_batches/<batch_id>/`，不进入 Git。

“生成训练视图”会在批次目录内创建独立数据视图和内部 `batch.yaml`，并记录 `old_raw_image_count=0`；“开始快速训练”会启动 GPU 后台任务，页面可查看状态和实时日志。训练产物始终标记为待校准候选，不会自动替换 production。后续仍需完成阈值校准、lock 复核和 `generation promote`。

相同能力可由 CLI 操作：

```bash
agile-agent incremental-data upload --archive /path/to/new_batch.zip --name 新批次
agile-agent incremental-data list
agile-agent incremental-data show --batch-id BATCH_ID
agile-agent incremental-data inject --batch-id BATCH_ID
agile-agent incremental-data train --batch-id BATCH_ID
agile-agent incremental-data jobs --batch-id BATCH_ID
agile-agent incremental-data logs --batch-id BATCH_ID --job-id JOB_ID
```

上传容量、保存目录、初始权重、GPU、`imgsz`、batch、epoch、学习率和早停参数均位于 `configs/agent_pipeline.yaml` 的 `incremental_workbench` 段，也可使用现有 `--set` 或 `config set` 接口修改。完整说明见 `docs/incremental-workbench.md`。

## 运行日志

Agent 将 Web 请求、推理、CLI、数据审计、数据视图生成、增量训练、dev 阈值校准、lock 解封与复核、generation 注册、production 切换、回滚和异常统一写为结构化 JSONL。每条记录包含 UTC 时间、级别、组件、事件、`trace_id`、批次/任务/实验/运行/协议/代际编号、耗时和脱敏详情；上传内容、口令和令牌不会写入日志。阶段原始 `events.jsonl` 与 manifest 继续保留，全局日志作为跨模块行为索引。日志默认位于 `reports/agent_logs/`，按大小轮转并保留14个文件。

```bash
agile-agent logs --limit 200
agile-agent logs --component training --batch-id BATCH_ID
agile-agent logs --job-id JOB_ID
agile-agent logs --experiment-id warship_3plus1 --run-id RUN_ID
agile-agent logs --protocol-id round_01
agile-agent logs --generation-id generation-1-recheck-v2
```

日志目录、单文件大小和保留数量由主 YAML 的 `logging` 段统一配置。Web 的增量批次详情页只展示与当前批次相关的脱敏操作时间线；完整详情保留在 CLI 和 JSONL 中。完整事件规范见 `docs/agent-audit-logging.md`。

当前 TensorRT engine 可用以下命令只读校验；缺失时去掉 `--verify-only` 才会按 YAML 导出，已有且哈希正确的资产不会覆盖：

```bash
python tools/80_export_tensorrt_engines.py --verify-only
```

## 统一参数配置

`configs/agent_pipeline.yaml` 是 Agent 持久参数的唯一事实源，包含 GPU、服务、推理、路由、融合、上传、缓存、界面、性能和验收门槛。CLI 的重复 `--set key=value` 只覆盖当前进程；`config set/unset` 会原子写回、备份并记录审计，重启后生效。production、权重哈希和类别所有权只能通过代际专用命令修改。

```bash
agile-agent config validate --config configs/agent_pipeline.yaml
agile-agent config show --config configs/agent_pipeline.yaml --effective
agile-agent config get routing.conflict_iou --config configs/agent_pipeline.yaml
agile-agent config set routing.conflict_iou 0.50 --config configs/agent_pipeline.yaml
agile-agent --config configs/agent_pipeline.yaml --set inference.confidence_default=0.60 serve
agile-agent generation recheck --candidate generation-1-recheck-v2
agile-agent generation promote --candidate generation-1-recheck-v2 --manifest reports/generation_rechecks/<run_id>/manifest.json
agile-agent generation rollback --to generation-0
```

## 数据与增量训练

数据文件名遵循 `{sensor}_r1_base_{scene}_{id}`，图像和 YOLO 标签同名配对。获得授权数据后，可依次运行 `tools/00_check_dataset.py`、`01_build_metadata.py` 和 `02_split_dataset.py`。

增量任务统一按增量目标检测处理，以类别增量为主，同时支持目标增量。训练、验证、早停和调参只能读取增量数据集，禁止旧样本 replay；旧类测试数据只允许在增量权重冻结后的评分阶段使用。机器规则位于 `configs/incremental_detection_policy.yaml`，完整流程见 `docs/compliant-incremental-learning.md`。

真正类别增量模型必须使用基础类别集合之外的全局类别 ID，并用增量验证集校准激活阈值。其候选框不依赖旧模型产生同类检测；目标增量模型则使用旧模型同类框进行空间一致性复核。场景识别只参与软路由排序，不会按场景硬拒绝新增能力。

### 可复现的舰船 3+1 实验

舰船协议的唯一通用配置为 `configs/incremental/warship_3plus1.yaml`。更换基础类别、新增类别、源划分或增加后续轮次时先修改该 YAML，再运行统一入口：

```bash
agile-agent experiment validate --config configs/incremental/warship_3plus1.yaml
agile-agent experiment run --config configs/incremental/warship_3plus1.yaml
agile-agent experiment reproduce --manifest runs/experiments/warship_3plus1/<run_id>/run_manifest.json
```

`validate` 只读数据且保持 lock 封存；`run` 会启动训练；`reproduce` 只有在源数据指纹与父实验一致时才创建新 run。当前执行适配器 v1 支持单轮 3+1，通用 schema 和模型代际注册表已预留多轮；新增更多类别前需扩展训练适配器，不得把“可描述多轮”误写成“已验证多轮”。逐文件哈希、状态机、阈值冻结和验收规则见 `docs/warship-3plus1-reproducibility.md`。

### 历史严格双折实验

仓库提供 small_aircraft 与 warship 两个严格留一类别实验。基础模型只有三个检测通道，基础训练图像与新增类训练图像完全隔离；人员和坦克因始终共现，不用于严格留一实验。所有参数位于 `configs/strict_class_incremental_3plus1.yaml`，在 4090 服务器执行：

```bash
python tools/70_run_strict_3plus1.py --config configs/strict_class_incremental_3plus1.yaml
```

双 GPU 可用时两个协议分别使用 GPU 0/1 并行，否则在 GPU 0 按顺序运行。lock-val 只会在基础权重、specialist 权重和阈值冻结后物化。历史实验档可由 CLI 显式复核，但不会绕过代际门禁替换 Web production：

```bash
python -m fair_agent.cli detect --profile strict-p01 --source path/to/aircraft.png
python -m fair_agent.cli detect --profile strict-p02 --source path/to/warship.png
```

历史 `strict-p02` 阈值 0.51 的原始舰船折虽然通过核心门槛，但 lock precision 约 0.5411、图像误激活率约 0.3378，因此仍作为未上线的 `generation-1` 证据保留。独立复核版本仅依据增量 dev 将阈值固定为 0.63，并加入跨类别冲突抑制；当前注册表记录的 `generation-1-recheck-v2` 完整 lock-val 指标为旧类 mAP50 0.82738、New-mAP50 0.79500、KRR 1.0、组合 mAP50 0.81929、precision 1.0 和误激活率 0.0，现已切入 production。`generation-0` 始终保留为回滚点。

`strict-p02` 属于“冻结基础模型 + 独立 specialist”的历史系统级证据，不作为最终单模型增量结论。新的 clean-room v2 方案将三类教师检测头扩展为一个四类学生检测头，仅开放新增类通道训练，最终只部署 `student_4class.pt`。训练前先运行只读预检：

```bash
python tools/70_run_strict_3plus1.py --config configs/clean_class_incremental_v2.yaml --check-only
```

配置和执行边界见 `configs/clean_class_incremental_v2.yaml` 与 `docs/clean-class-incremental-v2.md`；该候选在重新训练并通过全部门禁前不会进入 Web。

最新 `clean-ci-v2-warship-r02` 已完成合规训练，但未通过模型门禁：New-mAP50 为0.28421、KRR为0.99909、四类 mAP50为0.67547。它证明了快速更新、数据隔离和旧类保持闭环，不证明新增类别学习能力已经成立。

### 无旧样本方法比较

仓库提供同一基础权重和同一 p02 数据上的 `DuET-YOLO11s` 与 `YOLO-IOD-lite` 对照实验。全部参数位于 `configs/incremental_method_comparison.yaml`，唯一运行入口为：

```bash
python tools/71_compare_incremental_methods.py --check-only
python tools/71_compare_incremental_methods.py
```

两个方法分别在 GPU 1/2 并行执行；脚本自动核对基础权重 SHA256，并生成不可覆盖的逐方法指标和统一比较报告。实现边界、指标门禁和复核规则见 `docs/incremental-method-comparison.md`。两者均未通过全部门禁前，不注册为 Agent 的活动增量能力。

### 官方完整 YOLO-IOD 复现

为验证轻量适配失败是否来自实现简化，仓库另提供基于官方 YOLO-World(X) 的 strict-p02 复现。r05 延续 r04 的数据审计结论，禁用不适用的 CPR，保留 IKS 和 CAKD，并在 GPU 3 上通过梯度累积统一为有效 batch 16：

```bash
python tools/72_run_full_yolo_iod.py --config configs/full_yolo_iod_p02_r05_gpu3.yaml --check-only
python tools/72_run_full_yolo_iod.py --config configs/full_yolo_iod_p02_r05_gpu3.yaml
```

类别、数据隔离、三阶段训练、lock-val 冻结和验收说明见 `docs/full-yolo-iod-reproduction.md`。完整模型只作为方法参考；只有全部门禁通过后才讨论压缩或蒸馏回 YOLO11s。

Scene-SensorNet 的训练参数全部位于 `configs/scene_sensor_model.yaml`，训练入口为 `python tools/60_train_scene_sensor.py`。固定权重已经随仓库发布，日常启动不会重新训练。

## 常见问题

- `nvidia-smi` 不可用：先修复 WSL/Linux 的 NVIDIA 驱动可见性，再运行配置脚本。不要通过关闭 GPU 门禁继续安装。
- `doctor` 报告 TensorRT 版本或 engine 不兼容：不要绕过检查。发布 engine 只保证在 README 声明的组合上可用；其他 GPU 或 TensorRT 版本需要由维护者重新导出 engine、更新受保护哈希并重新执行发布验收。
- 端口 `8501` 已占用：激活记录在 `.agent-python` 中的环境后，使用 `agile-agent config set runtime.server_port 8502 --config configs/agent_pipeline.yaml` 修改端口，再重启 Agent。
- 新终端中找不到 `agile-agent`：先按“首次配置说明”激活 `.agent-python` 对应环境；直接运行 `start_agent.sh` 不需要激活。
- 首次启动比后续启动慢：TensorRT engine 和模型需要首次载入 GPU，等待终端出现工作台地址后再打开页面。

## 已知限制

- x86 版本默认使用 NVIDIA GPU；GPU 推理速度不代表 Ascend 310B 端侧 FPS。
- 仓库不包含竞赛数据、标签、PDF、SSH 凭据、预测结果或训练运行产物。
- 官方隐藏测试目录和提交格式尚未确认，因此正式提交门禁默认关闭。
- 尚未获得外部真实新增类别数据；当前舰船 3+1 已通过内部部署门禁，但外部数据泛化仍需官方隐藏测试验证。
- RTX 4060 上 TensorRT FP16 三轮 API 复核的中位轮平均为 `32.88 ms`、P95 为 `39.58 ms`、20张批量为 `47.81 FPS`，8并发请求全部成功，已通过 `33.3 ms / 50 ms / 30 FPS` 门禁。原始性能报告属于本地 `reports/` 产物，不随仓库分发。这些 engine 仅在 TensorRT `10.8.0.43` 与 RTX 4060 Laptop（SM `8.9`）完成验证；完整 C++ ABI 尚未切入 production。
- Ascend 310B 转换与推理接口仍待板卡就绪后补充。

Web 与 CLI 的操作说明见 `docs/agent-operations.md`。
