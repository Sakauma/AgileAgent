# 灵动Agent

面向 IR/SAR 时变场景目标检测的快速学习智能体。系统以场景与传感器认知为上下文，自动协调冻结基础检测器和增量检测器，并提供增量数据审计、快速训练、模型复核、代际切换与回滚能力。

项目同时提供面向检测用户的 Web 工作台和面向开发、运维及未来端侧集成的 CLI。当前发布版运行于 x86-64 WSL/Linux 与 NVIDIA GPU；Ascend 310B 适配将在硬件到位后完成。

## 核心能力

- **自动识别与路由**：识别 IR/SAR 传感器和 air/forest/sea/urban 场景，自动解析当前生产代际并执行对应模型。
- **增量目标检测**：支持类别增量和目标增量；训练、验证、早停和调参只读取本批增量数据。
- **双前端操作**：Web 提供单图、批量检测和增量数据工作台；CLI 提供完整决策轨迹、配置、实验、日志及代际管理。
- **可审计与可回滚**：数据、配置、权重、阈值和评测结果均记录哈希；候选模型通过门禁后才能切换 production。
- **配置驱动**：GPU、推理、路由、融合、上传、缓存、训练及验收参数统一由 YAML 管理，并支持 CLI 临时覆盖或持久修改。

## 系统架构

```mermaid
flowchart LR
    A["IR/SAR 图像"] --> B["场景与传感器认知"]
    B --> C["Agent 决策与代际解析"]
    C --> D["三类冻结基础检测器"]
    C --> E["已上线增量检测器"]
    D --> F["全局类别映射与逐框融合"]
    E --> F
    F --> G["检测结果与决策轨迹"]

    H["增量数据包"] --> I["安全解压与数据审计"]
    I --> J["类别绑定与训练视图"]
    J --> K["GPU 快速训练"]
    K --> L["候选模型"]
    L --> M["dev 校准 / lock 复核 / 人工确认"]
    M --> C
```

当前 production 为“增量检测生产代际”：基础检测器负责 `soldier`、`small_aircraft` 和 `tank`，增量检测器当前验证绑定为 `warship`。四类统一 YOLO11s 仅作为性能上限基准，不参与默认推理和回滚。

## 当前状态

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| x86 NVIDIA GPU 推理 | 可用 | 默认 TensorRT FP16，不提供 CPU 回退 |
| Web / CLI | 可用 | 支持检测、决策展示、增量数据管理和结构化日志 |
| 舰船 3+1 类别增量 | 已验证 | 单轮内部 lock-val 证据已通过门禁 |
| 多轮增量 | 工程接口已预留 | 当前训练适配器只完成单轮实证 |
| Ascend 310B | 待硬件验证 | 尚无 OM、AscendCL 和真实板端 FPS |
| 官方隐藏测试提交 | 待赛题信息 | 测试目录和提交格式确认前保持阻塞 |

## 快速开始

### 运行环境

- x86-64 WSL 2 或 Linux
- NVIDIA GPU 及可用的 `nvidia-smi`
- Python `3.10-3.12`
- 建议至少 `10 GB` 可用空间

仓库中的 TensorRT engine 使用 TensorRT `10.8.0.43` 构建，并在 RTX 4060 Laptop、计算能力 `8.9` 上验证。其他 TensorRT 版本或 GPU 架构需要重新导出 engine 并更新资产清单。

<details>
<summary><strong>在其他 NVIDIA 设备上导出 TensorRT engine</strong></summary>

<br>

TensorRT engine 与 TensorRT 版本、GPU 架构和构建参数绑定。不要直接复用不匹配设备上的 `.engine` 文件，也不要覆盖仓库附带的 4060 发布资产。以下流程使用独立配置和本地输出目录。

1. 准备已经安装 CUDA 版 PyTorch 和项目依赖的 Python 环境。若 `bootstrap_x86.sh` 只在最终 `doctor` 阶段报告 engine 或 GPU 不匹配，依赖通常已经安装完成，可继续使用传入的解释器：

   ```bash
   AGENT_PYTHON=/path/to/env/bin/python
   "$AGENT_PYTHON" -m pip install -e ".[workbench,inference,export,dev]"
   ```

   上述 `inference` 依赖组会安装本项目验证过的 TensorRT `10.8.0.43`。若必须使用其他 TensorRT 版本，应在独立环境中手动安装与驱动、CUDA 和 PyTorch 兼容的运行时，不要让安装命令重新覆盖它；该组合需要重新完成本文全部门禁后才能作为发布配置。

2. 查询当前设备和 TensorRT 信息：

   ```bash
   "$AGENT_PYTHON" - <<'PY'
   import tensorrt as trt
   import torch

   device = 0
   print("TensorRT:", trt.__version__)
   print("GPU:", torch.cuda.get_device_name(device))
   print("Compute capability:", ".".join(map(str, torch.cuda.get_device_capability(device))))
   PY
   ```

3. 复制一份设备专用配置，不直接修改已验证的默认配置：

   ```bash
   DEVICE_TAG=my_gpu_trt
   PROFILE="configs/agent_pipeline.${DEVICE_TAG}.yaml"
   cp configs/agent_pipeline.yaml "$PROFILE"
   ```

4. 编辑设备专用 YAML 的 `runtime` 和 `tensorrt_backend`：

   - 将 `runtime.default_device` 和 `tensorrt_backend.export.device` 设为目标 GPU 编号。
   - 将 `expected_version` 和 `expected_compute_capability` 改为上一步的实际值。
   - 保持 `precision: fp16`、`require_exact_gpu: true` 和 `export.overwrite: false`。
   - 将 `validated` 暂设为 `false`。
   - 将三个 engine 的 `path` 改到新目录，例如 `runs/engines/<device_tag>/`。
   - 导出前可将对应 `sha256` 暂写为 64 个 `0`；不要沿用 4060 engine 的哈希。

5. 按 YAML 一次性导出场景模型、三类基础检测器和增量检测器：

   ```bash
   "$AGENT_PYTHON" tools/80_export_tensorrt_engines.py --config "$PROFILE"
   ```

   命令输出会列出每个新 engine 的实际 SHA256。将这些值回填到 `context_engine.sha256` 和 `engines.*.sha256`，然后执行只读校验：

   ```bash
   "$AGENT_PYTHON" tools/80_export_tensorrt_engines.py \
     --config "$PROFILE" --verify-only
   ```

6. 哈希校验通过后，将设备专用配置的 `tensorrt_backend.validated` 设为 `true`，再完成环境、精度和性能复核：

   ```bash
   "$AGENT_PYTHON" -m fair_agent.cli --config "$PROFILE" doctor
   "$AGENT_PYTHON" -m fair_agent.cli --config "$PROFILE" \
     generation recheck --candidate incremental_detection_generation
   "$AGENT_PYTHON" -m fair_agent.cli --config "$PROFILE" benchmark-api
   ```

7. 全部门禁通过后，使用该配置启动服务：

   ```bash
   "$AGENT_PYTHON" -m fair_agent.cli --config "$PROFILE" serve
   ```

`runs/engines/` 是本地目录，不会进入 Git。只有需要正式发布新的设备档案时，才将 engine 移入 `models/engines/<device_tag>/`，更新主 YAML、`models/SHA256SUMS.txt` 和相关 manifest，并重新执行完整发布验收。导出成功不等于精度和性能已经通过，禁止跳过 `generation recheck` 与 `benchmark-api`。

</details>

### 首次配置

```bash
git clone https://github.com/Sakauma/AgileAgent.git
cd AgileAgent
chmod +x scripts/bootstrap_x86.sh scripts/start_agent.sh
./scripts/bootstrap_x86.sh
```

配置脚本会选择或创建兼容的 Python 环境，检查 CUDA、PyTorch、TensorRT 和项目依赖，随后执行发布校验与 GPU 冒烟测试。兼容环境已存在时可显式复用：

```bash
AGILE_AGENT_PYTHON=/path/to/env/bin/python ./scripts/bootstrap_x86.sh
```

已验证的参考组合为 Python `3.10.19`、PyTorch `2.5.1+cu124`、TorchVision `0.20.1+cu124`、Ultralytics `8.4.92` 和 TensorRT `10.8.0.43`。项目允许使用满足约束且通过 `doctor` 的兼容版本，不要求环境名称或安装路径一致。

### 一键启动

```bash
./scripts/start_agent.sh
```

浏览器访问 `http://127.0.0.1:8501`。日常启动脚本只执行环境门禁、刷新状态和启动服务，不安装或修改依赖。

无浏览器环境使用 CLI 工作台：

```bash
./scripts/start_agent.sh --cli
```

## 使用方式

### Web 工作台

Web 面向评委和检测用户，提供：

- 单图检测、自动场景理解和 Agent 决策摘要
- 批量检测、逐图预览和结果包导出
- 增量数据上传、样本浏览、类别补名、数据注入和后台训练
- 当前会话历史结果跳转

用户无需选择模型。Agent 会读取活动代际，执行类别所有者并完成全局 ID 映射和 class-aware NMS。未通过验收的候选模型不会进入默认检测链路。

### CLI

首次配置后，可激活记录在 `.agent-python` 中的环境：

```bash
AGENT_PYTHON="$(cat .agent-python)"
source "$(dirname "$AGENT_PYTHON")/activate"
```

常用命令：

```bash
agile-agent doctor
agile-agent status --format json --refresh
agile-agent console
agile-agent detect --source path/to/image.png --confidence 0.50
agile-agent decide --source path/to/image.png
agile-agent logs --limit 100
agile-agent benchmark-api
```

`detect` 输出完整上下文、候选协议、模型执行、跳过原因和融合记录，适合调试或接入其他程序。Web 仅显示经过简化的用户决策摘要。

## 增量学习工作台

上传包为 ZIP，采用 YOLO 图像和标签结构：

```text
new_batch.zip
├── data.yaml                 # 可选，建议提供 names
├── images/train/*
├── images/val/*              # 可选，缺失时确定性划分
├── labels/train/*
└── labels/val/*
```

支持标准五列标签 `class x_center y_center width height`，也兼容单一新增类别的四列框标签。数据集未提供类别名称时，Agent 会分配稳定全局 ID 和“类别N”临时名称；用户可在 Web 或 CLI 中补充真实名称，重命名不会改变标签映射、训练快照或已有权重。

工作台依次完成：

```text
上传 → 安全审计 → 类别绑定 → 训练视图 → GPU训练 → 待校准候选
```

训练完成不会自动替换 production。候选模型仍需完成增量 dev 阈值校准、冻结哈希、完整 lock 复核和代际 promotion。CLI 等价入口：

```bash
agile-agent incremental-data upload --archive /path/to/new_batch.zip --name 新批次
agile-agent incremental-data list
agile-agent incremental-data show --batch-id BATCH_ID
agile-agent incremental-data rename --batch-id BATCH_ID --class-names 新类别名称
agile-agent incremental-data inject --batch-id BATCH_ID
agile-agent incremental-data train --batch-id BATCH_ID
agile-agent incremental-data jobs --batch-id BATCH_ID
```

完整数据约定和状态流转见 [增量学习工作台](docs/incremental-workbench.md)。

## 模型与指标

| 模型 | 功能 | 内部 lock-val 结果 | 状态 |
| --- | --- | --- | --- |
| Scene-SensorNet | IR/SAR 与四场景认知 | sensor 0.98947 / scene 0.76842 | production 上下文模型 |
| 三类基础检测器 | 冻结旧类检测 | 旧类 mAP50 0.82738 | production 旧类所有者 |
| 增量检测器 | 新类别检测 | New-mAP50 0.79500 / KRR 1.000 | production，当前绑定舰船 |
| 四类统一 YOLO11s | 单模型上限参考 | mAP50 0.91202 | `benchmark_only` |

组合系统在 95 张内部 lock-val 上取得 mAP50 `0.81929`、舰船 precision `1.000`、recall `0.79012` 和误激活率 `0.000`。检测指标均指 mAP50；这些结果不是官方隐藏测试成绩。

RTX 4060 上已观察到超过 30 FPS 的批量吞吐，但单图 API 延迟仍受 GPU 状态和运行环境影响。x86 结果只用于工程验证，不能替代 Ascend 310B 实测。

## 配置管理

[`configs/agent_pipeline.yaml`](configs/agent_pipeline.yaml) 是运行参数的唯一持久事实源。CLI 的 `--set` 只覆盖当前进程；`config set` 会校验并原子写回 YAML：

```bash
agile-agent config validate --config configs/agent_pipeline.yaml
agile-agent config show --config configs/agent_pipeline.yaml --effective
agile-agent config get routing.conflict_iou
agile-agent config set routing.conflict_iou 0.50
agile-agent --set inference.confidence_default=0.60 serve
```

production 代际、类别所有权和权重身份属于受保护状态，只能通过 `generation recheck/promote/rollback` 修改。

## 可复现实验

舰船 3+1 实验通过单一 YAML 描述基础类别、增量类别、数据划分和验收门槛：

```bash
agile-agent experiment validate --config configs/incremental/warship_3plus1.yaml
agile-agent experiment run --config configs/incremental/warship_3plus1.yaml
agile-agent experiment reproduce --manifest runs/experiments/warship_3plus1/<run_id>/run_manifest.json
```

增量阶段禁止读取旧类原始图像、旧类标签和旧数据缓存特征。lock-val 只在权重与阈值冻结后解封。逐文件哈希、状态机和复现边界见 [舰船 3+1 可复现实验](docs/warship-3plus1-reproducibility.md)。

## 开发与验收

```bash
pytest -q
python scripts/verify_release.py
python scripts/smoke_models.py
```

- `pytest` 验证配置、路径门禁、路由融合、增量数据、日志和代际操作。
- `verify_release.py` 校验配置、权重、TensorRT engine 和公开证据哈希。
- `smoke_models.py` 在 GPU 上加载三种功能模型并运行完整自动编排链路。

当前代码基线为 `152 passed`。修改模型、推理后端或依赖后，必须重新执行三项验收。

## 项目结构

```text
fair_agent/
├── backends/       # CUDA/TensorRT 推理后端
├── core/           # 配置、黑板、manifest 和审计日志
├── modules/        # 数据、推理、增量实验和代际管理
├── policies/       # 动作选择与路由策略
├── executors/      # 受控动作执行器
├── web/            # Starlette Web 服务与静态前端
└── ui/             # 终端工作台
configs/            # 运行和实验 YAML
models/             # 冻结权重、engine、注册表和指标
scripts/            # 安装、启动和发布验收
tools/              # 数据处理、训练和导出入口
tests/              # 自动化测试
docs/               # 设计、操作和复现实验文档
```

竞赛数据、标签、运行报告、预测结果和本地凭据均被 Git 忽略。

## 文档

- [Agent 操作手册](docs/agent-operations.md)
- [三种功能模型与协同链路](docs/functional-models.md)
- [合规增量学习规则](docs/compliant-incremental-learning.md)
- [增量学习工作台](docs/incremental-workbench.md)
- [全流程审计日志](docs/agent-audit-logging.md)
- [舰船 3+1 可复现实验](docs/warship-3plus1-reproducibility.md)
- [增量方法比较](docs/incremental-method-comparison.md)
- [YOLO-IOD 完整复现](docs/full-yolo-iod-reproduction.md)

## 已知限制

- 尚未完成 Ascend 310B 的 OM 转换、AscendCL 集成和真实板端 FPS 验证。
- 当前只有一轮舰船类别增量的完整实验证据，多轮能力尚未通过连续真实注入验证。
- Web 训练当前产出待校准候选；校准、lock 复核和 production 切换仍由受审计的 CLI 门禁完成。
- 发布 TensorRT engine 只在 README 声明的软硬件组合上完成验证，不兼容时必须重新导出，禁止回退到 CPU。
- 仓库不包含竞赛数据集、官方测试集或正式提交格式。
