# 灵动Agent

面向 IR/SAR 时变场景目标检测的快速学习智能体。系统以场景与传感器认知为上下文，自动协调冻结基础检测器和增量检测器，并提供增量数据审计、快速训练、模型复核、代际切换与回滚能力。

项目同时提供面向检测用户的 Web 工作台和面向开发、运维及端侧集成的 CLI。本机开发、训练与板前代理验证运行于 x86-64 WSL/Linux 和 NVIDIA GPU；竞赛正式部署目标固定为华为 Ascend 310B。当前仓库已实现基于 PyACL 的三模型 OM 后端和可选 DVPP 编码图像预处理；据本轮 SSH 只读复核，开发板上的正式 release 已能提供真实 PNG 推理。正式 release 的89图记录为平均引擎 `57.849 ms`、墙钟 `71.491 ms`；当前最完整的端到端 API 证据来自 AIPP staging 候选，为 `19.53 FPS`。两者均未达到端到端 `30 FPS`，且 staging 结果不代表正式服务。TensorRT 不属于 310B 部署链路。

## 核心能力

- **自动识别与编排**：识别 IR/SAR 传感器和 air/forest/sea/urban 已知场景，自动解析当前生产代际，并让冻结基础 owner 与全部活动增量 owner 处理每张图像。
- **增量目标检测**：支持类别增量和目标增量；训练、验证、早停和调参只读取本批增量数据。
- **双前端操作**：Web 提供单图、批量检测和增量数据工作台；CLI 提供完整决策轨迹、配置、实验、日志及代际管理。
- **可审计与可回滚**：离线增量训练记录数据隔离、配置、权重、阈值和评测证据；候选模型通过门禁后才能切换 production，失败时保留原代际。在线无标签检测不计算图像摘要，也不依据训练集或测试集身份路由。
- **配置驱动**：本机 GPU、推理、路由、融合、上传、缓存、训练及验收参数统一由 YAML 管理，并支持 CLI 临时覆盖或持久修改；Ascend 设备参数在独立板端配置中冻结。

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
    L --> M["dev 校准 / lock 复核 / 受控上线"]
    M --> C
```

当前 production 固定为通过全量750张模拟3+1验收的“双检测器增量代际”。冻结三类基础检测器永久负责旧类别，增量专家从通用预训练表征初始化、只读取新增类 train/dev 并只负责新增类别；两个 owner 对每张未知图共同推理后做框级融合，不使用场景到目标类别的硬路由。候选必须同时通过赛题三项与完整性门禁（`competition_accepted`），以及新增类 precision 和老图误激活部署门禁（`deployment_accepted`），才能进入 production。

## 当前状态

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| x86 NVIDIA GPU 开发与验证 | 可用 | 默认 PyTorch CUDA 加载模型权重，仅用于训练、本机功能验证和板前代理，不是竞赛部署后端 |
| Web / CLI | 可用 | 支持检测、决策展示、增量数据管理和结构化日志 |
| 舰船 3+1 类别增量 | 当前750张模拟测试满分档且通过部署门禁 | 基础 mAP50 `0.81414` / New-mAP50 `0.63869` / KRR `1.00000` / 新类 precision `0.92453` / 老图误激活 `1/70`；尚不代表官方隐藏测试成绩 |
| 后续官方增量数据 | 模板就绪 | 替换类别与清单后重新训练，增量阶段仍只读取当轮新增类数据 |
| Ascend 310B | 板端推理可用，端到端性能未达标 | 据本轮 SSH 只读复核，正式 release 健康状态为 `ready`；89图平均引擎 `57.849 ms` / 墙钟 `71.491 ms`，该批次不是 HTTP 压测。AIPP staging 候选的真实 multipart PNG API 为平均 `51.203 ms` / P95 `63.9 ms` / `19.53 FPS`。只有已解码 Agent 核心候选达到 `31.11 FPS`，不等同于端到端达标 |
| 官方隐藏测试提交 | 待赛题信息 | 测试目录和提交格式确认前保持阻塞 |

当前唯一活动划分是覆盖全部750张图的固定3+1协议；不再施加赛题未要求的连续帧边界间距。历史划分只保存在 `archive/`，不参与配置、训练、选模或验收。

## 阅读导航

- [快速开始](#快速开始)：本机开发环境、Ascend 部署边界、首次配置和一键启动。
- [系统架构](docs/ARCHITECTURE.md)：主要组件、在线推理、增量生命周期和关键抽象。
- [配置参考](docs/CONFIGURATION.md)：完整 YAML、环境变量、覆盖顺序和保护边界。
- [开发指南](docs/DEVELOPMENT.md) 与 [测试指南](docs/TESTING.md)：开发环境、常用命令、CI 和测试范围。
- [使用方式](#使用方式)：Web 与 CLI 检测入口。
- [增量学习工作台](#增量学习工作台)：上传、训练、校准、复核与上线流程。
- [配置管理](#配置管理)：YAML 参数和 CLI 覆盖。
- [可复现实验](#可复现实验)：唯一舰船3+1训练与复核入口。
- [Ascend 310B 稳定加速设计](docs/ascend-310b-deployment.md)：固定形状 OM、AscendCL、预处理、量化边界与板端验收。
- [Ascend 310B 当前工程评估](docs/ascend-310b-current-status.md)：正式部署、精度、性能、实验候选、风险与后续优先级。
- [开发与验收](#开发与验收)：测试、发布检查和 GPU 冒烟命令。

## 快速开始

### 本机开发环境

- x86-64 WSL 2 或 Linux
- NVIDIA GPU 及可用的 `nvidia-smi`
- Python `3.10-3.12`
- 建议至少 `10 GB` 可用空间

该环境负责训练、功能测试、ONNX 导出、数值对齐和板前性能代理。默认配置使用 CUDA 版 PyTorch 和 Ultralytics 加载 `.pt` 权重；本机 CUDA/ONNX Runtime 结果不得作为 310B 性能证据。

版本库只同步源代码、公共配置、文档、训练权重，以及复核模型身份和指标所必需的校准、指标与 manifest。公开的 Ascend 部署配置模板 [`configs/agent_pipeline_ascend310b.yaml`](configs/agent_pipeline_ascend310b.yaml) 也由 Git 跟踪；设备本地覆盖、凭据、运行生成配置、数据视图、运行报告、预测结果、ONNX/OM 文件、原生构建目录和运行缓存留在本地。

### 正式部署后端边界

310B 正式链路固定为：

- 固定 shape ONNX 作为 ATC 输入。
- ATC 编译生成设备专用 OM。
- `fair_agent/backends/ascend_acl.py` 通过 PyACL/AscendCL 负责 OM 加载、固定缓冲和推理执行。
- 正式 release 的 `encoded_preprocessing` 固定为 `cpu`；仓库中的 DVPP/VPC 编码输入路径为默认关闭的实验能力，未完成完整89张精度门禁前不得切换。
- Python/NumPy 后处理完成 YOLO 解码、NMS、框级融合和场景软门禁；每张图都必须执行 Base、Incremental 和 Scene。

TensorRT 是 NVIDIA 专用推理运行时，华为 Ascend 310B 不使用 `.engine`、TensorRT builder 或 CUDA 原生后端。仓库中已有的 TensorRT 文件只保留为历史 x86 兼容实现，不参与竞赛部署、板端编译、精度验收或性能结论。310B 的完整操作与验收步骤以 [Ascend 310B 稳定加速设计](docs/ascend-310b-deployment.md) 为唯一部署依据。

### 首次配置

```bash
git clone https://github.com/Sakauma/AgileAgent.git
cd AgileAgent
chmod +x scripts/bootstrap_x86.sh scripts/start_agent.sh
./scripts/bootstrap_x86.sh
```

配置脚本会选择或创建兼容的 Python 环境，检查 CUDA、PyTorch 和项目依赖，随后执行发布校验与 GPU 冒烟测试。兼容环境已存在时可显式复用：

```bash
AGILE_AGENT_PYTHON=/path/to/env/bin/python ./scripts/bootstrap_x86.sh
```

兼容的第三方依赖会直接复用，不会重复安装。配置脚本还会单独确认 `agile-agent` 命令入口属于当前检出的仓库；仅当入口缺失或指向其他目录时，才以 `--no-deps` 方式重新注册当前项目。

已验证的参考组合为 Python `3.10.19`、PyTorch `2.5.1+cu124`、TorchVision `0.20.1+cu124` 和 Ultralytics `8.4.92`。项目允许使用满足约束且通过 `doctor` 的兼容版本，不要求环境名称或安装路径一致。

### 一键启动

```bash
./scripts/start_agent.sh
```

浏览器访问 `http://127.0.0.1:8501`。日常启动脚本只执行环境门禁、刷新状态和启动服务，不安装或修改依赖。

无浏览器环境使用 CLI 工作台：

```bash
./scripts/start_agent.sh --cli
```

## 竞赛数据与固定划分

默认 Web/CLI 检测使用仓库内发布权重，不要求下载竞赛训练集。API 性能验收、舰船 3+1 复现实验和指标复核需要原始竞赛训练数据。由于数据授权限制，仓库只发布固定划分清单，不发布图像和标签。

将数据放置为以下结构：

```text
datasets_r1_base_train/
├── classes.txt
├── ir_r1_base_air_000001.png
├── ir_r1_base_air_000001.txt
└── ...
```

数据集根目录中名为 classes.txt 的类别文件按行写入：

```text
soldier
small_aircraft
warship
tank
```

仓库中的 [`splits/`](splits/README.md) 是唯一活动划分，固定模拟 `warship` 为新增类别：基础训练/验证为 `441/70` 张且只含 `soldier`、`small_aircraft`、`tank`，增量训练/验证为 `132/18` 张且只含 `warship`；基础测试为70张纯旧类图，最终混合测试为89张（70张旧类图 + 19张新类图）。场景识别使用 `573/88/89` 张已知场景清单，以上源池互斥且完整覆盖全部750张图。旧划分只保存在 [`archive/`](archive/README.md)，不再作为活动入口。

### 官方固定评分口径（不得改写）

本仓库将赛题中的检测 `mAP` 固定实现为 `mAP@0.5`（下文记作 `mAP50`），正式验收只看以下三项：

| 计分项 | 正确计算口径 | 满分门槛 |
|---|---|---:|
| 基础目标检测 | 增量前的冻结基础模型，在不含新增类别的基础测试集上按三种基础类别计算 `base_test_map50` | `>= 0.80` |
| 新类别识别 | 增量后系统先在完整“旧类+新类”混合测试集上推理，再只按新增类别计算 `new_map50`（New-mAP）；旧类图上的新增类误检同样计入 | `>= 0.60` |
| 旧类别知识保持 | 在同一完整混合测试集、同一批图像上计算 `KRR = old_map50_after / old_map50_before` | `>= 0.95` |

`0.80` 是赛题基础检测满分线，不是训练提前停止条件。基础训练必须跑满160 epoch、增量训练必须跑满80 epoch，最终 `best.pt` 只在完整预算结束后按各自 dev mAP50 选择。

基础检测分档为：`mAP50 >= 0.80/0.70/0.65/0.60/0.50/0.40` 时分别得 `30/25/20/15/10/5` 分，低于 `0.40` 得 `0` 分。New-mAP 分档为：`>= 0.60/0.50/0.40` 时分别得 `10/7/4` 分，低于 `0.40` 得 `0` 分。KRR 分档为：`>= 0.95/0.90/0.80` 时分别得 `10/7/4` 分，低于 `0.80` 得 `0` 分。

四类总体 `full_map50` 只作为诊断指标，**不是**“New-mAP >= 0.60”的替代项，也不得参与上述三项通过门禁。`base_dev` 只允许用于 checkpoint 选择，不得冒充基础测试成绩。

KRR 是相对保持率，不代表旧类的绝对精度。即使增量前后的旧类 mAP 都只有0.75，只要二者相等，KRR 仍为1.0；因此 `base_test_map50 >= 0.80` 与 `KRR >= 0.95` 必须分别通过。基础项只在70张纯旧类图像上计算，而 KRR 的分子和分母都在完整89张混合集上计算旧类 mAP，所以两处旧类绝对数值也不要求相同。当前并行 Agent 冻结旧类 owner，使增量前后旧类预测逐框一致，但这不能替代基础绝对精度门槛。

官方 KRR 按旧类别 AP 计算，新类别框即使错误出现在旧类图上，也不会改变旧类别 AP，因此可能出现“KRR 为1但最终输出有错误新增框”。仓库不篡改官方定义，而是另设 production 部署门禁：新增类 precision 必须 `>= 0.90`，70张旧类图上的新增类误激活率必须 `<= 0.05`。两项不增加赛题分数，但任一失败都会令 `deployment_accepted=false` 并禁止上线。

混合测试的单张图像没有“调用基础模型”或“调用新增类模型”的先验标签。计分链路必须对全部89张混合图像执行冻结基础检测器和全部活动增量专家，且输入 stem 集合完全一致。权重、阈值和融合规则冻结后，先保存并哈希原始预测和融合预测，随后评分器才读取 `splits/strict_3plus1/base_test.txt` 和测试标签：基础 mAP50 在70张旧类子集上计算，New-mAP50 与 KRR 在完整89张混合集上计算。`splits/strict_3plus1/base_test.txt` 只用于评分，禁止用于图片级模型路由。场景识别始终是已知场景识别，不得硬绑定目标类别。

这里的“冻结”是训练期参数约束，不是图片级开关：进入增量训练前，基础 owner 的参数被排除出优化器并保持权重哈希不变；进入推理后本来就没有反向传播，因此系统既不需要也不允许判断某张图属于新类还是旧类。每张未知图片始终进入全部基础推理 pass 和增量 owner，最后只按各 owner 的全局类别所有权做框级合并。

`predict.conf=0.01` 只是在冻结预测时保留原始候选的检测下限，不是新增类别的上线阈值。当前 production 的新增类阈值只由18张 `increment_dev` 校准为 `0.63`；在此之后先执行框级 IoU+置信度冲突仲裁，再使用132张 `increment_train` 学到的已知场景先验施加最多 `+0.05` 的软阈值惩罚。场景证据不能跳过任何 owner，也不能单独激活新增类。当前默认未启用新增框覆盖率规则，因为消融显示它会过多删除真实舰船框。

数据就位后执行：

```bash
python tools/00_check_dataset.py
python tools/01_build_metadata.py
agile-agent experiment validate --config configs/incremental/warship_3plus1.yaml
```

三项均通过后，才运行依赖真实数据的命令：

```bash
agile-agent benchmark-api
agile-agent experiment run --config configs/incremental/warship_3plus1.yaml
```

`benchmark-api` 使用 `splits/strict_3plus1/mixed_test.txt` 中的 89 张旧类与新类混合图像；缺少原始数据时不能用合成图替代正式性能验收。严格训练入口固定为：

```bash
python tools/70_run_strict_3plus1.py --check-only
python tools/70_run_strict_3plus1.py --run-id UNIQUE_RUN_ID
```

唯一正式配方为 `YOLO11s`，基础和增量阶段都从仓库内的 `models/pretrained/yolo11s.pt` 初始化：基础 owner 使用 `imgsz=896 / batch=32 / epochs=160 / patience=0`，增量专家使用 `imgsz=640 / batch=32 / epochs=80 / patience=0`，均禁用提前停止并在完整预算后选择 best。当前基础 best 为 seed `20260705`、best epoch `85`；基础局部类别 `0/1/2` 映射到全局 `0/1/3`，增量局部类别 `0` 映射到全局 `2`。旧的四类 `imgsz=640` benchmark 权重已移除，不属于训练或运行入口。未来官方增量数据到达后只替换类别映射与清单并重新训练，不依赖 `warship` 语义。

## 使用方式

### Web 工作台

Web 面向评委和检测用户，提供：

- 单图检测、自动场景理解和 Agent 决策摘要
- 批量检测、逐图预览和结果包导出
- 增量数据上传、样本浏览、类别补名、数据注入和后台训练
- 当前会话历史结果跳转

用户无需选择模型。最终检测输入统一视为无标签图像，Agent 不依据文件名、内容哈希、训练集身份或测试集身份选择模型。每张图像都解析当前 production 代际，执行冻结基础检测器和全部活动类别所有者，再依据图像内容、场景软证据、逐类阈值和冲突仲裁完成全局 ID 映射与 class-aware NMS。未通过验收的候选模型不会进入默认检测链路。

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

赛题增量数据与基础训练集采用相同的五列 YOLO 标签：`class x_center y_center width height`，Agent 会直接读取类别 ID。为兼容异常或临时数据，四列无类别标签不会被直接拒绝，而会按单一待确认类别导入，并在批次后台和结构化日志中显示警告。数据集未提供类别名称时，Agent 会分配稳定全局 ID 和“类别N”临时名称；用户可在 Web 或 CLI 中补充真实名称，重命名不会改变标签映射、训练快照或已有权重。

工作台依次完成：

```text
上传 → 训练数据隔离审计 → 自动封存lock → GPU训练 → dev逐类校准 → 可选INT8 PTQ → lock复核 → shadow加载 → 受控上线
```

生产配置会在缺少显式 lock 时按固定种子和类别组合自动封存20%样本；封存内容不进入训练 YAML。训练完成后，增量学习守护器自动完成 dev 诊断、动态混淆图、逐类阈值校准、代际注册、独立 lock 复核和 shadow 预热。当前750张图严格3+1模拟以基础 mAP50、New-mAP50 和 KRR 为三项官方计分门槛；数据隔离与资产一致性仍是发布完整性前提。四类总体 mAP50 和 x86 时延只生成诊断；新增类 precision 与老图误激活率不改变官方分数，但属于 production 必过质量门禁。310B FPS 由板端部署验收单独判定。

批次通过门禁并晋升后，Agent 会登记新的类别所有者和 production 代际。后续所有无标签图像都执行该代际：旧类别继续由冻结基础模型负责，新增类别由活动增量模型负责。训练数据隔离记录只用于证明增量训练没有读取历史样本，绝不参与在线推理或模型选择。

同一批次包含多个新增类别时只训练一个多类增量检测器，并为每个全局类别保存独立阈值。推荐的统一 CLI 入口：

```bash
agile-agent incremental audit --batch /path/to/new_batch.zip
agile-agent incremental run --batch BATCH_ID
agile-agent incremental status --run-id TRAIN_JOB_ID
```

`incremental run` 是前台完整生命周期命令，会持续运行到上线、拒绝或回滚并返回相应退出码；Web 工作台使用同一状态机，但任务在后台执行。

底层分步命令仍可用于排障：

```bash
agile-agent incremental-data upload --archive /path/to/new_batch.zip --name 新批次
agile-agent incremental-data list
agile-agent incremental-data show --batch-id BATCH_ID
agile-agent incremental-data rename --batch-id BATCH_ID --class-names 新类别名称
agile-agent incremental-data inject --batch-id BATCH_ID
agile-agent incremental-data train --batch-id BATCH_ID
agile-agent incremental-data jobs --batch-id BATCH_ID
```

Web 与 CLI 使用同一状态机和配置，批次状态、任务日志及最终门禁结论可以通过对应的 `status` 和 `logs` 命令查询。

## 模型与指标

| 模型 | 功能 | 当前750张模拟验收 | 状态 |
| --- | --- | --- | --- |
| YOLO11s 通用预训练权重 | 基础与增量训练初始化 | 不直接计分 | `models/pretrained/yolo11s.pt`，由 Git 固定版本 |
| Scene-SensorNet | IR/SAR 与四场景认知 | sensor `0.96629` / scene `0.80899` / joint `0.77528` | 使用573/88/89场景划分训练/验证/测试，仅提供软证据 |
| 三类基础检测器 | 冻结旧类检测 | 基础 mAP50 `0.81414` | production 旧类 owner，推理尺寸 `896` |
| 增量检测器 | 新类别检测 | New-mAP50 `0.63869` / KRR `1.00000` / precision `0.92453` / 误激活率 `0.01429` | production 新类 owner，推理尺寸 `640`，dev阈值 `0.63`，当前模拟绑定舰船 |

最终基础权重 SHA256 为 `6cf015573f50fe49c7c42203c6ca587b890d81e75a349b26bbe696dc77470119`，增量权重 SHA256 为 `d27bda7cb89375788deb1f29366b037757f23f7b32ddf6c11e1aa778384dc957`。本次89张混合测试的四类总体 mAP50 为 `0.75390`，仅作诊断；上述结果不是官方隐藏测试成绩。复核时两个 owner 都对完整89张混合集推理，`label_aware_routing=false`、`filename_class_routing=false`、`scene_hard_routing=false`。

## 配置管理

[`configs/agent_pipeline.yaml`](configs/agent_pipeline.yaml) 是 x86/NVIDIA 本机运行的默认持久事实源；Ascend 正式部署使用独立的 [`configs/agent_pipeline_ascend310b.yaml`](configs/agent_pipeline_ascend310b.yaml)。CLI 的 `--set` 只覆盖当前进程；`config set` 会校验并原子写回所选 YAML：

```bash
agile-agent config validate --config configs/agent_pipeline.yaml
agile-agent config show --config configs/agent_pipeline.yaml --effective
agile-agent config get routing.conflict_iou
agile-agent config set routing.conflict_iou 0.50
agile-agent --set inference.confidence_default=0.60 serve
```

production 代际、类别所有权和权重身份属于受保护状态，只能通过 `generation recheck/promote/rollback` 修改。

## 可复现实验

舰船 3+1 实验通过单一 YAML 描述基础类别、增量类别、数据划分和验收门槛；配置只读取当前 `splits/`：

```bash
agile-agent experiment validate --config configs/incremental/warship_3plus1.yaml
agile-agent experiment run --config configs/incremental/warship_3plus1.yaml
agile-agent experiment reproduce --manifest runs/experiments/warship_3plus1/<run_id>/run_manifest.json
```

增量阶段禁止读取旧类原始图像、旧类标签和旧数据缓存特征。lock-val 只在权重与阈值冻结后解封。每次运行的配置快照、状态机、逐图预测和复现边界均保存在独立实验目录。

正式训练与复核只使用 `configs/strict_class_incremental_3plus1.yaml` 和 `tools/70_run_strict_3plus1.py`；仓库不再保留与当前赛题口径无关的 OOF、ensemble、困难类专家或多批次压力配置。

## 开发与验收

```bash
pytest -q
python scripts/verify_release.py
python scripts/smoke_models.py --load-only
```

- `pytest` 验证配置、路径门禁、路由融合、增量数据、日志和代际操作。
- `scripts/verify_release.py` 校验配置、模型权重和公开证据哈希。
- `scripts/smoke_models.py --load-only` 在 GPU 上校验并加载三种功能模型，不导入或启动 Web 服务；仅在需要额外验证 Web 自动编排链路时才省略 `--load-only`。

不启动 Web 的 Ascend 板前复核入口如下；ONNX、golden 和性能报告均写入 Git 忽略的 `runs/ascend310b/`：

```bash
python tools/90_ascend_preflight.py export --shape-mode rect --device 0
python tools/90_ascend_preflight.py raw-align --shape-mode rect --device 0 --samples 6
python tools/90_ascend_preflight.py metric-align --shape-mode rect --device 0
python tools/90_ascend_preflight.py golden --shape-mode rect --device 0 --samples 6
python tools/90_ascend_preflight.py benchmark --shape-mode rect --device 0 \
  --samples 6 --warmup 20 --rounds 100
python tools/90_ascend_preflight.py optimize --shape-mode rect --device 0 \
  --samples 6 --warmup 20 --rounds 100
```

`optimize` 对每轮全部候选使用同一张性能样本，并额外在完整89张混合集上做不读取标签的逐层等价性检查；基础、增量和场景三个 owner 对每张图都会执行。本机稳定候选组合为 OpenCV PNG 解码、固定地址 CUDA Graph 代理、NMS 候选预筛选和新增类最低阈值前移，实测平均 `30.337 ms`、P95 `31.918 ms`、按平均值折算 `32.96 FPS`。它只存在于板前基准路径，production 运行链路尚未切换；CUDA 代理结果也不能替代 310B 的 OM、ACL、DVPP、多 stream 和稳定性验收。

历史板前阶段曾使用混合 FP16 CUDA 代理暴露精度敏感点。以下命令从已验证的 FP32 ONNX 生成“内部 FP16、输入输出 FP32”的独立候选目录，不覆盖 FP32 基线；它们仍可用于复现历史板前实验，但不能替代当前板端结果：

```bash
python tools/90_ascend_preflight.py convert-fp16 \
  --source-root runs/ascend310b \
  --output-root runs/ascend310b_mixed_fp16 \
  --shape-mode rect --overwrite
python tools/90_ascend_preflight.py metric-align \
  --output-root runs/ascend310b_mixed_fp16 \
  --shape-mode rect --device 0 --provider cuda
python tools/90_ascend_preflight.py optimize \
  --output-root runs/ascend310b_mixed_fp16 \
  --shape-mode rect --device 0 --provider cuda --warmup 30 --rounds 100
```

历史混合 FP16 CUDA 代理的基础 mAP50 `0.81954`、New-mAP50 `0.63869`、KRR `1.00000`、新类 precision `0.92453`、老图误激活率 `0.01429`，五项门槛均通过；但严格输出一致候选为 `26.97 FPS`，未优于同配置 FP32 的 `27.63 FPS`。启用新增类最低阈值前移后虽可达 `31.81 FPS`，却改变 `1/89` 张最终结果，因此已拒绝。开发板现已完成独立 ATC/OM 和板端验收，当前结论以 [Ascend 310B 当前工程评估](docs/ascend-310b-current-status.md) 为准；CUDA 代理仍不能代表310B。

本轮共收集到 `221` 个 pytest 用例，并按板端范围执行了 Ascend 后端、Web 契约和静态发布校验相关的 `33` 项，结果为 `33 passed, 1 warning`；`python scripts/verify_release.py` 也已通过。该脚本通过只表示默认 x86 配置、资产和公开证据一致；当前 `configs/functional_models.yaml` 仍把三个功能模型的 `ascend_310b` 标为 `false`，因此静态状态仍报告 `ascend_310b_not_ready`，尚不能用它证明板端 release 已验收。部分测试和训练能力只适用于 x86 环境，未在本轮全部执行。修改模型、推理后端或依赖后，必须重新执行对应验收。

## 项目结构

```text
fair_agent/
├── backends/       # 本机开发推理与后端适配器
├── core/           # 配置、黑板、manifest 和审计日志
├── modules/        # 数据、推理、增量实验和代际管理
├── policies/       # 动作选择与路由策略
├── executors/      # 受控动作执行器
├── web/            # Starlette Web 服务与静态前端
└── ui/             # 终端工作台
native/             # 历史 x86/CUDA 兼容后端，不参与310B部署
native_ascend/      # 可选 C++ C ABI 与无 CANN contract stub；当前正式板端后端位于 fair_agent/backends/ascend_acl.py
configs/            # 运行和实验 YAML
models/             # 冻结权重、注册表和指标
splits/             # 唯一活动的全量750张严格 3+1 划分
archive/            # 仅供历史复现的旧划分
scripts/            # 安装、启动和发布验收
tools/              # 数据处理、训练和导出入口
tests/              # 自动化测试
```

竞赛图像、标签、运行报告、预测结果、设备部署产物、构建缓存和本地凭据均被 Git 忽略。固定数据划分清单由 Git 跟踪，用于在各设备上复现同一 train/dev/lock 边界。历史 x86 加速兼容代码不会进入310B构建；ONNX、OM 和板端运行产物均不进入版本控制。

## 已知限制

- Ascend 310B 已完成 OM、PyACL/AscendCL 集成和真实板端测试，但正式 release 的89图记录不是 HTTP 压测；AIPP staging 候选的真实 PNG API 仅为平均 `19.53 FPS`，尚未达到 `30 FPS`。一小时稳定性和 DVPP 完整89张精度门禁也尚未完成。
- 静态功能模型注册表仍保留板前状态 `ascend_310b: false`，导致发布校验继续报告 `ascend_310b_not_ready`；板端部署记录与仓库级发布元数据尚未形成同一套可复现证据。
- 两份运行配置中的 `policies.end_device` 仍是旧值 `paused_until_ascend_board_ready`；Ascend 配置的 `refresh_blackboard.required_artifacts` 也仍指向默认 x86 配置。这些元数据不决定 PyACL 服务能否启动，但会污染决策展示和证据追踪，后续应随 release manifest 一并修正。
- 当前750张模拟只验证了3个基础类别 + 1个新增类别；真实未知类别和多轮官方增量数据到达后必须按同一模板重新训练并复核。
- Web 与 CLI 均会从训练继续执行到逐类校准、lock复核和受控上线；任一门禁失败时保持原production。
- 仓库不包含竞赛数据集、官方测试集或正式提交格式。
