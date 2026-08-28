# Ascend310B 断网一键 4→4+2 增量学习演示

## 用途

这是现场演示的主入口：在已经封装好 production 和独立训练环境的 Ascend310B 板端，只给出当前赛题的 Increment 数据目录，一条命令完成：

```text
增量数据审计
  → 4→4+1→4+2 轮次与标签对齐
  → npu:0 真实反向传播和参数更新
  → Increment dev 选择
  → mixed lock 精度验收
  → ONNX / OM 导出
  → ACL 数值与延迟验收
  → 隔离演示通道部署
  → 启用 Adapter 后完整图像推理 FPS 复测
```

现场提供两条互补的增量路径：本页处理当前 `4→4+2`，类别 `4/5` 的检测候选由冻结专家产生，310B 在板端训练轻量置信度 Adapter；现场真正新增第 7 类及以后类别时，由 [`onsite-4plus2plusn.md`](onsite-4plus2plusn.md) 训练新的检测专家以获得新定位能力。

这不是方案设想，而是已经在 `Ascend310B1 + CANN 7.0.RC1` 上完成的整链实测。2026-08-26 的验收运行从同一条命令开始，在强制离线配置下依次完成两轮真实 NPU 反向传播、选模、精度门禁、ATC、ACL、隔离部署和正式运行时复测，最终 `passed: true`。

## 演示前一次性准备

板端应在断网前已经具备：

- 当前六类 Ascend production 资产与 CANN `7.0.RC1`；
- `/usr/local/miniconda3/envs/agileagent` production 环境；
- `~/agileagent/envs/agileagent_train` 独立 `torch + torch_npu` 训练环境；脚本也会自动识别已验证板上的 `edge_incremental_training_20260825/env`；
- 工程原有 Base dev/lock 评估图像和标签。

断网封装包含 production、独立训练环境、经过验证的 aarch64 wheels、固定评估资产和 CANN 身份。现场命令只使用这些本地资产并在训练前完成环境一致性预检。

## 一条命令完成演示

输入可以是完整数据集根目录，也可以直接是 `datasets_r2_inc_train` 目录：

```bash
cd ~/agileagent/repo

./scripts/run_ascend310b_incremental_demo.sh \
  /path/to/datasets_r2_inc_train
```

包装脚本会自动停止已登记的 Agent 服务，避免它和训练进程同时占用 NPU。脚本强制离线环境变量、移除代理变量，并固定在 `npu:0` 训练。

因此，现场只需要把当前 Increment 数据目录放到板端；Base 权重、Scene-SensorNet、固定 split、Base dev/lock 和两个 Python 环境由预封装工程提供，整条命令在离线边界内完成。

不训练、不停服务、不部署的预检：

```bash
./scripts/run_ascend310b_incremental_demo.sh \
  /path/to/datasets_r2_inc_train \
  --plan-only
```

如果工程的 Base 评估路径没有预先对齐，可显式补充；该路径只用于 dev/lock 联合评估，不进入增量训练：

```bash
./scripts/run_ascend310b_incremental_demo.sh \
  --incremental-data /path/to/datasets_r2_inc_train \
  --base-data /path/to/datasets_r1_base_train
```

## 一键入口保证的口径

| 阶段 | 可读数据 | 是否更新权重 |
| --- | --- | --- |
| 当轮增量训练 | Increment train/dev | 仅当轮 8 参数 Adapter |
| 系统选择 | mixed dev | 否，只选 Adapter 强度 |
| 联合验收 | mixed lock | 否，全部冻结 |
| 运行时性能 | 20 张 lock 图像 | 否 |

Base、原有 Incremental 检测器和 Scene-SensorNet 始终冻结。`input_audit.json` 会明确记录 `base_images_used_for_training: 0` 和 `old_raw_image_count: 0`。Scene-SensorNet 仍属于系统上下文，不计作增量学习器。

## 自动部署与验收

只有以下条件全部通过，候选才会写入隔离演示通道：

- Base mAP50 `>= 0.80`；
- New-mAP50 `>= 0.60`；
- KRR `>= 0.95`；
- Adapter OM 数值一致性通过；
- 启用 Adapter 后的全流程 aggregate FPS `>= 30`，按总帧数除以总墙钟耗时计算。

性能复测包含图像解码、Scene-SensorNet、决策、Base、Incremental 专家、Adapter、门控、融合和正式六列 TXT 写出。各轮 FPS 仅诊断，门禁统一按所有轮次总帧数除以全流程总墙钟耗时计算。

演示部署会产生独立配置，与满分 production 并列保存：

```text
runs/ascend_edge_incremental_demo/<run_id>/
├── demo_plan.json
├── demo_state.json
├── demo_report.json
├── workflow/
├── evaluation/runtime_benchmark.json
└── deployment/
    ├── edge_adapter_bank.om
    ├── adapter_manifest.json
    └── agent_pipeline_ascend310b_demo.yaml
```

需要在演示验收后继续用 CLI 查看效果时，使用报告中的 `demo_config` 启动：

```bash
AGILE_AGENT_CONFIG=/absolute/run/deployment/agent_pipeline_ascend310b_demo.yaml \
  ./scripts/start_agent.sh --cli
```

不设置 `AGILE_AGENT_CONFIG` 时，下次启动仍使用原 production。任一门禁失败时，Adapter 清单会被撤销为 `accepted: false`，演示配置也会因验收保护而拒绝加载。

一键脚本为了独占 NPU 会先停止已登记服务。演示结束后若要恢复原正式 CLI，直接运行：

```bash
./scripts/start_agent.sh --cli
```

若要查看本次学习后的隔离候选，才设置上面的 `AGILE_AGENT_CONFIG`。两者不会相互覆盖。

## 板端实测结果

2026-08-26 完整热态演练 `board-full-check-v6` 的结果如下。“热态”只表示 torch_npu 图编译缓存已经由前一次冷态运行建立，不跳过任何训练、导出或验收阶段。

| 内容 | 时间 |
| --- | ---: |
| 两轮 5 seed × 3 LR 真实 NPU 训练搜索 | `155.17 秒`（2 分 35 秒） |
| ONNX 导出 | `4.82 秒` |
| ATC 编译 OM | `87.67 秒`（1 分 28 秒） |
| 从输入审计到隔离部署及 FPS 门禁的完整命令 | `1007.07 秒`（16 分 47 秒） |

精度与运行时门禁结果：

| 指标 | 实测值 | 门禁 |
| --- | ---: | ---: |
| Base mAP50 | `0.816663` | `>= 0.80` |
| New-mAP50 | `0.624935` | `>= 0.60` |
| KRR | `1.000000` | `>= 0.95` |
| Full-mAP50 | `0.726497` | 记录项 |
| 新类误激活 | `17 / 75` | 记录项 |
| 旧 engine-only 图像链路三轮 FPS | `39.05 / 38.70 / 37.92` | 历史诊断 |
| 旧 engine-only 中位 FPS | `38.6995` | 非当前官方口径，待新工具重测 |
| production 身份保持 | `true` | 通过 |

首次冷态运行会为第一组训练候选编译 NPU 图。实测冷态的两轮训练搜索阶段为 `760.61 秒`（12 分 41 秒），比热态多约 10 分钟。现场应按冷态预留 **30 分钟**；重复演示通常约 17 分钟。不要依赖缓存作为是否通过的条件，缓存只影响等待时间，不改变种子、学习率、选中强度或验收结果。

首次 torch_npu 图编译占用了大部分冷启动时间。期间可能出现 CANN `vendors/customize` 权限 traceback；该板的已验收运行会继续完成图编译，并以进程退出码和 `npu_backward.json` 记录最终 NPU 反向传播状态。
