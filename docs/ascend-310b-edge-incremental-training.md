# Ascend310B 板端轻量增量训练

## 功能定位

该功能用于证明 Ascend310B 可以在端侧完成真实的增量参数更新。它读取当前 production 三模型产生的冻结候选，在每个注册新增类别上训练一个 8 参数置信度残差 Adapter：

```text
冻结 Base / Incremental / Scene-SensorNet
                  │
                  ▼
  候选置信度、框面积、场景概率、传感器概率（8 维）
                  │
                  ▼
       当轮新增类别 Residual Adapter（8 参数）
                  │
                  ▼
       dev 强度选择 → lock 评分 → ONNX / OM
```

当前 4+2 注册表包含两个新增类别，因此总计训练 16 个参数。反向传播和 SGD 参数更新真实发生在 `npu:0`；Base、已有 Incremental 检测器和 Scene-SensorNet 权重始终冻结。

该实现采用“冻结检测器 + 轻量 Adapter”的端侧增量形式，适合当前 `4→4+2` 已有检测候选的现场演示。真正新增且尚无定位候选的 `4+2+n` 类别由 [`onsite-4plus2plusn.md`](onsite-4plus2plusn.md) 的新检测专家链路负责。底层 `run_pipeline.sh` 产出候选与证据，现场主入口在完整门禁通过后将候选提升到独立演示配置；正式 production、CANN 和父代 release 保持可直接启动。

## 现场主入口

已经验证的板端训练能力现在已封装为当前 `4→4+2` 的断网一键演示：

```bash
./scripts/run_ascend310b_incremental_demo.sh /path/to/datasets_r2_inc_train
```

该入口自动对齐当前固定 split，调用本目录的底层流水线，然后把通过门禁的 Adapter 部署到隔离演示配置，最后将 Adapter 实际接入完整图像推理链路重测 FPS。隔离演示配置和满分 production 作为两个可选择的运行身份并存。现场操作、候选撤销和演示 CLI 启动方式见 [`ascend-310b-offline-incremental-demo.md`](ascend-310b-offline-incremental-demo.md)。

下文保留底层手工入口，用于研发调试和重现单个阶段。

## 数据与协议边界

类别、轮次、父子代际和 split 全部读取 [`configs/incremental_round_registry_4plus2.yaml`](../configs/incremental_round_registry_4plus2.yaml)，训练入口不固定写死类别 `4/5` 或两条 split 路径。当前 Adapter 实现要求每轮注册一个新增类别。

| 阶段 | 读取范围 | 标签用途 | 权重更新 |
| --- | --- | --- | --- |
| `training` probe | 各轮 Increment train/dev，共 126 图 | 不读标签，仅冻结原始候选 | 无 |
| `incremental_learning` | 当轮 Increment train/dev | train 训练、dev 选 seed/LR | 只更新当轮 8 参数 Adapter |
| `selection` probe | Base dev + 各轮 Increment dev，共 89 图 | probe 不读标签 | 无 |
| `system_calibration` | mixed dev | 选择逐类 Adapter 强度 | 不更新任何权重 |
| `lock` probe | Base lock + 各轮 Increment lock，共 89 图 | Adapter 冻结后才评分 | 无 |
| `all` 诊断（可选） | train/dev/lock 全部 890 图 | 只作二号诊断 | 无 |

注册表校验会拒绝 Base 未冻结、旧专家未冻结、`old_raw_image_count != 0`、非当轮标签投影、错误父子代际或多类别单轮。训练 probe 必须与注册范围完全一致，且不得包含 Base 图像；lock probe 只会在训练和 dev 强度选择结束后生成。

## 板端前提

- Atlas 200I DK A2 / Ascend310B1，`aarch64`；
- CANN `7.0.RC1` 已可用，`atc` 与 PyACL 正常；
- 当前 production release 已物化，`configs/agent_pipeline_ascend310b.yaml` 中的绝对模型与代际路径在板端有效；
- 数据目录已经放在仓库 split 所引用的位置，或以软链接映射到这些位置；
- production 服务建议先停止，避免同时占用模型和板端内存：

  ```bash
  ./scripts/stop_agent_ascend310b.sh
  ```

环境脚本复用板端已验证的 Python 3.9 production 环境作为离线克隆源，但只通过源码目录运行本额外功能，不会把主包安装回该环境。

## 1. 准备独立 Conda 环境

准备以下经过验证的 aarch64 wheel。文件名的 build 后缀可以不同，版本和 CPython ABI 必须一致：

```text
torch-2.0.1-cp39-cp39-*-aarch64.whl
torch_npu-2.0.1-cp39-cp39-*-aarch64.whl
numpy-1.24.4-cp39-cp39-*-aarch64*.whl
onnx-1.14.1-cp39-cp39-*-aarch64*.whl
protobuf-3.20.3-cp39-cp39-*-aarch64.whl
filelock-3.12.2-py3-none-any.whl
```

在仓库根目录执行：

```bash
chmod +x \
  extras/ascend_edge_incremental/bootstrap_env.sh \
  extras/ascend_edge_incremental/run_pipeline.sh

./extras/ascend_edge_incremental/bootstrap_env.sh \
  --wheel-dir /path/to/aarch64-wheels
```

默认行为：

- 从 `/usr/local/miniconda3/envs/agileagent` 离线克隆；
- 新环境写入 `~/agileagent/envs/agileagent_train`；
- production 环境只读，不卸载或替换其中的 PyTorch；
- 目标环境已存在且版本正确时跳过，版本不符时拒绝覆盖；
- 不下载或替换 CANN。

需要自定义路径时使用 `--prefix`、`--base-env` 和 `--conda`。训练入口也支持 `AGILE_EDGE_TRAINING_PYTHON` 与 `AGILE_EDGE_PRODUCTION_PYTHON`。

## 2. 预览完整计划

`plan` 只解析注册表和输出命令，不创建目录、不运行模型：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

/usr/local/miniconda3/envs/agileagent/bin/python \
  -m extras.ascend_edge_incremental.workflow plan \
  --output-root "$HOME/agileagent/edge_incremental_runs/plan-check" \
  --baseline-fps 38.2175 \
  --encoded
```

`--baseline-fps` 必须填写当前未叠加 Adapter 的实测完整推理 FPS。`38.2175` 是当前 release 的 mixed 20 图中位结果；production 变化后应先重测基线，不能沿用旧数值。

## 3. 执行 lock 门禁流水线

输出目录必须不存在，并应放在仓库 production 资产之外：

```bash
RUN_ROOT="$HOME/agileagent/edge_incremental_runs/$(date +%Y%m%d-%H%M%S)"

./extras/ascend_edge_incremental/run_pipeline.sh \
  --output-root "$RUN_ROOT" \
  --baseline-fps 38.2175 \
  --encoded
```

默认参数复现实验配置：5 个 seed、3 个学习率、每候选 80 epoch、4096 条平衡训练行、batch 256。包装脚本会加载现有 CANN 环境并默认使用该板已验证的原始 OPP。首次 torch_npu 图编译期间可能打印 `vendors/customize` root-only 文件的 Python traceback，实测中该警告非致命，约 176 秒后继续。不应为了消除警告而排除整个 `vendors`，否则会丢失 AutoTiling 注册。`--opp-source` 只作为其他板型的显式诊断选项，不再自动启用。

完整执行顺序是：

1. NPU forward/backward/optimizer 探测；
2. 冻结 Increment train/dev 的低阈值三模型候选；
3. 按注册轮次进行多 seed、多学习率 Adapter 训练；
4. 独立冻结 mixed dev 候选并选择 Adapter 强度；
5. 参数冻结后才生成 mixed lock probe，计算 Base mAP50、New-mAP50、KRR、Full-mAP50、P/R/FP 与误激活；
6. 导出无 MatMul ONNX，使用 ATC 编译 OM；
7. 在 ACL 上验证数值一致性、OM 延迟和保守串行叠加 FPS。

任一门禁失败即停止。流水线拒绝 CPU fallback，也拒绝把输出写入 `models/production/`、`models/ascend310b/`、`configs/`、`splits/` 或 `.git/`。

## 4. 增加 890 图二号诊断

需要完整对比时增加：

```bash
./extras/ascend_edge_incremental/run_pipeline.sh \
  --output-root "$RUN_ROOT" \
  --baseline-fps 38.2175 \
  --encoded \
  --include-all-diagnostics
```

`all` 结果只用于观察拟合、误激活和跨数据范围稳定性，不用于训练、dev 选参或比赛 lock 门禁。为了保持输出不可覆盖，同一 `RUN_ROOT` 不能先跑默认流程再追加诊断；应在首次执行时指定该选项，或使用新的输出目录。

## 输出目录

```text
RUN_ROOT/
├── workflow_plan.json
├── workflow_state.json
├── evidence/                   环境探测、输入范围和导出报告
├── probes/                     label-free 冻结候选与临时图像视图
├── training/
│   ├── <round_id>_adapter.pt
│   ├── combined_adapter_bank.pt
│   └── training_report.json
├── calibration/
│   └── adapter_scales.json
├── evaluation/
│   ├── lock/evaluation_report.json
│   ├── all/evaluation_report.json       可选
│   └── adapter_om_benchmark.json
└── export/
    ├── edge_adapter_bank.onnx
    └── edge_adapter_bank.om
```

这些目录包含数据路径、冻结响应和设备产物，应保留在板端工作区，不提交到 Git。

## 2026-08-25 底层独立实验

独立实验使用相同 4+2 注册顺序和当前 Ascend production 候选，选出的安全强度为 patrol_boat `0.0`、armored_vehicle `0.2`。

| mixed lock 指标 | production 基线 | 板端 Adapter | 变化 |
| --- | ---: | ---: | ---: |
| Base mAP50 | `0.816663` | `0.816663` | `+0.000000` |
| New-mAP50 | `0.611461` | `0.649306` | `+0.037845` |
| KRR | `1.000000` | `1.000000` | `+0.000000` |
| Full-mAP50 | `0.722005` | `0.736421` | `+0.014416` |
| 新类 precision@0.63 | `0.729167` | `0.708333` | `-0.020833` |
| 新类 TP / FP@0.63 | `54 / 18` | `55 / 21` | `+1 / +3` |
| 新类误激活图像 | `17/75` | `17/75` | `+0` |

| 890 图诊断 | production 基线 | 板端 Adapter | 变化 |
| --- | ---: | ---: | ---: |
| Base mAP50 | `0.844509` | `0.840010` | `-0.004499` |
| New-mAP50 | `0.684992` | `0.706947` | `+0.021955` |
| KRR | `1.000000` | `1.000000` | `+0.000000` |
| Full-mAP50 | `0.768291` | `0.774954` | `+0.006664` |
| 新类 precision@0.63 | `0.822169` | `0.799413` | `-0.022757` |
| 新类 TP / FP@0.63 | `520 / 104` | `540 / 134` | `+20 / +30` |
| 新类误激活图像 | `108/750` | `122/750` | `+14` |

OM 实测 wall 中位/P95 为 `0.174773 / 0.199086 ms`，最大绝对误差 `0.0005395`。将 wall 中位串行叠加到 `38.2175 FPS` 基线，保守预计为 `37.9639 FPS`。这组底层结果用于建立 Adapter 单体性能基线；下一节记录它接入实际完整图像链路后的实测。

## 2026-08-25 底层实验耗时与资源

| 项目 | 结果 |
| --- | ---: |
| 两轮 5 seed × 3 LR 完整训练搜索 | `556.26 s`（约 `9分16秒`） |
| 必需 NPU 探测 + 训练 + ONNX/OM 导出 | 约 `12分12秒` |
| 再加 890 图完整诊断 | 约 `20分` |
| 首次训练图编译候选 | `482.70 s` |
| 热启动候选中位 | `2.035 s` |
| 训练进程 RSS 峰值 | `1380.75 MB` |
| npu-smi 内存峰值 | `8088 MB` |
| 峰值功耗 / 温度 | `11.9 W / 61°C` |

首次图编译占绝大多数时间，不应把首轮约 8 分钟的停顿误判为卡死。完整流水线还会冻结不同数据范围的候选；具体总耗时受当前服务配置、存储和是否使用 encoded 路径影响，以 `workflow_state.json` 的逐阶段墙钟为准。

## 2026-08-26 完整运行时验收

`board-full-check-v6` 将 Adapter 接入 `WebInferenceEngine` 的冻结 score calibration 之前，并使用隔离演示配置完成真实完整图像链路复测。该次运行从同一条断网命令开始，逐项通过输入审计、NPU 训练、mixed lock、OM 数值、隔离部署和 FPS 门禁。

| mixed lock 指标 | 正式 production | 隔离演示 Adapter | 变化 |
| --- | ---: | ---: | ---: |
| Base mAP50 | `0.816663` | `0.816663` | `+0.000000` |
| New-mAP50 | `0.611461` | `0.624935` | `+0.013474` |
| KRR | `1.000000` | `1.000000` | `+0.000000` |
| Full-mAP50 | `0.722005` | `0.726497` | `+0.004492` |
| 新类误激活图像 | `17/75` | `17/75` | `+0` |

| 运行与导出 | 实测 |
| --- | ---: |
| 两轮 5 seed × 3 LR NPU 训练搜索 | `155.17 秒` |
| ONNX 导出 | `4.82 秒` |
| ATC 编译 OM | `87.67 秒` |
| Adapter OM 最大绝对误差 | `5.96e-08` |
| 完整图像链路三轮 FPS | `39.05 / 38.70 / 37.92` |
| 完整图像链路中位 FPS | `38.6995` |
| 输入审计至隔离部署和 FPS 的热态总耗时 | `1007.07 秒` |

训练审计为 `base_images_used_for_training=0`、`old_raw_image_count=0`；隔离候选状态为 `accepted`，production 修改标记为 `false`。演示配置保留当前满分 production 作为父代，CLI 通过 `AGILE_AGENT_CONFIG=<demo_config>` 显式选择学习后的运行身份。该设计已经完成了此前“独立 release、真实链路、mixed lock、KRR、Full-mAP50、误激活、数值对齐、整链 FPS 和父代保留”的完整闭环。

## 常见问题

- `torch.npu is unavailable`：确认使用独立训练 Python、wheel ABI 为 cp39/aarch64，并已加载与 CANN 7.0.RC1 匹配的环境；不要升级或替换 production CANN。
- `atc is unavailable`：先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，或直接使用 `run_pipeline.sh`。
- production 代际路径不存在：先按 [`ascend-310b-deployment.md`](ascend-310b-deployment.md) 物化当前 release，使板端配置中的绝对路径有效。
- 输出目录已存在：为每次实验使用新的 `RUN_ROOT`。不可覆盖是为了避免把不同轮次、probe 和指标混在一起。
- 流程中断：查看 `workflow_state.json` 和 `workflow_plan.json`。最稳妥的恢复方式是换一个新输出目录完整重跑；也可以人工核对后只执行 plan 中尚未完成的命令。
