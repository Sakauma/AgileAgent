# Ascend 310B 满分方法与复现手册

## 1. 当前结论

本手册是 Ascend310B1 比赛链路的活动入口。P0–P11 的逐阶段消融已经归档；更换数据集后应复用这里记录的结构、冻结约束、阈值搜索和四项评分门禁，而不是把 `0.05/0.30` 当成新数据集的固定答案。

2026-08-16 参考候选在提交 `bc8def938523bb5856d96aa90f397468f208a4b6` 上得到：

| 评分项 | 结果 | 满分门槛 |
| --- | ---: | ---: |
| Base mAP50 | `0.8049006528` | `≥0.80` |
| New-mAP50 | `0.6050327631` | `≥0.60` |
| KRR | `1.0000000000` | `≥0.95` |
| 20 图 batch | 首轮 `30.066 FPS`，复轮 `30.080 FPS` | `≥30 FPS` |

这是“候选达到满分评分门槛”，不是“正式服务已经切换”。正式 `127.0.0.1:8501` 仍保留原三 OM 发布链路；训练、构建和评分候选只能使用 `127.0.0.1:8502`。

机器可读的固定方法位于 [`configs/ascend310b/full_score_method.yaml`](../configs/ascend310b/full_score_method.yaml)，轻量证据位于 [`2026-08-16-full-score-evidence.json`](archive/ascend310b/2026-08-16-full-score-evidence.json)。

参考链路的核心 SHA256 为：实际导出的 `last.pt` `6d1e7098015134615b32a7cedeeab9352bb83adf812f227b85044a3e64da9c6a`、同轮 `best.pt` `fb44299eb6ad070e5330e88db975e1bebc1389cba229cd49466406dcf475b15d`、胜出 ONNX `5d6651a25cdc227a6feaf3135d754f1d132f0740117156f9b6d0651c32104c5e`、板端 OM `3dd053e041c36225059cf6624eefebe5945ba6b8ca5bc0ca9d914448c4a54c89`、training report `ce094cf5493ff76da92a62638292334126f4d39057a03499a52fcfd3ed1552dc`、export manifest `a97fae923f997beeea9b22df920cb4be058da196039a8317d2f36f18b9e4e5d1`。

## 2. 保留的满分链路

### 模型结构

- 复用正式 Base 的 backbone、neck/FPN 和 old Detect head，并冻结权重、BN 统计和 EMA。
- 在共享三尺度特征上训练 residual `1×1` adapter 和 new Detect head；训练输入只能来自新增类数据。
- 单个 `shared_backbone_dual_head_v1` OM 使用 Base 的 `896×736` AIPP 输入，一次执行返回 old `[1,7,13524]` 和 new `[1,5,13524]` 两个 raw head。
- Agent 层继续把 old head 记为 `frozen_base_model`，new head 记为 `incremental_model`，保持原融合、审计和响应 schema。
- `fixed_neutral_v1` 不执行 Scene/Sensor 推理：Sensor 概率固定为 `0.5/0.5`，Scene 四类概率固定为各 `0.25`，`neutral_context_score=0.5` 仅作为路由回退值。原 context OM 仍会加载并登记在 manifest 中，以便显式回滚，但正常候选路径不调用其前向推理。

### 运行时快路径

- CANN 固定 `7.0.RC1`，`mixed_float16`，不使用 INT8、降分辨率或跨请求流水线。
- batch 图片直接走 DVPP encoded 预处理。
- 使用 pageable memory 和 threaded execution。
- 保留有界 multipart 解析、neutral batch/schedule elision 和 image-copy elision。
- 正式计分关闭详细 event 插桩，避免测量扰动。

当前剩余瓶颈是 20 图 batch 中约 `656.3–657.8 ms` 的 Engine；解析约 `6.2–7.1 ms`，cache 约 `0.4–0.8 ms`。首轮距离 30 FPS 边界只有约 `0.22%`，因此新数据集必须重新搜索阈值并复测，不能只验证服务可启动。

## 3. 新数据集复现流程

### 3.1 数据和训练前置

1. 生成新的 base、increment、mixed/lock 划分和增量数据审计 manifest。
2. 在查看 lock 标签前固定划分、训练参数和候选编号。
3. 审计必须满足：旧类原始图像、旧类标签、旧类 feature cache 均为 `0`，原始数据未修改。
4. 若全局类别编号变化，只修改方法配置中的 old/new `class_map`；导出、构建、候选生成和 score gate 都从该配置继承映射，owner 语义不得改变。

本机训练和导出只使用 WSL 仓库既有 `.venv`，禁止安装依赖或下载 CPU PyTorch：

```bash
.venv/bin/python tools/107_train_shared_dual_head.py \
  --base-weight BASE.pt \
  --specialist-weight SPECIALIST.pt \
  --method-config configs/ascend310b/full_score_method.yaml \
  --data INCREMENT_DATASET.yaml \
  --dataset-manifest DATASET_AUDIT.json \
  --artifact-dir artifacts/ascend310b/TRAIN_ID \
  --run-root runs/detect \
  --run-name TRAIN_ID \
  --device 0
```

训练工具从方法 YAML 读取 residual adapter、输入尺寸、优化器、epoch/batch/seed/增强和 checkpoint 策略；同名 CLI 参数仅可作为相等断言，不能静默覆盖方法配置。报告同时登记 `best.pt` 和 `last.pt` 的 SHA256、New-mAP50 与共享参数漂移，两者都必须 `shared_max_drift=0`。训练阶段的 New-mAP50 可以用于提前停止明显失败项，但正式晋级仍以冻结后的 Agent 评分为准。

当前参考满分链路导出的是同一训练 run 的 `last.pt`，不是 training report 中旧格式唯一登记的 `best.pt`。新训练报告采用 schema v2 并显式授权 best/last；构建脚本只接受被报告授权且与 export manifest 一致的 checkpoint。旧 schema v1 只对本文件上方固定的 training report、export manifest 和 `last.pt` 哈希组合保留兼容，不形成通用旁路。

### 3.2 导出和板端构建

```bash
.venv/bin/python tools/108_export_ascend_dual_head.py \
  --base-weight BASE.pt \
  --new-head-weight EXPORT_CHECKPOINT.pt \
  --output-dir artifacts/ascend310b/EXPORT_ID \
  --device cuda:0
```

将 ONNX、训练权重和 export manifest 以 SHA256 校验方式同步到板端，然后在 CANN `7.0.RC1` 环境执行：

```bash
./scripts/build_ascend_dual_head_om.sh \
  shared_backbone_dual_head.onnx \
  EXPORT_CHECKPOINT.pt \
  training-report.json \
  export-manifest.json \
  FORMAL_CONTEXT_BUILD_MANIFEST.json \
  artifacts/ascend310b/BUILD_ID
```

构建脚本从 `full_score_method.yaml` 读取 SoC、精度、输入 shape、logical head、输出 shape 和 AIPP 契约，并从 CANN 版本文件或 `atc --version` 确认实际环境为 `7.0.RC1`；无法确认时拒绝生成 manifest。新 manifest 记录 dual/context、ATC 命令、Git SHA、方法配置和所有资产 SHA256。输出目录非空时拒绝覆盖。

### 3.3 阈值候选和评分

当前 `old=0.05`、`new=0.30` 只是搜索种子。新数据集按以下顺序执行，避免直接跑完整笛卡尔积：

1. old 固定 `0.05`，new 测试 `0.20/0.25/0.30/0.35/0.40`。
2. 第一阶段候选按最终确定性规则（最小精度余量、FPS 波动、中位 FPS）排名，固定胜出的 new 后，old 测试 `0.03/0.05/0.10/0.20/0.30`。
3. 对两个最优 old/new 组合做交叉复核。
4. 精度不满分的组合不再浪费板端 batch 测量时间。

每个组合先生成独立候选配置：

```bash
.venv/bin/python tools/109_materialize_ascend_full_score_candidate.py \
  --dual-om shared_backbone_dual_head.om \
  --context-om scene_sensor_net.om \
  --build-manifest build-manifest.json \
  --old-threshold 0.05 --new-threshold 0.30 \
  --output artifacts/ascend310b/CANDIDATE_ID/candidate.yaml
```

生成器会强制端口 `8502`、核对 OM/manifest SHA256，并拒绝 logical head 结构与构建清单不一致。old/new 阈值是 Host 运行时参数，不属于 OM 身份，因此同一个 OM/build manifest 可用于完整阈值搜索。随后执行：

```bash
./scripts/run_ascend310b_score_gate.sh \
  artifacts/ascend310b/CANDIDATE_ID/candidate.yaml \
  MIXED_IMAGES \
  MIXED_SPLIT.txt \
  BASE_SPLIT.txt \
  artifacts/ascend310b/CANDIDATE_ID/score-gate
```

脚本检查 `8501 ready`，并以受控候选环境变量授权预测冻结和精确的 `8502` 进程，按“无标签预测冻结 → 读取标签评分 → 30 次预热 → 三轮 20 图 batch”执行；它不运行板端 Web pytest，也不执行单请求诊断。输出目录和预测文件均拒绝覆盖。

评分图片有硬输入契约：仅读取 `MIXED_IMAGES` 根目录的 `*.png`，每张必须为 `640×512`、8-bit RGB（PNG color type 2）或 RGBA（color type 6），且文件 stem 唯一。新数据集需先完成该转换；不符合时 score gate 在加载候选前停止。

### 3.4 自动选优

为每个候选建立索引：

```json
{
  "schema_version": 1,
  "candidates": [
    {
      "id": "old005-new030",
      "score": "old005-new030/score.json",
      "benchmark": "old005-new030/benchmark.json",
      "repeat_benchmarks": [],
      "prerequisites": {
        "incremental_data_isolation": true,
        "asset_hashes_verified": true
      }
    }
  ]
}
```

```bash
.venv/bin/python tools/110_select_ascend_full_score_candidate.py \
  --candidates candidates.json \
  --output selection.json
```

选择器只用 Base mAP50、New-mAP50、KRR 和主 benchmark 的 20 图中位 FPS 判定满分。多个满分候选依次比较最小精度余量、batch FPS 波动和中位 FPS。若没有候选达到 30 FPS，最高 FPS 的精度通过项只标记为 `intermediate_only`。

## 4. 验收边界

硬评分项只有：

- Base mAP50 `≥0.80`；
- New-mAP50 `≥0.60`；
- KRR `≥0.95`；
- 三轮 20 图 batch 的中位 FPS `≥30`。

逐框/业务 JSON 差异、lock precision、误激活率、单请求均值/P95/P99 和 Scene/Sensor accuracy 必须留在报告中，但只能产生 warning，不能否决四项满分候选。

预测必须先冻结再打开标签；数据隔离、Base 零漂移和资产 SHA256 必须通过。这些检查保证结果真实可复现，不增加新的精度或性能阈值。

## 5. 回滚和正式切换

- score gate 的 trap 只停止其启动且命令行明确包含 `--port 8502` 的进程。
- `8501` 在候选开始和结束时都必须返回 `ready`；否则本轮无效。
- 当前 `configs/agent_pipeline_ascend310b.yaml` 和正式三 OM 继续作为回滚链路。
- 满分候选仍保持 `validated: false`。正式切换必须另行生成发布验证摘要并执行 release 校验，不在本手册的整理范围内。

历史消融和板端执行记录见 [`docs/archive/ascend310b/`](archive/ascend310b/)。本地已有的胜出 ONNX、training/export manifest、P10 中间 candidate/build manifest、冻结预测和评分摘要保存在被忽略的 `artifacts/archive/ascend310b/`；误归到 rejected 的两组必要输入已迁到 `2026-08-16-full-score/method-inputs/`，并由本地 archive manifest 标记为 `required_by_full_score`。胜出 OM、对应的 last checkpoint、与轻量证据哈希完全对应的最终 candidate/build manifest 以及两份 benchmark 原始报告没有同步回本机，本仓库只保留其 SHA256 和执行记录，不宣称这些板端独有资产已完成本地归档。
