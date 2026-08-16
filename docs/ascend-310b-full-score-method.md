# Ascend 310B 满分方法与复现手册

## 1. 当前结论

本手册是 Ascend310B1 比赛链路的活动入口。P0–P11 的逐阶段消融已经归档；更换数据集后应复用这里记录的结构、冻结约束、阈值搜索和四项评分门禁，而不是把 `0.05/0.30` 当成新数据集的固定答案。

2026-08-16 参考候选在运行时提交 `bc8def938523bb5856d96aa90f397468f208a4b6` 上得到：

| 评分项 | 结果 | 满分门槛 |
| --- | ---: | ---: |
| Base mAP50 | `0.8049006528` | `≥0.80` |
| New-mAP50 | `0.6050327631` | `≥0.60` |
| KRR | `1.0000000000` | `≥0.95` |
| 20 图 batch | 首轮 `30.066 FPS`，复轮 `30.080 FPS` | `≥30 FPS` |

该候选随后由发布工具提交 `1493b04161f6fbe052636a838a6baabcf6d9b9b8` 物化为正式 release，并于 2026-08-16 原子提升。公共 `127.0.0.1:8501` 当前路由到内部 `18501` 的共享双头主实例；原三 OM 监听器继续保留为即时回滚，训练、构建和后续评分候选仍只使用 `127.0.0.1:8502`。

发布后经公共 `8501` 执行 `30 + 3×20`，三轮为 `30.234/30.243/30.294 FPS`、中位 `30.243 FPS`。正式 release 为 `/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04`；配置、release manifest、validation summary 和发布后 benchmark SHA256 分别为 `39f6472094b3e7f61950a903a0ff914d1e620c557d9b9b747151fd9a502be490`、`ffca93c54aa600a268acc31cdee82e14a040f6313427a180c3597e07db5fc2dd`、`62234e2aba8921c07b8c8e0d66c87f912ffba8b00d8b43245a908524c3a56891`、`bb011d96b62f627d36388f4237017570afd4195e6327522162ab6a0fab15b4e5`。

机器可读的固定方法位于 [`configs/ascend310b/full_score_method.yaml`](../configs/ascend310b/full_score_method.yaml)，轻量证据位于 [`2026-08-16-full-score-evidence.json`](archive/ascend310b/2026-08-16-full-score-evidence.json)。可直接部署的完整正式模型包位于 [`models/ascend310b/full-score/20260816-full-score-1493b04/`](../models/ascend310b/full-score/20260816-full-score-1493b04/README.md)。

参考链路的核心 SHA256 为：实际导出的 `last.pt` `6d1e7098015134615b32a7cedeeab9352bb83adf812f227b85044a3e64da9c6a`、同轮 `best.pt` `fb44299eb6ad070e5330e88db975e1bebc1389cba229cd49466406dcf475b15d`、胜出 ONNX `5d6651a25cdc227a6feaf3135d754f1d132f0740117156f9b6d0651c32104c5e`、板端 OM `3dd053e041c36225059cf6624eefebe5945ba6b8ca5bc0ca9d914448c4a54c89`、training report `ce094cf5493ff76da92a62638292334126f4d39057a03499a52fcfd3ed1552dc`、export manifest `a97fae923f997beeea9b22df920cb4be058da196039a8317d2f36f18b9e4e5d1`。

## 2. 当前正式 release 的零训练复现

默认 310B 已安装 CANN `7.0.RC1` 和 `/usr/local/miniconda3/envs/agileagent`。克隆仓库后执行：

```bash
chmod +x scripts/materialize_ascend310b_full_score_release.sh
./scripts/materialize_ascend310b_full_score_release.sh

RELEASE=/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04
AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
curl -fsS http://127.0.0.1:8501/api/health
```

物化器校验包内全部 26 项 SHA256，复制已构建 OM 及运行源码，并执行正式 release 验证；它不训练、不导出、不运行 ATC、不安装依赖，也不操作端口。新板可以直接监听 `8501`；若板上已有旧三 OM 回滚 release，则使用 [`ascend-310b-deployment.md`](ascend-310b-deployment.md) 中的双实例拓扑，让主实例监听 `18501` 并保留旧 listener。两者都使用同一正式模型身份。

仓库不包含受授权约束的竞赛图像和标签。复现边界如下：

| 输入 | 可以复现的结果 |
| --- | --- |
| 仅仓库 | 模型/证据哈希、release 验证、服务 health、原始 score/benchmark 报告 |
| 20 张契约 PNG | 重新测量三轮 20 图 batch FPS |
| 合法取得的 89 图和标签 | 重新冻结预测并计算 Base/New/KRR |
| 同版 89 图标签 | 对包内冻结预测直接重新评分，无需训练或重新推理 |

## 3. 保留的满分链路

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

当前剩余瓶颈仍是 20 图 batch 的 Engine；候选阶段约 `656.3–657.8 ms`，发布后约 `651.7–653.2 ms`，而解析仅约 `6.8–7.2 ms`、cache 约 `0.4 ms`。发布后中位相对 30 FPS 约有 `0.81%` 余量，仍不宽裕，因此新数据集必须重新搜索阈值并复测，不能只验证服务可启动。

## 4. 新数据集训练与复现流程

### 4.1 数据和训练前置

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

### 4.2 导出和板端构建

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

### 4.3 阈值候选和评分

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

### 4.4 自动选优

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

## 5. 验收边界

硬评分项只有：

- Base mAP50 `≥0.80`；
- New-mAP50 `≥0.60`；
- KRR `≥0.95`；
- 三轮 20 图 batch 的中位 FPS `≥30`。

逐框/业务 JSON 差异、lock precision、误激活率、单请求均值/P95/P99 和 Scene/Sensor accuracy 必须留在报告中，但只能产生 warning，不能否决四项满分候选。

预测必须先冻结再打开标签；数据隔离、Base 零漂移和资产 SHA256 必须通过。这些检查保证结果真实可复现，不增加新的精度或性能阈值。

## 6. 回滚和正式切换

- score gate 的 trap 只停止其启动且命令行明确包含 `--port 8502` 的进程。
- `8501` 在候选开始和结束时都必须返回 `ready`；候选不得停止主线或回滚 listener。
- `tools/111_promote_ascend_full_score_release.py` 只接受 `8502`、`validation_candidate: true`、`validated: false` 的胜出配置，并核对 score schema v2、benchmark schema v5、训练隔离、Base 零漂移和全部资产哈希。
- 工具把所有必要资产复制到不可变 release，生成 `validated: true` 的正式配置和 validation summary；不得手工翻转验证状态。
- `scripts/install_ascend310b_primary_services.sh` 让满分主实例监听内部 `18501`，原三 OM service 继续监听 `8501`，随后以带固定 comment 的精确 iptables loopback 规则原子切换新连接。
- `scripts/manage_ascend310b_primary_route.sh remove 18501` 删除该唯一规则并立即恢复三 OM；`apply 18501` 仅在满分主实例健康且布局正确时重新提升。
- 三个 systemd unit 分别管理主实例、回滚 listener 和持久路由；`8502` 从不写入正式路由。

历史消融和板端执行记录见 [`docs/archive/ascend310b/`](archive/ascend310b/)。正式胜出 OM、实际导出的 `last.pt`、ONNX、AIPP、ATC 日志、training/export/build manifest、正式配置、冻结预测及原始 score/benchmark 报告现已统一版本化在 [`models/ascend310b/full-score/20260816-full-score-1493b04/`](../models/ascend310b/full-score/20260816-full-score-1493b04/README.md)，并由包内 `SHA256SUMS` 保护。P7/P10 失败或中间实验仍保存在本地 ignored archive，不属于部署当前满分 release 的必要资产。
