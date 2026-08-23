<!-- generated-by: gsd-doc-writer -->
# 快速上手

本指南覆盖当前 strict 4+2 正式流程：x86/CUDA 工作台、YOLO26s Base 训练、patrol_boat → armored_vehicle 两轮类别增量、Scene-SensorNet 系统级校准，以及 Ascend310B v2 release `20260823-4plus2-yolo26-content-gate-v2` 的板端启动。

## 前置条件

### x86/CUDA

| 项目 | 要求 |
| --- | --- |
| 操作系统 | x86-64 Linux 或 WSL |
| Python | `>=3.10,<3.13`，带 `pip`；也可使用 `uv` 创建环境 |
| GPU | 可由 `nvidia-smi` 识别的 NVIDIA GPU |
| 训练初始化 | YOLO26s 通用预训练权重 `.pt` |
| 数据 | 完整 4+2 数据根目录，包含 Base、Increment 和 `splits/strict_4plus2/` |

引导脚本默认准备 PyTorch `2.5.1+cu124` 与 TorchVision `0.20.1+cu124`，并按 [`pyproject.toml`](../pyproject.toml) 安装工作台、推理和开发依赖。训练工具要求每条队列只看到一张物理 GPU。

### Ascend310B v2

| 项目 | 正式 release 要求 |
| --- | --- |
| 设备 | Atlas 200I DK A2 / Ascend310B1 |
| CANN | `7.0.RC1`，环境脚本位于 `/usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Python 环境 | `/usr/local/miniconda3/envs/agileagent` |
| 仓库模型包 | `models/ascend310b/full-score/20260823-4plus2-yolo26-content-gate-v2/` |

## 安装 x86/CUDA 工作台

1. 克隆仓库并进入项目目录：

   ```bash
   git clone https://github.com/Sakauma/AgileAgent.git
   cd AgileAgent
   ```

2. 运行环境引导：

   ```bash
   chmod +x scripts/bootstrap_x86.sh scripts/start_agent.sh
   ./scripts/bootstrap_x86.sh
   ```

   脚本会选择 Python 3.10–3.12、检查 CUDA 与 TorchVision NMS、安装缺失依赖、以 editable 模式登记 `agile-agent`，并把解释器绝对路径写入 `.agent-python`。

3. 已有符合版本要求的解释器时，显式指定它：

   ```bash
   AGILE_AGENT_PYTHON=/path/to/python ./scripts/bootstrap_x86.sh
   ```

## 首次运行

启动 Web 工作台：

```bash
./scripts/start_agent.sh
```

该入口默认使用 `--config auto`。在 `x86_64/AMD64` 上自动加载 CUDA 配置与 `.pt` 模型；在 `aarch64/ARM64` 上自动加载 Ascend 配置与 `.om` 模型，并复用现有 `/usr/local/miniconda3/envs/agileagent` 和 CANN 环境。CLI 与 Web 使用同一个选择结果，不会发生一端使用 PT、另一端使用 OM 的分裂。

浏览器打开 `http://127.0.0.1:8501`，或在另一终端检查服务：

```bash
curl -fsS http://127.0.0.1:8501/api/health
```

无浏览器环境使用终端工作台：

```bash
./scripts/start_agent.sh --cli
```

也可以运行 `agile-agent config validate` 查看 `runtime.architecture`、`backend`、`model_format` 与选择来源；顶层 `--config PATH` 保留为显式覆盖。

单图检测会输出场景概率、六类检测结果、类别 owner 和模型执行轨迹：

```bash
agile-agent detect \
  --source /path/to/image.png \
  --confidence 0.10 \
  --profile incremental-detection
```

## 准备 strict 4+2 数据

训练工具以一个统一数据根目录为输入，目录结构如下：

```text
tiaozhanbei_4plus2_dataset_20260821/
├── datasets_r1_base_train/       # 750 张四类 Base 图像及标签
├── datasets_r2_inc_train/        # 140 张 Increment 图像及原始六类标签
└── splits/strict_4plus2/         # train/dev/lock 与两轮固定清单
```

在仓库根目录的同一 shell 中设置后续命令变量：

```bash
REPO_ROOT="$(pwd)"
AGENT_PYTHON="$(tr -d '\r\n' < .agent-python)"
DATA_ROOT=/path/to/tiaozhanbei_4plus2_dataset_20260821
YOLO26S_PRETRAIN=/path/to/yolo26s.pt
SEEDS=3407,20260821,8675309,42,20260822
TRAIN_GPU=0

BASE_RUN="$REPO_ROOT/runs/detect/base_4plus2_yolo26s"
INCREMENT_RUN="$REPO_ROOT/runs/detect/incremental_4plus2_sequential"
EVAL_ROOT="$REPO_ROOT/runs/evaluation/strict_4plus2_sequential"
SCENE_RUN="$REPO_ROOT/runs/context/scene_sensor_4plus2"

ROUND1=round_01_patrol_boat
ROUND2=round_02_armored_vehicle
```

检查解释器、数据和初始化权重：

```bash
test -x "$AGENT_PYTHON"
test -d "$DATA_ROOT/datasets_r1_base_train"
test -d "$DATA_ROOT/datasets_r2_inc_train"
test -f "$DATA_ROOT/splits/strict_4plus2/manifest.json"
test -f "$YOLO26S_PRETRAIN"
```

复核仓库固定清单，并按类别注册表物化两轮清单：

```bash
"$AGENT_PYTHON" tools/03_split_r2_4plus2.py --verify-only
"$AGENT_PYTHON" tools/11_prepare_incremental_round_splits.py \
  --data-root "$DATA_ROOT"
```

预期轮次计数为：

```text
round_01_patrol_boat: train=56 dev=7 lock=7
round_02_armored_vehicle: train=56 dev=7 lock=7
```

## 训练并选择 Base YOLO26s

正式训练采用 `1280` 输入、最多 `500` epoch、`50` epoch 无改善早停和多随机种子。`--batch 0.90` 表示由 Ultralytics 按 90% 可用显存选择批大小，也可替换为已验证的正整数。

```bash
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  "$AGENT_PYTHON" tools/04_train_base_4plus2.py \
    --data-root "$DATA_ROOT" \
    --model "$YOLO26S_PRETRAIN" \
    --model-tag yolo26s \
    --project "$BASE_RUN" \
    --seeds "$SEEDS" \
    --device 0 \
    --imgsz 1280 \
    --batch 0.90 \
    --epochs 500 \
    --patience 50 \
    --workers 6
```

按 Base dev mAP50 复评并选择权重：

```bash
BASE_DATASET="$BASE_RUN/_control/yolo26s/base_4plus2.yaml"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  "$AGENT_PYTHON" tools/05_select_base_4plus2.py \
    --project "$BASE_RUN" \
    --dataset-yaml "$BASE_DATASET" \
    --tags yolo26s \
    --seeds "$SEEDS" \
    --device 0 \
    --imgsz 1280 \
    --batch 16 \
    --workers 6

BASE_WEIGHT="$BASE_RUN/selection/selected/best_base.pt"
test -f "$BASE_WEIGHT"
```

## 执行两轮类别增量

两轮定义来自 [`configs/incremental_round_registry_4plus2.yaml`](../configs/incremental_round_registry_4plus2.yaml)。每轮训练视图只保留当轮新类并映射到专家局部 ID；Base 和上一轮专家权重在本轮保持冻结。

### Round 1：patrol_boat

训练与 dev 选模：

```bash
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  "$AGENT_PYTHON" tools/06_train_incremental_4plus2.py \
    --data-root "$DATA_ROOT" \
    --round-id "$ROUND1" \
    --model "$YOLO26S_PRETRAIN" \
    --model-tag yolo26s_generic \
    --queue-tag formal_round1 \
    --project "$INCREMENT_RUN" \
    --seeds "$SEEDS" \
    --device 0 \
    --imgsz 1280 \
    --batch 0.90 \
    --epochs 500 \
    --patience 50 \
    --workers 6

ROUND1_DATASET="$INCREMENT_RUN/_control/formal_round1/$ROUND1/incremental_round.yaml"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  "$AGENT_PYTHON" tools/07_select_incremental_4plus2.py \
    --project "$INCREMENT_RUN" \
    --dataset-yaml "$ROUND1_DATASET" \
    --round-id "$ROUND1" \
    --model-tag yolo26s_generic \
    --seeds "$SEEDS" \
    --device 0 \
    --imgsz 1280 \
    --batch 18 \
    --workers 6
```

冻结预测、累计评测并登记 Round 1 candidate：

```bash
ROUND1_WEIGHT="$INCREMENT_RUN/selection/$ROUND1/selected/best_$ROUND1.pt"
ROUND1_SELECTION="$INCREMENT_RUN/selection/$ROUND1/incremental_selection.json"
ROUND1_EVAL="$EVAL_ROOT/$ROUND1"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  "$AGENT_PYTHON" tools/08_evaluate_4plus2.py \
    --data-root "$DATA_ROOT" \
    --round-id "$ROUND1" \
    --base-weight "$BASE_WEIGHT" \
    --specialist-weight "$ROUND1=$ROUND1_WEIGHT" \
    --skip-context-check \
    --output-dir "$ROUND1_EVAL" \
    --device 0 \
    --base-imgsz 1280 \
    --specialist-imgsz 1280 \
    --batch 18

"$AGENT_PYTHON" tools/13_register_incremental_round_candidate.py \
  --round-id "$ROUND1" \
  --selection "$ROUND1_SELECTION" \
  --evaluation "$ROUND1_EVAL/metrics.json"
```

### Round 2：armored_vehicle

训练与 dev 选模：

```bash
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  "$AGENT_PYTHON" tools/06_train_incremental_4plus2.py \
    --data-root "$DATA_ROOT" \
    --round-id "$ROUND2" \
    --model "$YOLO26S_PRETRAIN" \
    --model-tag yolo26s_generic \
    --queue-tag formal_round2 \
    --project "$INCREMENT_RUN" \
    --seeds "$SEEDS" \
    --device 0 \
    --imgsz 1280 \
    --batch 0.90 \
    --epochs 500 \
    --patience 50 \
    --workers 6

ROUND2_DATASET="$INCREMENT_RUN/_control/formal_round2/$ROUND2/incremental_round.yaml"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  "$AGENT_PYTHON" tools/07_select_incremental_4plus2.py \
    --project "$INCREMENT_RUN" \
    --dataset-yaml "$ROUND2_DATASET" \
    --round-id "$ROUND2" \
    --model-tag yolo26s_generic \
    --seeds "$SEEDS" \
    --device 0 \
    --imgsz 1280 \
    --batch 18 \
    --workers 6
```

Round 2 累计评测同时加载两个冻结专家，然后登记最终 candidate：

```bash
ROUND2_WEIGHT="$INCREMENT_RUN/selection/$ROUND2/selected/best_$ROUND2.pt"
ROUND2_SELECTION="$INCREMENT_RUN/selection/$ROUND2/incremental_selection.json"
ROUND2_EVAL="$EVAL_ROOT/$ROUND2"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  "$AGENT_PYTHON" tools/08_evaluate_4plus2.py \
    --data-root "$DATA_ROOT" \
    --round-id "$ROUND2" \
    --base-weight "$BASE_WEIGHT" \
    --specialist-weight "$ROUND1=$ROUND1_WEIGHT" \
    --specialist-weight "$ROUND2=$ROUND2_WEIGHT" \
    --skip-context-check \
    --output-dir "$ROUND2_EVAL" \
    --device 0 \
    --base-imgsz 1280 \
    --specialist-imgsz 1280 \
    --batch 18

"$AGENT_PYTHON" tools/13_register_incremental_round_candidate.py \
  --round-id "$ROUND2" \
  --selection "$ROUND2_SELECTION" \
  --evaluation "$ROUND2_EVAL/metrics.json"
```

汇总并校验完整父子代际：

```bash
ROUND_SUMMARY="$EVAL_ROOT/round_summary"

"$AGENT_PYTHON" tools/12_summarize_incremental_rounds.py \
  --metrics "$ROUND1=$ROUND1_EVAL/metrics.json" \
  --metrics "$ROUND2=$ROUND2_EVAL/metrics.json" \
  --output-dir "$ROUND_SUMMARY"
```

## 训练 Scene-SensorNet 并完成系统级校准

Scene-SensorNet 与门控搜索属于 `system_calibration`。该阶段在两个检测器和两个增量专家均冻结后运行。

先检查数据和 GPU，再训练三个随机种子：

```bash
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  "$AGENT_PYTHON" tools/60_train_scene_sensor.py \
    --config configs/scene_sensor_model_4plus2.yaml \
    --seed 3407 \
    --device 0 \
    --run-dir "$SCENE_RUN/preflight" \
    --data-root "$DATA_ROOT" \
    --check-only

for SCENE_SEED in 3407 20260821 8675309; do
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
    "$AGENT_PYTHON" tools/60_train_scene_sensor.py \
      --config configs/scene_sensor_model_4plus2.yaml \
      --seed "$SCENE_SEED" \
      --device 0 \
      --run-dir "$SCENE_RUN/seed_$SCENE_SEED" \
      --data-root "$DATA_ROOT"
done

"$AGENT_PYTHON" tools/61_select_scene_sensor_4plus2.py \
  --project "$SCENE_RUN" \
  --seeds 3407,20260821,8675309

SCENE_WEIGHT="$SCENE_RUN/selection/selected/scene_sensor_net.pt"
test -f "$SCENE_WEIGHT"
```

使用 mixed dev 生成场景软阈值候选，冻结 `guarded_precision` 运行点后再执行 mixed lock 联合评估：

```bash
CAL_DEV="$EVAL_ROOT/system_calibration_dev"
CAL_LOCK="$EVAL_ROOT/joint_evaluation_lock"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  "$AGENT_PYTHON" tools/09_optimize_scene_aware_4plus2.py dev \
    --data-root "$DATA_ROOT" \
    --round-id "$ROUND2" \
    --evidence-dir "$ROUND2_EVAL" \
    --scene-weight "$SCENE_WEIGHT" \
    --output-dir "$CAL_DEV" \
    --device 0 \
    --batch 256

CANDIDATE="$CAL_DEV/candidates/guarded_precision.json"
test -f "$CANDIDATE"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  "$AGENT_PYTHON" tools/09_optimize_scene_aware_4plus2.py lock \
    --data-root "$DATA_ROOT" \
    --round-id "$ROUND2" \
    --evidence-dir "$ROUND2_EVAL" \
    --scene-weight "$SCENE_WEIGHT" \
    --output-dir "$CAL_LOCK" \
    --candidate "$CANDIDATE" \
    --device 0 \
    --batch 256
```

`lock_result.json` 通过比赛门禁后，将最终两轮模型组合切换到 production：

```bash
"$AGENT_PYTHON" tools/10_promote_scene_aware_4plus2.py \
  --round-id "$ROUND2" \
  --candidate "$CANDIDATE" \
  --lock-result "$CAL_LOCK/lock_result.json" \
  --dev-search "$CAL_DEV/dev_search.json" \
  --dev-report "$CAL_DEV/dev_search.md" \
  --round-evidence "$ROUND_SUMMARY/round_evidence.json"
```

## 物化并启动 Ascend310B v2

在已经配置 CANN 与 `/usr/local/miniconda3/envs/agileagent` 的板端，从仓库根目录执行：

```bash
chmod +x scripts/materialize_ascend310b_full_score_release.sh
./scripts/materialize_ascend310b_full_score_release.sh
```

物化目标固定为：

```text
/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2
```

启动正式服务：

```bash
RELEASE=/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2

AGILE_AGENT_ASCEND_RELEASE="$RELEASE" \
AGILE_AGENT_CONFIG="$RELEASE/configs/agent_pipeline_ascend310b.yaml" \
AGILE_AGENT_ASCEND_PORT=8501 \
  "$RELEASE/src/scripts/start_agent_ascend310b.sh"
```

在另一终端检查健康状态和单图推理：

```bash
curl -fsS http://127.0.0.1:8501/api/health
curl -fsS -F "file=@/path/to/sample.png;type=image/png" \
  http://127.0.0.1:8501/api/detect
```

健康响应应包含：

```json
{
  "status": "ready",
  "backend": "ascend_acl",
  "validated": true,
  "model_layout": "independent_yolo26_e2e_v1",
  "context_mode": "model"
}
```

目标 release 已存在时，执行只读复核：

```bash
./scripts/materialize_ascend310b_full_score_release.sh --verify-existing
```

## 常见设置问题

### `bootstrap_x86.sh` 报告未检测到 `nvidia-smi`

先确认当前 Linux/WSL 会话可见驱动：

```bash
nvidia-smi
```

命令失败时先恢复 WSL GPU 映射或 NVIDIA 驱动，再重新运行引导脚本；正式工作台推理后端要求 GPU ready。

### 训练工具报告可见多张 GPU

`tools/04_train_base_4plus2.py` 和 `tools/06_train_incremental_4plus2.py` 要求每条队列只看到一张物理卡。把物理卡号写入 `CUDA_VISIBLE_DEVICES`，工具内部仍使用逻辑设备 `0`：

```bash
CUDA_VISIBLE_DEVICES=2 "$AGENT_PYTHON" -c \
  'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))'
```

### 提示划分、图像或标签不存在

确认 `DATA_ROOT` 指向统一数据包根目录，而不是单独的 Base 或 Increment 子目录；随后重新执行：

```bash
"$AGENT_PYTHON" tools/11_prepare_incremental_round_splits.py \
  --data-root "$DATA_ROOT"
```

### 提示输出目录非空

训练、评测、校准和板端评分工具保留现有证据并拒绝静默覆盖。为新运行设置新的 `BASE_RUN`、`INCREMENT_RUN`、`EVAL_ROOT` 或 release candidate 目录。

### Ascend 物化提示目标 release 已存在

使用 `--verify-existing` 验证现有正式目录。服务端口已被占用时，先读取 [`ascend-310b-deployment.md`](ascend-310b-deployment.md) 中的服务与路由拓扑，再选择正式实例端口。

## 下一步

- 架构与三模型数据流：[`ARCHITECTURE.md`](ARCHITECTURE.md)
- 配置字段与环境覆盖：[`CONFIGURATION.md`](CONFIGURATION.md)
- 本地开发流程：[`DEVELOPMENT.md`](DEVELOPMENT.md)
- 测试与发布验证：[`TESTING.md`](TESTING.md)
- 增量学习阶段契约：[`compliant-incremental-learning.md`](compliant-incremental-learning.md)
- Ascend310B v2 方法：[`ascend-310b-full-score-method.md`](ascend-310b-full-score-method.md)
- 板端部署与路由：[`ascend-310b-deployment.md`](ascend-310b-deployment.md)
