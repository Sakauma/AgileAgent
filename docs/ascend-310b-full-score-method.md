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

## 4. 更换数据集后的完整训练、构建与评分流程

### 4.1 先判断能否直接使用现有 3+1 工具

现有活动工具不是任意数据集转换器。只有同时满足下列条件时，才能原样执行本节 4.2–4.7：

- 原始数据平铺在仓库根目录的 `datasets_r1_base_train/`，每张 `640×512`、8-bit RGB/RGBA PNG 有同 stem 的 YOLO TXT，另有 `classes.txt`；
- 文件名恰好为五段 `sensor_round_base_scene_id`，例如 `ir_r1_base_air_000001.png`；sensor 只能是 `ir/sar`，scene 只能是 `air/forest/sea/urban`；
- 总图片数恰好为 `750`，类别恰好为四类，当前全局顺序仍为 `soldier/small_aircraft/warship/tank`；
- 只模拟一轮“三个旧类 + 一个新类”，新增类训练/验证图不能与旧类共现。

这些限制分别来自 `fair_agent/dataset_utils.py`、`tools/02_split_dataset.py`、`fair_agent/modules/strict_incremental.py` 和 `fair_agent/modules/incremental_experiment.py`，不是文档约定。只更换同协议图像时走“兼容路径”；图片数量、命名、sensor/scene、类别名称、类别数量、新增类数量或增量轮数变化时，先执行 4.8 的代码迁移，不能只改 `class_map` 后继续训练。

### 4.2 从兼容原始数据生成 base/increment/mixed/lock

以下命令必须在 WSL 的仓库根目录运行，只使用仓库现有 `.venv`：

```bash
cd "/mnt/d/Ajax Mao/研二/近期工作/研二下/tiaozhanbei/AgileAgent"
PY=.venv/bin/python
test -x "$PY"
test -f datasets_r1_base_train/classes.txt

"$PY" tools/00_check_dataset.py
"$PY" tools/01_build_metadata.py

DATA_ID=newdata-YYYYMMDD
WORK="artifacts/ascend310b/$DATA_ID"
SPLIT_ROOT="$WORK/splits"
mkdir -p "$WORK"
"$PY" tools/02_split_dataset.py \
  --increment-class warship \
  --output-dir "$SPLIT_ROOT"
```

`00` 生成 `reports/data_audit_report.md` 和 `reports/data_audit_summary.json`，`01` 生成 `reports/metadata.csv`；`02` 的目标必须不存在或为空，它生成：

| 产物 | 作用 |
| --- | --- |
| `pool_train.txt` / `pool_dev.txt` | 后续再次按 owner 物化 Base 和 Increment 数据视图 |
| `mixed_test.txt` | 固定 mixed/lock 清单；训练前只允许读取图像清单，不允许读取其标签做调参 |
| `strict_3plus1/base_train.txt` / `base_dev.txt` | Base 训练/开发清单证据 |
| `strict_3plus1/increment_train.txt` / `increment_dev.txt` | Specialist 训练/开发清单证据 |
| `strict_3plus1/base_test.txt` | Base mAP50 的评分子集 |
| 两级 `manifest.json` | 源池分配和严格 3+1 协议审计 |

不要覆盖仓库活动配置。下面从模板生成本轮专用配置，并把实际 split 数量写回 `expected_incremental_counts`：

```bash
RUN_ID="$DATA_ID-strict"
PROTOCOL=warship-incremental
RUN_CONFIG="$WORK/strict-3plus1.yaml"

"$PY" - "$SPLIT_ROOT" "$WORK" "$RUN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import yaml

split_root = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
config = yaml.safe_load(
    Path("configs/strict_class_incremental_3plus1.yaml").read_text(encoding="utf-8")
)
strict = json.loads(
    (split_root / "strict_3plus1" / "manifest.json").read_text(encoding="utf-8")
)
config["paths"]["source_splits"] = {
    "train": str(split_root / "pool_train.txt"),
    "val": str(split_root / "pool_dev.txt"),
    "lock": str(split_root / "mixed_test.txt"),
}
config["paths"]["base_test_split"] = str(
    split_root / "strict_3plus1" / "base_test.txt"
)
config["paths"]["dataset_root"] = str(work / "protocol-data")
config["paths"]["run_root"] = str(work / "strict-runs")
config["paths"]["report_root"] = str(work / "strict-reports")
config["paths"]["freeze_root"] = str(work / "profiles")
config["protocols"][0]["expected_incremental_counts"] = {
    "train": strict["counts"]["increment_train"],
    "val": strict["counts"]["increment_dev"],
}
config["protocols"][0]["expected_base_test_count"] = strict["counts"]["base_test"]
output.write_text(
    yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
)
PY

"$PY" tools/70_run_strict_3plus1.py \
  --config "$RUN_CONFIG" --run-id "$RUN_ID" --check-only
"$PY" tools/70_run_strict_3plus1.py \
  --config "$RUN_CONFIG" --run-id "$RUN_ID"
```

`tools/70` 要求 WSL 的 NVIDIA GPU 可用。它先调用 `build_protocol_dataset()` 创建隔离数据视图和初始审计，再训练并冻结预测，最后才调用 `materialize_lock_data()` 物化 lock 标签。当前模板中的真实初始化关系是：

- `BASE.pt`：从 `models/pretrained/yolo11s.pt` 初始化、在 Base 数据上微调后的 `best.pt`；
- `SPECIALIST.pt`：独立从 `models/pretrained/yolo11s.pt` 初始化，只在新增类数据上训练后的 `best.pt`；它不是从 `BASE.pt` 初始化；
- shared dual-head 初始化：`tools/107` 把 `BASE.pt` 的 backbone/neck 复制进混合 checkpoint，同时保留 `SPECIALIST.pt` 的 Detect head，然后只训练 residual adapter 和 new head。

本轮实际输入路径不是字面量 `BASE.pt/SPECIALIST.pt/DATASET_AUDIT.json`，而是：

```bash
DATASET_AUDIT="$WORK/protocol-data/$RUN_ID/$PROTOCOL/manifest.json"
INCREMENT_DATASET="$WORK/protocol-data/$RUN_ID/$PROTOCOL/incremental/dataset.yaml"
BASE_PT="$WORK/strict-runs/$RUN_ID/$PROTOCOL/base/weights/best.pt"
SPECIALIST_PT="$WORK/strict-runs/$RUN_ID/$PROTOCOL/specialist/weights/best.pt"

test -f "$DATASET_AUDIT"
test -f "$INCREMENT_DATASET"
test -f "$BASE_PT"
test -f "$SPECIALIST_PT"
```

### 4.3 `DATASET_AUDIT.json` 的来源和完整 schema

`DATASET_AUDIT.json` 是文档中的角色名；真实文件就是上面的 `manifest.json`，不应手工拼一个四字段文件冒充。schema v1 的全部顶层字段如下，JSON 中整数 map 的键会序列化为字符串；为保持示例可读，stem/hash/path 数组用空数组表示类型，实际生成器会写入完整逐项值：

```json
{
  "schema_version": 1,
  "protocol": "warship-incremental",
  "incremental_mode": "class_incremental",
  "learning_data_scope": "incremental_dataset_only",
  "base_classes": ["soldier", "small_aircraft", "tank"],
  "new_class": "warship",
  "new_global_id": 2,
  "base_local_to_global": {"0": 0, "1": 1, "2": 3},
  "specialist_local_to_global": {"0": 2},
  "source_split_sha256": {"train": "...", "val": "...", "lock": "..."},
  "source_split_stems": {"train": [], "val": [], "lock": []},
  "source_split_intersections": {"train_val": [], "train_lock": [], "val_lock": []},
  "counts": {
    "base": {"train": 441, "val": 70, "test": 89},
    "incremental": {"train": 132, "val": 18, "test": 89}
  },
  "source_stems": {
    "base": {"train": [], "val": [], "test": []},
    "incremental": {"train": [], "val": [], "test": []}
  },
  "intersections": {
    "base_incremental_train": [],
    "base_incremental_val": [],
    "incremental_train_val": []
  },
  "base_nc": 3,
  "specialist_nc": 1,
  "student_nc": null,
  "unified_student_enabled": false,
  "old_raw_stems": [],
  "old_raw_content_hashes": [],
  "old_raw_label_hashes": [],
  "old_raw_image_paths": [],
  "old_raw_label_paths": [],
  "old_raw_image_count": 0,
  "old_raw_label_count": 0,
  "old_feature_cache_count": 0,
  "feature_cache_files": [],
  "original_data_modified": false,
  "lock_materialized_after_freeze": true,
  "base_dataset": ".../base/dataset.yaml",
  "incremental_dataset": ".../incremental/dataset.yaml",
  "student_dataset": null
}
```

生成器同时用 stem、图像 SHA256 和标签 SHA256 检查旧数据泄漏，并扫描 `.cache/.npy/.npz/.pt` feature cache。`tools/107` 当前机器强制的最小隔离门禁是 `old_raw_image_count=0`、`old_raw_label_count=0`、`old_feature_cache_count=0`、`original_data_modified=false`；其余字段用于重建来源、类别映射、集合互斥和 lock 解封时序，仍应完整保留。

### 4.4 训练 residual adapter/new head 并导出 ONNX

```bash
TRAIN_ID="$DATA_ID-dual-head"
TRAIN_ARTIFACT="$WORK/shared-training"
TRAIN_RUN_ROOT="$WORK/shared-runs"

"$PY" tools/107_train_shared_dual_head.py \
  --base-weight "$BASE_PT" \
  --specialist-weight "$SPECIALIST_PT" \
  --method-config configs/ascend310b/full_score_method.yaml \
  --data "$INCREMENT_DATASET" \
  --dataset-manifest "$DATASET_AUDIT" \
  --artifact-dir "$TRAIN_ARTIFACT" \
  --run-root "$TRAIN_RUN_ROOT" \
  --run-name "$TRAIN_ID" \
  --device 0

EXPORT_CHECKPOINT="$TRAIN_RUN_ROOT/$TRAIN_ID/weights/last.pt"
EXPORT_DIR="$WORK/export"
"$PY" tools/108_export_ascend_dual_head.py \
  --base-weight "$BASE_PT" \
  --new-head-weight "$EXPORT_CHECKPOINT" \
  --method-config configs/ascend310b/full_score_method.yaml \
  --output-dir "$EXPORT_DIR" \
  --device cuda:0
```

训练参数只能来自 `full_score_method.yaml`；对应 CLI 参数即使提供，也只能与配置相等。schema v2 training report 同时登记 `best.pt/last.pt`、New-mAP50 和共享参数漂移，两份 checkpoint 都必须 `shared_max_drift=0`。当前固定方法选择 `last.pt` 导出，不能看 lock 结果后在 best/last 之间临时挑选。构建脚本只接受 training report 已授权且与 export manifest 哈希一致的 checkpoint。

### 4.5 从 WSL 同步到板端并构建 OM

先在 WSL 建立只包含四个构建输入的传输目录和校验清单：

```bash
SYNC_DIR="$WORK/board-sync"
mkdir -p "$SYNC_DIR"
cp "$EXPORT_DIR/shared_backbone_dual_head.onnx" "$SYNC_DIR/"
cp "$EXPORT_CHECKPOINT" "$SYNC_DIR/export-checkpoint.pt"
cp "$TRAIN_ARTIFACT/training-report.json" "$SYNC_DIR/"
cp "$EXPORT_DIR/export-manifest.json" "$SYNC_DIR/"
(cd "$SYNC_DIR" && sha256sum \
  shared_backbone_dual_head.onnx export-checkpoint.pt \
  training-report.json export-manifest.json > SHA256SUMS)

BOARD=HwHiAiUser@192.168.137.100
REMOTE_INCOMING="/home/HwHiAiUser/agileagent/candidates/$DATA_ID/incoming"
ssh "$BOARD" "mkdir -p '$REMOTE_INCOMING'"
scp "$SYNC_DIR"/* "$BOARD:$REMOTE_INCOMING/"
ssh "$BOARD" "cd '$REMOTE_INCOMING' && sha256sum -c SHA256SUMS"
```

`FORMAL_CONTEXT_BUILD_MANIFEST.json` 不是新训练生成的文件，也不是 WSL 侧 export manifest。它必须是板端已物化正式 release 的：

```text
/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04/provenance/release-build-manifest.json
```

该清单的 `artifacts` 中恰好有一个 `role: context`，并登记 context weight/ONNX/AIPP/OM/ATC log 的板端绝对路径与 SHA256；构建脚本会逐项复核。仓库副本位于 `models/ascend310b/full-score/20260816-full-score-1493b04/provenance/release-build-manifest.json`，但其中同样保存板端路径，因此实际构建应使用已物化 release 内的文件。

登录板端后，在已同步到同一 `main` HEAD 的仓库中构建：

```bash
cd /home/HwHiAiUser/agileagent/repo
git fetch origin main
git merge --ff-only origin/main

DATA_ID=newdata-YYYYMMDD
REMOTE_INCOMING="/home/HwHiAiUser/agileagent/candidates/$DATA_ID/incoming"
REMOTE_BUILD="/home/HwHiAiUser/agileagent/candidates/$DATA_ID/build"
FORMAL_CONTEXT_BUILD_MANIFEST="/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04/provenance/release-build-manifest.json"
AGILE_AGENT_ASCEND_PYTHON=/usr/local/miniconda3/envs/agileagent/bin/python \
./scripts/build_ascend_dual_head_om.sh \
  "$REMOTE_INCOMING/shared_backbone_dual_head.onnx" \
  "$REMOTE_INCOMING/export-checkpoint.pt" \
  "$REMOTE_INCOMING/training-report.json" \
  "$REMOTE_INCOMING/export-manifest.json" \
  "$FORMAL_CONTEXT_BUILD_MANIFEST" \
  "$REMOTE_BUILD"
```

输出为 `$REMOTE_BUILD/shared_backbone_dual_head.om` 和 `$REMOTE_BUILD/build-manifest.json`。脚本固定 CANN `7.0.RC1`、`mixed_float16`、`[1,3,736,896]` 输入、AIPP 和双输出契约；无法确认 CANN 版本、输入哈希、checkpoint 授权、context 资产或 ATC 成功标志时都会停止。

### 4.6 score gate 前恢复正式 `8501 ready`

`run_ascend310b_score_gate.sh` 不会替用户启动正式服务。若此前临时停止了三个 unit，先在板端恢复主实例、回滚 listener 和精确路由：

```bash
sudo systemctl start agileagent-ascend310b-main.service
sudo systemctl start agileagent-ascend310b-rollback.service

for i in $(seq 1 180); do
  curl -fsS http://127.0.0.1:18501/api/health >/dev/null 2>&1 && \
  curl -fsS http://127.0.0.1:8501/api/health >/dev/null 2>&1 && break
  sleep 1
done

sudo systemctl start agileagent-ascend310b-route.service
sudo /usr/local/sbin/agileagent-ascend310b-primary-route status 18501
curl -fsS http://127.0.0.1:8501/api/health
ss -H -ltn 'sport = :8501 or sport = :18501 or sport = :8502'
```

公共 health 必须是 `status:"ready"`，并应显示 `validated:true`、`model_layout:"shared_backbone_dual_head_v1"`；`8502` 此时必须没有监听器。当前板端服务是人工停止状态时，不要为了阅读文档而执行这组命令；只在真正开始 score gate 前恢复。

### 4.7 阈值矩阵、`candidates.json` 和自动选优

当前 `old=0.05/new=0.30` 只是种子。下面的板端函数会为每个组合生成独立 `8502` 配置并运行完整“无标签预测冻结 → 三项精度评分 → 30 次预热 → 三轮 20 图 batch”：

```bash
PY=/usr/local/miniconda3/envs/agileagent/bin/python
SEARCH="/home/HwHiAiUser/agileagent/candidates/$DATA_ID/threshold-search"
DUAL_OM="$REMOTE_BUILD/shared_backbone_dual_head.om"
BUILD_MANIFEST="$REMOTE_BUILD/build-manifest.json"
CONTEXT_OM="/home/HwHiAiUser/agileagent/releases/20260816-full-score-1493b04/om/scene_sensor_net.om"
MIXED_IMAGES=/path/to/new-mixed-png
MIXED_SPLIT=/path/to/new-mixed-test.txt
BASE_SPLIT=/path/to/new-base-test.txt

run_pair() {
  local stage="$1" id="$2" old="$3" new="$4"
  local root="$SEARCH/$stage/$id"
  mkdir -p "$root"
  "$PY" tools/109_materialize_ascend_full_score_candidate.py \
    --dual-om "$DUAL_OM" \
    --context-om "$CONTEXT_OM" \
    --build-manifest "$BUILD_MANIFEST" \
    --old-threshold "$old" --new-threshold "$new" \
    --report-root "reports/ascend310b/$DATA_ID/$stage/$id" \
    --output "$root/candidate.yaml"
  ./scripts/run_ascend310b_score_gate.sh \
    "$root/candidate.yaml" "$MIXED_IMAGES" "$MIXED_SPLIT" "$BASE_SPLIT" \
    "$root/gate"
}

for new in 0.20 0.25 0.30 0.35 0.40; do
  tag=${new/./}
  run_pair stage1 "old005-new$tag" 0.05 "$new"
done
```

每完成一个阶段，用下面的命令从真实 `score.json/benchmark.json` 和 candidate YAML 汇总索引；它不会伪造结果：

```bash
make_index() {
  local stage="$1"
  SEARCH_STAGE="$SEARCH/$stage" "$PY" - <<'PY'
import json
import os
from pathlib import Path

import yaml

root = Path(os.environ["SEARCH_STAGE"])
rows = []
for gate in sorted(root.glob("*/gate")):
    candidate = yaml.safe_load((gate.parent / "candidate.yaml").read_text(encoding="utf-8"))
    model = next(iter(candidate["ascend_backend"]["models"].values()))
    heads = model["logical_heads"]
    rows.append({
        "id": gate.parent.name,
        "old_threshold": float(heads["old"]["candidate_confidence"]),
        "new_threshold": float(heads["new"]["candidate_confidence"]),
        "score": str((gate / "score.json").relative_to(root)),
        "benchmark": str((gate / "benchmark.json").relative_to(root)),
        "repeat_benchmarks": [],
        "prerequisites": {
            "incremental_data_isolation": True,
            "asset_hashes_verified": True
        }
    })
(root / "candidates.json").write_text(
    json.dumps({"schema_version": 1, "candidates": rows}, indent=2) + "\n",
    encoding="utf-8",
)
PY
  "$PY" tools/110_select_ascend_full_score_candidate.py \
    --candidates "$SEARCH/$stage/candidates.json" \
    --output "$SEARCH/$stage/selection.json" || true
}

make_index stage1
NEW_WIN=$("$PY" - "$SEARCH/stage1/selection.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
winner = p["selected_candidate"]
if winner is None:
    raise SystemExit("stage1没有同时通过有效性与精度门禁的候选")
print(next(row["source"]["new_threshold"] for row in p["candidates"] if row["id"] == winner))
PY
)

for old in 0.03 0.05 0.10 0.20 0.30; do
  tag=${old/./}
  run_pair stage2 "old$tag-new${NEW_WIN/./}" "$old" "$NEW_WIN"
done
make_index stage2
```

合并两个阶段并按选择器的正式规则产生待复核前两名：

```bash
mkdir -p "$SEARCH/final"
"$PY" - "$SEARCH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for stage in ("stage1", "stage2"):
    payload = json.loads((root / stage / "candidates.json").read_text(encoding="utf-8"))
    for row in payload["candidates"]:
        copied = dict(row)
        copied["id"] = f"{stage}-{row['id']}"
        copied["score"] = f"../{stage}/{row['score']}"
        copied["benchmark"] = f"../{stage}/{row['benchmark']}"
        copied["repeat_benchmarks"] = []
        rows.append(copied)
(root / "final" / "precheck-candidates.json").write_text(
    json.dumps({"schema_version": 1, "candidates": rows}, indent=2) + "\n",
    encoding="utf-8",
)
PY

"$PY" tools/110_select_ascend_full_score_candidate.py \
  --candidates "$SEARCH/final/precheck-candidates.json" \
  --output "$SEARCH/final/precheck-selection.json" || true

"$PY" - "$SEARCH/final/precheck-selection.json" "$SEARCH/final/top2.tsv" <<'PY'
import json
import sys
from pathlib import Path

payload = json.load(open(sys.argv[1], encoding="utf-8"))
full = [row for row in payload["candidates"] if row["full_score"]]
accuracy_pass = [
    row for row in payload["candidates"]
    if row["validity_passed"] and row["accuracy_passed"]
]
if full:
    ranked = sorted(
        full,
        key=lambda row: (
            -row["minimum_accuracy_headroom"],
            row["batch_fps_spread"],
            -row["batch_median_fps"],
            row["id"],
        ),
    )
else:
    ranked = sorted(
        accuracy_pass,
        key=lambda row: (
            -row["batch_median_fps"],
            -row["minimum_accuracy_headroom"],
            row["batch_fps_spread"],
            row["id"],
        ),
    )
lines = [
    f"{row['id']}\t{row['source']['old_threshold']}\t{row['source']['new_threshold']}"
    for row in ranked[:2]
]
Path(sys.argv[2]).write_text(
    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
)
PY

while IFS=$'\t' read -r source_id old new; do
  test -n "$source_id" || continue
  run_pair crosscheck "$source_id-repeat" "$old" "$new"
done < "$SEARCH/final/top2.tsv"
```

把交叉复核的 benchmark 作为原候选的 `repeat_benchmarks`，再执行最终选择：

```bash
"$PY" - "$SEARCH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = json.loads(
    (root / "final" / "precheck-candidates.json").read_text(encoding="utf-8")
)
top2 = {}
for line in (root / "final" / "top2.tsv").read_text(encoding="utf-8").splitlines():
    candidate_id, old, new = line.split("\t")
    top2[candidate_id] = f"../crosscheck/{candidate_id}-repeat/gate/benchmark.json"
for row in payload["candidates"]:
    row["repeat_benchmarks"] = [top2[row["id"]]] if row["id"] in top2 else []
(root / "final" / "candidates.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

"$PY" tools/110_select_ascend_full_score_candidate.py \
  --candidates "$SEARCH/final/candidates.json" \
  --output "$SEARCH/final/selection.json"
```

不要复用非空 gate 目录。完整索引条目必须包含 `id`、`score`、`benchmark`、`repeat_benchmarks` 和两个为真的 prerequisites；额外保存的 `old_threshold/new_threshold` 会原样进入 selection 的 `source`，便于复核。`tools/110` 在尚无满分项时会以退出码 `1` 返回 `intermediate_only/no_eligible_candidate`，所以仅生成前两名复核清单的 precheck 调用显式使用 `|| true`，最终正式选择不得忽略退出码。

选择器只以 Base mAP50、New-mAP50、KRR 和主 benchmark 的 20 图中位 FPS 判定满分。多个满分候选依次按最小精度余量、所有主/复核轮次 FPS 波动和主 benchmark 中位 FPS 排名；没有效率满分项时，最高 FPS 的精度通过项只能是 `intermediate_only`。

评分图片仍有硬输入契约：`MIXED_IMAGES` 根目录只能有 stem 唯一的 `*.png`，每张必须为 `640×512`、8-bit RGB（color type 2）或 RGBA（color type 6）。score gate 只停止它自己启动且命令行明确含 `--port 8502` 的进程，不运行板端 Web pytest，并在退出时再次确认 `8501 ready`。

### 4.8 类别数量、类别名称或增量轮次变化时的迁移清单

当前完整自动链路实质上是单轮、单新增类的 3+1。Ascend raw dual-head 运行时会按 `class_count` 计算通道，但数据、训练和评分入口尚未全部通用；因此下面所有项目必须在一次专门代码变更中同步完成并通过测试：

1. 数据入口：去除 `dataset_utils.py` 的固定数据目录、五段文件名、sensor/scene 和 `EXPECTED_IMAGE_COUNT=750`；把 `tools/02` 的“恰好四类”和单新增类共现规则改为显式数据协议配置。
2. 全局类别：把 `strict_incremental.py` 的 `GLOBAL_CLASS_NAMES` 从硬编码改为配置；`base_local_to_global` 的本地键仍须从 `0` 连续编号，old/new 全局 ID 必须互斥。
3. 训练适配器：`compile_training_adapter()` 当前明确拒绝非“三旧类、一轮、一个新类”；多新增类要生成 `specialist_local_to_global={0..N_new-1}` 和对应 YOLO dataset `names/nc`，多轮增量还需定义每一轮是新增 logical head、合并进 old head，还是重新蒸馏，不能复用双 head 假装多轮已支持。
4. checkpoint 结构：Base Detect head 的 `nc=N_old`，Specialist/new head 的 `nc=N_new`；`build_shared_head_training_checkpoint()`、residual adapter 冻结检查和 EMA 零漂移必须用这两个真实 checkpoint 复测。
5. 方法配置：同步更新 old/new `class_map`、`class_count`、`output_shape` 和评分类别集合。固定 `896×736` 时 anchor count 仍为 `13524`，raw 输出 shape 必须分别为 `[1,4+N_old,13524]`、`[1,4+N_new,13524]`。
6. 导出与构建：重新导出 ONNX，让实际两个输出 shape 与方法 YAML 完全一致；随后重跑 ATC，生成新的 export/build manifest 和哈希。只编辑 YAML 或复用旧 OM 会被结构校验拒绝。
7. 阈值：`tools/109` 当前每个 logical head 只有一个 scalar threshold。若一个 new head 含多个新增类，必须决定并实现“共用阈值”或“逐类阈值”，并同步候选 schema、矩阵生成和排名证据。
8. 评分：`tools/94` 目前只有单数 `--new-class-id`，score gate 还显式要求 new `class_map` 只有一个值。多新增类必须改为 `--new-class-ids`（或方法配置集合），定义 New-mAP50 的多类平均口径，并更新 KRR、预测冻结和 score schema v2 兼容逻辑。
9. 测试：更新 strict split、训练 adapter、shared dual head、配置校验、score gate、selector、release/promote 和模型 package 测试；至少加入 `N_old!=3`、`N_new>1`、类别重排和不支持多轮时 fail-closed 的用例。

完成以上迁移前，类别数量变化的数据只能停在“待适配”状态，不能声称现有命令可以重训或达到满分。owner 语义仍保持 old=`frozen_base_model`、new=`incremental_model`；四项评分门禁不变，具体类别集合从新协议读取。

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
