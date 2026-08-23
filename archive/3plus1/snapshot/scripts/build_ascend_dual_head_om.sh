#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  ./scripts/build_ascend_dual_head_om.sh ONNX SOURCE_WEIGHT TRAINING_REPORT EXPORT_MANIFEST CONTEXT_MANIFEST OUTPUT_DIR [MANIFEST]

在已安装 CANN 7.0.RC1 的 Ascend310B1 主机上构建共享双逻辑头 OM。
CONTEXT_MANIFEST 必须提供一个已验证的 context 资产，作为 fixed_neutral_v1 的显式回滚项。
输出目录必须不存在或为空，脚本不会安装、升级或覆盖任何运行时。
EOF
}

if (( $# < 6 || $# > 7 )); then
  usage >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD_CONFIG="${AGILE_AGENT_FULL_SCORE_METHOD:-$ROOT/configs/ascend310b/full_score_method.yaml}"
ONNX="$(readlink -f "$1")"
SOURCE_WEIGHT="$(readlink -f "$2")"
TRAINING_REPORT="$(readlink -f "$3")"
EXPORT_MANIFEST="$(readlink -f "$4")"
CONTEXT_MANIFEST="$(readlink -f "$5")"
OUTPUT_DIR="$6"
MANIFEST="${7:-$OUTPUT_DIR/build-manifest.json}"
ASCEND_PYTHON="${AGILE_AGENT_ASCEND_PYTHON:-/usr/local/miniconda3/envs/agileagent/bin/python}"

for path in "$ONNX" "$SOURCE_WEIGHT" "$TRAINING_REPORT" "$EXPORT_MANIFEST" "$CONTEXT_MANIFEST" "$METHOD_CONFIG"; do
  test -f "$path" || { printf '缺少构建输入：%s\n' "$path" >&2; exit 1; }
done
METHOD_CONFIG="$(readlink -f "$METHOD_CONFIG")"
test -x "$ASCEND_PYTHON" || { printf '板端Python不可执行：%s\n' "$ASCEND_PYTHON" >&2; exit 1; }

mapfile -t METHOD_VALUES < <("$ASCEND_PYTHON" - "$METHOD_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
method = yaml.safe_load(path.read_text(encoding="utf-8"))
if not isinstance(method, dict) or method.get("kind") != "ascend310b_full_score_method":
    raise RuntimeError("满分方法配置schema/kind非法")
target = method["target"]
export = method["export"]
heads = json.loads(json.dumps(export["logical_heads"]))
for head in heads.values():
    head.pop("output_shape", None)
aipp = Path(str(export["aipp_config"]))
if aipp.is_absolute() or ".." in aipp.parts:
    raise RuntimeError("AIPP配置必须是仓库内相对路径")
values = (
    target["soc_version"],
    target["cann_version"],
    target["precision"],
    export["input_name"],
    ",".join(str(value) for value in export["input_shape_nchw"]),
    json.dumps(export["input_shape_nchw"], separators=(",", ":")),
    json.dumps(export["input_shape_aipp_nhwc"], separators=(",", ":")),
    json.dumps(
        [heads[name] for name in ("old", "new")],
        separators=(",", ":"),
        sort_keys=True,
    ),
    json.dumps(
        [export["logical_heads"][name]["output_shape"] for name in ("old", "new")],
        separators=(",", ":"),
    ),
    export["model_layout"],
    export["output_contract"],
    str(export["opset"]),
    json.dumps(heads, separators=(",", ":"), sort_keys=True),
    str(aipp),
)
print("\n".join(str(value) for value in values))
PY
)
if (( ${#METHOD_VALUES[@]} != 14 )); then
  printf '无法从满分方法配置读取完整构建契约：%s\n' "$METHOD_CONFIG" >&2
  exit 1
fi
SOC_VERSION="${METHOD_VALUES[0]}"
CANN_VERSION="${METHOD_VALUES[1]}"
PRECISION="${METHOD_VALUES[2]}"
INPUT_NAME="${METHOD_VALUES[3]}"
INPUT_SHAPE_CSV="${METHOD_VALUES[4]}"
INPUT_SHAPE_JSON="${METHOD_VALUES[5]}"
AIPP_SHAPE_JSON="${METHOD_VALUES[6]}"
HEAD_LIST_JSON="${METHOD_VALUES[7]}"
OUTPUT_SHAPES_JSON="${METHOD_VALUES[8]}"
MODEL_LAYOUT="${METHOD_VALUES[9]}"
OUTPUT_CONTRACT="${METHOD_VALUES[10]}"
OPSET="${METHOD_VALUES[11]}"
LOGICAL_HEADS_JSON="${METHOD_VALUES[12]}"
AIPP="$(readlink -f "$ROOT/${METHOD_VALUES[13]}")"
test -f "$AIPP" || { printf '方法配置指定的AIPP不存在：%s\n' "$AIPP" >&2; exit 1; }
if test -e "$OUTPUT_DIR" && test -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)"; then
  printf '输出目录非空，拒绝覆盖：%s\n' "$OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(readlink -f "$OUTPUT_DIR")"
mkdir -p -- "$(dirname "$MANIFEST")"
MANIFEST="$(readlink -f "$MANIFEST")"
test ! -e "$MANIFEST" || { printf '构建清单已存在，拒绝覆盖：%s\n' "$MANIFEST" >&2; exit 1; }

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
command -v atc >/dev/null || { printf 'CANN ATC不可用。\n' >&2; exit 1; }
CANN_ROOT="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
CANN_RESOLVED="$(readlink -f "$CANN_ROOT")"
CANN_VERSION_EVIDENCE="$(
  shopt -s nullglob
  version_files=(
    "$CANN_RESOLVED/version.info"
    "$CANN_RESOLVED"/*-linux/ascend_toolkit_install.info
  )
  for version_file in "${version_files[@]}"; do
    printf '%s: ' "$version_file"
    tr '\n' ' ' <"$version_file"
    printf '\n'
  done
  atc --version 2>&1 || true
)"
if [[ "$CANN_VERSION_EVIDENCE" != *"$CANN_VERSION"* ]]; then
  printf '无法从CANN版本文件或ATC确认要求版本%s；root=%s evidence=%s\n' \
    "$CANN_VERSION" "$CANN_RESOLVED" "$CANN_VERSION_EVIDENCE" >&2
  exit 1
fi

OUTPUT_PREFIX="$OUTPUT_DIR/shared_backbone_dual_head"
OM="$OUTPUT_PREFIX.om"
LOG="$OUTPUT_DIR/atc_shared_backbone_dual_head.log"
COMMAND_FILE="$OUTPUT_DIR/atc_shared_backbone_dual_head.command.txt"
ARGS=(
  "--model=$ONNX"
  "--framework=5"
  "--output=$OUTPUT_PREFIX"
  "--input_format=NCHW"
  "--input_shape=$INPUT_NAME:$INPUT_SHAPE_CSV"
  "--soc_version=$SOC_VERSION"
  "--precision_mode_v2=$PRECISION"
  "--insert_op_conf=$AIPP"
)
EXPECTED_COMMAND="atc"
for argument in "${ARGS[@]}"; do
  printf -v quoted_argument '%q' "$argument"
  EXPECTED_COMMAND+=" $quoted_argument"
done
printf '%s\n' "$EXPECTED_COMMAND" >"$COMMAND_FILE"
atc "${ARGS[@]}" 2>&1 | tee "$LOG"
test -s "$OM" || { printf 'ATC未生成双头OM：%s\n' "$OM" >&2; exit 1; }
grep -q 'ATC run success' "$LOG" || { printf 'ATC日志没有成功标志：%s\n' "$LOG" >&2; exit 1; }

AGILE_AGENT_BUILD_ROOT="$ROOT" \
AGILE_AGENT_METHOD_CONFIG="$METHOD_CONFIG" \
AGILE_AGENT_DUAL_ONNX="$ONNX" \
AGILE_AGENT_DUAL_SOURCE="$SOURCE_WEIGHT" \
AGILE_AGENT_TRAINING_REPORT="$TRAINING_REPORT" \
AGILE_AGENT_EXPORT_MANIFEST="$EXPORT_MANIFEST" \
AGILE_AGENT_CONTEXT_MANIFEST="$CONTEXT_MANIFEST" \
AGILE_AGENT_DUAL_AIPP="$AIPP" \
AGILE_AGENT_DUAL_OM="$OM" \
AGILE_AGENT_DUAL_LOG="$LOG" \
AGILE_AGENT_DUAL_COMMAND_FILE="$COMMAND_FILE" \
AGILE_AGENT_BUILD_MANIFEST="$MANIFEST" \
AGILE_AGENT_SOC_VERSION="$SOC_VERSION" \
AGILE_AGENT_CANN_VERSION="$CANN_VERSION" \
AGILE_AGENT_CANN_VERSION_EVIDENCE="$CANN_VERSION_EVIDENCE" \
AGILE_AGENT_PRECISION="$PRECISION" \
AGILE_AGENT_INPUT_SHAPE_JSON="$INPUT_SHAPE_JSON" \
AGILE_AGENT_AIPP_SHAPE_JSON="$AIPP_SHAPE_JSON" \
AGILE_AGENT_HEAD_LIST_JSON="$HEAD_LIST_JSON" \
AGILE_AGENT_OUTPUT_SHAPES_JSON="$OUTPUT_SHAPES_JSON" \
AGILE_AGENT_MODEL_LAYOUT="$MODEL_LAYOUT" \
AGILE_AGENT_OUTPUT_CONTRACT="$OUTPUT_CONTRACT" \
AGILE_AGENT_OPSET="$OPSET" \
AGILE_AGENT_LOGICAL_HEADS_JSON="$LOGICAL_HEADS_JSON" \
"$ASCEND_PYTHON" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_entry(entry: object, label: str) -> dict:
    if not isinstance(entry, dict):
        raise RuntimeError(f"context manifest缺少{label}")
    path = Path(str(entry.get("path") or ""))
    if not path.is_file() or sha256(path) != entry.get("sha256"):
        raise RuntimeError(f"context资产缺失或SHA256不一致：{label}:{path}")
    return dict(entry)


root = Path(os.environ["AGILE_AGENT_BUILD_ROOT"])
method_config = Path(os.environ["AGILE_AGENT_METHOD_CONFIG"])
onnx = Path(os.environ["AGILE_AGENT_DUAL_ONNX"])
source = Path(os.environ["AGILE_AGENT_DUAL_SOURCE"])
training_report_path = Path(os.environ["AGILE_AGENT_TRAINING_REPORT"])
export_manifest_path = Path(os.environ["AGILE_AGENT_EXPORT_MANIFEST"])
context_manifest_path = Path(os.environ["AGILE_AGENT_CONTEXT_MANIFEST"])
aipp = Path(os.environ["AGILE_AGENT_DUAL_AIPP"])
om = Path(os.environ["AGILE_AGENT_DUAL_OM"])
log = Path(os.environ["AGILE_AGENT_DUAL_LOG"])
command_file = Path(os.environ["AGILE_AGENT_DUAL_COMMAND_FILE"])
manifest_path = Path(os.environ["AGILE_AGENT_BUILD_MANIFEST"])
soc_version = os.environ["AGILE_AGENT_SOC_VERSION"]
cann_version = os.environ["AGILE_AGENT_CANN_VERSION"]
cann_version_evidence = os.environ["AGILE_AGENT_CANN_VERSION_EVIDENCE"]
precision = os.environ["AGILE_AGENT_PRECISION"]
input_shape = json.loads(os.environ["AGILE_AGENT_INPUT_SHAPE_JSON"])
aipp_shape = json.loads(os.environ["AGILE_AGENT_AIPP_SHAPE_JSON"])
head_list = json.loads(os.environ["AGILE_AGENT_HEAD_LIST_JSON"])
output_shapes = json.loads(os.environ["AGILE_AGENT_OUTPUT_SHAPES_JSON"])
model_layout = os.environ["AGILE_AGENT_MODEL_LAYOUT"]
output_contract = os.environ["AGILE_AGENT_OUTPUT_CONTRACT"]
opset = int(os.environ["AGILE_AGENT_OPSET"])
method_heads = json.loads(os.environ["AGILE_AGENT_LOGICAL_HEADS_JSON"])

method = yaml.safe_load(method_config.read_text(encoding="utf-8"))
if not isinstance(method, dict) or method.get("kind") != "ascend310b_full_score_method":
    raise RuntimeError("满分方法配置schema/kind非法")
training_contract = method.get("training") or {}
reference_result = method.get("reference_result") or {}
allowed_checkpoints = list(training_contract.get("export_checkpoints") or [])
if not allowed_checkpoints or any(
    name not in {"best", "last"} for name in allowed_checkpoints
):
    raise RuntimeError("满分方法没有合法的best/last导出checkpoint契约")

source_sha256 = sha256(source)
training_report_sha256 = sha256(training_report_path)
export_manifest_sha256 = sha256(export_manifest_path)
training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
required_audit = {
    "old_raw_image_count": 0,
    "old_raw_label_count": 0,
    "old_feature_cache_count": 0,
    "original_data_modified": False,
}
if (
    training_report.get("kind") != "shared_backbone_dual_head_training"
    or training_report.get("method") != training_contract.get("method")
    or float(training_report.get("shared_max_drift", -1.0)) != 0.0
    or {key: (training_report.get("dataset_audit") or {}).get(key) for key in required_audit}
    != required_audit
):
    raise RuntimeError("training report未通过训练方法、数据隔离或零漂移门禁")

checkpoint_role = None
checkpoint_authorization = None
if int(training_report.get("schema_version", 0)) >= 2:
    method_entry = training_report.get("method_config") or {}
    checkpoints = training_report.get("checkpoints") or {}
    if (
        method_entry.get("sha256") != sha256(method_config)
        or training_report.get("training_contract") != training_contract
        or not isinstance(checkpoints, dict)
        or any(name not in checkpoints for name in allowed_checkpoints)
    ):
        raise RuntimeError("training report与满分方法或checkpoint契约不一致")
    for name in allowed_checkpoints:
        row = checkpoints[name]
        if (
            not isinstance(row, dict)
            or len(str(row.get("sha256") or "")) != 64
            or float(row.get("shared_max_drift", -1.0)) != 0.0
        ):
            raise RuntimeError(f"training report的{name} checkpoint未通过零漂移/哈希门禁")
        if row["sha256"] == source_sha256:
            checkpoint_role = name
    checkpoint_authorization = "training_report_v2"
else:
    historical_match = (
        training_report_sha256 == reference_result.get("training_report_sha256")
        and export_manifest_sha256 == reference_result.get("export_manifest_sha256")
        and source_sha256 == reference_result.get("export_checkpoint_sha256")
        and training_report.get("best_weight_sha256")
        == reference_result.get("training_best_checkpoint_sha256")
        and reference_result.get("export_checkpoint") in allowed_checkpoints
    )
    if historical_match:
        checkpoint_role = str(reference_result["export_checkpoint"])
        checkpoint_authorization = "fixed_reference_evidence_compatibility"
if checkpoint_role is None:
    raise RuntimeError("SOURCE_WEIGHT未获training report或固定历史证据授权")

export_heads = export_manifest.get("logical_heads")
if (
    export_manifest.get("kind") != model_layout
    or export_manifest.get("opset") != opset
    or export_manifest.get("input_shape") != input_shape
    or export_manifest.get("output_shapes") != output_shapes
    or float((export_manifest.get("provenance") or {}).get("shared_max_drift", -1.0))
    != 0.0
    or (export_manifest.get("provenance") or {}).get("new_head_weight_sha256")
    != source_sha256
    or (export_manifest.get("onnx") or {}).get("sha256") != sha256(onnx)
    or not isinstance(export_heads, dict)
    or set(export_heads) != {"old", "new"}
    or export_heads != method_heads
    or [export_heads[name] for name in ("old", "new")] != head_list
):
    raise RuntimeError("export manifest的结构、零漂移或输入资产哈希不一致")

context_manifest = json.loads(context_manifest_path.read_text(encoding="utf-8"))
context_rows = [
    row
    for row in (context_manifest.get("artifacts") or {}).values()
    if isinstance(row, dict) and row.get("role") == "context"
]
if len(context_rows) != 1:
    raise RuntimeError("CONTEXT_MANIFEST必须恰好包含一个context资产")
context = dict(context_rows[0])
for name in ("source_weight", "onnx", "aipp", "om", "atc_log"):
    context[name] = checked_entry(context.get(name), f"context.{name}")
command = str(context.get("atc_command") or "")
for marker in (
    "--framework=5",
    f"--soc_version={soc_version}",
    f"--precision_mode_v2={precision}",
):
    if marker not in command:
        raise RuntimeError(f"context ATC命令缺少{marker}")

try:
    git_sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
except (OSError, subprocess.CalledProcessError):
    git_sha = "unknown"

payload = {
    "schema_version": 1,
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "git_sha": git_sha,
    "soc_version": soc_version,
    "cann_version": cann_version,
    "cann_version_evidence": cann_version_evidence,
    "precision": precision,
    "validated": False,
    "validation_basis": "competition_score_pending",
    "model_layout": model_layout,
    "method_config": {
        "path": str(method_config),
        "sha256": sha256(method_config),
    },
    "training_report": {
        "path": str(training_report_path),
        "sha256": sha256(training_report_path),
    },
    "export_manifest": {
        "path": str(export_manifest_path),
        "sha256": sha256(export_manifest_path),
    },
    "artifacts": {
        "shared_backbone_dual_head": {
            "role": "dual_detector",
            "source_weight": {
                "path": str(source),
                "sha256": source_sha256,
                "checkpoint_role": checkpoint_role,
                "authorization": checkpoint_authorization,
            },
            "onnx": {"path": str(onnx), "sha256": sha256(onnx)},
            "aipp": {"path": str(aipp), "sha256": sha256(aipp)},
            "om": {"path": str(om), "sha256": sha256(om)},
            "atc_log": {"path": str(log), "sha256": sha256(log)},
            "atc_command": command_file.read_text(encoding="utf-8").strip(),
            "input_contract": {
                "dtype": "uint8",
                "layout": "NHWC",
                "shape": aipp_shape,
            },
            "output_contract": output_contract,
            "logical_heads": export_heads,
        },
        "scene_sensor_net": context,
    },
}
manifest_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({
    "manifest": str(manifest_path),
    "manifest_sha256": sha256(manifest_path),
    "om": str(om),
    "om_sha256": sha256(om),
}, ensure_ascii=False, indent=2))
PY
