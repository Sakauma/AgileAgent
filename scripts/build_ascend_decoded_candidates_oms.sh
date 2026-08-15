#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  bash scripts/build_ascend_decoded_candidates_oms.sh \
    ONNX_DIR OUTPUT_DIR BASE_MANIFEST SEMANTIC_PROBE_REPORT

在固定 CANN 7.0.RC1 / Ascend310B1 上构建 P9 decoded_candidates_v1
Base 和 Specialist OM。Scene OM 从 BASE_MANIFEST 复用，不覆盖任何产物。
EOF
}

if (( $# != 4 )); then
  usage >&2
  exit 2
fi

if [[ "${AGILE_AGENT_P9_SEMANTIC_GATE:-}" != "passed" ]]; then
  printf '%s\n' \
    'P9设备解码严格语义门禁未授权；拒绝构建decoded_candidates_v1候选。' \
    '保存通过报告后显式设置AGILE_AGENT_P9_SEMANTIC_GATE=passed。' >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_DIR="$(readlink -f "$1")"
OUTPUT_DIR="$2"
BASE_MANIFEST="$(readlink -f "$3")"
SEMANTIC_REPORT="$(readlink -f "$4")"
AIPP_DIR="${AGILE_AGENT_AIPP_DIR:-$ROOT/configs/ascend310b/aipp}"
AIPP_DIR="$(readlink -f "$AIPP_DIR")"
ASCEND_PYTHON="${AGILE_AGENT_ASCEND_PYTHON:-/usr/local/miniconda3/envs/agileagent/bin/python}"

test -d "$ONNX_DIR" || { printf 'ONNX目录不存在：%s\n' "$ONNX_DIR" >&2; exit 1; }
test -f "$BASE_MANIFEST" || { printf '基础构建清单不存在：%s\n' "$BASE_MANIFEST" >&2; exit 1; }
test -f "$SEMANTIC_REPORT" || { printf '语义探针报告不存在：%s\n' "$SEMANTIC_REPORT" >&2; exit 1; }
test -x "$ASCEND_PYTHON" || { printf 'Ascend Python不存在：%s\n' "$ASCEND_PYTHON" >&2; exit 1; }
for name in base_detector incremental_detector; do
  test -f "$ONNX_DIR/$name.onnx" || { printf '缺少P9 ONNX：%s\n' "$name" >&2; exit 1; }
  test -f "$AIPP_DIR/$name.cfg" || { printf '缺少AIPP配置：%s\n' "$name" >&2; exit 1; }
done
if test -e "$OUTPUT_DIR" && test -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)"; then
  printf '候选输出目录非空，拒绝覆盖：%s\n' "$OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(readlink -f "$OUTPUT_DIR")"

"$ASCEND_PYTHON" - "$SEMANTIC_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("kind") != "ascend_decoded_candidates_v1_probe_verification":
    raise SystemExit("P9语义探针报告类型错误")
if report.get("passed") is not True:
    raise SystemExit("P9语义探针未通过")
PY

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export PATH="$(dirname "$ASCEND_PYTHON"):$PATH"
command -v atc >/dev/null || { printf 'CANN ATC不可用。\n' >&2; exit 1; }

build_one() {
  local model_id="$1"
  local input_shape="$2"
  local output_prefix="$OUTPUT_DIR/$model_id"
  local log="$OUTPUT_DIR/atc_$model_id.log"
  local command_file="$OUTPUT_DIR/atc_$model_id.command.txt"
  local args=(
    "--model=$ONNX_DIR/$model_id.onnx"
    "--framework=5"
    "--output=$output_prefix"
    "--input_format=NCHW"
    "--input_shape=images:$input_shape"
    "--soc_version=Ascend310B1"
    "--precision_mode_v2=mixed_float16"
    "--insert_op_conf=$AIPP_DIR/$model_id.cfg"
  )
  local expected_command="atc"
  local argument quoted_argument
  for argument in "${args[@]}"; do
    printf -v quoted_argument '%q' "$argument"
    expected_command+=" $quoted_argument"
  done
  printf '%s\n' "$expected_command" >"$command_file"
  atc "${args[@]}" >"$log" 2>&1
  test -s "$output_prefix.om" || { printf 'ATC未生成OM：%s\n' "$model_id" >&2; exit 1; }
}

build_one base_detector 1,3,736,896
build_one incremental_detector 1,3,512,640

AGILE_AGENT_BUILD_ROOT="$ROOT" \
AGILE_AGENT_ONNX_DIR="$ONNX_DIR" \
AGILE_AGENT_AIPP_DIR="$AIPP_DIR" \
AGILE_AGENT_OUTPUT_DIR="$OUTPUT_DIR" \
AGILE_AGENT_BASE_MANIFEST="$BASE_MANIFEST" \
AGILE_AGENT_SEMANTIC_REPORT="$SEMANTIC_REPORT" \
"$ASCEND_PYTHON" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(os.environ["AGILE_AGENT_BUILD_ROOT"])
onnx_dir = Path(os.environ["AGILE_AGENT_ONNX_DIR"])
aipp_dir = Path(os.environ["AGILE_AGENT_AIPP_DIR"])
output_dir = Path(os.environ["AGILE_AGENT_OUTPUT_DIR"])
base_manifest_path = Path(os.environ["AGILE_AGENT_BASE_MANIFEST"])
semantic_report = Path(os.environ["AGILE_AGENT_SEMANTIC_REPORT"])
base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
scene = base_manifest["artifacts"]["scene_sensor_net"]
try:
    git_sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
except (OSError, subprocess.CalledProcessError):
    git_sha = "unknown"

specs = {
    "base_detector": {
        "role": "base",
        "source": root / "models/production/incremental_detection/three_class_base_detector.pt",
        "input_shape": [1, 736, 896, 3],
        "candidate_capacity": 4096,
        "anchor_count": 13524,
        "class_count": 3,
    },
    "incremental_detector": {
        "role": "specialist",
        "source": root / "models/production/incremental_detection/incremental_detector.pt",
        "input_shape": [1, 512, 640, 3],
        "candidate_capacity": 2048,
        "anchor_count": 6720,
        "class_count": 1,
    },
}
artifacts = {}
for model_id, spec in specs.items():
    source = Path(spec["source"])
    onnx = onnx_dir / f"{model_id}.onnx"
    aipp = aipp_dir / f"{model_id}.cfg"
    om = output_dir / f"{model_id}.om"
    log = output_dir / f"atc_{model_id}.log"
    command_file = output_dir / f"atc_{model_id}.command.txt"
    for path in (source, onnx, aipp, om, log, command_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    capacity = int(spec["candidate_capacity"])
    anchors = int(spec["anchor_count"])
    classes = int(spec["class_count"])
    outputs = {
        "boxes": {"shape": [capacity, 4], "dtype": "float32"},
        "scores": {"shape": [capacity], "dtype": "float32"},
        "class_ids": {"shape": [capacity], "dtype": "int32"},
        "anchor_ids": {"shape": [capacity], "dtype": "int32"},
        "valid_count": {"shape": [1], "dtype": "int32"},
        "overflow": {"shape": [1], "dtype": "int32"},
        "raw_output": {
            "shape": [1, 4 + classes, anchors],
            "dtype": "float32",
        },
    }
    artifacts[model_id] = {
        "role": spec["role"],
        "source_weight": {"path": str(source), "sha256": sha256(source)},
        "onnx": {"path": str(onnx), "sha256": sha256(onnx)},
        "aipp": {"path": str(aipp), "sha256": sha256(aipp)},
        "om": {"path": str(om), "sha256": sha256(om)},
        "atc_log": {"path": str(log), "sha256": sha256(log)},
        "atc_command": command_file.read_text(encoding="utf-8").strip(),
        "input_contract": {
            "dtype": "uint8",
            "layout": "NHWC",
            "shape": spec["input_shape"],
        },
        "output_contract": "decoded_candidates_v1",
        "postprocess_contract": {
            "candidate_confidence": 0.01,
            "candidate_capacity": capacity,
            "anchor_count": anchors,
            "class_count": classes,
            "outputs": outputs,
        },
    }
artifacts["scene_sensor_net"] = scene
payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "git_sha": git_sha,
    "soc_version": "Ascend310B1",
    "cann_version": "7.0.RC1",
    "precision": "mixed_float16",
    "validated": False,
    "parent_build_manifest": {
        "path": str(base_manifest_path),
        "sha256": sha256(base_manifest_path),
    },
    "semantic_probe": {
        "path": str(semantic_report),
        "sha256": sha256(semantic_report),
    },
    "artifacts": artifacts,
}
manifest = output_dir / "build-manifest.json"
manifest.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "om_sha256": {
                name: row["om"]["sha256"]
                for name, row in artifacts.items()
            },
        },
        ensure_ascii=False,
        indent=2,
    )
)
PY
