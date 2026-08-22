#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  ./scripts/build_ascend_yolo26_e2e_oms.sh ONNX_DIR OUTPUT_DIR CONTEXT_BUILD_MANIFEST

在 CANN 7.0.RC1 / Ascend310B1 上构建 4+2 YOLO26 E2E Base 与
Specialist OM。两个模型固定输入 608x736，固定输出 [1,300,6]；Scene
资产只从给定父构建清单复用，fixed_neutral_v1 运行时不会执行其前向。

设置 AGILE_AGENT_RESUME=1 可复用命令和成功日志均严格匹配的现有 OM。
EOF
}

if (( $# != 3 )); then
  usage >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_DIR="$(readlink -f "$1")"
OUTPUT_DIR="$2"
CONTEXT_BUILD_MANIFEST="$(readlink -f "$3")"
AIPP_DIR="${AGILE_AGENT_AIPP_DIR:-$ROOT/configs/ascend310b/aipp}"
AIPP_DIR="$(readlink -f "$AIPP_DIR")"
ASCEND_PYTHON="${AGILE_AGENT_ASCEND_PYTHON:-/usr/local/miniconda3/envs/agileagent/bin/python}"
RESUME="${AGILE_AGENT_RESUME:-0}"

test -d "$ONNX_DIR" || { printf 'ONNX目录不存在：%s\n' "$ONNX_DIR" >&2; exit 1; }
test -f "$CONTEXT_BUILD_MANIFEST" || {
  printf 'Scene父构建清单不存在：%s\n' "$CONTEXT_BUILD_MANIFEST" >&2
  exit 1
}
test -x "$ASCEND_PYTHON" || { printf 'Ascend Python不存在：%s\n' "$ASCEND_PYTHON" >&2; exit 1; }
for name in base specialist; do
  test -f "$ONNX_DIR/$name.onnx" || { printf '缺少ONNX：%s.onnx\n' "$name" >&2; exit 1; }
done
for name in base_detector incremental_detector; do
  test -f "$AIPP_DIR/$name.cfg" || { printf '缺少AIPP：%s.cfg\n' "$name" >&2; exit 1; }
done

if test -e "$OUTPUT_DIR" && test -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)"; then
  if test "$RESUME" != "1"; then
    printf '候选输出目录非空，拒绝覆盖：%s\n' "$OUTPUT_DIR" >&2
    exit 1
  fi
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(readlink -f "$OUTPUT_DIR")"
MANIFEST="$OUTPUT_DIR/build-manifest.json"
test ! -e "$MANIFEST" || { printf '构建清单已存在，拒绝覆盖：%s\n' "$MANIFEST" >&2; exit 1; }

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export PATH="$(dirname "$ASCEND_PYTHON"):$PATH"
command -v atc >/dev/null || { printf 'CANN ATC不可用。\n' >&2; exit 1; }

build_one() {
  local model_id="$1"
  local onnx_name="$2"
  local output_prefix="$OUTPUT_DIR/$model_id"
  local log="$OUTPUT_DIR/atc_$model_id.log"
  local command_file="$OUTPUT_DIR/atc_$model_id.command.txt"
  local args=(
    "--model=$ONNX_DIR/$onnx_name.onnx"
    "--framework=5"
    "--output=$output_prefix"
    "--input_format=NCHW"
    "--input_shape=images:1,3,608,736"
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
  if test -e "$output_prefix.om"; then
    test "$RESUME" = "1" || { printf 'OM已存在：%s\n' "$output_prefix.om" >&2; exit 1; }
    test "$(cat "$command_file")" = "$expected_command" || {
      printf '续跑命令不匹配：%s\n' "$model_id" >&2
      exit 1
    }
    grep -q 'ATC run success' "$log" || {
      printf '续跑日志没有成功标记：%s\n' "$model_id" >&2
      exit 1
    }
    printf '复用已验证OM：%s\n' "$output_prefix.om"
    return
  fi
  test ! -e "$command_file" && test ! -e "$log" || {
    printf 'OM缺失但命令或日志已存在：%s\n' "$model_id" >&2
    exit 1
  }
  printf '%s\n' "$expected_command" >"$command_file"
  atc "${args[@]}" 2>&1 | tee "$log"
  test "${PIPESTATUS[0]}" = "0" && test -s "$output_prefix.om" || {
    printf 'ATC未成功生成OM：%s\n' "$model_id" >&2
    exit 1
  }
}

build_one base_detector base
build_one incremental_detector specialist

AGILE_AGENT_BUILD_ROOT="$ROOT" \
AGILE_AGENT_ONNX_DIR="$ONNX_DIR" \
AGILE_AGENT_AIPP_DIR="$AIPP_DIR" \
AGILE_AGENT_OUTPUT_DIR="$OUTPUT_DIR" \
AGILE_AGENT_CONTEXT_BUILD_MANIFEST="$CONTEXT_BUILD_MANIFEST" \
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
context_manifest_path = Path(os.environ["AGILE_AGENT_CONTEXT_BUILD_MANIFEST"])
context_manifest = json.loads(context_manifest_path.read_text(encoding="utf-8"))
context_rows = [
    row
    for row in (context_manifest.get("artifacts") or {}).values()
    if isinstance(row, dict) and row.get("role") == "context"
]
if len(context_rows) != 1:
    raise RuntimeError("Scene父构建清单必须恰好包含一个context资产")
context = context_rows[0]
for field in ("source_weight", "onnx", "aipp", "om", "atc_log"):
    entry = context.get(field) or {}
    path = Path(str(entry.get("path") or ""))
    if not path.is_file() or sha256(path) != entry.get("sha256"):
        raise RuntimeError(f"Scene父资产校验失败：{field}")

git_sha = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()
specs = {
    "base_detector": {
        "role": "base",
        "onnx_name": "base",
        "source": root / "models/production/incremental_detection/four_class_base_detector.pt",
        "class_count": 4,
    },
    "incremental_detector": {
        "role": "specialist",
        "onnx_name": "specialist",
        "source": root / "models/production/incremental_detection/incremental_detector.pt",
        "class_count": 2,
    },
}
artifacts = {}
for model_id, spec in specs.items():
    source = Path(spec["source"])
    onnx = onnx_dir / f"{spec['onnx_name']}.onnx"
    aipp = aipp_dir / f"{model_id}.cfg"
    om = output_dir / f"{model_id}.om"
    log = output_dir / f"atc_{model_id}.log"
    command_file = output_dir / f"atc_{model_id}.command.txt"
    for path in (source, onnx, aipp, om, log, command_file):
        if not path.is_file():
            raise FileNotFoundError(path)
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
            "shape": [1, 608, 736, 3],
        },
        "output_contract": "yolo26_e2e_v1",
        "postprocess_contract": {
            "max_det": 300,
            "class_count": int(spec["class_count"]),
            "outputs": {
                "detections": {"shape": [1, 300, 6], "dtype": "float32"}
            },
        },
    }
artifacts["scene_sensor_net"] = context
payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "git_sha": git_sha,
    "soc_version": "Ascend310B1",
    "cann_version": "7.0.RC1",
    "precision": "mixed_float16",
    "model_layout": "independent_yolo26_e2e_v1",
    "validated": False,
    "validation_basis": "competition_score_pending",
    "parent_context_build_manifest": {
        "path": str(context_manifest_path),
        "sha256": sha256(context_manifest_path),
    },
    "artifacts": artifacts,
}
manifest = output_dir / "build-manifest.json"
manifest.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({
    "manifest": str(manifest),
    "manifest_sha256": sha256(manifest),
    "om_sha256": {
        name: row["om"]["sha256"] for name, row in artifacts.items()
    },
}, ensure_ascii=False, indent=2))
PY
