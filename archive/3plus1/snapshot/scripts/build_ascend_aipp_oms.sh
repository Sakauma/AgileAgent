#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  ./scripts/build_ascend_aipp_oms.sh ONNX_DIR OUTPUT_DIR [MANIFEST]

在安装 CANN 7.0.RC1 的 Ascend310B1 主机上，从固定 ONNX 和受控 AIPP
配置构建三个候选 OM。OUTPUT_DIR 必须不存在或为空；脚本不会覆盖候选产物。
EOF
}

if (( $# < 2 || $# > 3 )); then
  usage >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_DIR="$(readlink -f "$1")"
OUTPUT_DIR="$2"
MANIFEST="${3:-$OUTPUT_DIR/build-manifest.json}"
AIPP_DIR="${AGILE_AGENT_AIPP_DIR:-$ROOT/configs/ascend310b/aipp}"
AIPP_DIR="$(readlink -f "$AIPP_DIR")"
ASCEND_PYTHON="${AGILE_AGENT_ASCEND_PYTHON:-/usr/local/miniconda3/envs/agileagent/bin/python}"
RESUME="${AGILE_AGENT_RESUME:-0}"

test -d "$ONNX_DIR" || { printf 'ONNX目录不存在：%s\n' "$ONNX_DIR" >&2; exit 1; }
test -x "$ASCEND_PYTHON" || { printf '正式Python不存在：%s\n' "$ASCEND_PYTHON" >&2; exit 1; }
for name in base_detector incremental_detector scene_sensor_net; do
  test -f "$ONNX_DIR/$name.onnx" || {
    printf '缺少固定ONNX：%s\n' "$ONNX_DIR/$name.onnx" >&2
    exit 1
  }
  test -f "$AIPP_DIR/$name.cfg" || {
    printf '缺少受控AIPP配置：%s\n' "$AIPP_DIR/$name.cfg" >&2
    exit 1
  }
done

if test -e "$OUTPUT_DIR" && test -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)"; then
  if test "$RESUME" != "1"; then
    printf '候选输出目录非空，拒绝覆盖；如需校验后续跑请设置AGILE_AGENT_RESUME=1：%s\n' "$OUTPUT_DIR" >&2
    exit 1
  fi
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(readlink -f "$OUTPUT_DIR")"
mkdir -p "$(dirname "$MANIFEST")"
MANIFEST="$(readlink -f "$MANIFEST")"
test ! -e "$MANIFEST" || { printf '构建清单已存在，拒绝覆盖：%s\n' "$MANIFEST" >&2; exit 1; }

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
  if test -e "$output_prefix.om"; then
    if test "$RESUME" != "1"; then
      printf '候选OM已存在，拒绝覆盖：%s\n' "$output_prefix.om" >&2
      exit 1
    fi
    test -s "$command_file" || { printf '续跑缺少命令记录：%s\n' "$command_file" >&2; exit 1; }
    test -s "$log" || { printf '续跑缺少ATC日志：%s\n' "$log" >&2; exit 1; }
    test "$(cat "$command_file")" = "$expected_command" || {
      printf '续跑命令与已有产物不一致：%s\n' "$model_id" >&2
      exit 1
    }
    grep -q 'ATC run success' "$log" || {
      printf '续跑拒绝复用未成功的OM：%s\n' "$output_prefix.om" >&2
      exit 1
    }
    printf '已校验并复用候选OM：%s\n' "$output_prefix.om"
    return
  fi
  test ! -e "$command_file" && test ! -e "$log" || {
    printf '候选命令或日志已存在但OM缺失，拒绝覆盖：%s\n' "$model_id" >&2
    exit 1
  }
  printf '%s\n' "$expected_command" >"$command_file"
  atc "${args[@]}" 2>&1 | tee "$log"
  test -s "$output_prefix.om" || { printf 'ATC未生成OM：%s\n' "$output_prefix.om" >&2; exit 1; }
}

build_one base_detector 1,3,736,896
build_one incremental_detector 1,3,512,640
build_one scene_sensor_net 1,3,160,160

AGILE_AGENT_BUILD_ROOT="$ROOT" \
AGILE_AGENT_ONNX_DIR="$ONNX_DIR" \
AGILE_AGENT_AIPP_DIR="$AIPP_DIR" \
AGILE_AGENT_OUTPUT_DIR="$OUTPUT_DIR" \
AGILE_AGENT_MANIFEST="$MANIFEST" \
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
manifest_path = Path(os.environ["AGILE_AGENT_MANIFEST"])
try:
    git_sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
except (OSError, subprocess.CalledProcessError):
    git_sha = os.environ.get("AGILE_AGENT_GIT_SHA", "unknown")

specs = {
    "base_detector": {
        "role": "base",
        "source": root / "models/production/incremental_detection/three_class_base_detector.pt",
        "shape": [1, 736, 896, 3],
    },
    "incremental_detector": {
        "role": "specialist",
        "source": root / "models/production/incremental_detection/incremental_detector.pt",
        "shape": [1, 512, 640, 3],
    },
    "scene_sensor_net": {
        "role": "context",
        "source": root / "models/context/scene_sensor_net.pt",
        "shape": [1, 160, 160, 3],
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
            "shape": spec["shape"],
        },
    }

payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "git_sha": git_sha,
    "soc_version": "Ascend310B1",
    "cann_version": "7.0.RC1",
    "precision": "mixed_float16",
    "validated": False,
    "artifacts": artifacts,
}
manifest_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({
    "manifest": str(manifest_path),
    "manifest_sha256": sha256(manifest_path),
    "om_sha256": {name: row["om"]["sha256"] for name, row in artifacts.items()},
}, ensure_ascii=False, indent=2))
PY
