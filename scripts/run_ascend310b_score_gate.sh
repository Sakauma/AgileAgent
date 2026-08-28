#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  ./scripts/run_ascend310b_score_gate.sh CONFIG IMAGE_ROOT MIXED_SPLIT BASE_SPLIT OUTPUT_DIR

只在 127.0.0.1:8502 启动候选，依次冻结无标签预测、评分并执行
满分方法配置指定的预热与 batch 协议。正式 8501 必须在执行前后保持 ready。
EOF
}

if (( $# != 5 )); then
  usage >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD_CONFIG="${AGILE_AGENT_FULL_SCORE_METHOD:-$ROOT/configs/ascend310b/full_score_method.yaml}"
CONFIG="$(readlink -f "$1")"
IMAGE_ROOT="$(readlink -f "$2")"
MIXED_SPLIT="$(readlink -f "$3")"
BASE_SPLIT="$(readlink -f "$4")"
OUTPUT_DIR="$5"
PYTHON="${AGILE_AGENT_ASCEND_PYTHON:-/usr/local/miniconda3/envs/agileagent/bin/python}"

for path in "$CONFIG" "$MIXED_SPLIT" "$BASE_SPLIT" "$METHOD_CONFIG"; do
  test -f "$path" || { printf '缺少score gate输入：%s\n' "$path" >&2; exit 1; }
done
METHOD_CONFIG="$(readlink -f "$METHOD_CONFIG")"
test -d "$IMAGE_ROOT" || { printf '图像目录不存在：%s\n' "$IMAGE_ROOT" >&2; exit 1; }
test -x "$PYTHON" || { printf '板端Python不可执行：%s\n' "$PYTHON" >&2; exit 1; }
if test -e "$OUTPUT_DIR" && test -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)"; then
  printf 'score gate输出目录非空，拒绝覆盖：%s\n' "$OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(readlink -f "$OUTPUT_DIR")"

health_ready() {
  local url="$1"
  "$PYTHON" - "$url" <<'PY'
import json
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1] + "/api/health", timeout=3) as response:
    payload = json.load(response)
if payload.get("status") != "ready":
    raise SystemExit(1)
PY
}

port_is_free() {
  local port="$1"
  "$PYTHON" - "$port" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    probe.settimeout(1.0)
    if probe.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0:
        raise SystemExit(1)
PY
}

mapfile -t CONFIG_VALUES < <("$PYTHON" - "$CONFIG" "$METHOD_CONFIG" <<'PY'
import sys

import yaml

payload = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
method = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
if (
    not isinstance(method, dict)
    or method.get("schema_version") != 2
    or method.get("kind") != "ascend310b_full_score_method"
):
    raise RuntimeError("满分方法配置schema/kind非法")
target = method["target"]
benchmark = method["benchmark"]
performance = method["competition"]["performance_gate"]
image = benchmark["image_contract"]
if (
    benchmark["batch_probe_size"] != performance["batch_image_count"]
    or benchmark["batch_rounds"] != performance["batch_rounds"]
    or float(benchmark["target_batch_fps"]) != float(performance["aggregate_fps_min"])
    or benchmark.get("fps_calculation")
    != "total_frames_divided_by_total_elapsed_seconds"
    or benchmark.get("includes_result_persistence") is not True
    or performance.get("calculation")
    != "total_frames_divided_by_total_elapsed_seconds"
    or performance.get("includes_result_persistence") is not True
    or performance.get("required_components")
    != [
        "image_decode",
        "scene_model",
        "decision_model",
        "base_detector",
        "incremental_detector",
        "postprocess",
        "formal_result_write",
    ]
):
    raise RuntimeError("benchmark与competition performance gate不一致")
if target["candidate_port"] != 8502 or target["formal_port"] != 8501:
    raise RuntimeError("满分方法必须保护8501且只使用8502候选")
if (
    benchmark["base_url"] != f"http://127.0.0.1:{target['candidate_port']}"
    or benchmark["official_url"] != f"http://127.0.0.1:{target['formal_port']}"
):
    raise RuntimeError("benchmark URL与满分方法端口不一致")
ascend = payload["ascend_backend"]
if payload["runtime"]["server_port"] != target["candidate_port"]:
    raise RuntimeError("候选配置端口与满分方法不一致")
if (
    ascend["model_layout"] != method["export"]["model_layout"]
    or ascend["context_mode"] != method["runtime"]["context_mode"]
    or ascend["cann_version"] != target["cann_version"]
    or ascend.get("validation_candidate") is not True
    or ascend.get("validated") is not False
):
    raise RuntimeError("候选配置的布局/context/CANN/验证状态与满分方法不一致")
print(ascend["build_manifest"])
layout = ascend["model_layout"]
if layout == "independent_yolo26_e2e_v1":
    models = ascend["models"]
    base_rows = [
        row for key, row in models.items()
        if str(key).endswith("four_class_base_detector.pt")
    ]
    specialist_rows = [
        row for key, row in models.items()
        if str(key).endswith("incremental_detector.pt")
    ]
    if len(base_rows) != 1 or len(specialist_rows) != 1:
        raise RuntimeError("4+2独立布局必须各登记一个Base与Specialist模型")
    if any(
        row.get("output_contract") != "yolo26_e2e_v1"
        for row in (*base_rows, *specialist_rows)
    ):
        raise RuntimeError("4+2独立布局必须使用yolo26_e2e_v1输出契约")
    method_models = method["export"]["models"]
    old_map = method_models["base"]["class_map"]
    new_map = method_models["specialist"]["class_map"]
else:
    raise RuntimeError(f"score gate不支持模型布局：{layout}")
old_values = [old_map[key] for key in sorted(old_map, key=int)]
new_values = [new_map[key] for key in sorted(new_map, key=int)]
if not old_values or not new_values or set(old_values) & set(new_values):
    raise RuntimeError("score gate新旧类别映射非法")
print(",".join(str(value) for value in old_values))
print(",".join(str(value) for value in new_values))
print(target["candidate_port"])
print(target["formal_port"])
print(target["cann_version"])
method_confidence = float(
    method["threshold_search"]["scoring_request_confidence"]
)
configured_confidence = float(payload["inference"]["confidence_default"])
scoring_confidence = min(method_confidence, configured_confidence)
if not (
    float(payload["inference"]["confidence_min"])
    <= scoring_confidence
    <= float(payload["inference"]["confidence_max"])
):
    raise RuntimeError("score gate请求置信度不在候选配置范围内")
print(scoring_confidence)
print(benchmark["warmup_requests"])
print(benchmark["batch_probe_size"])
print(benchmark["batch_rounds"])
print(benchmark["target_batch_fps"])
print("1" if benchmark["encoded"] else "0")
print(image["root_glob"])
print(image["width"])
print(image["height"])
print(image["bit_depth"])
print(",".join(str(value) for value in image["color_types"]))
PY
)
if (( ${#CONFIG_VALUES[@]} != 17 )); then
  printf '无法从候选配置和满分方法读取完整score gate契约。\n' >&2
  exit 1
fi
BUILD_MANIFEST="${CONFIG_VALUES[0]}"
OLD_CLASS_IDS="${CONFIG_VALUES[1]}"
NEW_CLASS_IDS="${CONFIG_VALUES[2]}"
CANDIDATE_PORT="${CONFIG_VALUES[3]}"
FORMAL_PORT="${CONFIG_VALUES[4]}"
CANN_VERSION="${CONFIG_VALUES[5]}"
SCORING_CONFIDENCE="${CONFIG_VALUES[6]}"
WARMUP_REQUESTS="${CONFIG_VALUES[7]}"
BATCH_PROBE_SIZE="${CONFIG_VALUES[8]}"
BATCH_ROUNDS="${CONFIG_VALUES[9]}"
TARGET_BATCH_FPS="${CONFIG_VALUES[10]}"
ENCODED="${CONFIG_VALUES[11]}"
IMAGE_GLOB="${CONFIG_VALUES[12]}"
IMAGE_WIDTH="${CONFIG_VALUES[13]}"
IMAGE_HEIGHT="${CONFIG_VALUES[14]}"
IMAGE_BIT_DEPTH="${CONFIG_VALUES[15]}"
IMAGE_COLOR_TYPES="${CONFIG_VALUES[16]}"
OFFICIAL_URL="http://127.0.0.1:$FORMAL_PORT"
CANDIDATE_URL="http://127.0.0.1:$CANDIDATE_PORT"

health_ready "$OFFICIAL_URL" || { printf '正式8501不是ready，停止score gate。\n' >&2; exit 1; }
if ! port_is_free "$CANDIDATE_PORT"; then
  printf '候选8502已被占用，拒绝复用或停止未知进程。\n' >&2
  exit 1
fi

test -f "$BUILD_MANIFEST" || { printf '候选build manifest不存在：%s\n' "$BUILD_MANIFEST" >&2; exit 1; }
EXPECTED_IMAGES="$("$PYTHON" - "$IMAGE_ROOT" "$IMAGE_GLOB" "$IMAGE_WIDTH" "$IMAGE_HEIGHT" "$IMAGE_BIT_DEPTH" "$IMAGE_COLOR_TYPES" <<'PY'
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
pattern = sys.argv[2]
width = int(sys.argv[3])
height = int(sys.argv[4])
bit_depth = int(sys.argv[5])
color_types = {int(value) for value in sys.argv[6].split(",")}
if pattern != "*.png":
    raise RuntimeError("score gate只支持根目录*.png输入契约")
paths = sorted(path for path in root.glob(pattern) if path.is_file())
if len({path.stem for path in paths}) != len(paths):
    raise RuntimeError("评分PNG的stem必须唯一")
for path in paths:
    header = path.read_bytes()[:29]
    if len(header) < 29 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"不是有效PNG：{path}")
    actual_width, actual_height, actual_depth, actual_color = struct.unpack(
        ">IIBB", header[16:26]
    )
    if (
        actual_width != width
        or actual_height != height
        or actual_depth != bit_depth
        or actual_color not in color_types
    ):
        raise RuntimeError(
            f"PNG输入契约不符：{path.name}:"
            f"{actual_width}x{actual_height}/bit={actual_depth}/color={actual_color}"
        )
print(len(paths))
PY
)"
if [[ ! "$EXPECTED_IMAGES" =~ ^[0-9]+$ ]] || (( EXPECTED_IMAGES <= 0 )); then
  printf '无法确定评分PNG数量：%s\n' "$IMAGE_ROOT" >&2
  exit 1
fi

CANDIDATE_PID=""
cleanup() {
  local status="$1"
  trap - EXIT INT TERM
  if [[ -n "$CANDIDATE_PID" ]] && kill -0 "$CANDIDATE_PID" 2>/dev/null; then
    command_line="$(tr '\0' ' ' < "/proc/$CANDIDATE_PID/cmdline")"
    if [[ "$command_line" == *"uvicorn fair_agent.web.app:app"* && "$command_line" == *"--port $CANDIDATE_PORT"* ]]; then
      kill "$CANDIDATE_PID"
      wait "$CANDIDATE_PID" 2>/dev/null || true
    else
      printf '拒绝停止不匹配的候选进程：PID=%s CMD=%s\n' "$CANDIDATE_PID" "$command_line" >&2
      status=1
    fi
  fi
  if ! health_ready "$OFFICIAL_URL"; then
    printf 'score gate结束时正式8501不是ready。\n' >&2
    status=1
  fi
  exit "$status"
}
trap 'cleanup $?' EXIT
trap 'exit 130' INT TERM

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
CANN_ROOT="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
CANN_RESOLVED="$(readlink -f "$CANN_ROOT")"
CANN_VERSION_EVIDENCE="$(
  shopt -s nullglob
  version_files=(
    "$CANN_RESOLVED/version.info"
    "$CANN_RESOLVED"/*-linux/ascend_toolkit_install.info
  )
  for version_file in "${version_files[@]}"; do
    tr '\n' ' ' <"$version_file"
    printf '\n'
  done
  atc --version 2>&1 || true
)"
if [[ "$CANN_VERSION_EVIDENCE" != *"$CANN_VERSION"* ]]; then
  printf 'score gate无法确认要求的CANN版本%s；root=%s evidence=%s\n' \
    "$CANN_VERSION" "$CANN_RESOLVED" "$CANN_VERSION_EVIDENCE" >&2
  exit 1
fi
cd "$ROOT"

# Freeze in a short-lived isolated engine first. Starting the HTTP candidate
# afterwards avoids loading the detector OMs twice during prediction.
ENCODED_FLAG=()
if [[ "$ENCODED" == "1" ]]; then
  ENCODED_FLAG=(--encoded)
fi
AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION=1 \
"$PYTHON" tools/98_freeze_ascend_predictions.py \
  --config "$CONFIG" \
  --image-root "$IMAGE_ROOT" \
  --output "$OUTPUT_DIR/predictions.jsonl" \
  --summary "$OUTPUT_DIR/predictions-summary.json" \
  --confidence "$SCORING_CONFIDENCE" \
  --expected-images "$EXPECTED_IMAGES" \
  "${ENCODED_FLAG[@]}"

"$PYTHON" tools/94_score_ascend_agent.py \
  --predictions "$OUTPUT_DIR/predictions.jsonl" \
  --method-config "$METHOD_CONFIG" \
  --mixed-split "$MIXED_SPLIT" \
  --base-split "$BASE_SPLIT" \
  --expected-images "$EXPECTED_IMAGES" \
  --old-class-ids "$OLD_CLASS_IDS" \
  --new-class-ids "$NEW_CLASS_IDS" \
  --output "$OUTPUT_DIR/score.json"

AGILE_AGENT_CONFIG="$CONFIG" \
AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION=1 \
"$PYTHON" -m uvicorn fair_agent.web.app:app \
  --host 127.0.0.1 --port "$CANDIDATE_PORT" --no-access-log \
  >"$OUTPUT_DIR/candidate.log" 2>&1 &
CANDIDATE_PID=$!

for _ in $(seq 1 120); do
  if health_ready "$CANDIDATE_URL" 2>/dev/null; then
    break
  fi
  kill -0 "$CANDIDATE_PID" 2>/dev/null || {
    printf '候选进程提前退出，查看%s\n' "$OUTPUT_DIR/candidate.log" >&2
    exit 1
  }
  sleep 1
done
health_ready "$CANDIDATE_URL" || { printf '候选8502未在120秒内ready。\n' >&2; exit 1; }

"$PYTHON" tools/97_benchmark_ascend_api.py \
  --base-url "$CANDIDATE_URL" \
  --image-root "$IMAGE_ROOT" \
  --output "$OUTPUT_DIR/benchmark.json" \
  --confidence "$SCORING_CONFIDENCE" \
  --warmup-requests "$WARMUP_REQUESTS" \
  --rounds 0 \
  --skip-single-requests \
  --gate-profile score \
  --batch-probe-size "$BATCH_PROBE_SIZE" \
  --batch-rounds "$BATCH_ROUNDS" \
  --target-batch-fps "$TARGET_BATCH_FPS" \
  --expected-images "$EXPECTED_IMAGES" \
  --config "$CONFIG" \
  --build-manifest "$BUILD_MANIFEST"

printf 'Ascend310B满分score gate通过：%s\n' "$OUTPUT_DIR"
