#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  materialize_ascend310b_full_score_release.sh [--verify-existing]

从仓库内的预构建模型包物化正式 Ascend310B1 release。
不训练、不运行 ATC、不升级 CANN，也不修改 8501/8502 上的进程。

固定目标：
  /home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1
EOF
}

VERIFY_EXISTING=0
if (( $# > 1 )); then
  usage >&2
  exit 2
fi
if (( $# == 1 )); then
  if [[ "$1" != "--verify-existing" ]]; then
    usage >&2
    exit 2
  fi
  VERIFY_EXISTING=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RELEASE_ID=20260824-4plus2-yolo26-runtime-calibration-v1
PACKAGE_ROOT="${REPO_ROOT}/models/ascend310b/full-score/${RELEASE_ID}"
RELEASE_PARENT=/home/HwHiAiUser/agileagent/releases
RELEASE_ROOT="${RELEASE_PARENT}/${RELEASE_ID}"
PYTHON=/usr/local/miniconda3/envs/agileagent/bin/python
CONFIG="${RELEASE_ROOT}/configs/agent_pipeline_ascend310b.yaml"
STAGING=""

cleanup() {
  local status="$?"
  if [[ -n "$STAGING" && -d "$STAGING" ]]; then
    case "$STAGING" in
      "${RELEASE_PARENT}"/.materialize-full-score-*) rm -rf -- "$STAGING" ;;
      *) printf '拒绝清理非预期临时目录：%s\n' "$STAGING" >&2 ;;
    esac
  fi
  exit "$status"
}
trap cleanup EXIT

test -d "$PACKAGE_ROOT" || { printf '预构建模型包不存在：%s\n' "$PACKAGE_ROOT" >&2; exit 1; }
test -f "$PACKAGE_ROOT/SHA256SUMS" || { printf '模型包缺少SHA256SUMS。\n' >&2; exit 1; }
test -x "$PYTHON" || { printf '板端Python不可执行：%s\n' "$PYTHON" >&2; exit 1; }
test -f /usr/local/Ascend/ascend-toolkit/set_env.sh || {
  printf '板端CANN环境不存在。\n' >&2
  exit 1
}

(
  cd "$PACKAGE_ROOT"
  sha256sum -c SHA256SUMS
)

verify_release() {
  AGILE_AGENT_CONFIG="$CONFIG" \
    "$PYTHON" "$RELEASE_ROOT/src/tools/95_verify_ascend_release.py" \
      --config "$CONFIG" --require-validation
}

if [[ -e "$RELEASE_ROOT" ]]; then
  if (( VERIFY_EXISTING == 0 )); then
    printf '目标release已存在：%s\n' "$RELEASE_ROOT" >&2
    printf '如需只读复核，请使用 --verify-existing。\n' >&2
    exit 1
  fi
  verify_release
  printf '已有正式release验证通过：%s\n' "$RELEASE_ROOT"
  exit 0
fi

mkdir -p "$RELEASE_PARENT"
STAGING="$(mktemp -d "${RELEASE_PARENT}/.materialize-full-score-XXXXXX")"
cp -a "$PACKAGE_ROOT/." "$STAGING/"
mkdir -p "$STAGING/src"

# 只复制Git跟踪的源码。板端运行不需要x86 PT，也不在src/内
# 重复嵌套本模型包；发布所需OM/权重/ONNX已在release顶层。
while IFS= read -r -d '' relative; do
  case "$relative" in
    models/*.pt|models/*/*.pt|models/*/*/*.pt|models/ascend310b/*) continue ;;
  esac
  source_path="${REPO_ROOT}/${relative}"
  target_path="${STAGING}/src/${relative}"
  if [[ -f "$source_path" ]]; then
    mkdir -p "$(dirname "$target_path")"
    cp -a "$source_path" "$target_path"
  fi
done < <(git -C "$REPO_ROOT" ls-files -z)

mv -- "$STAGING" "$RELEASE_ROOT"
STAGING=""
verify_release

printf '正式满分release已物化：%s\n' "$RELEASE_ROOT"
printf '本步未启动服务；按部署文档选择直接8501或主/回滚双实例。\n'
