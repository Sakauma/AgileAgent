#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PRODUCTION_PYTHON="${AGILE_EDGE_PRODUCTION_PYTHON:-/usr/local/miniconda3/envs/agileagent/bin/python}"
TRAINING_PYTHON="${AGILE_EDGE_TRAINING_PYTHON:-}"
CANN_ENV="${AGILE_EDGE_CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

if [[ -z "${TRAINING_PYTHON}" ]]; then
  for candidate in \
    "${HOME}/agileagent/envs/agileagent_train/bin/python" \
    "${HOME}/agileagent/edge_incremental_training_20260825/env/bin/python"; do
    if [[ -x "${candidate}" ]]; then
      TRAINING_PYTHON="${candidate}"
      break
    fi
  done
fi

usage() {
  printf '%s\n' \
    '用法：' \
    '  ./scripts/run_ascend310b_incremental_demo.sh /path/to/datasets_r2_inc_train' \
    '  ./scripts/run_ascend310b_incremental_demo.sh --incremental-data DIR [其他选项]' \
    '' \
    '在 Ascend310B 断网环境一键执行当前 4->4+2 增量学习演示。' \
    '自动完成数据审计、NPU 训练、dev 选择、lock 精度、ONNX/OM、' \
    '隔离演示部署和完整图像推理 FPS 验收。'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ "$(uname -m)" != "aarch64" ]]; then
  printf '该演示入口只允许在 aarch64 Ascend310B 板端执行。\n' >&2
  exit 2
fi
if [[ ! -x "${PRODUCTION_PYTHON}" || ! -x "${TRAINING_PYTHON}" ]]; then
  printf '未找到 production 或独立训练 Python。\n' >&2
  printf '训练环境应预先离线准备在：%s\n' "${TRAINING_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${CANN_ENV}" ]]; then
  printf '未找到 CANN 环境脚本：%s\n' "${CANN_ENV}" >&2
  exit 2
fi

set +u
# shellcheck disable=SC1090
source "${CANN_ENV}"
set -u

export PIP_NO_INDEX=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export YOLO_CONFIG_DIR="${REPO_ROOT}/runs/yolo_config_offline"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

cd "${REPO_ROOT}"
plan_only=false
for argument in "$@"; do
  if [[ "${argument}" == "--plan-only" ]]; then
    plan_only=true
    break
  fi
done
if [[ "${plan_only}" == false ]]; then
  "${SCRIPT_DIR}/stop_agent_ascend310b.sh"
fi

arguments=("$@")
if [[ "${1}" != --* ]]; then
  arguments=(--incremental-data "${1}" "${@:2}")
fi
printf '\n◆ Ascend310B 离线 4->4+2 增量学习演示\n'
printf '  网络：禁用    训练设备：npu:0    production：不改写\n\n'
printf '  首次 NPU 图编译约需 3 分钟；CANN custom vendor 权限 traceback 为已知非致命警告。\n\n'
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  exec "${PRODUCTION_PYTHON}" -m extras.ascend_edge_incremental.demo \
    --training-python "${TRAINING_PYTHON}" \
    --production-python "${PRODUCTION_PYTHON}" \
    "${arguments[@]}"
