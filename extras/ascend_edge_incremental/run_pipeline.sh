#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TRAINING_PYTHON="${AGILE_EDGE_TRAINING_PYTHON:-${HOME}/agileagent/envs/agileagent_train/bin/python}"
PRODUCTION_PYTHON="${AGILE_EDGE_PRODUCTION_PYTHON:-/usr/local/miniconda3/envs/agileagent/bin/python}"
CANN_ENV="${AGILE_EDGE_CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  printf '板端训练入口只允许在 aarch64 Ascend310B 上执行。\n' >&2
  exit 2
fi
if [[ ! -x "${TRAINING_PYTHON}" ]]; then
  printf '未找到独立训练 Python：%s\n请先运行 bootstrap_env.sh。\n' "${TRAINING_PYTHON}" >&2
  exit 2
fi
if [[ ! -x "${PRODUCTION_PYTHON}" || ! -f "${CANN_ENV}" ]]; then
  printf 'production Python 或 CANN 环境不可用。\n' >&2
  exit 2
fi

source "${CANN_ENV}"
cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  exec "${PRODUCTION_PYTHON}" -m extras.ascend_edge_incremental.workflow \
    run \
    --repo-root "${REPO_ROOT}" \
    --training-python "${TRAINING_PYTHON}" \
    --production-python "${PRODUCTION_PYTHON}" \
    "$@"
