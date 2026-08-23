#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="${AGILE_AGENT_ASCEND_RELEASE:-/home/HwHiAiUser/agileagent/releases/20260824-4plus2-yolo26-runtime-calibration-v1}"
SOURCE_ROOT="${RELEASE_ROOT}/src"
CONDA_ENV="${AGILE_AGENT_ASCEND_ENV:-/usr/local/miniconda3/envs/agileagent}"
PYTHON="${CONDA_ENV}/bin/python"
CONFIG="${AGILE_AGENT_CONFIG:-${SOURCE_ROOT}/configs/agent_pipeline_ascend310b.yaml}"
PID_FILE="${AGILE_AGENT_ASCEND_PID_FILE:-${RELEASE_ROOT}/agent-web.pid}"
PORT="${AGILE_AGENT_ASCEND_PORT:-8501}"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Ascend命名环境不存在或Python不可执行：%s\n' "${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${CONDA_ENV}/conda-meta/history" ]]; then
  printf 'Ascend路径不是有效Conda环境：%s\n' "${CONDA_ENV}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  printf 'Ascend配置不存在：%s\n' "${CONFIG}" >&2
  exit 1
fi
if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  printf 'Ascend服务端口非法：%s\n' "${PORT}" >&2
  exit 1
fi
if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if [[ "${OLD_PID}" =~ ^[0-9]+$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    printf 'Ascend Agent已经运行，PID=%s\n' "${OLD_PID}"
    exit 0
  fi
  rm -f -- "${PID_FILE}"
fi

# Load only the existing CANN runtime.  This script never installs or upgrades
# CANN, the driver or firmware.
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

export AGILE_AGENT_CONFIG="${CONFIG}"
cd "${SOURCE_ROOT}"
printf '%s\n' "$$" > "${PID_FILE}"
exec "${PYTHON}" -m uvicorn fair_agent.web.app:app \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --no-access-log
