#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MACHINE_ARCH_RAW="$(uname -m)"
MACHINE_ARCH="${MACHINE_ARCH_RAW,,}"
case "${MACHINE_ARCH}" in
  x86_64|amd64|i386|i486|i586|i686)
    AGENT_PLATFORM=x86
    ;;
  aarch64|arm64|arm|armv7l|armv8l)
    AGENT_PLATFORM=arm
    ;;
  *)
    printf '不支持的设备架构：%s（仅支持 x86/x86_64 与 ARM/aarch64）。\n' "${MACHINE_ARCH_RAW}" >&2
    exit 1
    ;;
esac

if [[ -n "${AGILE_AGENT_PYTHON:-}" ]]; then
  AGENT_PYTHON="${AGILE_AGENT_PYTHON}"
elif [[ "${AGENT_PLATFORM}" == x86 && -f "${ROOT_DIR}/.agent-python" ]]; then
  IFS= read -r AGENT_PYTHON < "${ROOT_DIR}/.agent-python"
elif [[ "${AGENT_PLATFORM}" == arm ]]; then
  ASCEND_ENV="${AGILE_AGENT_ASCEND_ENV:-/usr/local/miniconda3/envs/agileagent}"
  AGENT_PYTHON="${ASCEND_ENV}/bin/python"
else
  AGENT_PYTHON="${ROOT_DIR}/.venv/bin/python"
fi
if [[ ! -x "${AGENT_PYTHON}" ]]; then
  if [[ "${AGENT_PLATFORM}" == arm ]]; then
    printf '未找到 ARM/Ascend Python：%s\n请配置 AGILE_AGENT_ASCEND_ENV 或 AGILE_AGENT_PYTHON。\n' "${AGENT_PYTHON}" >&2
  else
    printf '未找到已配置的 Python：%s\n请先运行 scripts/bootstrap_x86.sh。\n' "${AGENT_PYTHON}" >&2
  fi
  exit 1
fi
export AGILE_AGENT_PYTHON="${AGENT_PYTHON}"

if [[ "${AGENT_PLATFORM}" == arm ]]; then
  CANN_ENV_SCRIPT="${AGILE_AGENT_CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
  if [[ ! -f "${CANN_ENV_SCRIPT}" ]]; then
    printf '未找到 CANN 环境脚本：%s\n' "${CANN_ENV_SCRIPT}" >&2
    exit 1
  fi
  set +u
  # shellcheck disable=SC1090
  source "${CANN_ENV_SCRIPT}"
  set -u
fi

"${AGENT_PYTHON}" -m fair_agent.cli doctor --quiet

if [[ "${1:-}" == "--cli" ]]; then
  exec "${AGENT_PYTHON}" -m fair_agent.cli console
fi

if [[ $# -gt 0 ]]; then
  printf '未知参数：%s\n用法：scripts/start_agent.sh [--cli]\n' "$1" >&2
  exit 2
fi

"${AGENT_PYTHON}" -m fair_agent.cli refresh
"${AGENT_PYTHON}" -m fair_agent.cli decide

AGENT_HOST="$("${AGENT_PYTHON}" -m fair_agent.cli config get runtime.server_host)"
AGENT_PORT="$("${AGENT_PYTHON}" -m fair_agent.cli config get runtime.server_port)"
printf '\n正在启动灵动Agent工作台：http://%s:%s\n按 Ctrl+C 可停止服务。\n' "${AGENT_HOST}" "${AGENT_PORT}"
exec "${AGENT_PYTHON}" -m fair_agent.cli serve
