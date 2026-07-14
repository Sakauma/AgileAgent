#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${AGILE_AGENT_PYTHON:-}" ]]; then
  AGENT_PYTHON="${AGILE_AGENT_PYTHON}"
elif [[ -f "${ROOT_DIR}/.agent-python" ]]; then
  IFS= read -r AGENT_PYTHON < "${ROOT_DIR}/.agent-python"
else
  AGENT_PYTHON="${ROOT_DIR}/.venv/bin/python"
fi
if [[ ! -x "${AGENT_PYTHON}" ]]; then
  printf '未找到已配置的 Python：%s\n请先运行 scripts/bootstrap_x86.sh。\n' "${AGENT_PYTHON}" >&2
  exit 1
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
