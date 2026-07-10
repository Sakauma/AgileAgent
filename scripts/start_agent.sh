#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

AGENT_PYTHON="${AGILE_AGENT_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${AGENT_PYTHON}" ]]; then
  printf '未找到已配置的 Python：%s\n请先运行 scripts/bootstrap_x86.sh。\n' "${AGENT_PYTHON}" >&2
  exit 1
fi

"${AGENT_PYTHON}" -m fair_agent.cli doctor
"${AGENT_PYTHON}" -m fair_agent.cli refresh
"${AGENT_PYTHON}" -m fair_agent.cli decide

printf '\n正在启动 AgileAgent 工作台：http://127.0.0.1:8501\n按 Ctrl+C 可停止服务。\n'
exec "${AGENT_PYTHON}" -m fair_agent.cli serve
