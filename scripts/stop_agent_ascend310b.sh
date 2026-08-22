#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="${AGILE_AGENT_ASCEND_RELEASE:-/home/HwHiAiUser/agileagent/releases/20260823-4plus2-yolo26-content-gate-v2}"
PID_FILE="${AGILE_AGENT_ASCEND_PID_FILE:-${RELEASE_ROOT}/agent-web.pid}"

if [[ ! -f "${PID_FILE}" ]]; then
  printf '服务未登记PID：%s\n' "${PID_FILE}"
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if [[ "${PID}" =~ ^[0-9]+$ ]] && kill -0 "${PID}" 2>/dev/null; then
  COMMAND="$(tr '\0' ' ' < "/proc/${PID}/cmdline")"
  if [[ "${COMMAND}" != *"uvicorn fair_agent.web.app:app"* ]]; then
    printf '拒绝停止PID文件指向的非Agent进程：PID=%s CMD=%s\n' "${PID}" "${COMMAND}" >&2
    exit 1
  fi
  kill "${PID}"
  for _ in $(seq 1 40); do
    kill -0 "${PID}" 2>/dev/null || break
    sleep 0.25
  done
  if kill -0 "${PID}" 2>/dev/null; then
    printf 'Agent进程未在10秒内退出：PID=%s\n' "${PID}" >&2
    exit 1
  fi
fi
rm -f -- "${PID_FILE}"
