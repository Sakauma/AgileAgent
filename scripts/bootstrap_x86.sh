#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[workbench,inference,dev]"
python -m fair_agent.cli doctor
python scripts/smoke_models.py

printf '\nAgileAgent 已准备就绪。使用以下命令启动界面：\n  .venv/bin/python -m fair_agent.cli refresh\n  .venv/bin/python -m fair_agent.cli serve\n'
