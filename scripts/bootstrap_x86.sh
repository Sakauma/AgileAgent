#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
if [[ ! -x .venv/bin/python ]]; then
  "${PYTHON_BIN}" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
if ! python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  python -m pip install --upgrade --force-reinstall torch torchvision --index-url "${PYTORCH_INDEX_URL}"
fi
python -m pip install -e ".[workbench,inference,dev]"
python -m fair_agent.cli doctor
python scripts/smoke_models.py

printf '\n环境配置完成。日常启动请运行：./scripts/start_agent.sh\n'
