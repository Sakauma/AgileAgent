#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  printf '仅支持 x86-64 Linux/WSL。\n' >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf '未检测到 NVIDIA 驱动工具 nvidia-smi。\n' >&2
  exit 1
fi

python_supported() {
  "$1" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' >/dev/null 2>&1
}

PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0+cu128}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0+cu128}"
if [[ -x .venv/bin/python ]]; then
  if ! python_supported .venv/bin/python || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    printf '现有 .venv 不完整或 Python 版本不受支持。请移走该目录后重新运行配置脚本。\n' >&2
    exit 1
  fi
elif [[ -n "${PYTHON_BIN:-}" ]]; then
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 || ! python_supported "${PYTHON_BIN}"; then
    printf 'PYTHON_BIN 必须指向可用的 Python 3.10-3.12：%s\n' "${PYTHON_BIN}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m venv .venv
elif command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 --seed .venv
else
  for candidate in python3.12 python3.11 python3.10; do
    if command -v "${candidate}" >/dev/null 2>&1 && python_supported "${candidate}"; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    printf '未找到 Python 3.10-3.12。请先安装兼容版本，或通过 PYTHON_BIN 指定。\n' >&2
    exit 1
  fi
  if ! "${PYTHON_BIN}" -m venv .venv; then
    printf '无法创建虚拟环境。请为 %s 安装 venv/ensurepip，或安装 uv 后重试。\n' "${PYTHON_BIN}" >&2
    exit 1
  fi
fi
source .venv/bin/activate
python -m pip install --upgrade pip
if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.__version__ == '${TORCH_VERSION}' else 1)" >/dev/null 2>&1; then
  python -m pip install --upgrade --force-reinstall "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${PYTORCH_INDEX_URL}"
fi
python -m pip install -c constraints-agent.txt -e ".[workbench,inference,dev]"
python -m fair_agent.cli doctor
python scripts/smoke_models.py

printf '\n环境配置完成。日常启动请运行：./scripts/start_agent.sh\n'
