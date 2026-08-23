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

PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1+cu124}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1+cu124}"
AGENT_PYTHON=""

if [[ -n "${AGILE_AGENT_PYTHON:-}" ]]; then
  if [[ ! -x "${AGILE_AGENT_PYTHON}" ]] || ! python_supported "${AGILE_AGENT_PYTHON}" || ! "${AGILE_AGENT_PYTHON}" -m pip --version >/dev/null 2>&1; then
    printf 'AGILE_AGENT_PYTHON 必须指向带 pip 的 Python 3.10-3.12：%s\n' "${AGILE_AGENT_PYTHON}" >&2
    exit 1
  fi
  AGENT_PYTHON="${AGILE_AGENT_PYTHON}"
elif [[ -x .venv/bin/python ]]; then
  if ! python_supported .venv/bin/python || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    printf '现有 .venv 不完整或 Python 版本不受支持。请移走该目录后重新运行配置脚本。\n' >&2
    exit 1
  fi
  AGENT_PYTHON="${ROOT_DIR}/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]] && python_supported "${VIRTUAL_ENV}/bin/python"; then
  AGENT_PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]] && python_supported "${CONDA_PREFIX}/bin/python"; then
  AGENT_PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -n "${PYTHON_BIN:-}" ]]; then
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 || ! python_supported "${PYTHON_BIN}"; then
    printf 'PYTHON_BIN 必须指向可用的 Python 3.10-3.12：%s\n' "${PYTHON_BIN}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m venv .venv
  AGENT_PYTHON="${ROOT_DIR}/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  UV_PYTHON=""
  for candidate in python3.12 python3.11 python3.10; do
    if command -v "${candidate}" >/dev/null 2>&1 && python_supported "${candidate}"; then
      UV_PYTHON="$(command -v "${candidate}")"
      break
    fi
  done
  uv venv --python "${UV_PYTHON:-3.12}" --seed .venv
  AGENT_PYTHON="${ROOT_DIR}/.venv/bin/python"
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
  AGENT_PYTHON="${ROOT_DIR}/.venv/bin/python"
fi

# Persist the selected interpreter before dependency/model checks so a later
# diagnostic failure does not make start_agent.sh fall back to .venv.
printf '%s\n' "${AGENT_PYTHON}" > .agent-python

torch_stack_compatible() {
  "${AGENT_PYTHON}" - <<'PY' >/dev/null 2>&1
import re
import torch
import torchvision

match = re.match(r"(\d+)\.(\d+)", torch.__version__)
if match is None or tuple(map(int, match.groups())) < (2, 0):
    raise SystemExit(1)
if not torch.cuda.is_available() or not torch.version.cuda:
    raise SystemExit(1)

boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cuda")
scores = torch.tensor([1.0], device="cuda")
torchvision.ops.nms(boxes, scores, 0.5)
PY
}

dependencies_compatible() {
  "${AGENT_PYTHON}" - <<'PY' >/dev/null 2>&1
import re

from importlib import import_module

requirements = {
    "yaml": (6, 0),
    "PIL": (9, 5),
    "pandas": (2, 0),
    "starlette": (0, 40),
    "uvicorn": (0, 30),
    "multipart": (0, 0, 9),
    "ultralytics": (8, 3),
    "cv2": (4, 8),
    "pytest": (8, 0),
    "httpx2": (2, 5),
}

for module_name, minimum in requirements.items():
    module = import_module(module_name)
    raw_version = getattr(module, "__version__", None)
    if raw_version is None and module_name == "PIL":
        raw_version = getattr(module, "PILLOW_VERSION", None)
    numbers = tuple(int(value) for value in re.findall(r"\d+", str(raw_version))[:len(minimum)])
    if len(numbers) < len(minimum) or numbers < minimum:
        raise SystemExit(1)

PY
  "${AGENT_PYTHON}" -m pip check >/dev/null 2>&1
}

project_entrypoint_compatible() {
  "${AGENT_PYTHON}" - "${ROOT_DIR}" <<'PY' >/dev/null 2>&1
import json
import sys
from importlib import metadata
from pathlib import Path
from sysconfig import get_paths
from urllib.parse import unquote, urlparse

root = Path(sys.argv[1]).resolve()
site_paths = sorted({get_paths()["purelib"], get_paths()["platlib"]})
distribution = next(
    (
        item
        for item in metadata.distributions(path=site_paths)
        if str(item.metadata.get("Name", "")).lower().replace("_", "-") == "agile-agent"
    ),
    None,
)
if distribution is None:
    raise SystemExit(1)

raw_direct_url = distribution.read_text("direct_url.json")
if not raw_direct_url:
    raise SystemExit(1)
direct_url = json.loads(raw_direct_url)
if direct_url.get("dir_info", {}).get("editable") is not True:
    raise SystemExit(1)
project_path = Path(unquote(urlparse(str(direct_url.get("url", ""))).path)).resolve()
entrypoint = Path(sys.executable).resolve().parent / "agile-agent"
if project_path != root or not entrypoint.is_file():
    raise SystemExit(1)
PY
}

printf '使用 Python 环境：%s\n' "${AGENT_PYTHON}"
if torch_stack_compatible; then
  printf '现有 PyTorch/CUDA/torchvision 兼容，跳过安装。\n'
else
  printf 'PyTorch 栈缺失或不兼容，安装经过验证的默认组合。\n'
  "${AGENT_PYTHON}" -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${PYTORCH_INDEX_URL}"
  if ! torch_stack_compatible; then
    printf 'PyTorch 安装后仍无法通过 CUDA 兼容性检查。\n' >&2
    exit 1
  fi
fi

if dependencies_compatible; then
  printf '现有 Agent 依赖完整且无冲突，跳过安装。\n'
else
  printf '检测到缺失或不兼容依赖，按最低兼容范围补充安装。\n'
  "${AGENT_PYTHON}" -m pip install -e ".[workbench,inference,dev]"
  if ! dependencies_compatible; then
    printf '依赖安装后仍存在缺失、版本过低或包冲突。\n' >&2
    "${AGENT_PYTHON}" -m pip check >&2 || true
    exit 1
  fi
fi
if project_entrypoint_compatible; then
  printf '当前仓库的 agile-agent 命令入口已注册，跳过项目注册。\n'
else
  printf '注册当前仓库的 agile-agent 命令入口，不重装第三方依赖。\n'
  "${AGENT_PYTHON}" -m pip install -e . --no-deps
  if ! project_entrypoint_compatible; then
    printf '当前仓库的 agile-agent 命令入口注册失败。\n' >&2
    exit 1
  fi
fi
"${AGENT_PYTHON}" -m fair_agent.cli doctor
"${AGENT_PYTHON}" scripts/smoke_models.py --load-only

printf '\n环境配置完成。日常启动请运行：./scripts/start_agent.sh\n'
