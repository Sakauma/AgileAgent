#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ENV_PREFIX="${HOME}/agileagent/envs/agileagent_train"
BASE_ENV="/usr/local/miniconda3/envs/agileagent"
WHEEL_DIR="${AGILE_EDGE_WHEEL_DIR:-}"
CONDA_BIN="${CONDA_EXE:-}"

usage() {
  printf '%s\n' \
    '用法：bootstrap_env.sh --wheel-dir DIR [--prefix DIR] [--base-env DIR] [--conda PATH]' \
    '' \
    '在独立 Conda 前缀中安装已验证的 Ascend 训练轮组。' \
    '不会修改 production 环境、CANN 或模型。'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheel-dir) WHEEL_DIR="$2"; shift 2 ;;
    --prefix) ENV_PREFIX="$2"; shift 2 ;;
    --base-env) BASE_ENV="$2"; shift 2 ;;
    --conda) CONDA_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf '未知参数：%s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$(uname -m)" != "aarch64" ]]; then
  printf '该环境只允许在 aarch64 Ascend310B 板端创建。\n' >&2
  exit 2
fi
if [[ -z "${CONDA_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [[ -x /usr/local/miniconda3/bin/conda ]]; then
    CONDA_BIN=/usr/local/miniconda3/bin/conda
  else
    printf '未找到 Miniconda；请用 --conda 指定 conda。\n' >&2
    exit 2
  fi
fi
if [[ ! -x "${CONDA_BIN}" || ! -x "${BASE_ENV}/bin/python" ]]; then
  printf 'Conda 或 production 基础环境不可用。\n' >&2
  exit 2
fi
if [[ ! -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  printf '未找到 CANN 环境脚本，拒绝创建不匹配的训练环境。\n' >&2
  exit 2
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ -z "${WHEEL_DIR}" || ! -d "${WHEEL_DIR}" ]]; then
  printf '请用 --wheel-dir 提供已验证的 aarch64 离线 wheel 目录。\n' >&2
  exit 2
fi

if [[ -e "${ENV_PREFIX}" ]]; then
  if "${ENV_PREFIX}/bin/python" - <<'PY'
import importlib.metadata as metadata
expected = {"torch": "2.0.1", "torch-npu": "2.0.1", "numpy": "1.24.4", "onnx": "1.14.1"}
assert all(metadata.version(name) == version for name, version in expected.items())
import torch
import torch_npu  # noqa: F401
assert hasattr(torch, "npu")
assert torch.npu.is_available()
PY
  then
    printf '独立训练环境已满足要求，跳过安装：%s\n' "${ENV_PREFIX}"
    exit 0
  fi
  printf '目标前缀已存在但版本不匹配，拒绝覆盖：%s\n' "${ENV_PREFIX}" >&2
  exit 2
fi

shopt -s nullglob
torch_wheels=("${WHEEL_DIR}"/torch-2.0.1-cp39-cp39-*aarch64.whl)
npu_wheels=("${WHEEL_DIR}"/torch_npu-2.0.1-cp39-cp39-*aarch64.whl)
numpy_wheels=("${WHEEL_DIR}"/numpy-1.24.4-cp39-cp39-*aarch64*.whl)
onnx_wheels=("${WHEEL_DIR}"/onnx-1.14.1-cp39-cp39-*aarch64*.whl)
protobuf_wheels=("${WHEEL_DIR}"/protobuf-3.20.3-cp39-cp39-*aarch64.whl)
filelock_wheels=("${WHEEL_DIR}"/filelock-3.12.2-py3-none-any.whl)
for group in torch_wheels npu_wheels numpy_wheels onnx_wheels protobuf_wheels filelock_wheels; do
  declare -n matches="${group}"
  if [[ ${#matches[@]} -ne 1 ]]; then
    printf 'wheel 目录中的 %s 必须唯一匹配，实际 %d 个。\n' "${group}" "${#matches[@]}" >&2
    exit 2
  fi
done

printf '克隆 production Python 到独立前缀（production 不会被修改）……\n'
"${CONDA_BIN}" create --yes --offline --clone "${BASE_ENV}" --prefix "${ENV_PREFIX}"
"${ENV_PREFIX}/bin/python" -m pip install \
  --no-index --no-deps --force-reinstall \
  "${filelock_wheels[0]}" \
  "${protobuf_wheels[0]}" \
  "${numpy_wheels[0]}" \
  "${onnx_wheels[0]}" \
  "${torch_wheels[0]}" \
  "${npu_wheels[0]}"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${ENV_PREFIX}/bin/python" - <<'PY'
import importlib.metadata as metadata
import torch
import torch_npu  # noqa: F401
expected = {"torch": "2.0.1", "torch-npu": "2.0.1", "numpy": "1.24.4", "onnx": "1.14.1"}
assert all(metadata.version(name) == version for name, version in expected.items())
assert hasattr(torch, "npu")
assert torch.npu.is_available()
print("Ascend 独立训练环境已就绪。")
PY
printf '环境路径：%s\n' "${ENV_PREFIX}"
