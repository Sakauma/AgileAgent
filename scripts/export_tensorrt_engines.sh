#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -ne 1 ]]; then
  printf '用法：%s <设备专用配置.yaml>\n' "$0" >&2
  exit 2
fi

PROFILE="$(realpath -m "$1")"
DEFAULT_PROFILE="$(readlink -f configs/agent_pipeline.yaml)"
if [[ ! -f "${PROFILE}" ]]; then
  printf '设备专用配置不存在：%s\n' "${PROFILE}" >&2
  exit 1
fi
if [[ "${PROFILE}" == "${DEFAULT_PROFILE}" ]]; then
  printf '拒绝使用默认发布配置导出；请先复制并编辑设备专用 YAML。\n' >&2
  exit 1
fi

if [[ -n "${AGILE_AGENT_PYTHON:-}" ]]; then
  AGENT_PYTHON="${AGILE_AGENT_PYTHON}"
elif [[ -f "${ROOT_DIR}/.agent-python" ]]; then
  IFS= read -r AGENT_PYTHON < "${ROOT_DIR}/.agent-python"
else
  AGENT_PYTHON="${ROOT_DIR}/.venv/bin/python"
fi
if [[ ! -x "${AGENT_PYTHON}" ]]; then
  printf '未找到已配置的 Python：%s\n请先完成 README 中的环境配置。\n' "${AGENT_PYTHON}" >&2
  exit 1
fi

"${AGENT_PYTHON}" - "${PROFILE}" <<'PY'
import sys

import tensorrt as trt
import torch

from fair_agent.core.config import load_config

profile = sys.argv[1]
config = load_config(profile, allow_unverified_tensorrt_hashes=True)
backend = config["tensorrt_backend"]
device = int(backend["export"]["device"])
if backend["validated"] is not False:
    raise SystemExit("设备专用配置必须设置 tensorrt_backend.validated: false。")
if not torch.cuda.is_available() or device >= torch.cuda.device_count():
    raise SystemExit(f"配置的 GPU {device} 不可用。")
actual_version = str(trt.__version__)
actual_capability = ".".join(map(str, torch.cuda.get_device_capability(device)))
if actual_version != str(backend["expected_version"]):
    raise SystemExit(
        f"TensorRT 版本不匹配：配置 {backend['expected_version']}，当前 {actual_version}。"
    )
if backend["require_exact_gpu"] and actual_capability != str(backend["expected_compute_capability"]):
    raise SystemExit(
        "GPU 计算能力不匹配："
        f"配置 {backend['expected_compute_capability']}，当前 {actual_capability}。"
    )
print(f"设备配置：{profile}")
print(f"运行环境：TensorRT {actual_version}，{torch.cuda.get_device_name(device)}，SM {actual_capability}")
PY

printf '\n[1/2] 导出 TensorRT engine 并登记真实 SHA256\n'
"${AGENT_PYTHON}" tools/80_export_tensorrt_engines.py --config "${PROFILE}"

printf '\n[2/2] 执行只读完整性校验\n'
"${AGENT_PYTHON}" tools/80_export_tensorrt_engines.py --config "${PROFILE}" --verify-only

printf '\nTensorRT engine 导出与完整性校验完成。\n'
printf '设备配置仍保持 validated: false；完成精度和性能门禁后再批准为运行配置。\n'
