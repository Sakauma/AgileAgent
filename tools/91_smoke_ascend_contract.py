#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="验证Ascend原生后端C ABI contract stub。")
    parser.add_argument("library", type=Path)
    args = parser.parse_args()
    library_path = args.library.resolve()
    if not library_path.is_file():
        raise FileNotFoundError(library_path)

    library = ctypes.CDLL(str(library_path))
    required = {
        "agile_agent_ascend_backend_version",
        "agile_agent_ascend_create",
        "agile_agent_ascend_destroy",
        "agile_agent_ascend_ready",
        "agile_agent_ascend_warmup",
        "agile_agent_ascend_predict",
        "agile_agent_ascend_free_result",
        "agile_agent_ascend_last_error",
    }
    missing = sorted(name for name in required if not hasattr(library, name))
    if missing:
        raise RuntimeError("Ascend ABI缺少符号：" + ", ".join(missing))

    library.agile_agent_ascend_backend_version.restype = ctypes.c_uint32
    library.agile_agent_ascend_create.argtypes = [ctypes.c_char_p]
    library.agile_agent_ascend_create.restype = ctypes.c_void_p
    library.agile_agent_ascend_destroy.argtypes = [ctypes.c_void_p]
    library.agile_agent_ascend_ready.argtypes = [ctypes.c_void_p]
    library.agile_agent_ascend_ready.restype = ctypes.c_int
    library.agile_agent_ascend_warmup.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    library.agile_agent_ascend_warmup.restype = ctypes.c_int
    library.agile_agent_ascend_predict.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
    ]
    library.agile_agent_ascend_predict.restype = ctypes.c_void_p
    library.agile_agent_ascend_last_error.argtypes = [ctypes.c_void_p]
    library.agile_agent_ascend_last_error.restype = ctypes.c_char_p

    version = int(library.agile_agent_ascend_backend_version())
    if version != 1:
        raise RuntimeError(f"Ascend ABI版本错误：{version}")
    handle = library.agile_agent_ascend_create(b"{}")
    if not handle:
        raise RuntimeError("contract stub必须返回可析构的Not Ready handle")
    try:
        ready = int(library.agile_agent_ascend_ready(handle))
        warmup = int(library.agile_agent_ascend_warmup(handle, None, 0, 20))
        prediction = library.agile_agent_ascend_predict(handle, None, 0, b"{}")
        error = (library.agile_agent_ascend_last_error(handle) or b"").decode(
            "utf-8", errors="replace"
        )
        if ready != 0 or warmup == 0 or prediction:
            raise RuntimeError("contract stub不得进入Ready或返回CPU替代推理结果")
        if "CANN runtime is unavailable" not in error:
            raise RuntimeError(f"contract stub错误信息不明确：{error}")
    finally:
        library.agile_agent_ascend_destroy(handle)

    print(
        json.dumps(
            {
                "library": str(library_path),
                "abi_version": version,
                "ready": False,
                "warmup_failed": True,
                "predict_failed": True,
                "cpu_fallback": False,
                "error": error,
                "passed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
