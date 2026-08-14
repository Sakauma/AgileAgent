#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.backends.ascend_acl import (
    ACL_MEMCPY_DEVICE_TO_HOST,
    _numpy_pointer,
    _require,
    context_tensor,
    detector_tensor,
)
from fair_agent.core.config import load_config
from fair_agent.core.hashes import sha256_file
from fair_agent.web.app import AtomicEngineProvider, build_web_settings


def difference(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"输入shape不一致：{reference.shape} != {candidate.shape}")
    expected = reference.astype(np.int16)
    actual = candidate.astype(np.int16)
    delta = np.abs(actual - expected)
    return {
        "shape": list(reference.shape),
        "max_abs": int(delta.max(initial=0)),
        "mean_abs": float(delta.mean()),
        "different_elements": int(np.count_nonzero(delta)),
        "different_ratio": float(np.count_nonzero(delta) / delta.size),
        "reference_channel_means": [
            float(value) for value in expected.reshape(-1, 3).mean(axis=0)
        ],
        "candidate_channel_means": [
            float(value) for value in actual.reshape(-1, 3).mean(axis=0)
        ],
    }


def copy_input(model: Any) -> np.ndarray:
    acl = model.runtime.acl
    host = np.empty(model.input_size, dtype=np.uint8)
    _require(
        acl.rt.memcpy(
            _numpy_pointer(acl, host),
            host.nbytes,
            model.input_device,
            model.input_size,
            ACL_MEMCPY_DEVICE_TO_HOST,
        ),
        f"acl.rt.memcpy({model.path.name} input D2H)",
    )
    return host.reshape(model.input_shape)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从310B DVPP设备缓冲回读Base/Specialist/Scene输入并与CPU契约逐元素比较。"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"DVPP输入报告已存在，拒绝覆盖：{args.output}")
    if len({path.resolve() for path in args.image}) != len(args.image):
        raise ValueError("--image包含重复路径。")
    config = load_config(args.config)
    engine = AtomicEngineProvider._build_engine(build_web_settings(config))
    preprocessor = getattr(engine, "encoded_preprocessor", None)
    if preprocessor is None:
        engine.close()
        raise RuntimeError("配置未创建DVPP encoded preprocessor。")

    if args.dump_dir is not None:
        args.dump_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        for path in args.image:
            data = path.read_bytes()
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            queued_ms = preprocessor.prepare(data)
            started = time.perf_counter_ns()
            _require(
                preprocessor.runtime.acl.rt.synchronize_stream(preprocessor.stream),
                "acl.rt.synchronize_stream(DVPP validation)",
            )
            device_ms = (time.perf_counter_ns() - started) / 1_000_000.0

            actual_specialist = copy_input(preprocessor.specialist_model)
            actual_base = copy_input(preprocessor.base_model)
            actual_context = copy_input(preprocessor.context_model)
            expected_specialist = np.ascontiguousarray(np.asarray(image), dtype=np.uint8)[
                None, ...
            ]
            expected_base, _info = detector_tensor(
                image, 736, 896, input_mode="nhwc_uint8_aipp"
            )
            expected_context = context_tensor(
                image, 160, input_mode="nhwc_uint8_aipp"
            )
            context_rgb = difference(expected_context, actual_context)
            context_bgr = difference(expected_context, actual_context[..., ::-1])
            row = {
                "image": str(path.resolve()),
                "image_sha256": sha256_file(path),
                "dvpp_queue_ms": queued_ms,
                "dvpp_synchronize_ms": device_ms,
                "specialist": difference(expected_specialist, actual_specialist),
                "base": difference(expected_base, actual_base),
                "context": {
                    "rgb": context_rgb,
                    "candidate_channel_reversed": context_bgr,
                    "channel_order_closer": (
                        "rgb"
                        if context_rgb["mean_abs"] <= context_bgr["mean_abs"]
                        else "bgr"
                    ),
                },
            }
            rows.append(row)
            if args.dump_dir is not None:
                for role, expected, actual in (
                    ("specialist", expected_specialist, actual_specialist),
                    ("base", expected_base, actual_base),
                    ("context", expected_context, actual_context),
                ):
                    np.save(args.dump_dir / f"{path.stem}-{role}-reference.npy", expected)
                    np.save(args.dump_dir / f"{path.stem}-{role}-dvpp.npy", actual)
    finally:
        engine.close()

    report = {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "image_count": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
