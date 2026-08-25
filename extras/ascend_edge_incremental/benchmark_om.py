#!/usr/bin/env python3
"""Measure Adapter OM accuracy and conservative serial pipeline latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .core import FEATURE_DIM, load_adapter_bank, load_scales
from .protocol import load_protocol


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--om", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=Path)
    parser.add_argument("--candidate-slots", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--baseline-fps", type=float, required=True)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if min(args.candidate_slots, args.warmup, args.iterations) <= 0:
        raise ValueError("candidate slots, warmup and iterations must be positive")
    if args.baseline_fps <= 0:
        raise ValueError("baseline FPS must be positive")
    repo_root = args.repo_root.expanduser().resolve()
    protocol = load_protocol(args.registry, repo_root)
    from fair_agent.backends.ascend_acl import (  # noqa: PLC0415
        AscendAclModel,
        AscendAclRuntime,
    )

    _, states = load_adapter_bank(args.checkpoint, protocol)
    scales, scale_source = load_scales(
        args.scales, protocol.new_class_ids, protocol.protocol_id
    )
    weights = np.stack(
        [
            states[class_id]["weights"].numpy() * scales[class_id]
            for class_id in protocol.new_class_ids
        ],
        axis=1,
    ).astype(np.float32)
    rng = np.random.default_rng(20260825)
    features = rng.normal(size=(args.candidate_slots, FEATURE_DIM)).astype(np.float32)
    expected = features @ weights
    runtime = AscendAclRuntime.acquire(0)
    model = AscendAclModel(
        runtime, args.om.expanduser().resolve(), execution_mode="synchronous"
    )
    wall_ms: list[float] = []
    device_ms: list[float] = []
    try:
        for _ in range(args.warmup):
            model.execute(features)
        observed = None
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            outputs, inference_ms = model.execute(features)
            wall_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
            device_ms.append(float(inference_ms))
            observed = outputs[0]
        assert observed is not None
    finally:
        model.close()
        runtime.close()
    max_abs_error = float(np.max(np.abs(observed - expected)))
    allclose = bool(np.allclose(observed, expected, atol=args.atol, rtol=args.rtol))
    median_wall = statistics.median(wall_ms)
    baseline_engine_ms = 1000.0 / args.baseline_fps
    projected_engine_ms = baseline_engine_ms + median_wall
    projected_fps = 1000.0 / projected_engine_ms
    report = {
        "schema_version": 1,
        "platform": "Ascend310B1",
        "protocol_id": protocol.protocol_id,
        "om": str(args.om.expanduser().resolve()),
        "class_order": list(protocol.new_class_ids),
        "scales": {str(key): value for key, value in scales.items()},
        "scale_source": scale_source,
        "input_shape": list(features.shape),
        "output_shape": list(observed.shape),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "numerical_equivalence": {
            "max_abs_error": max_abs_error,
            "atol": args.atol,
            "rtol": args.rtol,
            "passed": allclose,
        },
        "latency_ms": {
            "wall_median": median_wall,
            "wall_p95": percentile(wall_ms, 0.95),
            "device_median": statistics.median(device_ms),
            "device_p95": percentile(device_ms, 0.95),
        },
        "projected_integrated_pipeline": {
            "baseline_engine_ms": baseline_engine_ms,
            "baseline_fps": args.baseline_fps,
            "baseline_plus_measured_adapter_wall_ms": projected_engine_ms,
            "projected_fps": projected_fps,
            "fps_gate_30_passed": projected_fps >= 30.0,
            "projection_is_conservative_sequential_execution": True,
            "integrated_service_remeasured": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if allclose and projected_fps >= 30.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
