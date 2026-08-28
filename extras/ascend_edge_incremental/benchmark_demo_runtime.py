#!/usr/bin/env python3
"""Measure the complete image pipeline with the isolated demo Adapter active."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from fair_agent.core.config import load_config
from fair_agent.modules.formal_results import (
    validate_formal_prediction_files,
    write_formal_prediction_files,
)
from fair_agent.web.app import AtomicEngineProvider

from .protocol import load_protocol


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在 Ascend310B 上验收启用 Adapter 后的完整图像推理 FPS。"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-output-root", type=Path)
    parser.add_argument("--probe-size", type=int, default=20)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    if min(args.probe_size, args.warmup_rounds, args.rounds) <= 0:
        raise ValueError("probe size, warmup rounds and rounds must be positive")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("confidence must be between zero and one")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    result_output_root = (
        args.result_output_root
        if args.result_output_root is not None
        else output.parent / f"{output.stem}-formal-results"
    ).expanduser().resolve()
    if result_output_root.exists():
        raise FileExistsError(result_output_root)
    result_output_root.mkdir(parents=True, exist_ok=False)
    repo_root = args.repo_root.expanduser().resolve()
    protocol = load_protocol(args.registry, repo_root)
    paths = list(protocol.image_paths("lock"))[: args.probe_size]
    if len(paths) != args.probe_size:
        raise RuntimeError(
            f"runtime benchmark requires {args.probe_size} lock images, got {len(paths)}"
        )
    items = [(path.read_bytes(), path.name) for path in paths]
    config = load_config(args.config.expanduser().resolve())
    provider = AtomicEngineProvider(config)
    engines = provider.low_latency_pool()
    adapter_status = dict(engines[0].edge_incremental_adapter_status)
    if adapter_status.get("active") is not True:
        for engine in engines:
            engine.close()
        raise RuntimeError("isolated demo config did not activate the edge Adapter")

    round_rows: list[dict[str, float | int]] = []
    last_results = []
    try:
        for _ in range(args.warmup_rounds):
            provider.predict_encoded_low_latency_batch(
                items,
                args.confidence,
                "auto",
            )
        for index in range(args.rounds):
            started = time.perf_counter_ns()
            last_results = provider.predict_encoded_low_latency_batch(
                items,
                args.confidence,
                "auto",
            )
            write_started = time.perf_counter_ns()
            result_dir = result_output_root / f"round-{index + 1:02d}"
            result_paths = write_formal_prediction_files(
                result_dir,
                last_results,
                [path.name for path in paths],
            )
            result_write_ms = (
                time.perf_counter_ns() - write_started
            ) / 1_000_000.0
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            round_rows.append(
                {
                    "round": index + 1,
                    "images": len(items),
                    "full_pipeline_wall_ms": elapsed_ms,
                    "result_write_ms": result_write_ms,
                    "fps": len(items) * 1000.0 / elapsed_ms,
                    "result_output_dir": str(result_dir),
                    "result_file_count": len(result_paths),
                    "formal_results_valid": validate_formal_prediction_files(
                        result_paths,
                        len(items),
                    ),
                }
            )
    finally:
        for engine in engines:
            engine.close()

    decisions_active = all(
        (result.get("agent") or {})
        .get("decision", {})
        .get("edge_incremental_adapter", {})
        .get("active")
        is True
        for result in last_results
    )
    total_frames = sum(int(row["images"]) for row in round_rows)
    total_elapsed_ms = sum(
        float(row["full_pipeline_wall_ms"]) for row in round_rows
    )
    aggregate_fps = total_frames * 1000.0 / total_elapsed_ms
    median_fps_diagnostic = statistics.median(
        float(row["fps"]) for row in round_rows
    )
    formal_results_valid = all(
        row.get("formal_results_valid") is True for row in round_rows
    )
    passed = (
        decisions_active
        and formal_results_valid
        and aggregate_fps >= args.target_fps
    )
    report = {
        "schema_version": 2,
        "platform": "Ascend310B1",
        "measurement": "official_full_pipeline_with_formal_result_write",
        "input": "encoded_png",
        "confidence": args.confidence,
        "execution_profile": "configured_low_latency_pool",
        "engine_replicas": len(engines),
        "image_count": len(items),
        "warmup_rounds": args.warmup_rounds,
        "rounds": round_rows,
        "fps_calculation": "total_frames_divided_by_total_elapsed_seconds",
        "total_frames": total_frames,
        "total_elapsed_ms": total_elapsed_ms,
        "aggregate_fps": aggregate_fps,
        "median_round_fps_diagnostic": median_fps_diagnostic,
        "target_fps": args.target_fps,
        "adapter": adapter_status,
        "adapter_visible_in_every_result": decisions_active,
        "timed_components": [
            "image_decode",
            "scene_model",
            "decision_model",
            "base_detector",
            "incremental_detector",
            "postprocess",
            "formal_result_write",
        ],
        "formal_result_format": (
            "class_id x_center y_center width height confidence"
        ),
        "formal_result_output_root": str(result_output_root),
        "formal_results_valid": formal_results_valid,
        "includes_result_persistence": True,
        "passed": passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
