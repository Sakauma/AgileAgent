#!/usr/bin/env python3
"""Freeze low-threshold Base, Specialist and Scene responses without labels."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from fair_agent.core.config import load_config, resolve_path
from fair_agent.web.app import AtomicEngineProvider, build_web_settings


def public_record(result: Mapping[str, Any]) -> dict[str, Any]:
    agent = result.get("agent") or {}
    return {
        "image_id": Path(str(result["filename"])).stem,
        "filename": result["filename"],
        "context": result["context"],
        "detections": result["detections"],
        "class_counts": result["class_counts"],
        "detection_count": result["detection_count"],
        "models_used": agent.get("models_used") or [],
        "decision": agent.get("decision") or {},
        "timings": result.get("timings") or {},
    }


def neutralize_policy(
    config: dict[str, Any], temporary_root: Path, confidence_floor: float
) -> dict[str, Any]:
    registry_source = resolve_path(config["web"]["generation_registry"])
    registry = json.loads(registry_source.read_text(encoding="utf-8"))
    model_audit: dict[str, Any] = {}
    for model in registry.get("models", []):
        owned = [int(value) for value in model.get("owns_classes", [])]
        if owned:
            model["per_class_thresholds"] = {
                str(class_id): confidence_floor for class_id in owned
            }
        model["context_prior"] = {}
        model["context_gate"] = {"enabled": False}
        if model.get("role") in {
            "class_incremental_expert",
            "target_incremental_expert",
        }:
            model["content_execution_gate"] = {"enabled": False}
        model_audit[str(model.get("id"))] = {
            "owns_classes": owned,
            "threshold": confidence_floor,
            "context_gate_enabled": False,
            "content_execution_gate_enabled": False,
        }
    probe_registry = temporary_root / "probe-generations.json"
    probe_registry.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config["web"]["generation_registry"] = str(probe_registry)
    config["generation"]["registry"] = str(probe_registry)
    config["inference"]["confidence_min"] = min(
        float(config["inference"]["confidence_min"]), confidence_floor
    )
    config["inference"]["confidence_default"] = confidence_floor
    routing = config["routing"]
    routing["fusion_iou"] = 1.0
    routing["conflict_iou"] = 1.0
    routing["conflict_incremental_coverage"] = None
    routing["conflict_base_confidence"] = 1.0
    routing["specialist_margin"] = 0.0
    routing["score_calibration"] = {"enabled": False}
    routing["edge_incremental_adapter"] = {"enabled": False}
    routing["cross_class_suppression"] = {
        "enabled": False,
        "strategy": "highest_confidence",
        "scope": "all_classes",
        "iou": 1.0,
        "smaller_box_coverage": None,
        "incremental_over_base_margin": 0.0,
    }
    return {
        "registry_source": str(registry_source),
        "confidence_floor": confidence_floor,
        "labels_read": False,
        "policy_layers_neutralized": [
            "per_class_thresholds",
            "known_scene_soft_gates",
            "content_execution_gate",
            "score_calibration",
            "edge_incremental_adapter",
            "old_new_conflict_arbitration",
            "cross_class_suppression",
        ],
        "models": model_audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在隔离进程中冻结无标签 Ascend 原始候选，用于板端 Adapter。"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--confidence", type=float, default=0.00001)
    parser.add_argument("--encoded", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    if output.exists() or summary_path.exists():
        raise FileExistsError("frozen probe or summary already exists")
    if not 0.00001 <= args.confidence <= 0.01:
        raise ValueError("edge calibration probe confidence must be in [0.00001, 0.01]")
    image_root = args.image_root.expanduser().resolve()
    paths = sorted(image_root.glob("*.png"))
    if (
        len(paths) != args.expected_images
        or len({path.stem for path in paths}) != args.expected_images
    ):
        raise RuntimeError(
            f"probe input must contain {args.expected_images} unique PNG images"
        )
    config = copy.deepcopy(load_config(args.config.expanduser().resolve()))
    records: list[dict[str, Any]] = []
    wall_ms: list[float] = []
    with tempfile.TemporaryDirectory(prefix="agileagent-edge-probe-") as directory:
        audit = neutralize_policy(config, Path(directory), args.confidence)
        engine = AtomicEngineProvider._build_engine(build_web_settings(config))
        try:
            for path in paths:
                started = time.perf_counter_ns()
                if args.encoded:
                    data = path.read_bytes()
                    if not engine.accepts_encoded(data):
                        raise RuntimeError(f"Ascend engine rejected encoded PNG: {path}")
                    result = engine.predict_encoded(
                        data,
                        path.name,
                        confidence=args.confidence,
                        incremental_protocol="auto",
                    )
                else:
                    with Image.open(path) as opened:
                        image = opened.convert("RGB")
                    result = engine.predict(
                        image,
                        path.name,
                        confidence=args.confidence,
                        incremental_protocol="auto",
                    )
                wall_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
                records.append(public_record(result))
        finally:
            engine.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "input_mode": "unlabeled_images",
        "labels_read": False,
        "config": str(args.config.expanduser().resolve()),
        "confidence": args.confidence,
        "encoded": args.encoded,
        "probe_audit": audit,
        "image_count": len(records),
        "prediction_count": sum(len(row["detections"]) for row in records),
        "mean_wall_ms": sum(wall_ms) / len(wall_ms),
        "predictions": str(output),
        "passed": len(records) == args.expected_images,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
