#!/usr/bin/env python3
"""Select adapter strengths on the registered mixed dev scope."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .core import (
    adapt_probe,
    image_sizes,
    load_adapter_bank,
    load_calibration_module,
    load_context_prior,
    load_method,
    raw_logit,
    require_exact_probe,
    score_view,
)
from .protocol import load_protocol


def scaled_probe(
    calibration: Any,
    raw_probe: Any,
    adapted_probe: Any,
    scales: Mapping[int, float],
) -> Any:
    records = []
    for raw, adapted in zip(raw_probe.records, adapted_probe.records):
        class_id = int(raw["class_id"])
        if raw.get("source") != "incremental_model" or class_id not in scales:
            records.append(raw)
            continue
        residual = float(adapted.get("adapter_residual_logit", 0.0)) * float(
            scales[class_id]
        )
        confidence = 1.0 / (
            1.0 + math.exp(-(raw_logit(float(raw["confidence"])) + residual))
        )
        row = dict(raw)
        row["confidence"] = confidence
        row["adapter_residual_logit"] = residual
        records.append(row)
    return calibration.ProbeData(
        tuple(records), dict(raw_probe.contexts), frozenset(raw_probe.image_ids)
    )


def grid(center: float | None = None) -> list[float]:
    if center is None:
        return [round(index / 10.0, 6) for index in range(11)]
    return sorted(
        {
            round(min(1.0, max(0.0, center + offset * 0.025)), 6)
            for offset in range(-4, 5)
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="只在注册 mixed dev 上选择 Adapter 强度，不更新任何权重。"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--context-prior", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-map50-min", type=float)
    parser.add_argument("--new-map50-min", type=float)
    parser.add_argument("--krr-min", type=float)
    parser.add_argument("--false-activation-increase", type=int, default=0)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    repo_root = args.repo_root.expanduser().resolve()
    protocol = load_protocol(args.registry, repo_root)
    from fair_agent.modules.strict_incremental import (  # noqa: PLC0415
        evaluate_ap50,
        precision_recall,
        retention_metrics,
        subset_rows,
        yolo_ground_truth,
    )

    calibration = load_calibration_module(repo_root)
    full_probe = calibration.load_probe(args.probe.expanduser().resolve())
    dev_paths = protocol.image_paths("selection")
    require_exact_probe(full_probe, dev_paths, "selection")
    base_ids = protocol.base_image_ids("selection")
    sizes = image_sizes(dev_paths)
    _, states = load_adapter_bank(args.checkpoint, protocol)
    full_adapter_probe = adapt_probe(calibration, full_probe, states, sizes)
    ground_truth = yolo_ground_truth(dev_paths)
    context_prior = load_context_prior(args.context_prior.expanduser().resolve())
    method = load_method(args.method_config.expanduser().resolve())
    default_gates = {
        "base_map50": float(method["competition"]["accuracy_gates"]["base_map50_min"]),
        "new_map50": float(method["competition"]["accuracy_gates"]["new_map50_min"]),
        "krr": float(method["competition"]["accuracy_gates"]["krr_min"]),
    }
    gates = {
        "base_map50": (
            args.base_map50_min
            if args.base_map50_min is not None
            else default_gates["base_map50"]
        ),
        "new_map50": (
            args.new_map50_min
            if args.new_map50_min is not None
            else default_gates["new_map50"]
        ),
        "krr": args.krr_min if args.krr_min is not None else default_gates["krr"],
    }
    class_ids = protocol.new_class_ids
    cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def evaluate(values: tuple[float, ...]) -> dict[str, Any]:
        if values in cache:
            return cache[values]
        scales = dict(zip(class_ids, values))
        candidate_probe = scaled_probe(
            calibration, full_probe, full_adapter_probe, scales
        )
        metrics, _ = score_view(
            calibration,
            protocol,
            method,
            candidate_probe,
            ground_truth,
            base_ids,
            context_prior,
            evaluate_ap50,
            precision_recall,
            retention_metrics,
            subset_rows,
        )
        row = {
            "scales": {str(key): value for key, value in scales.items()},
            "metrics": metrics,
        }
        cache[values] = row
        return row

    baseline = evaluate(tuple(0.0 for _ in class_ids))
    baseline_fa = int(baseline["metrics"]["new_false_activation"]["count"])

    def passing(row: Mapping[str, Any]) -> bool:
        metrics = row["metrics"]
        return bool(
            float(metrics["base_map50"]) >= gates["base_map50"]
            and float(metrics["new_map50"]) >= gates["new_map50"]
            and float(metrics["krr"]) >= gates["krr"]
            and int(metrics["new_false_activation"]["count"])
            <= baseline_fa + args.false_activation_increase
        )

    def objective(row: Mapping[str, Any]) -> tuple[float, ...]:
        metrics = row["metrics"]
        scales = row["scales"]
        return (
            float(metrics["full_map50"]),
            float(metrics["new_map50"]),
            float(metrics["new_macro_precision_at_0_63"]),
            -sum(float(scales[str(class_id)]) for class_id in class_ids),
        )

    coarse = [evaluate(values) for values in itertools.product(grid(), repeat=len(class_ids))]
    coarse_passing = [row for row in coarse if passing(row)]
    coarse_selected = max(coarse_passing or coarse, key=objective)
    centers = tuple(
        float(coarse_selected["scales"][str(class_id)]) for class_id in class_ids
    )
    for values in itertools.product(*(grid(center) for center in centers)):
        evaluate(tuple(values))
    all_passing = [row for row in cache.values() if passing(row)]
    selected = max(all_passing or cache.values(), key=objective)
    top = sorted(all_passing or cache.values(), key=objective, reverse=True)[:20]
    payload = {
        "schema_version": 1,
        "phase": "system_calibration",
        "platform": "Ascend310B1",
        "protocol_id": protocol.protocol_id,
        "selection_scope": "registered_mixed_dev_only",
        "lock_labels_opened": False,
        "detector_weights_updated": False,
        "adapter_weights_updated": False,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "constraints": {
            **{f"{key}_min": value for key, value in gates.items()},
            "new_false_activation_count_max": (
                baseline_fa + args.false_activation_increase
            ),
        },
        "objective": "maximize Full-mAP50 then New-mAP50 under all constraints",
        "evaluated_candidate_count": len(cache),
        "passing_candidate_count": len(all_passing),
        "baseline": baseline,
        "coarse_selected": coarse_selected,
        "selected": selected,
        "top_candidates": top,
        "competition_passed_on_dev": passing(selected),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["competition_passed_on_dev"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
