#!/usr/bin/env python3
"""Freeze the final per-class base-ensemble policy from base-dev artifacts."""

from __future__ import annotations

import argparse
import json
import random
import runpy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import (
    evaluate_ap50,
    read_split,
    sha256_file,
    yolo_ground_truth,
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def subset(rows: Sequence[Mapping[str, Any]], image_ids: set[str]) -> list[Dict[str, Any]]:
    return [dict(row) for row in rows if str(row["image_id"]) in image_ids]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    source_path = resolve_path(config["paths"]["selection_source_report"])
    destination = resolve_path(config["paths"]["selection_report"])
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已冻结 base-dev 选择报告：{destination}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_split = str(source.get("split", ""))
    lowered = source_split.lower()
    if (
        source.get("selection_scope") != "base_dev_only"
        or bool(source.get("lock_data_access", True))
        or "mixed_test" in lowered
        or "lock" in lowered
        or "base_test" in lowered
    ):
        raise ValueError("源预测必须来自无 lock/test 访问的 base_dev")

    images = read_split(Path(source_split))
    ground_truth = yolo_ground_truth(images)
    local_to_global = {
        int(key): int(value) for key, value in config["base_local_to_global"].items()
    }
    global_to_local = {global_id: local_id for local_id, global_id in local_to_global.items()}
    cache_dir = source_path.parent / "predictions"
    model_names = sorted(
        {
            str(policy["primary"])
            for policy in config["base_fusion"]["class_policies"].values()
        }
        | {
            str(name)
            for policy in config["base_fusion"]["class_policies"].values()
            for name in dict(policy.get("secondary_scales", {}))
        }
    )
    model_selection_audit = None
    model_selection_value = config.get("paths", {}).get("model_selection_report")
    if model_selection_value:
        model_selection_path = resolve_path(model_selection_value)
        model_selection = json.loads(model_selection_path.read_text(encoding="utf-8"))
        if model_selection.get("selection_scope") != "base_dev_only" or bool(
            model_selection.get("lock_data_access", True)
        ):
            raise ValueError("support owner 选模报告必须仅来自 base_dev")
        if str(model_selection.get("source_report")) != str(source_path):
            raise ValueError("support owner 选模报告与预测源报告不一致")
        selected_support = str(model_selection.get("selected_support"))
        if selected_support not in model_names:
            raise ValueError("base fusion 未使用 base_dev 选中的 support owner")
        model_selection_audit = {
            "path": str(model_selection_path),
            "sha256": sha256_file(model_selection_path),
            "selected_support": selected_support,
        }
    predictions = {
        name: read_jsonl(cache_dir / f"{name}.jsonl") for name in model_names
    }
    ensemble_module = runpy.run_path(str(ROOT / "tools" / "72_select_base_ensemble.py"))
    evaluator_module = runpy.run_path(str(ROOT / "tools" / "73_evaluate_base_ensemble.py"))
    local_policies = {
        global_to_local[int(class_id)]: dict(policy)
        for class_id, policy in config["base_fusion"]["class_policies"].items()
    }
    fused = evaluator_module["fuse_base_classes"](
        predictions,
        local_policies,
        ensemble_module["fuse_focus_class"],
    )
    class_ids = sorted(local_to_global)
    primary = str(next(iter(config["base_fusion"]["class_policies"].values()))["primary"])
    baseline = predictions[primary]
    baseline_metrics = evaluate_ap50(baseline, ground_truth, class_ids)
    final_metrics = evaluate_ap50(fused, ground_truth, class_ids)

    image_ids = [path.stem for path in images]
    strata: dict[str, list[str]] = defaultdict(list)
    for image_id in image_ids:
        strata[image_id.rsplit("_", 1)[0]].append(image_id)
    baseline_by_image: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    fused_by_image: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    ground_truth_by_image: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in baseline:
        baseline_by_image[str(row["image_id"])].append(dict(row))
    for row in fused:
        fused_by_image[str(row["image_id"])].append(dict(row))
    for row in ground_truth:
        ground_truth_by_image[str(row["image_id"])].append(dict(row))
    rng = random.Random(20260807)
    deltas = []
    for iteration in range(1000):
        sampled_baseline = []
        sampled_fused = []
        sampled_ground_truth = []
        sample_index = 0
        for values in strata.values():
            for original_id in rng.choices(values, k=len(values)):
                sampled_id = f"bootstrap:{iteration}:{sample_index}"
                sample_index += 1
                sampled_baseline.extend(
                    {**row, "image_id": sampled_id}
                    for row in baseline_by_image[original_id]
                )
                sampled_fused.extend(
                    {**row, "image_id": sampled_id} for row in fused_by_image[original_id]
                )
                sampled_ground_truth.extend(
                    {**row, "image_id": sampled_id}
                    for row in ground_truth_by_image[original_id]
                )
        deltas.append(
            evaluate_ap50(sampled_fused, sampled_ground_truth, class_ids)["map50"]
            - evaluate_ap50(sampled_baseline, sampled_ground_truth, class_ids)["map50"]
        )

    model_audit = {}
    source_models = {str(row["name"]): row for row in source["models"]}
    for name in model_names:
        row = source_models[name]
        model_audit[name] = {
            "weight": row["weight"],
            "weight_sha256": row["weight_sha256"],
            "imgsz": int(
                row.get(
                    "imgsz",
                    source.get("imgsz", config["models"][name]["imgsz"]),
                )
            ),
            "inference_mode": str(row.get("inference_mode", "full_frame")),
            "tile": row.get("tile"),
            "predictions_path": str(cache_dir / f"{name}.jsonl"),
            "predictions_sha256": row["predictions_sha256"],
        }
    report = {
        "schema_version": 1,
        "candidate_id": config["candidate_id"],
        "selection_scope": "base_dev_only",
        "lock_data_access": False,
        "split": source_split,
        "split_sha256": sha256_file(Path(source_split)),
        "image_count": len(images),
        "source_report": str(source_path),
        "source_report_sha256": sha256_file(source_path),
        "model_selection": model_selection_audit,
        "config": str(config_path),
        "config_sha256_at_selection": sha256_file(config_path),
        "models": model_audit,
        "base_fusion": config["base_fusion"],
        "baseline": {
            "model": primary,
            "map50": float(baseline_metrics["map50"]),
            "per_class_ap50": baseline_metrics["per_class_ap50"],
        },
        "selected": {
            "map50": float(final_metrics["map50"]),
            "per_class_ap50": final_metrics["per_class_ap50"],
            "delta_map50": float(final_metrics["map50"] - baseline_metrics["map50"]),
        },
        "paired_stratified_bootstrap": {
            "iterations": len(deltas),
            "seed": 20260807,
            "delta_median": float(np.median(deltas)),
            "delta_ci95_low": float(np.percentile(deltas, 2.5)),
            "delta_ci95_high": float(np.percentile(deltas, 97.5)),
            "positive_fraction": float(sum(value > 0 for value in deltas) / len(deltas)),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
