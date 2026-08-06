#!/usr/bin/env python3
"""Select only the support-model seed for an internally frozen fusion policy."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import (
    evaluate_ap50,
    read_split,
    sha256_file,
    subset_rows,
    yolo_ground_truth,
)


FORBIDDEN_MARKERS = ("mixed_test", "base_test", "lock")


def reject_lock_reference(path: Path) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"基础 owner 选模不得引用 test/lock：{path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def subgroup_metrics(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    images: Sequence[Path],
    class_ids: Sequence[int],
) -> dict[str, Any]:
    sequences = sorted({path.stem.rsplit("_", 1)[0] for path in images})
    output = {}
    for sequence in sequences:
        image_ids = {
            path.stem for path in images if path.stem.startswith(f"{sequence}_")
        }
        output[sequence] = evaluate_ap50(
            subset_rows(predictions, image_ids),
            subset_rows(targets, image_ids),
            class_ids,
        )
    return output


def evaluate_support_candidates(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    targets: Sequence[Mapping[str, Any]],
    images: Sequence[Path],
    primary: str,
    fixed_secondaries: Sequence[str],
    support_candidates: Sequence[str],
    focus_class: int,
    class_ids: Sequence[int],
    fusion_iou: float,
    secondary_scale: float,
    agreement_bonus: float,
    weighted_boxes: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ensemble = runpy.run_path(str(ROOT / "tools" / "72_select_base_ensemble.py"))
    baseline_rows = list(predictions[primary])
    baseline = evaluate_ap50(baseline_rows, targets, class_ids)
    baseline_subgroups = subgroup_metrics(baseline_rows, targets, images, class_ids)
    rows = []
    for support in support_candidates:
        secondary_names = [*fixed_secondaries, support]
        fused = ensemble["fuse_focus_class"](
            predictions,
            primary,
            secondary_names,
            int(focus_class),
            float(fusion_iou),
            float(secondary_scale),
            float(agreement_bonus),
            bool(weighted_boxes),
        )
        metrics = evaluate_ap50(fused, targets, class_ids)
        subgroups = subgroup_metrics(fused, targets, images, class_ids)
        deltas = {
            name: float(item["map50"]) - float(baseline_subgroups[name]["map50"])
            for name, item in subgroups.items()
        }
        rows.append(
            {
                "support": support,
                "secondaries": secondary_names,
                "map50": float(metrics["map50"]),
                "per_class_ap50": metrics["per_class_ap50"],
                "delta_map50": float(metrics["map50"] - baseline["map50"]),
                "subgroups": subgroups,
                "subgroup_delta_vs_primary": deltas,
                "worst_subgroup_delta_vs_primary": min(deltas.values()) if deltas else 0.0,
                "degraded_subgroup_count": sum(delta < -0.01 for delta in deltas.values()),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["map50"]),
            int(row["degraded_subgroup_count"]),
            -float(row["worst_subgroup_delta_vs_primary"]),
            str(row["support"]),
        )
    )
    return rows, {
        "model": primary,
        "map50": float(baseline["map50"]),
        "per_class_ap50": baseline["per_class_ap50"],
        "subgroups": baseline_subgroups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--fixed-secondary", action="append", required=True)
    parser.add_argument("--support-candidate", action="append", required=True)
    parser.add_argument("--focus-class", type=int, default=0)
    parser.add_argument("--class-ids", default="0,1,2")
    parser.add_argument("--fusion-iou", type=float, required=True)
    parser.add_argument("--secondary-scale", type=float, required=True)
    parser.add_argument("--agreement-bonus", type=float, required=True)
    parser.add_argument("--weighted-boxes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source_report.expanduser().resolve()
    output = args.output.expanduser().resolve()
    reject_lock_reference(source_path)
    reject_lock_reference(output)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有 owner 选模报告：{output}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("selection_scope") != "base_dev_only" or bool(
        source.get("lock_data_access", True)
    ):
        raise ValueError("源预测报告必须仅来自 base_dev 且未读取 lock")
    split = Path(str(source["split"])).resolve()
    reject_lock_reference(split)
    images = read_split(split)
    class_ids = [int(value) for value in args.class_ids.split(",")]
    targets = yolo_ground_truth(images, class_ids)

    names = [args.primary, *args.fixed_secondary, *args.support_candidate]
    if len(names) != len(set(names)):
        raise ValueError("primary、fixed secondary 和 support candidate 名称必须唯一")
    source_models = {str(row["name"]): dict(row) for row in source["models"]}
    missing = sorted(set(names) - set(source_models))
    if missing:
        raise ValueError(f"源报告缺少 owner 预测：{missing}")
    cache_dir = source_path.parent / "predictions"
    predictions = {name: read_jsonl(cache_dir / f"{name}.jsonl") for name in names}
    ranking, baseline = evaluate_support_candidates(
        predictions,
        targets,
        images,
        args.primary,
        args.fixed_secondary,
        args.support_candidate,
        int(args.focus_class),
        class_ids,
        float(args.fusion_iou),
        float(args.secondary_scale),
        float(args.agreement_bonus),
        bool(args.weighted_boxes),
    )
    report = {
        "schema_version": 1,
        "selection_scope": "base_dev_only",
        "lock_data_access": False,
        "source_report": str(source_path),
        "source_report_sha256": sha256_file(source_path),
        "split": str(split),
        "split_sha256": sha256_file(split),
        "image_count": len(images),
        "policy_source": "base_train_internal_temporal_holdout",
        "policy": {
            "primary": args.primary,
            "fixed_secondaries": args.fixed_secondary,
            "focus_class": int(args.focus_class),
            "fusion_iou": float(args.fusion_iou),
            "secondary_scale": float(args.secondary_scale),
            "agreement_bonus": float(args.agreement_bonus),
            "weighted_boxes": bool(args.weighted_boxes),
        },
        "models": {name: source_models[name] for name in names},
        "baseline": baseline,
        "ranking": ranking,
        "selected_support": ranking[0]["support"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
