#!/usr/bin/env python3
"""Freeze and score a fixed generic + small-object OOF fusion policy."""

from __future__ import annotations

import argparse
import json
import runpy
import statistics
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
CLASS_IDS = [0, 1, 3]


def reject_test_reference(path: Path, role: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"OOF fusion {role} 不得引用 test/lock：{path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def validate_source(path: Path, expected_mode: str) -> dict[str, Any]:
    reject_test_reference(path, "source report")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("selection_scope") != "base_train_and_dev_oof_only"
        or bool(report.get("lock_data_access", True))
        or str(report.get("inference_mode")) != expected_mode
    ):
        raise ValueError(f"OOF 源报告范围或推理模式错误：{path}")
    return report


def subgroup_metrics(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    image_ids: Sequence[str],
) -> dict[str, Any]:
    sequences = sorted({image_id.rsplit("_", 1)[0] for image_id in image_ids})
    output = {}
    for sequence in sequences:
        selected = {image_id for image_id in image_ids if image_id.startswith(f"{sequence}_")}
        output[sequence] = evaluate_ap50(
            subset_rows(predictions, selected),
            subset_rows(targets, selected),
            CLASS_IDS,
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic-report", type=Path, required=True)
    parser.add_argument("--crop-full-report", type=Path, required=True)
    parser.add_argument("--crop-tile-report", type=Path, required=True)
    parser.add_argument(
        "--secondary",
        action="append",
        choices=("crop_full", "crop_tile"),
        help="参与固定融合的 secondary；默认同时使用 full 与 tile。",
    )
    parser.add_argument("--fusion-iou", type=float, required=True)
    parser.add_argument("--secondary-scale", type=float, required=True)
    parser.add_argument("--agreement-bonus", type=float, required=True)
    parser.add_argument("--weighted-boxes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    secondaries = list(args.secondary or ["crop_full", "crop_tile"])
    if len(secondaries) != len(set(secondaries)):
        raise ValueError("secondary 不得重复")

    paths = {
        "generic": args.generic_report.resolve(),
        "crop_full": args.crop_full_report.resolve(),
        "crop_tile": args.crop_tile_report.resolve(),
    }
    output = args.output.resolve()
    reject_test_reference(output, "output")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有 OOF fusion：{output}")
    reports = {
        "generic": validate_source(paths["generic"], "full_frame"),
        "crop_full": validate_source(paths["crop_full"], "full_frame"),
        "crop_tile": validate_source(paths["crop_tile"], "sliding_window"),
    }
    manifest_hashes = {str(report["manifest_sha256"]) for report in reports.values()}
    image_counts = {int(report["image_count"]) for report in reports.values()}
    if len(manifest_hashes) != 1 or len(image_counts) != 1:
        raise ValueError("三个 OOF 源没有使用同一非测试 fold manifest")
    for fold_name in reports["crop_full"]["models"]:
        full_hash = reports["crop_full"]["models"][fold_name]["weight_sha256"]
        tile_hash = reports["crop_tile"]["models"][fold_name]["weight_sha256"]
        if full_hash != tile_hash:
            raise ValueError(f"{fold_name} crop full/tile 不是同一 best.pt")

    manifest_path = Path(reports["generic"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = {f"fold_{int(row['fold'])}": dict(row) for row in manifest["folds"]}
    ensemble = runpy.run_path(str(ROOT / "tools" / "72_select_base_ensemble.py"))
    output.mkdir(parents=True)
    prediction_root = output / "predictions"
    prediction_root.mkdir(parents=True)

    all_generic = []
    all_fused = []
    all_targets = []
    all_image_ids = []
    fold_metrics = {}
    for fold_name in sorted(folds, key=lambda name: int(name.rsplit("_", 1)[1])):
        source_rows = {
            name: read_jsonl(paths[name].parent / "predictions" / f"{fold_name}.jsonl")
            for name in paths
        }
        fused = ensemble["fuse_focus_class"](
            source_rows,
            "generic",
            secondaries,
            0,
            float(args.fusion_iou),
            float(args.secondary_scale),
            float(args.agreement_bonus),
            bool(args.weighted_boxes),
        )
        images = read_split(Path(folds[fold_name]["val_split"]))
        targets = yolo_ground_truth(images, CLASS_IDS)
        generic_metrics = evaluate_ap50(source_rows["generic"], targets, CLASS_IDS)
        fused_metrics = evaluate_ap50(fused, targets, CLASS_IDS)
        fold_metrics[fold_name] = {
            "image_count": len(images),
            "generic_map50": float(generic_metrics["map50"]),
            "fused_map50": float(fused_metrics["map50"]),
            "delta_map50": float(fused_metrics["map50"] - generic_metrics["map50"]),
            "per_class_ap50": fused_metrics["per_class_ap50"],
        }
        (prediction_root / f"{fold_name}.jsonl").write_text(
            "\n".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) for row in fused
            )
            + "\n",
            encoding="utf-8",
        )
        all_generic.extend(source_rows["generic"])
        all_fused.extend(fused)
        all_targets.extend(targets)
        all_image_ids.extend(image.stem for image in images)

    generic_metrics = evaluate_ap50(all_generic, all_targets, CLASS_IDS)
    fused_metrics = evaluate_ap50(all_fused, all_targets, CLASS_IDS)
    generic_subgroups = subgroup_metrics(all_generic, all_targets, all_image_ids)
    fused_subgroups = subgroup_metrics(all_fused, all_targets, all_image_ids)
    subgroup_deltas = {
        name: float(fused_subgroups[name]["map50"])
        - float(generic_subgroups[name]["map50"])
        for name in generic_subgroups
    }
    fold_scores = [float(item["fused_map50"]) for item in fold_metrics.values()]
    report = {
        "schema_version": 1,
        "selection_scope": "base_train_and_dev_oof_only",
        "lock_data_access": False,
        "manifest": str(manifest_path),
        "manifest_sha256": next(iter(manifest_hashes)),
        "image_count": next(iter(image_counts)),
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "policy": {
            "primary": "generic",
            "secondaries": secondaries,
            "focus_class": 0,
            "fusion_iou": float(args.fusion_iou),
            "secondary_scale": float(args.secondary_scale),
            "agreement_bonus": float(args.agreement_bonus),
            "weighted_boxes": bool(args.weighted_boxes),
            "policy_source": "previous_base_train_internal_temporal_holdout",
        },
        "folds": fold_metrics,
        "fold_summary": {
            "minimum_map50": min(fold_scores),
            "mean_map50": statistics.fmean(fold_scores),
            "median_map50": statistics.median(fold_scores),
            "maximum_map50": max(fold_scores),
        },
        "generic_oof": generic_metrics,
        "fused_oof": fused_metrics,
        "delta_map50": float(fused_metrics["map50"] - generic_metrics["map50"]),
        "generic_subgroups": generic_subgroups,
        "fused_subgroups": fused_subgroups,
        "subgroup_delta": subgroup_deltas,
        "worst_subgroup_delta": min(subgroup_deltas.values()),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
