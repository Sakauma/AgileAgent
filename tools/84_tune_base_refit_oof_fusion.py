#!/usr/bin/env python3
"""Tune base small-object fusion on OOF folds and validate on later OOF folds."""

from __future__ import annotations

import argparse
import itertools
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import evaluate_ap50, read_split, sha256_file, yolo_ground_truth


CLASS_IDS = [0, 1, 3]
FORBIDDEN_MARKERS = ("mixed_test", "base_test", "lock")


def reject_test_reference(path: Path, role: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"OOF fusion {role} 不得引用 test/lock：{path}")


def float_grid(value: str) -> list[float]:
    try:
        values = [float(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("参数网格必须是逗号分隔浮点数") from error
    if not values:
        raise argparse.ArgumentTypeError("参数网格不得为空")
    return values


def int_set(value: str) -> set[int]:
    try:
        values = {int(item) for item in value.split(",") if item}
    except ValueError as error:
        raise argparse.ArgumentTypeError("folds 必须是逗号分隔整数") from error
    if not values:
        raise argparse.ArgumentTypeError("folds 不得为空")
    return values


def named_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("额外 OOF 报告必须使用 NAME=PATH")
    name, path = value.split("=", 1)
    if not name or name == "generic" or not path:
        raise argparse.ArgumentTypeError("额外 OOF 报告必须使用非 generic 的 NAME=PATH")
    return name, Path(path)


def secondary_set(value: str) -> tuple[str, ...]:
    names = tuple(item for item in value.split(",") if item)
    if not names or len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("secondary-set 必须是非空且不重复的逗号分隔名称")
    return names


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def concatenate(
    rows_by_fold: Mapping[int, Sequence[Mapping[str, Any]]], folds: set[int]
) -> list[dict[str, Any]]:
    return [dict(row) for fold in sorted(folds) for row in rows_by_fold[fold]]


def eligible_candidates(
    candidates: Sequence[Mapping[str, Any]], max_degraded_tuning_folds: int
) -> list[Mapping[str, Any]]:
    if max_degraded_tuning_folds < 0:
        raise ValueError("max-degraded-tuning-folds 不得为负")
    return [
        row
        for row in candidates
        if int(row["degraded_tuning_fold_count"]) <= max_degraded_tuning_folds
        and float(row["tuning_delta_vs_generic"]) > 0.0
    ]


def open_fold_targets(
    fold_images: Mapping[int, Sequence[Path]], selected_folds: set[int]
) -> dict[int, list[dict[str, Any]]]:
    return {
        fold_index: yolo_ground_truth(fold_images[fold_index], CLASS_IDS)
        for fold_index in sorted(selected_folds)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic-report", type=Path, required=True)
    parser.add_argument("--crop-full-report", type=Path, required=True)
    parser.add_argument("--crop-tile-report", type=Path, required=True)
    parser.add_argument(
        "--extra-full-report",
        type=named_report,
        action="append",
        help="注册额外完整帧 OOF owner，使用 NAME=REPORT；可重复。",
    )
    parser.add_argument(
        "--secondary",
        action="append",
        help="参与网格选择的已注册 secondary 名称；默认使用 crop_full 与 crop_tile。",
    )
    parser.add_argument(
        "--secondary-set",
        type=secondary_set,
        action="append",
        help="把一组 owner 注册为待比较组合，例如 crop_full,recent_crop；可重复。",
    )
    parser.add_argument("--tuning-folds", type=int_set, default=int_set("0,1,2"))
    parser.add_argument("--validation-folds", type=int_set, default=int_set("3,4"))
    parser.add_argument("--fusion-ious", type=float_grid, default=float_grid("0.25,0.35,0.45,0.55"))
    parser.add_argument(
        "--secondary-scales", type=float_grid, default=float_grid("0.50,0.65,0.80,1.00")
    )
    parser.add_argument(
        "--agreement-bonuses", type=float_grid, default=float_grid("0.00,0.05,0.10,0.15,0.20")
    )
    parser.add_argument("--weighted-box-options", default="false,true")
    parser.add_argument("--max-degraded-tuning-folds", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "generic": args.generic_report.resolve(),
        "crop_full": args.crop_full_report.resolve(),
        "crop_tile": args.crop_tile_report.resolve(),
    }
    for name, path in args.extra_full_report or []:
        if name in paths:
            raise ValueError(f"额外 OOF owner 名称重复：{name}")
        paths[name] = path.resolve()
    if args.secondary and args.secondary_set:
        raise ValueError("--secondary 与 --secondary-set 不得同时使用")
    secondary_sets = [
        list(values)
        for values in (
            args.secondary_set
            or [tuple(args.secondary or ["crop_full", "crop_tile"])]
        )
    ]
    if len({tuple(values) for values in secondary_sets}) != len(secondary_sets):
        raise ValueError("secondary-set 不得重复")
    registered_secondaries = set(paths) - {"generic"}
    unknown_secondaries = sorted(
        {name for values in secondary_sets for name in values} - registered_secondaries
    )
    if unknown_secondaries:
        raise ValueError(f"secondary 未注册 OOF 报告：{unknown_secondaries}")
    output = args.output.resolve()
    reject_test_reference(output, "output")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有 OOF fusion tune：{output}")
    weighted_options = []
    for value in args.weighted_box_options.split(","):
        lowered = value.strip().lower()
        if lowered not in {"true", "false"}:
            raise ValueError("weighted-box-options 只允许 false,true")
        weighted_options.append(lowered == "true")

    reports = {
        name: validate_source(
            path,
            "sliding_window" if name == "crop_tile" else "full_frame",
        )
        for name, path in paths.items()
    }
    manifest_hashes = {str(report["manifest_sha256"]) for report in reports.values()}
    if len(manifest_hashes) != 1:
        raise ValueError("OOF 源没有使用同一 fold manifest")
    manifest_path = Path(reports["generic"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = {int(row["fold"]): dict(row) for row in manifest["folds"]}
    expected_folds = set(folds)
    tuning_folds = set(args.tuning_folds)
    validation_folds = set(args.validation_folds)
    if tuning_folds & validation_folds or tuning_folds | validation_folds != expected_folds:
        raise ValueError("tuning/validation folds 必须互斥且完整覆盖 manifest")

    predictions: dict[str, dict[int, list[dict[str, Any]]]] = {
        name: {} for name in paths
    }
    fold_images: dict[int, list[Path]] = {}
    for fold_index, fold in sorted(folds.items()):
        fold_name = f"fold_{fold_index}"
        for name, path in paths.items():
            predictions[name][fold_index] = read_jsonl(
                path.parent / "predictions" / f"{fold_name}.jsonl"
            )
        # The image list and frozen prediction artifacts are label-blind.  Do
        # not open this fold's labels here: validation labels remain sealed
        # until the tuning-only ranking has selected exactly one policy.
        fold_images[fold_index] = read_split(Path(fold["val_split"]))

    ensemble = runpy.run_path(str(ROOT / "tools" / "72_select_base_ensemble.py"))
    targets = open_fold_targets(fold_images, tuning_folds)
    generic_tune = concatenate(predictions["generic"], tuning_folds)
    targets_tune = concatenate(targets, tuning_folds)
    generic_tune_map = float(evaluate_ap50(generic_tune, targets_tune, CLASS_IDS)["map50"])
    candidates = []
    for selected_secondaries, iou, scale, bonus, weighted in itertools.product(
        secondary_sets,
        args.fusion_ious,
        args.secondary_scales,
        args.agreement_bonuses,
        weighted_options,
    ):
        key = (float(iou), float(scale), float(bonus), bool(weighted))
        by_fold = {}
        fold_scores = {}
        fold_deltas = {}
        for fold_index in tuning_folds:
            source = {name: predictions[name][fold_index] for name in predictions}
            fused = ensemble["fuse_focus_class"](
                source,
                "generic",
                selected_secondaries,
                0,
                key[0],
                key[1],
                key[2],
                key[3],
            )
            by_fold[fold_index] = fused
            if fold_index in tuning_folds:
                fused_map = float(
                    evaluate_ap50(fused, targets[fold_index], CLASS_IDS)["map50"]
                )
                generic_map = float(
                    evaluate_ap50(
                        predictions["generic"][fold_index], targets[fold_index], CLASS_IDS
                    )["map50"]
                )
                fold_scores[fold_index] = fused_map
                fold_deltas[fold_index] = fused_map - generic_map
        tune_metrics = evaluate_ap50(
            concatenate(by_fold, tuning_folds), targets_tune, CLASS_IDS
        )
        candidates.append(
            {
                "secondaries": list(selected_secondaries),
                "fusion_iou": key[0],
                "secondary_scale": key[1],
                "agreement_bonus": key[2],
                "weighted_boxes": key[3],
                "tuning_map50": float(tune_metrics["map50"]),
                "tuning_per_class_ap50": tune_metrics["per_class_ap50"],
                "tuning_delta_vs_generic": float(tune_metrics["map50"]) - generic_tune_map,
                "minimum_tuning_fold_map50": min(fold_scores.values()),
                "worst_tuning_fold_delta": min(fold_deltas.values()),
                "degraded_tuning_fold_count": sum(value < -0.01 for value in fold_deltas.values()),
            }
        )
    candidates.sort(
        key=lambda row: (
            -float(row["tuning_map50"]),
            int(row["degraded_tuning_fold_count"]),
            -float(row["minimum_tuning_fold_map50"]),
            -float(row["worst_tuning_fold_delta"]),
            float(row["fusion_iou"]),
            float(row["secondary_scale"]),
            float(row["agreement_bonus"]),
            bool(row["weighted_boxes"]),
            tuple(row["secondaries"]),
        )
    )
    eligible = eligible_candidates(candidates, int(args.max_degraded_tuning_folds))
    if not eligible:
        raise RuntimeError("没有同时改善调参集且满足逐折退化上限的融合策略")
    selected = eligible[0]
    selected_key = (
        float(selected["fusion_iou"]),
        float(selected["secondary_scale"]),
        float(selected["agreement_bonus"]),
        bool(selected["weighted_boxes"]),
    )
    selected_secondaries = list(selected["secondaries"])
    selected_by_fold = {}
    # Freeze the one selected policy's predictions on every fold before any
    # validation-fold label is opened.
    for fold_index in expected_folds:
        source = {name: predictions[name][fold_index] for name in predictions}
        selected_by_fold[fold_index] = ensemble["fuse_focus_class"](
            source,
            "generic",
            selected_secondaries,
            0,
            selected_key[0],
            selected_key[1],
            selected_key[2],
            selected_key[3],
        )
    validation_targets = open_fold_targets(fold_images, validation_folds)
    targets.update(validation_targets)

    def metrics_for(selected_folds: set[int]) -> dict[str, Any]:
        fused_rows = concatenate(selected_by_fold, selected_folds)
        generic_rows = concatenate(predictions["generic"], selected_folds)
        target_rows = concatenate(targets, selected_folds)
        fused = evaluate_ap50(fused_rows, target_rows, CLASS_IDS)
        generic = evaluate_ap50(generic_rows, target_rows, CLASS_IDS)
        return {
            "image_count": sum(int(folds[index]["val_count"]) for index in selected_folds),
            "generic_map50": float(generic["map50"]),
            "generic_per_class_ap50": generic["per_class_ap50"],
            "fused_map50": float(fused["map50"]),
            "fused_per_class_ap50": fused["per_class_ap50"],
            "delta_map50": float(fused["map50"] - generic["map50"]),
        }

    report = {
        "schema_version": 1,
        "selection_scope": "base_train_and_dev_oof_tune_validate",
        "lock_data_access": False,
        "manifest": str(manifest_path),
        "manifest_sha256": next(iter(manifest_hashes)),
        "tuning_folds": sorted(tuning_folds),
        "validation_folds": sorted(validation_folds),
        "secondaries": selected_secondaries,
        "candidate_secondary_sets": secondary_sets,
        "grid_size": len(candidates),
        "eligible_grid_size": len(eligible),
        "max_degraded_tuning_folds": int(args.max_degraded_tuning_folds),
        "validation_labels_opened_after_policy_selection": True,
        "validation_predictions_frozen_before_labels": True,
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "selected_policy": selected,
        "tuning": metrics_for(tuning_folds),
        "validation": metrics_for(validation_folds),
        "all_oof_diagnostic": metrics_for(expected_folds),
        "top_tuning_candidates": candidates[:20],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
