#!/usr/bin/env python3
"""Select non-focus base-class owners on tuning folds, then validate them."""

from __future__ import annotations

import argparse
import json
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
    yolo_ground_truth,
)


FORBIDDEN_MARKERS = ("mixed_test", "base_test", "lock")


def reject_test_reference(path: Path, role: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"{role} 不得引用 test/lock：{path}")


def named_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source 必须使用 NAME=REPORT")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("source 必须使用 NAME=REPORT")
    return name, Path(path)


def int_set(value: str) -> set[int]:
    try:
        values = {int(item) for item in value.split(",") if item}
    except ValueError as error:
        raise argparse.ArgumentTypeError("folds 必须是逗号分隔整数") from error
    if not values:
        raise argparse.ArgumentTypeError("folds 不得为空")
    return values


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def concatenate(
    rows_by_fold: Mapping[int, Sequence[Mapping[str, Any]]], folds: set[int]
) -> list[dict[str, Any]]:
    return [dict(row) for fold in sorted(folds) for row in rows_by_fold[fold]]


def rank_candidates(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Prefer worst-fold robustness, then pooled tuning AP."""
    return sorted(
        rows,
        key=lambda row: (
            -float(row["minimum_tuning_fold_ap50"]),
            -float(row["tuning_ap50"]),
            -float(row["mean_tuning_fold_ap50"]),
            str(row["name"]),
        ),
    )


def eligible_owner_candidates(
    rows: Sequence[Mapping[str, Any]], baseline_name: str
) -> list[Mapping[str, Any]]:
    baseline = next(
        (row for row in rows if str(row["name"]) == str(baseline_name)), None
    )
    if baseline is None:
        raise ValueError("owner candidates 缺少 baseline")
    return [
        row
        for row in rows
        if float(row["minimum_tuning_fold_ap50"])
        >= float(baseline["minimum_tuning_fold_ap50"])
        and float(row["tuning_ap50"]) >= float(baseline["tuning_ap50"])
    ]


def metrics_for(
    predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    targets: Mapping[int, Sequence[Mapping[str, Any]]],
    folds: set[int],
    class_id: int,
) -> dict[str, Any]:
    pooled = evaluate_ap50(
        concatenate(predictions, folds), concatenate(targets, folds), [class_id]
    )
    fold_ap50 = {
        str(index): float(
            evaluate_ap50(predictions[index], targets[index], [class_id])["map50"]
        )
        for index in sorted(folds)
    }
    return {
        "ap50": float(pooled["map50"]),
        "fold_ap50": fold_ap50,
        "minimum_fold_ap50": min(fold_ap50.values()),
        "mean_fold_ap50": sum(fold_ap50.values()) / len(fold_ap50),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=named_report, action="append", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--class-id", type=int, action="append", default=[])
    parser.add_argument("--tuning-folds", type=int_set, default=int_set("0,1,2"))
    parser.add_argument("--validation-folds", type=int_set, default=int_set("3,4"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_items = args.source or []
    if len(source_items) < 2 or len({name for name, _path in source_items}) != len(
        source_items
    ):
        raise ValueError("至少需要两个名称唯一的 OOF source")
    paths = {name: path.resolve() for name, path in source_items}
    if args.baseline not in paths:
        raise ValueError("baseline 必须是已注册 source")
    class_ids = args.class_id or [1, 3]
    if len(class_ids) != len(set(class_ids)) or any(value == 0 for value in class_ids):
        raise ValueError("class-id 必须唯一且不包含 focus class 0")
    output = args.output.resolve()
    reject_test_reference(output, "class owner selection output")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有 class owner selection：{output}")

    reports = {}
    for name, path in paths.items():
        reject_test_reference(path, f"OOF source {name}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("selection_scope") != "base_train_and_dev_oof_only"
            or bool(report.get("lock_data_access", True))
            or str(report.get("inference_mode")) != "full_frame"
        ):
            raise ValueError(f"OOF source 范围或推理模式错误：{name}")
        reports[name] = report
    manifest_hashes = {str(report["manifest_sha256"]) for report in reports.values()}
    manifest_paths = {str(Path(report["manifest"]).resolve()) for report in reports.values()}
    if len(manifest_hashes) != 1 or len(manifest_paths) != 1:
        raise ValueError("OOF sources 必须使用同一 fold manifest")
    manifest_path = Path(next(iter(manifest_paths)))
    reject_test_reference(manifest_path, "fold manifest")
    if sha256_file(manifest_path) != next(iter(manifest_hashes)):
        raise ValueError("fold manifest 哈希不一致")
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
    fold_images = {}
    for fold_index, fold in sorted(folds.items()):
        fold_name = f"fold_{fold_index}"
        fold_images[fold_index] = read_split(Path(fold["val_split"]))
        for name, path in paths.items():
            prediction_path = path.parent / "predictions" / f"{fold_name}.jsonl"
            reject_test_reference(prediction_path, f"OOF prediction {name}/{fold_name}")
            predictions[name][fold_index] = read_jsonl(prediction_path)

    # Only tuning-fold labels are opened while owner names are selected.
    targets = {
        index: yolo_ground_truth(fold_images[index], class_ids)
        for index in sorted(tuning_folds)
    }
    selected_names: dict[int, str] = {}
    tuning_rankings = {}
    for class_id in class_ids:
        rows = []
        for name in sorted(paths):
            metrics = metrics_for(
                predictions[name], targets, tuning_folds, int(class_id)
            )
            rows.append(
                {
                    "name": name,
                    "tuning_ap50": metrics["ap50"],
                    "tuning_fold_ap50": metrics["fold_ap50"],
                    "minimum_tuning_fold_ap50": metrics["minimum_fold_ap50"],
                    "mean_tuning_fold_ap50": metrics["mean_fold_ap50"],
                }
            )
        ranking = rank_candidates(rows)
        eligible = rank_candidates(eligible_owner_candidates(rows, args.baseline))
        selected_names[int(class_id)] = str(eligible[0]["name"])
        eligible_names = {str(row["name"]) for row in eligible}
        tuning_rankings[str(class_id)] = [
            {
                "rank": rank,
                "eligible_against_baseline": str(row["name"]) in eligible_names,
                **dict(row),
            }
            for rank, row in enumerate(ranking, start=1)
        ]

    # Freeze all selected owner predictions before validation labels are opened.
    selected_predictions = {
        class_id: {
            fold_index: [dict(row) for row in predictions[name][fold_index]]
            for fold_index in expected_folds
        }
        for class_id, name in selected_names.items()
    }
    targets.update(
        {
            index: yolo_ground_truth(fold_images[index], class_ids)
            for index in sorted(validation_folds)
        }
    )

    class_results = {}
    gates = {}
    for class_id, selected_name in selected_names.items():
        tuning = metrics_for(
            selected_predictions[class_id], targets, tuning_folds, class_id
        )
        validation = metrics_for(
            selected_predictions[class_id], targets, validation_folds, class_id
        )
        all_oof = metrics_for(
            selected_predictions[class_id], targets, expected_folds, class_id
        )
        baseline_tuning = metrics_for(
            predictions[args.baseline], targets, tuning_folds, class_id
        )
        baseline_validation = metrics_for(
            predictions[args.baseline], targets, validation_folds, class_id
        )
        class_gates = {
            "minimum_tuning_fold_not_worse": tuning["minimum_fold_ap50"]
            >= baseline_tuning["minimum_fold_ap50"],
            "tuning_ap50_not_worse": tuning["ap50"] >= baseline_tuning["ap50"],
            "validation_ap50_not_worse": validation["ap50"]
            >= baseline_validation["ap50"],
        }
        if not all(class_gates.values()):
            raise RuntimeError(
                f"class {class_id} 的 tuning-selected owner 未通过后置验证：{class_gates}"
            )
        gates[str(class_id)] = class_gates
        class_results[str(class_id)] = {
            "selected_owner": selected_name,
            "tuning": tuning,
            "validation": validation,
            "all_oof": all_oof,
            "baseline_owner": args.baseline,
            "baseline_tuning": baseline_tuning,
            "baseline_validation": baseline_validation,
        }

    report = {
        "schema_version": 1,
        "selection_scope": "base_train_and_dev_oof_class_owner_selection",
        "lock_data_access": False,
        "selection_basis": "maximin_tuning_fold_ap50_then_pooled_tuning_ap50",
        "manifest": str(manifest_path),
        "manifest_sha256": next(iter(manifest_hashes)),
        "tuning_folds": sorted(tuning_folds),
        "validation_folds": sorted(validation_folds),
        "validation_labels_opened_after_owner_selection": True,
        "validation_predictions_frozen_before_labels": True,
        "baseline": str(args.baseline),
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "selected_owners": {str(key): value for key, value in selected_names.items()},
        "tuning_rankings": tuning_rankings,
        "class_results": class_results,
        "gates": gates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
