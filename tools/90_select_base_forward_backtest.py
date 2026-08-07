#!/usr/bin/env python3
"""Select a base candidate on the tuning window of a forward backtest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import sha256_file


FORBIDDEN_MARKERS = ("mixed_test", "base_test", "lock")
BASE_LOCAL_TO_GLOBAL = {0: 0, 1: 1, 2: 3}


def reject_test_reference(path: Path, role: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"前向回测 {role} 不得引用 test/lock：{path}")


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("report 必须使用 NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("report 必须使用 NAME=PATH")
    return name, Path(path)


def global_per_class(values: Mapping[str, Any]) -> dict[str, float]:
    local = {int(key): float(value) for key, value in values.items()}
    if set(local) != set(BASE_LOCAL_TO_GLOBAL):
        raise ValueError("前向回测必须且只能包含三个基础局部类别")
    return {
        str(BASE_LOCAL_TO_GLOBAL[key]): local[key]
        for key in sorted(BASE_LOCAL_TO_GLOBAL)
    }


def validate_training_report(
    name: str,
    path: Path,
    tuning: Mapping[str, Any],
    required_epochs: int,
    required_batch: int,
) -> dict[str, Any]:
    path = path.resolve()
    reject_test_reference(path, f"candidate {name} report")
    report = json.loads(path.read_text(encoding="utf-8"))
    training = dict(report.get("training", {}))
    arguments = dict(training.get("arguments", {}))
    dataset_audit = dict(report.get("dataset_audit", {}))
    evaluation = dict(report.get("evaluation", {}))
    if (
        str(report.get("candidate", "")) != name
        or report.get("selection_scope") != "base_train_and_dev_only"
        or bool(report.get("lock_data_access", True))
        or int(training.get("requested_epochs", -1)) != required_epochs
        or int(training.get("completed_epochs", -1)) != required_epochs
        or bool(training.get("stopped_early", True))
        or int(arguments.get("epochs", -1)) != required_epochs
        or int(arguments.get("batch", -1)) != required_batch
        or int(arguments.get("patience", -1)) != 0
        or int(arguments.get("imgsz", -1)) != 896
        or int(evaluation.get("imgsz", -1)) != 896
        or int(dataset_audit.get("dev_count", -1)) != int(tuning["val_count"])
        or bool(dataset_audit.get("source_declared_test_split", True))
        or bool(dataset_audit.get("training_declared_test_split", True))
        or list(dataset_audit.get("train_dev_overlap", ["missing"]))
    ):
        raise ValueError(f"candidate {name} 未通过完整前向训练审计")

    dataset_yaml = Path(str(dataset_audit.get("dataset_yaml", ""))).resolve()
    dataset_manifest = dataset_yaml.parent / "manifest.json"
    reject_test_reference(dataset_manifest, f"candidate {name} dataset manifest")
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if (
        dataset.get("selection_scope") != "base_train_and_base_dev_only"
        or bool(dataset.get("lock_data_access", True))
        or str(dataset.get("split_mode", "")) != "external"
        or str(dataset.get("source_split_sha256", ""))
        != str(tuning["train_split_sha256"])
        or str(dataset.get("val_source_split_sha256", ""))
        != str(tuning["val_split_sha256"])
        or int(dataset.get("train_source_count", -1)) != int(tuning["train_count"])
        or int(dataset.get("val_source_count", -1)) != int(tuning["val_count"])
        or list(dataset.get("source_train_val_overlap", ["missing"]))
    ):
        raise ValueError(f"candidate {name} 数据没有使用相同前向调参窗口")

    weight = Path(str(report.get("best_weight", ""))).resolve()
    if sha256_file(weight) != str(report.get("best_weight_sha256", "")):
        raise ValueError(f"candidate {name} best.pt 哈希不一致")
    return {
        "name": name,
        "report": str(path),
        "report_sha256": sha256_file(path),
        "dataset_manifest": str(dataset_manifest),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "best_weight": str(weight),
        "best_weight_sha256": sha256_file(weight),
        "completed_epochs": int(training["completed_epochs"]),
        "best_epoch": int(training["best_epoch"]),
        "map50": float(evaluation["map50"]),
        "per_class_ap50": global_per_class(dict(evaluation["per_class_ap50"])),
        "dataset_recipe": {
            "recent_fraction": dataset.get("recent_fraction"),
            "recent_full_repeats": int(dataset.get("recent_full_repeats", 0)),
            "crop_enabled": bool(dataset.get("crop_enabled", False)),
            "crop_strategy": dataset.get("crop_strategy"),
            "crop_size": dataset.get("crop_size"),
            "crop_overlap": dataset.get("crop_overlap"),
            "jitter_fraction": dataset.get("jitter_fraction"),
            "min_visible_fraction": dataset.get("min_visible_fraction"),
        },
        "training_overrides": {
            key: arguments.get(key)
            for key in (
                "lr0",
                "weight_decay",
                "mosaic",
                "translate",
                "scale",
                "close_mosaic",
            )
        },
    }


def rank_candidates(
    rows: Sequence[Mapping[str, Any]],
    baseline_name: str,
    max_class_drop: float,
) -> list[dict[str, Any]]:
    baseline = next(
        (dict(row) for row in rows if str(row["name"]) == baseline_name), None
    )
    if baseline is None:
        raise ValueError("前向候选缺少 baseline")
    output = []
    for row in rows:
        item = dict(row)
        class_deltas = {
            key: float(item["per_class_ap50"][key])
            - float(baseline["per_class_ap50"][key])
            for key in baseline["per_class_ap50"]
        }
        item["delta_map50_vs_baseline"] = float(item["map50"]) - float(
            baseline["map50"]
        )
        item["per_class_delta_vs_baseline"] = class_deltas
        item["minimum_class_delta_vs_baseline"] = min(class_deltas.values())
        item["eligible"] = (
            item["delta_map50_vs_baseline"] >= 0.0
            and item["minimum_class_delta_vs_baseline"] >= -max_class_drop
        )
        output.append(item)
    return sorted(
        output,
        key=lambda row: (
            not bool(row["eligible"]),
            -float(row["map50"]),
            -float(row["minimum_class_delta_vs_baseline"]),
            str(row["name"]),
        ),
    )


def select_forward_candidate(
    manifest_path: Path,
    report_paths: Mapping[str, Path],
    baseline_name: str,
    output: Path,
    required_epochs: int = 160,
    required_batch: int = 32,
    max_class_drop: float = 0.01,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output = output.resolve()
    reject_test_reference(manifest_path, "manifest")
    reject_test_reference(output, "output")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有前向选择报告：{output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("selection_scope") != "base_train_and_dev_forward_only"
        or bool(manifest.get("lock_data_access", True))
        or manifest.get("strategy") != "expanding_window_temporal_backtest"
        or not bool(
            manifest.get(
                "post_validation_labels_must_remain_closed_during_tuning", False
            )
        )
    ):
        raise ValueError("manifest 不是无测试前向回测边界")
    tuning = dict(manifest["tuning"])
    rows = [
        validate_training_report(
            name,
            path,
            tuning,
            int(required_epochs),
            int(required_batch),
        )
        for name, path in sorted(report_paths.items())
    ]
    ranking = rank_candidates(rows, baseline_name, float(max_class_drop))
    eligible = [row for row in ranking if bool(row["eligible"])]
    if not eligible:
        raise RuntimeError("没有通过前向调参门禁的候选")
    selected = dict(eligible[0])
    report = {
        "schema_version": 1,
        "selection_scope": "base_train_and_dev_forward_tuning_only",
        "lock_data_access": False,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "baseline": baseline_name,
        "selection_basis": "highest_tuning_map50_with_per_class_regression_guard",
        "max_class_drop": float(max_class_drop),
        "selected": selected,
        "ranking": ranking,
        "post_validation": {
            "status": "sealed",
            "validation_fold": int(manifest["post_validation"]["validation_fold"]),
            "labels_opened": False,
            "candidate_must_be_frozen_before_training_and_scoring": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=named_path, action="append", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-epochs", type=int, default=160)
    parser.add_argument("--required-batch", type=int, default=32)
    parser.add_argument("--max-class-drop", type=float, default=0.01)
    args = parser.parse_args()
    report_paths = dict(args.report)
    if len(report_paths) != len(args.report):
        raise ValueError("report 名称不得重复")
    report = select_forward_candidate(
        args.manifest,
        report_paths,
        args.baseline,
        args.output,
        int(args.required_epochs),
        int(args.required_batch),
        float(args.max_class_drop),
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
