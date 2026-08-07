#!/usr/bin/env python3
"""Validate a frozen forward-backtest candidate on the final temporal block."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import sha256_file


FORBIDDEN_MARKERS = ("mixed_test", "base_test", "lock")
BASE_LOCAL_TO_GLOBAL = {0: 0, 1: 1, 2: 3}


def reject_test_reference(path: Path, role: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"前向后置验证 {role} 不得引用 test/lock：{path}")


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("baseline-report 必须使用 NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("baseline-report 必须使用 NAME=PATH")
    return name, Path(path)


def global_per_class(values: Mapping[str, Any]) -> dict[str, float]:
    local = {int(key): float(value) for key, value in values.items()}
    if set(local) != set(BASE_LOCAL_TO_GLOBAL):
        raise ValueError("后置验证必须且只能包含三个基础局部类别")
    return {
        str(BASE_LOCAL_TO_GLOBAL[key]): local[key]
        for key in sorted(BASE_LOCAL_TO_GLOBAL)
    }


def dataset_recipe(dataset: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "recent_fraction": dataset.get("recent_fraction"),
        "recent_full_repeats": int(dataset.get("recent_full_repeats", 0)),
        "crop_enabled": bool(dataset.get("crop_enabled", False)),
        "crop_strategy": dataset.get("crop_strategy"),
        "crop_size": dataset.get("crop_size"),
        "crop_overlap": dataset.get("crop_overlap"),
        "jitter_fraction": dataset.get("jitter_fraction"),
        "min_visible_fraction": dataset.get("min_visible_fraction"),
    }


def validate_report(
    name: str,
    path: Path,
    window: Mapping[str, Any],
    required_epochs: int,
    required_batch: int,
    selection_path: Path | None = None,
    selected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    reject_test_reference(path, f"report {name}")
    report = json.loads(path.read_text(encoding="utf-8"))
    training = dict(report.get("training", {}))
    arguments = dict(training.get("arguments", {}))
    audit = dict(report.get("dataset_audit", {}))
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
        or int(audit.get("dev_count", -1)) != int(window["val_count"])
        or bool(audit.get("source_declared_test_split", True))
        or bool(audit.get("training_declared_test_split", True))
        or list(audit.get("train_dev_overlap", ["missing"]))
    ):
        raise ValueError(f"后置验证 report {name} 未通过完整预算审计")

    source_dataset = Path(str(audit.get("dataset_yaml", ""))).resolve()
    manifest_path = source_dataset.parent / "manifest.json"
    reject_test_reference(manifest_path, f"dataset {name}")
    dataset = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        dataset.get("selection_scope") != "base_train_and_base_dev_only"
        or bool(dataset.get("lock_data_access", True))
        or str(dataset.get("split_mode", "")) != "external"
        or str(dataset.get("source_split_sha256", ""))
        != str(window["train_split_sha256"])
        or str(dataset.get("val_source_split_sha256", ""))
        != str(window["val_split_sha256"])
        or int(dataset.get("train_source_count", -1)) != int(window["train_count"])
        or int(dataset.get("val_source_count", -1)) != int(window["val_count"])
        or list(dataset.get("source_train_val_overlap", ["missing"]))
    ):
        raise ValueError(f"后置验证 report {name} 数据边界错误")

    if selection_path is not None and selected is not None:
        lineage = dict(report.get("lineage_selection", {}))
        if (
            Path(str(lineage.get("path", ""))).resolve() != selection_path
            or str(lineage.get("sha256", "")) != sha256_file(selection_path)
            or str(lineage.get("selected_candidate", "")) != name
            or not bool(lineage.get("post_validation_was_sealed", False))
            or dataset_recipe(dataset) != dict(selected["dataset_recipe"])
        ):
            raise ValueError("后置候选没有继承冻结的调参选择与数据配方")
        for key, expected in dict(selected["training_overrides"]).items():
            if expected is not None and arguments.get(key) != expected:
                raise ValueError(f"后置候选训练参数漂移：{key}")

    weight = Path(str(report.get("best_weight", ""))).resolve()
    if sha256_file(weight) != str(report.get("best_weight_sha256", "")):
        raise ValueError(f"后置验证 report {name} best.pt 哈希不一致")
    return {
        "name": name,
        "report": str(path),
        "report_sha256": sha256_file(path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "best_weight": str(weight),
        "best_weight_sha256": sha256_file(weight),
        "completed_epochs": int(training["completed_epochs"]),
        "best_epoch": int(training["best_epoch"]),
        "map50": float(evaluation["map50"]),
        "per_class_ap50": global_per_class(dict(evaluation["per_class_ap50"])),
        "dataset_recipe": dataset_recipe(dataset),
    }


def validate_forward_backtest(
    selection_path: Path,
    candidate_report: Path,
    baseline_name: str,
    baseline_report: Path,
    output: Path,
    required_epochs: int = 160,
    required_batch: int = 32,
    max_class_drop: float = 0.01,
) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    output = output.resolve()
    reject_test_reference(selection_path, "selection")
    reject_test_reference(output, "output")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有前向后置验证：{output}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = dict(selection.get("selected", {}))
    if (
        selection.get("selection_scope")
        != "base_train_and_dev_forward_tuning_only"
        or bool(selection.get("lock_data_access", True))
        or str(selection.get("post_validation", {}).get("status", "")) != "sealed"
        or bool(selection.get("post_validation", {}).get("labels_opened", True))
    ):
        raise ValueError("前向选择报告未在后置标签封存状态冻结候选")
    manifest_path = Path(str(selection.get("manifest", ""))).resolve()
    if sha256_file(manifest_path) != str(selection.get("manifest_sha256", "")):
        raise ValueError("前向选择报告的 manifest 哈希不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    window = dict(manifest["post_validation"])
    candidate = validate_report(
        str(selected["name"]),
        candidate_report,
        window,
        int(required_epochs),
        int(required_batch),
        selection_path,
        selected,
    )
    baseline = validate_report(
        baseline_name,
        baseline_report,
        window,
        int(required_epochs),
        int(required_batch),
    )
    per_class_delta = {
        key: float(candidate["per_class_ap50"][key])
        - float(baseline["per_class_ap50"][key])
        for key in baseline["per_class_ap50"]
    }
    delta_map50 = float(candidate["map50"]) - float(baseline["map50"])
    gates = {
        "post_validation_map50_not_worse": delta_map50 >= 0.0,
        "post_validation_class_regression_guard": min(per_class_delta.values())
        >= -float(max_class_drop),
        "full_epoch_budget": candidate["completed_epochs"] == int(required_epochs),
        "selection_frozen_before_post_validation": True,
    }
    report = {
        "schema_version": 1,
        "selection_scope": "base_train_and_dev_forward_tune_then_validate",
        "lock_data_access": False,
        "performance_evidence": False,
        "independent_test_evidence": False,
        "evidence_kind": "forward_temporal_development_backtest",
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
            "selected_candidate": str(selected["name"]),
        },
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "baseline": baseline,
        "candidate": candidate,
        "delta_map50_vs_baseline": delta_map50,
        "per_class_delta_vs_baseline": per_class_delta,
        "max_class_drop": float(max_class_drop),
        "gates": gates,
        "accepted": all(gates.values()),
        "post_validation_labels_opened_after_candidate_freeze": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-epochs", type=int, default=160)
    parser.add_argument("--required-batch", type=int, default=32)
    parser.add_argument("--max-class-drop", type=float, default=0.01)
    args = parser.parse_args()
    baseline_name, baseline_path = args.baseline_report
    report = validate_forward_backtest(
        args.selection,
        args.candidate_report,
        baseline_name,
        baseline_path,
        args.output,
        int(args.required_epochs),
        int(args.required_batch),
        float(args.max_class_drop),
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
