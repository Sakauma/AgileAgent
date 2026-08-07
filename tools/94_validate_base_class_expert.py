#!/usr/bin/env python3
"""Evaluate a frozen difficult-class owner on a declared regression window."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import sha256_file


SELECTOR = runpy.run_path(str(ROOT / "tools" / "93_select_base_class_expert.py"))
FORWARD_VALIDATOR = runpy.run_path(
    str(ROOT / "tools" / "91_validate_base_forward_backtest.py")
)
FORBIDDEN_MARKERS = ("mixed_test", "base_test", "lock")


def reject_test_reference(path: Path, role: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"困难类回归 {role} 不得引用 test/lock：{path}")


def validate_expert_lineage(
    report_path: Path,
    selection_path: Path,
    selected: Mapping[str, Any],
    regression_status: str,
) -> Mapping[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lineage = dict(report.get("lineage_selection", {}))
    if (
        Path(str(lineage.get("path", ""))).resolve() != selection_path
        or str(lineage.get("sha256", "")) != sha256_file(selection_path)
        or str(lineage.get("selected_candidate", "")) != str(selected["name"])
        or str(lineage.get("regression_status", "")) != regression_status
        or bool(lineage.get("post_validation_was_sealed", False))
        != (regression_status == "sealed")
    ):
        raise ValueError("困难类后置 report 没有继承冻结选择边界")
    return report


def validate_class_expert_regression(
    selection_path: Path,
    forward_manifest_path: Path,
    expert_report_path: Path,
    baseline_name: str,
    baseline_report_path: Path,
    output: Path,
    required_epochs: int = 160,
    required_batch: int = 32,
    min_focus_gain: float = 0.01,
    min_composite_map50: float = 0.85,
) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    forward_manifest_path = forward_manifest_path.resolve()
    expert_report_path = expert_report_path.resolve()
    output = output.resolve()
    for path, role in (
        (selection_path, "selection"),
        (forward_manifest_path, "forward manifest"),
        (expert_report_path, "expert report"),
        (baseline_report_path.resolve(), "baseline report"),
        (output, "output"),
    ):
        reject_test_reference(path, role)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖困难类回归报告：{output}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = dict(selection.get("selected", {}))
    regression = dict(selection.get("regression_window", {}))
    regression_status = str(regression.get("status", ""))
    if (
        selection.get("selection_scope")
        != "base_train_and_dev_rolling_class_expert_selection"
        or bool(selection.get("lock_data_access", True))
        or regression_status not in {"sealed", "reused_not_independent"}
        or not bool(selection.get("candidate_selected_without_regression_reports", False))
        or not bool(selection.get("all_unknown_images_must_execute_expert", False))
        or bool(selection.get("image_level_class_routing", True))
    ):
        raise ValueError("困难类 selection 没有冻结无图片路由候选")
    focus_local = int(selection["focus_local_class"])
    focus_global = int(selection["focus_global_class"])

    rolling_manifest_path = Path(str(selection.get("manifest", ""))).resolve()
    if sha256_file(rolling_manifest_path) != str(selection.get("manifest_sha256", "")):
        raise ValueError("困难类 selection 的 rolling manifest 哈希不一致")
    rolling = json.loads(rolling_manifest_path.read_text(encoding="utf-8"))
    forward = json.loads(forward_manifest_path.read_text(encoding="utf-8"))
    window = dict(forward.get("post_validation", {}))
    if (
        str(rolling.get("source_manifest_sha256", ""))
        != str(forward.get("source_manifest_sha256", ""))
        or int(window.get("validation_fold", -1))
        != int(regression.get("validation_fold", -2))
    ):
        raise ValueError("困难类回归窗口与滚动选择来源不一致")

    raw_expert = validate_expert_lineage(
        expert_report_path,
        selection_path,
        selected,
        regression_status,
    )
    expert = SELECTOR["validate_origin_report"](
        "regression",
        str(selected["name"]),
        expert_report_path,
        window,
        focus_local,
        int(required_epochs),
        int(required_batch),
    )
    if (
        expert["dataset_recipe"] != dict(selected["dataset_recipe"])
        or expert["training_overrides"] != dict(selected["training_overrides"])
    ):
        raise ValueError("困难类回归数据配方或训练参数发生漂移")
    baseline = FORWARD_VALIDATOR["validate_report"](
        baseline_name,
        baseline_report_path,
        window,
        int(required_epochs),
        int(required_batch),
    )
    if focus_global not in {int(key) for key in baseline["per_class_ap50"]}:
        raise ValueError("baseline 缺少 focus global class")

    baseline_focus = float(baseline["per_class_ap50"][str(focus_global)])
    expert_focus = float(expert["ap50"])
    focus_gain = expert_focus - baseline_focus
    composite_per_class = dict(baseline["per_class_ap50"])
    composite_per_class[str(focus_global)] = expert_focus
    composite_map50 = sum(composite_per_class.values()) / len(composite_per_class)
    gates = {
        "focus_gain": focus_gain >= float(min_focus_gain),
        "composite_not_worse": composite_map50 >= float(baseline["map50"]),
        "composite_engineering_line": composite_map50 >= float(min_composite_map50),
        "full_epoch_budget": expert["completed_epochs"] == int(required_epochs),
        "candidate_selected_without_regression_reports": True,
        "all_unknown_images_execute_expert": True,
    }
    report = {
        "schema_version": 1,
        "selection_scope": "base_class_expert_development_regression",
        "lock_data_access": False,
        "performance_evidence": False,
        "independent_test_evidence": False,
        "evidence_kind": regression_status,
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
            "selected_candidate": str(selected["name"]),
        },
        "forward_manifest": str(forward_manifest_path),
        "forward_manifest_sha256": sha256_file(forward_manifest_path),
        "focus_local_class": focus_local,
        "focus_global_class": focus_global,
        "baseline": baseline,
        "expert": expert,
        "expert_lineage": dict(raw_expert.get("lineage_selection", {})),
        "baseline_focus_ap50": baseline_focus,
        "expert_focus_ap50": expert_focus,
        "focus_gain": focus_gain,
        "composite_per_class_ap50": composite_per_class,
        "composite_map50": composite_map50,
        "min_focus_gain": float(min_focus_gain),
        "min_composite_map50": float(min_composite_map50),
        "gates": gates,
        "accepted": all(gates.values()),
        "image_level_class_routing": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("baseline-report 必须使用 NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("baseline-report 必须使用 NAME=PATH")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--forward-manifest", type=Path, required=True)
    parser.add_argument("--expert-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-epochs", type=int, default=160)
    parser.add_argument("--required-batch", type=int, default=32)
    parser.add_argument("--min-focus-gain", type=float, default=0.01)
    parser.add_argument("--min-composite-map50", type=float, default=0.85)
    args = parser.parse_args()
    baseline_name, baseline_path = args.baseline_report
    report = validate_class_expert_regression(
        args.selection,
        args.forward_manifest,
        args.expert_report,
        baseline_name,
        baseline_path,
        args.output,
        int(args.required_epochs),
        int(args.required_batch),
        float(args.min_focus_gain),
        float(args.min_composite_map50),
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
