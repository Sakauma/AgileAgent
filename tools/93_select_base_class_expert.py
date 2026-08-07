#!/usr/bin/env python3
"""Select a difficult-base-class owner across sealed rolling origins."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import sha256_file


FORBIDDEN_MARKERS = ("mixed_test", "base_test", "lock")
BASE_LOCAL_TO_GLOBAL = {0: 0, 1: 1, 2: 3}
TRAINING_OVERRIDE_KEYS = (
    "classes",
    "imgsz",
    "lr0",
    "weight_decay",
    "mosaic",
    "translate",
    "scale",
    "close_mosaic",
    "cls",
    "box",
)


def reject_test_reference(path: Path, role: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"困难类 owner {role} 不得引用 test/lock：{path}")


def report_spec(value: str) -> tuple[str, str, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("report 必须使用 ORIGIN:CANDIDATE=PATH")
    left, path = value.split("=", 1)
    origin, candidate = left.split(":", 1)
    if not origin or not candidate or not path:
        raise argparse.ArgumentTypeError("report 必须使用 ORIGIN:CANDIDATE=PATH")
    return origin, candidate, Path(path)


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


def validate_origin_report(
    origin: str,
    candidate: str,
    path: Path,
    window: Mapping[str, Any],
    focus_local_class: int,
    required_epochs: int,
    required_batch: int,
) -> dict[str, Any]:
    path = path.resolve()
    reject_test_reference(path, f"report {origin}:{candidate}")
    report = json.loads(path.read_text(encoding="utf-8"))
    training = dict(report.get("training", {}))
    arguments = dict(training.get("arguments", {}))
    audit = dict(report.get("dataset_audit", {}))
    evaluation = dict(report.get("evaluation", {}))
    class_filter = [int(value) for value in evaluation.get("class_filter") or []]
    train_classes = [int(value) for value in arguments.get("classes") or []]
    per_class = {
        int(key): float(value)
        for key, value in dict(evaluation.get("per_class_ap50", {})).items()
    }
    if (
        str(report.get("candidate", "")) != candidate
        or report.get("selection_scope") != "base_train_and_dev_only"
        or bool(report.get("lock_data_access", True))
        or int(training.get("requested_epochs", -1)) != required_epochs
        or int(training.get("completed_epochs", -1)) != required_epochs
        or bool(training.get("stopped_early", True))
        or int(arguments.get("epochs", -1)) != required_epochs
        or int(arguments.get("batch", -1)) != required_batch
        or int(arguments.get("patience", -1)) != 0
        or train_classes != [focus_local_class]
        or class_filter != [focus_local_class]
        or set(per_class) != {focus_local_class}
        or int(audit.get("dev_count", -1)) != int(window["val_count"])
        or bool(audit.get("source_declared_test_split", True))
        or bool(audit.get("training_declared_test_split", True))
        or list(audit.get("train_dev_overlap", ["missing"]))
    ):
        raise ValueError(f"report {origin}:{candidate} 未通过困难类完整训练审计")

    dataset_yaml = Path(str(audit.get("dataset_yaml", ""))).resolve()
    dataset_manifest = dataset_yaml.parent / "manifest.json"
    reject_test_reference(dataset_manifest, f"dataset {origin}:{candidate}")
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
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
        raise ValueError(f"dataset {origin}:{candidate} 没有使用指定滚动窗口")

    weight = Path(str(report.get("best_weight", ""))).resolve()
    if sha256_file(weight) != str(report.get("best_weight_sha256", "")):
        raise ValueError(f"report {origin}:{candidate} best.pt 哈希不一致")
    return {
        "origin": origin,
        "candidate": candidate,
        "report": str(path),
        "report_sha256": sha256_file(path),
        "dataset_manifest": str(dataset_manifest),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "best_weight": str(weight),
        "best_weight_sha256": sha256_file(weight),
        "completed_epochs": int(training["completed_epochs"]),
        "best_epoch": int(training["best_epoch"]),
        "ap50": per_class[focus_local_class],
        "dataset_recipe": dataset_recipe(dataset),
        "training_overrides": {
            key: arguments.get(key) for key in TRAINING_OVERRIDE_KEYS
        },
    }


def rank_candidates(
    rows: Sequence[Mapping[str, Any]],
    origins: Sequence[str],
    baseline_name: str,
    max_origin_drop: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate"]), {})[str(row["origin"])] = row
    if baseline_name not in grouped:
        raise ValueError("困难类候选缺少 baseline")
    expected = set(origins)
    for name, by_origin in grouped.items():
        if set(by_origin) != expected:
            raise ValueError(f"candidate {name} 没有完整覆盖全部滚动起点")

    baseline = grouped[baseline_name]
    output = []
    for name, by_origin in grouped.items():
        ap_by_origin = {
            origin: float(by_origin[origin]["ap50"]) for origin in origins
        }
        deltas = {
            origin: ap_by_origin[origin] - float(baseline[origin]["ap50"])
            for origin in origins
        }
        recipes = [by_origin[origin]["dataset_recipe"] for origin in origins]
        overrides = [by_origin[origin]["training_overrides"] for origin in origins]
        if any(value != recipes[0] for value in recipes[1:]):
            raise ValueError(f"candidate {name} 的滚动数据配方不一致")
        if any(value != overrides[0] for value in overrides[1:]):
            raise ValueError(f"candidate {name} 的滚动训练参数不一致")
        mean_ap = statistics.fmean(ap_by_origin.values())
        baseline_mean = statistics.fmean(
            float(baseline[origin]["ap50"]) for origin in origins
        )
        item = {
            "name": name,
            "origin_ap50": ap_by_origin,
            "minimum_origin_ap50": min(ap_by_origin.values()),
            "mean_origin_ap50": mean_ap,
            "origin_delta_vs_baseline": deltas,
            "minimum_origin_delta_vs_baseline": min(deltas.values()),
            "mean_delta_vs_baseline": mean_ap - baseline_mean,
            "dataset_recipe": recipes[0],
            "training_overrides": overrides[0],
            "reports": {
                origin: dict(by_origin[origin]) for origin in origins
            },
        }
        item["eligible"] = (
            item["minimum_origin_delta_vs_baseline"] >= -float(max_origin_drop)
            and item["mean_delta_vs_baseline"] >= 0.0
        )
        output.append(item)
    return sorted(
        output,
        key=lambda row: (
            not bool(row["eligible"]),
            -float(row["minimum_origin_ap50"]),
            -float(row["mean_origin_ap50"]),
            str(row["name"]),
        ),
    )


def select_class_expert(
    manifest_path: Path,
    report_paths: Mapping[tuple[str, str], Path],
    baseline_name: str,
    focus_local_class: int,
    output: Path,
    required_epochs: int = 160,
    required_batch: int = 32,
    max_origin_drop: float = 0.01,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output = output.resolve()
    reject_test_reference(manifest_path, "manifest")
    reject_test_reference(output, "selection output")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖困难类 owner 选择报告：{output}")
    if focus_local_class not in BASE_LOCAL_TO_GLOBAL:
        raise ValueError("focus local class 不在基础映射中")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    regression = dict(manifest.get("regression_window", {}))
    regression_status = str(regression.get("status", ""))
    regression_boundary_valid = (
        regression_status == "sealed"
        and not bool(regression.get("labels_opened", True))
    ) or (
        regression_status == "reused_not_independent"
        and bool(regression.get("labels_opened", False))
    )
    if (
        manifest.get("selection_scope")
        != "base_train_and_dev_rolling_forward_only"
        or bool(manifest.get("lock_data_access", True))
        or manifest.get("strategy")
        != "multi_origin_expanding_window_temporal_backtest"
        or not regression_boundary_valid
        or not bool(
            regression.get("must_not_participate_in_candidate_selection", False)
        )
        or bool(regression.get("independent_evidence", True))
    ):
        raise ValueError("manifest 不是 sealed 滚动前向开发协议")
    windows = dict(manifest.get("selection_windows", {}))
    origins = list(windows)
    if not origins:
        raise ValueError("rolling manifest 没有 selection windows")

    rows = []
    for (origin, candidate), path in sorted(report_paths.items()):
        if origin not in windows:
            raise ValueError(f"未知滚动起点：{origin}")
        rows.append(
            validate_origin_report(
                origin,
                candidate,
                path,
                dict(windows[origin]),
                int(focus_local_class),
                int(required_epochs),
                int(required_batch),
            )
        )
    ranking = rank_candidates(rows, origins, baseline_name, float(max_origin_drop))
    eligible = [row for row in ranking if bool(row["eligible"])]
    if not eligible:
        raise RuntimeError("没有通过多起点稳健门禁的困难类 owner")
    selected = dict(eligible[0])
    report = {
        "schema_version": 1,
        "selection_scope": "base_train_and_dev_rolling_class_expert_selection",
        "lock_data_access": False,
        "performance_evidence": False,
        "independent_test_evidence": False,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "focus_local_class": int(focus_local_class),
        "focus_global_class": BASE_LOCAL_TO_GLOBAL[int(focus_local_class)],
        "baseline": baseline_name,
        "selection_basis": "maximin_origin_ap50_then_mean_with_regression_guard",
        "max_origin_drop": float(max_origin_drop),
        "selected": selected,
        "ranking": ranking,
        "regression_window": {
            **regression,
            "candidate_frozen_before_opening": regression_status == "sealed",
        },
        "candidate_selected_without_regression_reports": True,
        "all_unknown_images_must_execute_expert": True,
        "image_level_class_routing": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=report_spec, action="append", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--focus-local-class", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-epochs", type=int, default=160)
    parser.add_argument("--required-batch", type=int, default=32)
    parser.add_argument("--max-origin-drop", type=float, default=0.01)
    args = parser.parse_args()
    report_paths = {(origin, candidate): path for origin, candidate, path in args.report}
    if len(report_paths) != len(args.report):
        raise ValueError("report origin/candidate 组合不得重复")
    report = select_class_expert(
        args.manifest,
        report_paths,
        args.baseline,
        int(args.focus_local_class),
        args.output,
        int(args.required_epochs),
        int(args.required_batch),
        float(args.max_origin_drop),
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
