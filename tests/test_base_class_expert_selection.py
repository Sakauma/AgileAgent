from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "93_select_base_class_expert.py"))
TRAINER = runpy.run_path(str(ROOT / "tools" / "71_sweep_base_dev.py"))
VALIDATOR = runpy.run_path(str(ROOT / "tools" / "94_validate_base_class_expert.py"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_manifest(tmp_path: Path, reused: bool = False) -> Path:
    windows = {}
    for fold_id in (2, 3):
        windows[f"origin_{fold_id}"] = {
            "train_count": fold_id * 4,
            "val_count": 4,
            "train_split_sha256": f"train-{fold_id}",
            "val_split_sha256": f"val-{fold_id}",
        }
    path = tmp_path / "rolling" / "manifest.json"
    write_json(
        path,
        {
            "selection_scope": "base_train_and_dev_rolling_forward_only",
            "lock_data_access": False,
            "strategy": "multi_origin_expanding_window_temporal_backtest",
            "source_manifest_sha256": "source-sha",
            "selection_windows": windows,
            "regression_window": {
                "status": "reused_not_independent" if reused else "sealed",
                "validation_fold": 4,
                "image_count": 4,
                "labels_opened": reused,
                "must_not_participate_in_candidate_selection": True,
                "independent_evidence": False,
            },
        },
    )
    return path


def make_report(
    tmp_path: Path,
    origin: str,
    candidate: str,
    ap50: float,
) -> Path:
    fold_id = int(origin.rsplit("_", 1)[1])
    dataset_root = tmp_path / "datasets" / origin
    dataset_yaml = dataset_root / "dataset.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text("names: [a, b, c]\n", encoding="utf-8")
    manifest = dataset_root / "manifest.json"
    if not manifest.exists():
        write_json(
            manifest,
            {
                "selection_scope": "base_train_and_base_dev_only",
                "lock_data_access": False,
                "split_mode": "external",
                "source_split_sha256": f"train-{fold_id}",
                "val_source_split_sha256": f"val-{fold_id}",
                "train_source_count": fold_id * 4,
                "val_source_count": 4,
                "source_train_val_overlap": [],
                "recent_fraction": None,
                "recent_full_repeats": 0,
                "crop_enabled": False,
                "crop_strategy": "smallest",
                "crop_size": {"width": 320, "height": 256},
                "crop_overlap": None,
                "jitter_fraction": 0.1,
                "min_visible_fraction": 0.5,
            },
        )
    weight = tmp_path / "weights" / f"{origin}-{candidate}.pt"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(f"{origin}-{candidate}".encode())
    report = tmp_path / "reports" / origin / f"{candidate}.json"
    write_json(
        report,
        {
            "candidate": candidate,
            "selection_scope": "base_train_and_dev_only",
            "lock_data_access": False,
            "dataset_audit": {
                "dataset_yaml": str(dataset_yaml),
                "dev_count": 4,
                "source_declared_test_split": False,
                "training_declared_test_split": False,
                "train_dev_overlap": [],
            },
            "training": {
                "requested_epochs": 160,
                "completed_epochs": 160,
                "stopped_early": False,
                "best_epoch": 80,
                "arguments": {
                    "epochs": 160,
                    "batch": 32,
                    "patience": 0,
                    "classes": [0],
                    "imgsz": 896,
                    "lr0": 0.001,
                    "weight_decay": 0.0005,
                    "mosaic": 0.8,
                    "translate": 0.15,
                    "scale": 0.5,
                    "close_mosaic": 20,
                    "cls": 0.5,
                    "box": 7.5,
                },
            },
            "evaluation": {
                "imgsz": 896,
                "class_filter": [0],
                "map50": ap50,
                "per_class_ap50": {"0": ap50},
            },
            "best_weight": str(weight),
            "best_weight_sha256": MODULE["sha256_file"](weight),
        },
    )
    return report


def test_selects_maximin_expert_without_opening_last_fold(tmp_path: Path) -> None:
    reports = {}
    values = {
        "expert_default": (0.60, 0.62),
        "expert_robust": (0.65, 0.68),
        "expert_flashy": (0.80, 0.60),
    }
    for candidate, scores in values.items():
        for origin, score in zip(("origin_2", "origin_3"), scores):
            reports[(origin, candidate)] = make_report(
                tmp_path, origin, candidate, score
            )

    result = MODULE["select_class_expert"](
        make_manifest(tmp_path),
        reports,
        "expert_default",
        0,
        tmp_path / "selection.json",
    )

    assert result["selected"]["name"] == "expert_robust"
    assert result["focus_global_class"] == 0
    assert result["regression_window"]["labels_opened"] is False
    assert result["candidate_selected_without_regression_reports"] is True
    assert result["all_unknown_images_must_execute_expert"] is True
    flashy = next(row for row in result["ranking"] if row["name"] == "expert_flashy")
    assert flashy["eligible"] is False


def test_rejects_candidate_missing_an_origin(tmp_path: Path) -> None:
    reports = {
        (origin, "expert_default"): make_report(
            tmp_path, origin, "expert_default", 0.6
        )
        for origin in ("origin_2", "origin_3")
    }
    reports[("origin_2", "incomplete")] = make_report(
        tmp_path, "origin_2", "incomplete", 0.7
    )

    with pytest.raises(ValueError, match="完整覆盖"):
        MODULE["select_class_expert"](
            make_manifest(tmp_path),
            reports,
            "expert_default",
            0,
            tmp_path / "selection.json",
        )


def test_trainer_accepts_frozen_class_expert_lineage(tmp_path: Path) -> None:
    selection = tmp_path / "expert_selection.json"
    write_json(
        selection,
        {
            "selection_scope": "base_train_and_dev_rolling_class_expert_selection",
            "lock_data_access": False,
            "selected": {
                "name": "expert_robust",
                "training_overrides": {"classes": [0], "lr0": 0.00075},
            },
            "regression_window": {
                "status": "reused_not_independent",
                "labels_opened": True,
                "must_not_participate_in_candidate_selection": True,
                "candidate_frozen_before_opening": False,
            },
            "candidate_selected_without_regression_reports": True,
            "all_unknown_images_must_execute_expert": True,
            "image_level_class_routing": False,
        },
    )

    audit = TRAINER["validate_lineage_selection"](
        selection,
        "expert_robust",
        {"classes": [0], "lr0": 0.00075},
    )

    assert audit["selection_scope"].endswith("class_expert_selection")
    with pytest.raises(ValueError, match="参数漂移"):
        TRAINER["validate_lineage_selection"](
            selection,
            "expert_robust",
            {"classes": [0], "lr0": 0.001},
        )


def make_baseline_report(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "datasets" / "origin_4"
    dataset_yaml = dataset_root / "dataset.yaml"
    report = tmp_path / "reports" / "origin_4" / "fold_4.json"
    weight = tmp_path / "weights" / "fold_4.pt"
    weight.write_bytes(b"fold-4")
    write_json(
        report,
        {
            "candidate": "fold_4",
            "selection_scope": "base_train_and_dev_only",
            "lock_data_access": False,
            "dataset_audit": {
                "dataset_yaml": str(dataset_yaml),
                "dev_count": 4,
                "source_declared_test_split": False,
                "training_declared_test_split": False,
                "train_dev_overlap": [],
            },
            "training": {
                "requested_epochs": 160,
                "completed_epochs": 160,
                "stopped_early": False,
                "best_epoch": 90,
                "arguments": {
                    "epochs": 160,
                    "batch": 32,
                    "patience": 0,
                    "imgsz": 896,
                },
            },
            "evaluation": {
                "imgsz": 896,
                "map50": 0.86,
                "per_class_ap50": {"0": 0.65, "1": 0.99, "2": 0.94},
            },
            "best_weight": str(weight),
            "best_weight_sha256": MODULE["sha256_file"](weight),
        },
    )
    return report


def test_validates_reused_regression_without_claiming_independence(
    tmp_path: Path,
) -> None:
    reports = {}
    for candidate, scores in {
        "expert_default": (0.60, 0.62),
        "expert_robust": (0.65, 0.68),
    }.items():
        for origin, score in zip(("origin_2", "origin_3"), scores):
            reports[(origin, candidate)] = make_report(
                tmp_path, origin, candidate, score
            )
    selection_path = tmp_path / "selection.json"
    selection = MODULE["select_class_expert"](
        make_manifest(tmp_path, reused=True),
        reports,
        "expert_default",
        0,
        selection_path,
    )

    expert_path = make_report(tmp_path, "origin_4", "expert_robust", 0.75)
    expert = json.loads(expert_path.read_text(encoding="utf-8"))
    expert["lineage_selection"] = {
        "path": str(selection_path.resolve()),
        "sha256": MODULE["sha256_file"](selection_path),
        "selected_candidate": "expert_robust",
        "post_validation_was_sealed": False,
        "regression_status": "reused_not_independent",
    }
    write_json(expert_path, expert)
    forward = tmp_path / "forward" / "manifest.json"
    write_json(
        forward,
        {
            "source_manifest_sha256": "source-sha",
            "post_validation": {
                "validation_fold": 4,
                "train_count": 16,
                "val_count": 4,
                "train_split_sha256": "train-4",
                "val_split_sha256": "val-4",
            },
        },
    )

    result = VALIDATOR["validate_class_expert_regression"](
        selection_path,
        forward,
        expert_path,
        "fold_4",
        make_baseline_report(tmp_path),
        tmp_path / "regression.json",
    )

    assert selection["regression_window"]["status"] == "reused_not_independent"
    assert result["accepted"] is True
    assert result["expert_focus_ap50"] == pytest.approx(0.75)
    assert result["composite_map50"] == pytest.approx((0.75 + 0.99 + 0.94) / 3)
    assert result["independent_test_evidence"] is False
