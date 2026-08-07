from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "90_select_base_forward_backtest.py"))
TRAINER = runpy.run_path(str(ROOT / "tools" / "71_sweep_base_dev.py"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    train_source = tmp_path / "forward" / "tuning" / "train_source.txt"
    val_source = tmp_path / "forward" / "tuning" / "val_source.txt"
    train_source.parent.mkdir(parents=True)
    train_source.write_text("train.png\n", encoding="utf-8")
    val_source.write_text("val.png\n", encoding="utf-8")
    manifest = {
        "selection_scope": "base_train_and_dev_forward_only",
        "lock_data_access": False,
        "strategy": "expanding_window_temporal_backtest",
        "post_validation_labels_must_remain_closed_during_tuning": True,
        "tuning": {
            "train_count": 1,
            "val_count": 1,
            "train_split_sha256": MODULE["sha256_file"](train_source),
            "val_split_sha256": MODULE["sha256_file"](val_source),
        },
        "post_validation": {"validation_fold": 4},
    }
    manifest_path = tmp_path / "forward" / "manifest.json"
    write_json(manifest_path, manifest)

    reports = {}
    values = {
        "forward_baseline": (0.84, {"0": 0.60, "1": 0.99, "2": 0.93}),
        "robust": (0.86, {"0": 0.62, "1": 0.99, "2": 0.95}),
        "class_regression": (0.87, {"0": 0.57, "1": 0.99, "2": 0.95}),
    }
    for name, (map50, per_class) in values.items():
        dataset_root = tmp_path / "datasets" / name
        dataset_yaml = dataset_root / "dataset.yaml"
        dataset_yaml.parent.mkdir(parents=True)
        dataset_yaml.write_text("names: [a, b, c]\n", encoding="utf-8")
        write_json(
            dataset_root / "manifest.json",
            {
                "selection_scope": "base_train_and_base_dev_only",
                "lock_data_access": False,
                "split_mode": "external",
                "source_split_sha256": MODULE["sha256_file"](train_source),
                "val_source_split_sha256": MODULE["sha256_file"](val_source),
                "train_source_count": 1,
                "val_source_count": 1,
                "source_train_val_overlap": [],
                "recent_full_repeats": 0,
                "crop_enabled": False,
            },
        )
        weight = tmp_path / "runs" / name / "best.pt"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(name.encode())
        report_path = tmp_path / "reports" / f"{name}.json"
        write_json(
            report_path,
            {
                "candidate": name,
                "selection_scope": "base_train_and_dev_only",
                "lock_data_access": False,
                "dataset_audit": {
                    "dataset_yaml": str(dataset_yaml),
                    "dev_count": 1,
                    "source_declared_test_split": False,
                    "training_declared_test_split": False,
                    "train_dev_overlap": [],
                },
                "training": {
                    "requested_epochs": 160,
                    "completed_epochs": 160,
                    "stopped_early": False,
                    "best_epoch": 100,
                    "arguments": {
                        "epochs": 160,
                        "batch": 32,
                        "patience": 0,
                        "imgsz": 896,
                    },
                },
                "evaluation": {
                    "imgsz": 896,
                    "map50": map50,
                    "per_class_ap50": per_class,
                },
                "best_weight": str(weight),
                "best_weight_sha256": MODULE["sha256_file"](weight),
            },
        )
        reports[name] = report_path
    return manifest_path, reports


def test_selects_best_candidate_without_class_regression(tmp_path: Path) -> None:
    manifest, reports = fixture(tmp_path)
    output = tmp_path / "selection.json"

    result = MODULE["select_forward_candidate"](
        manifest, reports, "forward_baseline", output
    )

    assert result["selected"]["name"] == "robust"
    assert result["selected"]["map50"] == pytest.approx(0.86)
    assert result["ranking"][0]["name"] == "robust"
    assert result["ranking"][-1]["name"] == "class_regression"
    assert result["post_validation"] == {
        "status": "sealed",
        "validation_fold": 4,
        "labels_opened": False,
        "candidate_must_be_frozen_before_training_and_scoring": True,
    }


def test_rejects_incomplete_epoch_budget(tmp_path: Path) -> None:
    manifest, reports = fixture(tmp_path)
    report = json.loads(reports["robust"].read_text(encoding="utf-8"))
    report["training"]["completed_epochs"] = 159
    write_json(reports["robust"], report)

    with pytest.raises(ValueError, match="完整前向训练审计"):
        MODULE["select_forward_candidate"](
            manifest,
            reports,
            "forward_baseline",
            tmp_path / "selection.json",
        )


def test_post_validation_lineage_locks_selected_arguments(tmp_path: Path) -> None:
    path = tmp_path / "forward_selection.json"
    write_json(
        path,
        {
            "selection_scope": "base_train_and_dev_forward_tuning_only",
            "lock_data_access": False,
            "selected": {
                "name": "robust",
                "training_overrides": {"lr0": 0.001, "weight_decay": 0.0005},
            },
            "post_validation": {
                "status": "sealed",
                "labels_opened": False,
                "candidate_must_be_frozen_before_training_and_scoring": True,
            },
        },
    )

    audit = TRAINER["validate_lineage_selection"](
        path, "robust", {"lr0": 0.001, "weight_decay": 0.0005}
    )

    assert audit["selected_candidate"] == "robust"
    assert audit["post_validation_was_sealed"] is True
    with pytest.raises(ValueError, match="参数漂移"):
        TRAINER["validate_lineage_selection"](
            path, "robust", {"lr0": 0.00075, "weight_decay": 0.0005}
        )
