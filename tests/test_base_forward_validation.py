from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "91_validate_base_forward_backtest.py"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_report(
    tmp_path: Path,
    name: str,
    map50: float,
    source_hash: str,
    val_hash: str,
    lineage: dict | None,
) -> Path:
    dataset_root = tmp_path / "datasets" / name
    dataset_yaml = dataset_root / "dataset.yaml"
    dataset_yaml.parent.mkdir(parents=True)
    dataset_yaml.write_text("names: [a, b, c]\n", encoding="utf-8")
    recipe = {
        "recent_fraction": 0.25 if lineage else None,
        "recent_full_repeats": 1 if lineage else 0,
        "crop_enabled": False,
        "crop_strategy": "smallest",
        "crop_size": {"width": 320, "height": 256},
        "crop_overlap": None,
        "jitter_fraction": 0.1,
        "min_visible_fraction": 0.5,
    }
    write_json(
        dataset_root / "manifest.json",
        {
            "selection_scope": "base_train_and_base_dev_only",
            "lock_data_access": False,
            "split_mode": "external",
            "source_split_sha256": source_hash,
            "val_source_split_sha256": val_hash,
            "train_source_count": 4,
            "val_source_count": 1,
            "source_train_val_overlap": [],
            **recipe,
        },
    )
    weight = tmp_path / "weights" / f"{name}.pt"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(name.encode())
    report = tmp_path / "reports" / f"{name}.json"
    write_json(
        report,
        {
            "candidate": name,
            "selection_scope": "base_train_and_dev_only",
            "lock_data_access": False,
            "lineage_selection": lineage,
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
                "best_epoch": 90,
                "arguments": {
                    "epochs": 160,
                    "batch": 32,
                    "patience": 0,
                    "imgsz": 896,
                    "lr0": 0.001,
                    "weight_decay": 0.0005,
                },
            },
            "evaluation": {
                "imgsz": 896,
                "map50": map50,
                "per_class_ap50": {"0": 0.7, "1": 0.99, "2": 0.94},
            },
            "best_weight": str(weight),
            "best_weight_sha256": MODULE["sha256_file"](weight),
        },
    )
    return report


def test_validates_frozen_candidate_on_later_window(tmp_path: Path) -> None:
    manifest = tmp_path / "forward" / "manifest.json"
    source_hash = "source-post-hash"
    val_hash = "val-post-hash"
    write_json(
        manifest,
        {
            "post_validation": {
                "train_count": 4,
                "val_count": 1,
                "train_split_sha256": source_hash,
                "val_split_sha256": val_hash,
            }
        },
    )
    selection = tmp_path / "selection.json"
    recipe = {
        "recent_fraction": 0.25,
        "recent_full_repeats": 1,
        "crop_enabled": False,
        "crop_strategy": "smallest",
        "crop_size": {"width": 320, "height": 256},
        "crop_overlap": None,
        "jitter_fraction": 0.1,
        "min_visible_fraction": 0.5,
    }
    write_json(
        selection,
        {
            "selection_scope": "base_train_and_dev_forward_tuning_only",
            "lock_data_access": False,
            "manifest": str(manifest),
            "manifest_sha256": MODULE["sha256_file"](manifest),
            "selected": {
                "name": "robust",
                "dataset_recipe": recipe,
                "training_overrides": {"lr0": 0.001, "weight_decay": 0.0005},
            },
            "post_validation": {"status": "sealed", "labels_opened": False},
        },
    )
    lineage = {
        "path": str(selection.resolve()),
        "sha256": MODULE["sha256_file"](selection),
        "selected_candidate": "robust",
        "post_validation_was_sealed": True,
    }
    candidate = make_report(
        tmp_path, "robust", 0.87, source_hash, val_hash, lineage
    )
    baseline = make_report(
        tmp_path, "baseline", 0.85, source_hash, val_hash, None
    )

    result = MODULE["validate_forward_backtest"](
        selection,
        candidate,
        "baseline",
        baseline,
        tmp_path / "validation.json",
    )

    assert result["accepted"] is True
    assert result["delta_map50_vs_baseline"] == pytest.approx(0.02)
    assert result["post_validation_labels_opened_after_candidate_freeze"] is True
