from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "79_evaluate_base_refit_cv.py"))


def test_training_report_requires_full_budget_and_matching_weight(tmp_path: Path) -> None:
    weight = tmp_path / "best.pt"
    weight.write_bytes(b"weight")
    digest = MODULE["sha256_file"](weight)
    report_path = tmp_path / "fold_0.json"
    report = {
        "candidate": "fold_0",
        "selection_scope": "base_train_and_dev_only",
        "lock_data_access": False,
        "dataset_audit": {
            "dev_count": 20,
            "source_declared_test_split": False,
            "training_declared_test_split": False,
        },
        "training": {
            "requested_epochs": 160,
            "completed_epochs": 160,
            "stopped_early": False,
            "best_epoch": 42,
            "best_metric_value": 0.87,
        },
        "best_weight_sha256": digest,
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    audit = MODULE["validate_training_report"](
        "fold_0", report_path, weight, 20, 160
    )
    assert audit["completed_epochs"] == 160
    assert audit["best_epoch"] == 42

    report["training"]["completed_epochs"] = 159
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="完整非测试 OOF"):
        MODULE["validate_training_report"](
            "fold_0", report_path, weight, 20, 160
        )


def test_tile_size_parser_rejects_invalid_values() -> None:
    assert MODULE["parse_tile_size"]("320x256") == (320, 256)
    with pytest.raises(Exception, match="WIDTHxHEIGHT"):
        MODULE["parse_tile_size"]("320")
