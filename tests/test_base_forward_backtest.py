from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "89_build_base_forward_backtest.py"))


def write_split(path: Path, rows: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(row) for row in rows) + "\n", encoding="utf-8")


def source_manifest(tmp_path: Path) -> Path:
    folds = []
    for fold_id in range(5):
        split = tmp_path / "source" / f"fold_{fold_id}" / "val.txt"
        rows = [
            tmp_path / "images" / f"ir_base_forest_{fold_id * 10 + index:06d}.png"
            for index in range(2)
        ] + [
            tmp_path / "images" / f"sar_base_urban_{fold_id * 10 + index:06d}.png"
            for index in range(2)
        ]
        write_split(split, rows)
        folds.append(
            {
                "fold": fold_id,
                "val_count": len(rows),
                "val_split": str(split),
                "val_split_sha256": MODULE["sha256_file"](split),
            }
        )
    manifest = {
        "selection_scope": "base_train_and_dev_only",
        "lock_data_access": False,
        "strategy": "per_sequence_contiguous_block_kfold",
        "fold_count": 5,
        "combined_non_test_count": 20,
        "folds": folds,
    }
    path = tmp_path / "source" / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_builds_nested_strictly_forward_windows(tmp_path: Path) -> None:
    output = tmp_path / "forward"
    report = MODULE["build_forward_backtest"](
        source_manifest(tmp_path), output, 3, 4
    )

    assert report["selection_scope"] == "base_train_and_dev_forward_only"
    assert report["lock_data_access"] is False
    assert report["tuning"]["training_folds"] == [0, 1, 2]
    assert report["tuning"]["validation_fold"] == 3
    assert report["tuning"]["train_count"] == 12
    assert report["tuning"]["val_count"] == 4
    assert report["post_validation"]["training_folds"] == [0, 1, 2, 3]
    assert report["post_validation"]["validation_fold"] == 4
    assert report["post_validation"]["train_count"] == 16
    assert report["post_validation"]["val_count"] == 4
    assert all(
        row["train_last_frame"] < row["validation_first_frame"]
        for row in report["post_validation"]["sequence_audit"].values()
    )


def test_rejects_nonfinal_post_validation_fold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="最后一个连续时间块"):
        MODULE["build_forward_backtest"](
            source_manifest(tmp_path), tmp_path / "forward", 2, 3
        )


def test_rejects_test_named_source_manifest(tmp_path: Path) -> None:
    path = source_manifest(tmp_path)
    forbidden = tmp_path / "mixed_test_manifest.json"
    forbidden.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="不得引用"):
        MODULE["build_forward_backtest"](
            forbidden, tmp_path / "forward", 3, 4
        )
