from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "92_build_base_rolling_forward.py"))


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
    path = tmp_path / "source" / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "selection_scope": "base_train_and_dev_only",
                "lock_data_access": False,
                "strategy": "per_sequence_contiguous_block_kfold",
                "fold_count": 5,
                "combined_non_test_count": 20,
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_builds_two_selection_origins_and_keeps_last_fold_sealed(
    tmp_path: Path,
) -> None:
    report = MODULE["build_rolling_forward"](
        source_manifest(tmp_path),
        tmp_path / "rolling",
        [2, 3],
        4,
    )

    assert report["selection_scope"] == "base_train_and_dev_rolling_forward_only"
    assert list(report["selection_windows"]) == ["origin_2", "origin_3"]
    assert report["selection_windows"]["origin_2"]["training_folds"] == [0, 1]
    assert report["selection_windows"]["origin_3"]["training_folds"] == [0, 1, 2]
    assert report["regression_window"] == {
        "status": "sealed",
        "validation_fold": 4,
        "image_count": 4,
        "labels_opened": False,
        "must_not_participate_in_candidate_selection": True,
        "independent_evidence": False,
    }
    assert not (tmp_path / "rolling" / "origin_4").exists()


def test_rejects_sealed_fold_as_selection_origin(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="严格早于"):
        MODULE["build_rolling_forward"](
            source_manifest(tmp_path),
            tmp_path / "rolling",
            [2, 4],
            4,
        )


def test_rejects_unsorted_or_duplicate_origins(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="升序且不重复"):
        MODULE["build_rolling_forward"](
            source_manifest(tmp_path),
            tmp_path / "rolling",
            [3, 2, 2],
            4,
        )


def test_records_previously_opened_last_fold_as_reused(tmp_path: Path) -> None:
    report = MODULE["build_rolling_forward"](
        source_manifest(tmp_path),
        tmp_path / "rolling",
        [2, 3],
        4,
        regression_already_opened=True,
    )

    assert report["regression_window"]["status"] == "reused_not_independent"
    assert report["regression_window"]["labels_opened"] is True
    assert report["regression_window"]["independent_evidence"] is False
