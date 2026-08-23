from __future__ import annotations

from pathlib import Path

import pytest

from scripts.smoke_models import select_runtime_artifacts


def test_x86_smoke_selects_only_pytorch_artifacts() -> None:
    entry = {
        "id": "incremental_detector",
        "artifacts": [
            {
                "path": "models/production/incremental_detector.pt",
                "runtime": "x86_gpu",
            },
            {
                "path": "models/ascend310b/incremental_detector.om",
                "runtime": "ascend_310b",
            },
        ],
    }

    selected = select_runtime_artifacts(entry, "x86_gpu", ".pt")

    assert selected == [
        Path(__file__).resolve().parents[1]
        / "models/production/incremental_detector.pt"
    ]
    assert all(path.suffix == ".pt" for path in selected)


def test_x86_smoke_rejects_mislabeled_ascend_artifact() -> None:
    entry = {
        "id": "incremental_detector",
        "artifacts": [
            {
                "path": "models/ascend310b/incremental_detector.om",
                "runtime": "x86_gpu",
            }
        ],
    }

    with pytest.raises(ValueError, match=r"x86_gpu 产物必须为 \.pt"):
        select_runtime_artifacts(entry, "x86_gpu", ".pt")


def test_single_model_roles_reject_duplicate_x86_artifacts() -> None:
    entry = {
        "id": "base_detector",
        "artifacts": [
            {"path": "models/base-a.pt", "runtime": "x86_gpu"},
            {"path": "models/base-b.pt", "runtime": "x86_gpu"},
        ],
    }

    with pytest.raises(ValueError, match="数量应为 1，实际为 2"):
        select_runtime_artifacts(
            entry, "x86_gpu", ".pt", expected_count=1
        )
