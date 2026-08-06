from __future__ import annotations

import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "76_evaluate_small_object_inference.py"))


def test_sliding_starts_cover_frame_and_are_deterministic() -> None:
    assert MODULE["sliding_starts"](640, 320, 0.20) == [0, 160, 320]
    assert MODULE["sliding_starts"](512, 256, 0.20) == [0, 128, 256]
    assert MODULE["sliding_starts"](640, 384, 0.20) == [0, 256]
    assert MODULE["sliding_starts"](256, 320, 0.20) == [0]


def test_selection_rejects_any_lock_or_test_path() -> None:
    for path in (
        Path("splits/mixed_test.txt"),
        Path("reports/base_test/candidate"),
        Path("reports/LOCK/candidate"),
    ):
        with pytest.raises(ValueError, match="test/lock"):
            MODULE["reject_lock_reference"](path, "selection")


def test_parse_sized_passthrough_model() -> None:
    name, path, imgsz = MODULE["parse_sized_model"]("tank_owner=/tmp/best.pt@896")

    assert name == "tank_owner"
    assert path.name == "best.pt"
    assert imgsz == 896
