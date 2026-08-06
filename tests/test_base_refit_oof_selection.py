from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "83_select_base_refit_oof.py"))


def test_oof_source_requires_expected_inference_mode(tmp_path: Path) -> None:
    report = tmp_path / "oof_report.json"
    report.write_text(
        json.dumps(
            {
                "selection_scope": "base_train_and_dev_oof_only",
                "lock_data_access": False,
                "inference_mode": "full_frame",
            }
        ),
        encoding="utf-8",
    )
    assert MODULE["validate_source"](report, "full_frame")["lock_data_access"] is False
    with pytest.raises(ValueError, match="推理模式错误"):
        MODULE["validate_source"](report, "sliding_window")
