from __future__ import annotations

import inspect
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "84_tune_base_refit_oof_fusion.py"))


def test_oof_tuning_grid_parsers() -> None:
    assert MODULE["float_grid"]("0.25,0.5") == [0.25, 0.5]
    assert MODULE["int_set"]("0,1,2") == {0, 1, 2}
    with pytest.raises(Exception, match="不得为空"):
        MODULE["float_grid"]("")


def test_oof_tuning_accepts_named_full_frame_owner() -> None:
    name, path = MODULE["named_report"]("recent_crop=reports/recent.json")

    assert name == "recent_crop"
    assert path == Path("reports/recent.json")
    with pytest.raises(Exception, match="NAME=PATH"):
        MODULE["named_report"]("recent_crop")
    with pytest.raises(Exception, match="非 generic"):
        MODULE["named_report"]("generic=reports/generic.json")


def test_oof_tuning_parses_predeclared_secondary_sets() -> None:
    assert MODULE["secondary_set"]("crop_full,recent_crop") == (
        "crop_full",
        "recent_crop",
    )
    with pytest.raises(Exception, match="非空且不重复"):
        MODULE["secondary_set"]("")
    with pytest.raises(Exception, match="非空且不重复"):
        MODULE["secondary_set"]("crop_full,crop_full")


def test_oof_selection_rejects_higher_score_with_degraded_fold() -> None:
    candidates = [
        {
            "name": "higher_but_degraded",
            "tuning_delta_vs_generic": 0.03,
            "degraded_tuning_fold_count": 1,
        },
        {
            "name": "stable",
            "tuning_delta_vs_generic": 0.02,
            "degraded_tuning_fold_count": 0,
        },
        {
            "name": "no_gain",
            "tuning_delta_vs_generic": 0.0,
            "degraded_tuning_fold_count": 0,
        },
    ]

    eligible = MODULE["eligible_candidates"](candidates, 0)

    assert [row["name"] for row in eligible] == ["stable"]
    with pytest.raises(ValueError, match="不得为负"):
        MODULE["eligible_candidates"](candidates, -1)


def test_validation_labels_open_only_after_policy_and_predictions_are_frozen() -> None:
    source = inspect.getsource(MODULE["main"])

    selection = source.index("selected = eligible[0]")
    prediction_freeze = source.index("selected_by_fold = {}")
    validation_label_open = source.index(
        "validation_targets = open_fold_targets(fold_images, validation_folds)"
    )

    assert selection < prediction_freeze < validation_label_open
