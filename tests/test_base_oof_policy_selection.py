from __future__ import annotations

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "86_select_base_oof_policy.py"))


def write_candidate(
    path: Path,
    tuning_map50: float,
    validation_map50: float,
    all_oof_map50: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "selection_scope": "base_train_and_dev_oof_tune_validate",
                "lock_data_access": False,
                "manifest_sha256": "same-manifest",
                "tuning_folds": [0, 1, 2],
                "validation_folds": [3, 4],
                "secondaries": ["crop"],
                "selected_policy": {
                    "secondaries": ["crop"],
                    "tuning_map50": tuning_map50,
                    "tuning_delta_vs_generic": 0.02,
                    "minimum_tuning_fold_map50": 0.81,
                    "worst_tuning_fold_delta": 0.001,
                    "degraded_tuning_fold_count": 0,
                },
                "tuning": {"fused_map50": tuning_map50},
                "validation": {
                    "fused_map50": validation_map50,
                    "delta_map50": 0.01,
                },
                "all_oof_diagnostic": {"fused_map50": all_oof_map50},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_policy_selection_uses_tuning_not_higher_validation(tmp_path: Path) -> None:
    tuning_winner = tmp_path / "candidate_a.json"
    validation_winner = tmp_path / "candidate_b.json"
    write_candidate(tuning_winner, 0.86, 0.86, 0.86)
    write_candidate(validation_winner, 0.85, 0.95, 0.90)

    report = MODULE["select_policy"](
        [("tuning_winner", tuning_winner), ("validation_winner", validation_winner)],
        0,
        0.85,
        0.85,
    )

    assert report["selected"]["name"] == "tuning_winner"
    assert report["selection_basis"] == "tuning_folds_only"
    assert report["post_selection_validation"]["fused_map50"] == 0.86
