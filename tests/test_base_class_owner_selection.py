from __future__ import annotations

import inspect
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "88_select_base_class_owners.py"))


def test_owner_ranking_prefers_worst_fold_robustness() -> None:
    ranked = MODULE["rank_candidates"](
        [
            {
                "name": "higher_pooled_brittle",
                "minimum_tuning_fold_ap50": 0.80,
                "tuning_ap50": 0.95,
                "mean_tuning_fold_ap50": 0.93,
            },
            {
                "name": "robust",
                "minimum_tuning_fold_ap50": 0.90,
                "tuning_ap50": 0.94,
                "mean_tuning_fold_ap50": 0.92,
            },
        ]
    )

    assert ranked[0]["name"] == "robust"


def test_owner_eligibility_requires_both_pooled_and_worst_fold_non_degradation() -> None:
    rows = [
        {
            "name": "baseline",
            "minimum_tuning_fold_ap50": 0.88,
            "tuning_ap50": 0.93,
        },
        {
            "name": "better_worst_but_lower_pooled",
            "minimum_tuning_fold_ap50": 0.89,
            "tuning_ap50": 0.92,
        },
        {
            "name": "pareto_improvement",
            "minimum_tuning_fold_ap50": 0.90,
            "tuning_ap50": 0.94,
        },
    ]

    eligible = MODULE["eligible_owner_candidates"](rows, "baseline")

    assert {row["name"] for row in eligible} == {"baseline", "pareto_improvement"}


def test_validation_labels_open_only_after_owner_predictions_freeze() -> None:
    source = inspect.getsource(MODULE["main"])

    freeze = source.index("selected_predictions =")
    validation_labels = source.index("targets.update(", freeze)

    assert freeze < validation_labels
