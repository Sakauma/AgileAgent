from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs" / "incremental" / "multibatch_stress_matrix.yaml"
AVAILABLE = {"ir": 78, "sar": 48}


def test_stress_matrix_has_three_distinct_scenarios() -> None:
    raw = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    scenarios = raw["matrix"]["scenarios"]
    assert len(scenarios) == 3
    assert len({row["id"] for row in scenarios}) == 3


def test_stress_scenarios_are_disjoint_and_fit_source_pool() -> None:
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))["matrix"]
    for scenario in matrix["scenarios"]:
        config_path = ROOT / scenario["config"]
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        plans = config["dataset"]["round_plan"]
        assert len(plans) == config["experiment"]["rounds"]
        totals = {sensor: 0 for sensor in AVAILABLE}
        for plan in plans:
            assert set(plan) == {"train", "val", "lock"}
            for sensor in AVAILABLE:
                assert int(plan["train"][sensor]) >= 1
                assert int(plan["val"][sensor]) >= 1
                assert int(plan["lock"][sensor]) >= 1
                totals[sensor] += sum(int(plan[split][sensor]) for split in plan)
        assert all(totals[sensor] <= AVAILABLE[sensor] for sensor in AVAILABLE)


def test_stress_scenarios_stop_after_official_gate_failure() -> None:
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))["matrix"]
    for scenario in matrix["scenarios"]:
        config = yaml.safe_load((ROOT / scenario["config"]).read_text(encoding="utf-8"))
        assert config["experiment"]["continue_after_rejection"] is False
