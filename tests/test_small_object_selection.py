from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "77_select_small_object_agent.py"))


def prediction(image_id: str, confidence: float, xyxy: list[float]) -> dict:
    return {
        "image_id": image_id,
        "class_id": 0,
        "confidence": confidence,
        "xyxy": xyxy,
    }


def test_support_selection_keeps_policy_fixed_and_ranks_seed() -> None:
    images = [Path("sequence_000001.png"), Path("sequence_000002.png")]
    targets = [{"image_id": "sequence_000001", "class_id": 0, "xyxy": [0, 0, 10, 10]}]
    predictions = {
        "primary": [
            prediction("sequence_000002", 0.90, [20, 20, 30, 30]),
            prediction("sequence_000001", 0.50, [0, 0, 10, 10]),
        ],
        "fixed": [],
        "seed_a": [prediction("sequence_000001", 0.95, [0, 0, 10, 10])],
        "seed_b": [],
    }

    ranking, baseline = MODULE["evaluate_support_candidates"](
        predictions,
        targets,
        images,
        "primary",
        ["fixed"],
        ["seed_a", "seed_b"],
        0,
        [0],
        0.50,
        1.0,
        0.0,
        False,
    )

    assert ranking[0]["support"] == "seed_a"
    assert ranking[0]["map50"] > baseline["map50"]
    assert ranking[0]["secondaries"] == ["fixed", "seed_a"]
