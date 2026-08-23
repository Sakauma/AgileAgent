from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fair_agent.modules.detection_fusion import calibrate_record_confidences
from fair_agent.modules.web_inference import remap_base_records


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/113_optimize_ascend_runtime_calibration.py"
SPEC = importlib.util.spec_from_file_location("ascend_runtime_calibration", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
CALIBRATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CALIBRATION
SPEC.loader.exec_module(CALIBRATION)


def test_base_records_are_owned_before_runtime_score_calibration() -> None:
    records = remap_base_records(
        [{"class_id": 0, "confidence": 0.8, "xyxy": [0, 0, 10, 10]}],
        {0: 3},
        {3: "tank"},
    )
    calibrated = calibrate_record_confidences(
        records,
        {
            "enabled": True,
            "method": "logit_affine",
            "sources": {
                "frozen_base_model": {"temperature": 2.0, "bias": 0.0}
            },
        },
    )

    assert calibrated[0]["source"] == "frozen_base_model"
    assert calibrated[0]["protocol_id"] is None
    assert calibrated[0]["raw_confidence"] == 0.8
    assert calibrated[0]["confidence"] != 0.8


def test_constrained_runtime_policy_can_remove_false_activation_without_map_loss() -> None:
    probe = CALIBRATION.ProbeData(
        records=(
            {
                "image_id": "base",
                "class_id": 0,
                "class_name": "soldier",
                "confidence": 0.90,
                "xyxy": [0, 0, 20, 20],
                "source": "frozen_base_model",
                "protocol_id": None,
            },
            {
                "image_id": "base",
                "class_id": 4,
                "class_name": "patrol_boat",
                "confidence": 0.20,
                "xyxy": [40, 40, 60, 60],
                "source": "incremental_model",
                "protocol_id": "incremental_detector",
            },
            {
                "image_id": "increment",
                "class_id": 4,
                "class_name": "patrol_boat",
                "confidence": 0.90,
                "xyxy": [10, 10, 30, 30],
                "source": "incremental_model",
                "protocol_id": "incremental_detector",
            },
        ),
        contexts={
            "base": {"scene_probabilities": {"sea": 0.1, "air": 0.1}},
            "increment": {"scene_probabilities": {"sea": 0.9, "air": 0.1}},
        },
        image_ids=frozenset({"base", "increment"}),
    )
    ground_truth = [
        {"image_id": "base", "class_id": 0, "xyxy": [0, 0, 20, 20]},
        {"image_id": "increment", "class_id": 4, "xyxy": [10, 10, 30, 30]},
    ]
    prior = {
        "source_split": "incremental_train_only",
        "per_class": {"4": {"scene": {"sea": 1.0, "air": 0.0}}},
    }
    gates = {"base_map50": 0.8, "new_map50": 0.6, "krr": 0.95}
    baseline = CALIBRATION.SearchParameters(thresholds=(0.10,) * 6)
    thresholds = list(baseline.thresholds)
    thresholds[4] = 0.50
    guarded = CALIBRATION.replace(baseline, thresholds=tuple(thresholds))

    before = CALIBRATION.score_parameters(
        probe,
        baseline,
        prior,
        ground_truth,
        {"base"},
        gates,
        content_gate_enabled=False,
        content_gate_scene_probability=0.5,
    )
    after = CALIBRATION.score_parameters(
        probe,
        guarded,
        prior,
        ground_truth,
        {"base"},
        gates,
        content_gate_enabled=False,
        content_gate_scene_probability=0.5,
    )

    assert before["competition_passed"] is True
    assert after["competition_passed"] is True
    assert before["metrics"]["false_activation_rate"] == 1.0
    assert after["metrics"]["false_activation_rate"] == 0.0
    assert after["metrics"]["new_map50"] == before["metrics"]["new_map50"]


def test_search_batch_evaluator_matches_sequential_cache() -> None:
    baseline = CALIBRATION.SearchParameters(thresholds=(0.10,) * 6)

    def evaluator(parameters):
        threshold = parameters.thresholds[4]
        return {
            "parameters": parameters,
            "metrics": {
                "base_map50": 0.8,
                "new_map50": 0.7,
                "krr": 1.0,
                "full_map50": 0.75,
                "new_precision": threshold,
                "false_activation_rate": 1.0 - threshold,
                "prediction_count": 1,
            },
            "competition_gates": {
                "base_map50": True,
                "new_map50": True,
                "krr": True,
            },
            "competition_passed": True,
            "constraint_violation": 0.0,
            "objective": (0.0, 1.0 - threshold, -0.7, -0.75, -threshold, 1),
        }

    options = {
        "beam_size": 2,
        "threshold_values": (0.10, 0.20),
        "temperature_values": (1.0,),
        "specialist_bias_values": (0.0,),
        "scene_penalty_values": (0.0,),
        "source_margin_values": (0.0,),
        "overlap_iou_values": (0.9,),
        "conflict_iou_values": (0.5,),
        "specialist_margin_values": (0.15,),
    }
    sequential, sequential_trace = CALIBRATION.search(
        baseline, evaluator, **options
    )
    batches = []

    def batch_evaluator(parameters):
        batches.append(tuple(parameters))
        return [evaluator(item) for item in parameters]

    parallel, parallel_trace = CALIBRATION.search(
        baseline,
        evaluator,
        batch_evaluator=batch_evaluator,
        **options,
    )

    assert parallel == sequential
    assert parallel_trace == sequential_trace
    assert any(len(batch) > 1 for batch in batches)
