from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fair_agent.modules.strict_incremental import (
    CLASS_NAMES,
    box_iou,
    calibrate_threshold,
    class_aware_nms,
    discover_experiment_profiles,
    evaluate_ap50,
    fuse_old_new_predictions,
    load_experiment_profile,
    read_yolo_labels,
    retention_metrics,
    source_label,
    yolo_ground_truth,
)


def prediction(
    image_id: str,
    class_id: int,
    confidence: float,
    xyxy: list[float],
) -> dict:
    return {
        "image_id": image_id,
        "class_id": class_id,
        "confidence": confidence,
        "xyxy": xyxy,
    }


def test_strict_4plus2_class_contract_is_complete() -> None:
    assert CLASS_NAMES == {
        0: "soldier",
        1: "small_aircraft",
        2: "warship",
        3: "tank",
        4: "patrol_boat",
        5: "armored_vehicle",
    }


def test_yolo_ground_truth_supports_canonical_images_labels_layout(
    tmp_path: Path,
) -> None:
    image = tmp_path / "dataset" / "images" / "lock" / "sample.png"
    label = tmp_path / "dataset" / "labels" / "lock" / "sample.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), "black").save(image)
    label.write_text("4 0.5 0.5 0.4 0.5\n", encoding="utf-8")

    assert source_label(image) == label
    assert read_yolo_labels(label) == [(4, 0.5, 0.5, 0.4, 0.5)]
    assert yolo_ground_truth([image]) == [
        {
            "image_id": "sample",
            "class_id": 4,
            "xyxy": pytest.approx([30.0, 20.0, 70.0, 60.0]),
        }
    ]


def test_ap50_and_krr_are_recomputed_from_frozen_prediction_rows() -> None:
    ground_truth = [
        {"image_id": "a", "class_id": 0, "xyxy": [0, 0, 10, 10]},
        {"image_id": "b", "class_id": 1, "xyxy": [0, 0, 10, 10]},
    ]
    before = [
        prediction("a", 0, 0.9, [0, 0, 10, 10]),
        prediction("b", 1, 0.8, [0, 0, 10, 10]),
    ]
    unchanged = [
        *before,
        prediction("a", 4, 0.7, [20, 20, 30, 30]),
    ]
    reduced = [before[0]]

    perfect = evaluate_ap50(before, ground_truth, [0, 1])
    assert perfect["map50"] == pytest.approx(1.0)

    retained = retention_metrics(before, unchanged, ground_truth, [0, 1])
    assert retained["krr"] == pytest.approx(1.0)
    assert retained["old_prediction_equivalent"] is True

    degraded = retention_metrics(before, reduced, ground_truth, [0, 1])
    assert degraded["old_map50_after"] < degraded["old_map50_before"]
    assert degraded["krr"] == pytest.approx(
        degraded["old_map50_after"] / degraded["old_map50_before"]
    )
    assert degraded["old_prediction_equivalent"] is False


def test_fusion_never_renms_the_frozen_base_owner_stream() -> None:
    old = [
        prediction("a", 0, 0.9, [0, 0, 10, 10]),
        prediction("a", 0, 0.8, [1, 1, 9, 9]),
    ]
    new = [
        prediction("a", 4, 0.7, [20, 20, 30, 30]),
        prediction("a", 4, 0.6, [21, 21, 29, 29]),
    ]

    fused, decisions = fuse_old_new_predictions(old, new, nms_iou=0.5)

    assert fused[:2] == old
    assert len(fused) == 3
    assert fused[-1]["class_id"] == 4
    assert decisions == []
    assert len(class_aware_nms(new, 0.5)) == 1
    assert box_iou(new[0]["xyxy"], new[1]["xyxy"]) > 0.5


def test_threshold_calibration_can_meet_a_precision_target() -> None:
    ground_truth = [
        {"image_id": "a", "class_id": 4, "xyxy": [0, 0, 10, 10]}
    ]
    predictions = [
        prediction("a", 4, 0.9, [0, 0, 10, 10]),
        prediction("b", 4, 0.4, [20, 20, 30, 30]),
    ]

    result = calibrate_threshold(
        predictions,
        ground_truth,
        4,
        minimum=0.1,
        maximum=0.9,
        step=0.1,
        target_precision=0.9,
    )

    assert result["passed"] is True
    assert result["reason"] == "target_precision_reached"
    assert result["selected"]["precision"] == pytest.approx(1.0)
    assert result["selected"]["recall"] == pytest.approx(1.0)


def test_current_4plus2_profile_loads_verified_production_evidence() -> None:
    profile = load_experiment_profile("incremental-detection")

    assert profile["base_local_to_global"] == {0: 0, 1: 1, 2: 2, 3: 3}
    assert profile["specialist_local_to_global"] == {0: 4, 1: 5}
    assert profile["new_global_ids"] == [4, 5]
    assert profile["new_map50"] == pytest.approx(0.7733677094868956)
    assert profile["krr"] == pytest.approx(0.9731258182061283)
    assert profile["full_map50"] == pytest.approx(0.7949935544547209)
    assert profile["phase_contract"]["incremental_learning"][
        "training_data_scope"
    ] == "incremental_dataset_only"


def test_profile_discovery_exposes_only_verified_4plus2_production() -> None:
    result = discover_experiment_profiles()

    assert result["core_verified_count"] == 1
    assert result["verified_count"] == 1
    assert result["true_class_incremental_verified"] is True
    assert [row["profile_id"] for row in result["profiles"]] == [
        "incremental-detection"
    ]
    assert result["errors"] == []
