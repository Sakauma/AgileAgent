from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from fair_agent.modules.detection_fusion import (
    apply_incremental_candidate_gates,
    context_adjusted_threshold,
    learn_context_prior,
)
from fair_agent.modules.incremental_rejection import (
    apply_positive_prototype,
    calibrate_positive_prototype,
    fit_positive_prototype,
)
from fair_agent.modules.strict_incremental import fuse_old_new_predictions
from fair_agent.modules.web_inference import apply_unified_class_gates


def make_target_image(path: Path, box: list[int]) -> Path:
    image = Image.new("RGB", (100, 100), (210, 210, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, fill=(20, 20, 20))
    image.save(path)
    return path


def row(path: Path, box: list[int], confidence: float = 1.0) -> dict:
    return {
        "image_id": path.stem,
        "class_id": 2,
        "confidence": confidence,
        "xyxy": box,
    }


def test_positive_prototype_uses_only_incremental_images_and_rejects_outlier(
    tmp_path: Path,
) -> None:
    train = make_target_image(tmp_path / "train.png", [10, 40, 90, 60])
    dev = make_target_image(tmp_path / "dev.png", [12, 40, 88, 60])
    lock = make_target_image(tmp_path / "lock.png", [10, 40, 90, 60])
    ground_truth = [row(train, [10, 40, 90, 60])]
    prototype = fit_positive_prototype(
        [train], ground_truth, 2, grid_size=8, minimum_scale=0.05
    )
    prototype = calibrate_positive_prototype(
        prototype,
        [dev],
        [row(dev, [12, 40, 88, 60], 0.9)],
        [row(dev, [12, 40, 88, 60])],
        target_recall=1.0,
        safety_factor=1.05,
    )
    candidates = [
        row(lock, [10, 40, 90, 60], 0.9),
        row(lock, [40, 10, 60, 90], 0.8),
    ]

    kept, rejected = apply_positive_prototype(candidates, [lock], prototype)

    assert prototype["learning_data_scope"] == "incremental_dataset_only"
    assert prototype["train_positive_count"] == 1
    assert prototype["dev_positive_count"] == 1
    assert [item["confidence"] for item in kept] == [0.9]
    assert [item["confidence"] for item in rejected] == [0.8]
    assert rejected[0]["reason"] == "outside_incremental_positive_prototype"


def test_cross_class_fusion_suppresses_only_overlapping_incremental_candidate() -> None:
    old = [
        {
            "image_id": "one",
            "class_id": 1,
            "confidence": 0.80,
            "xyxy": [0, 0, 20, 20],
        }
    ]
    new = [
        {
            "image_id": "one",
            "class_id": 2,
            "confidence": 0.90,
            "xyxy": [1, 1, 19, 19],
        },
        {
            "image_id": "one",
            "class_id": 2,
            "confidence": 0.70,
            "xyxy": [40, 40, 60, 60],
        },
    ]

    fused, decisions = fuse_old_new_predictions(
        old,
        new,
        nms_iou=0.60,
        cross_class={
            "enabled": True,
            "iou": 0.50,
            "base_confidence": 0.50,
            "incremental_margin": 0.15,
            "preserve_base_class_owners": True,
        },
    )

    assert len(fused) == 2
    assert {int(item["class_id"]) for item in fused} == {1, 2}
    assert decisions[0]["action"] == "reject_specialist"


def test_cross_class_fusion_detects_incremental_box_covered_by_large_old_box() -> None:
    old = [
        {
            "image_id": "one",
            "class_id": 3,
            "confidence": 0.80,
            "xyxy": [0, 0, 100, 100],
        }
    ]
    new = [
        {
            "image_id": "one",
            "class_id": 2,
            "confidence": 0.70,
            "xyxy": [10, 10, 20, 20],
        }
    ]

    fused, decisions = fuse_old_new_predictions(
        old,
        new,
        nms_iou=0.60,
        cross_class={
            "enabled": True,
            "iou": 0.50,
            "incremental_coverage": 0.80,
            "base_confidence": 0.50,
            "incremental_margin": 0.15,
            "preserve_base_class_owners": True,
        },
    )

    assert fused == old
    assert decisions[0]["iou"] == 0.01
    assert decisions[0]["incremental_coverage"] == 1.0
    assert decisions[0]["action"] == "reject_specialist"


def test_covered_incremental_box_with_stronger_evidence_is_not_deleted() -> None:
    old = [
        {"image_id": "one", "class_id": 3, "confidence": 0.80, "xyxy": [0, 0, 100, 100]}
    ]
    new = [
        {"image_id": "one", "class_id": 2, "confidence": 0.96, "xyxy": [10, 10, 20, 20]}
    ]

    fused, decisions = fuse_old_new_predictions(
        old,
        new,
        nms_iou=0.60,
        cross_class={
            "enabled": True,
            "iou": 0.50,
            "incremental_coverage": 0.80,
            "base_confidence": 0.50,
            "incremental_margin": 0.15,
            "preserve_base_class_owners": True,
        },
    )

    assert fused == old + new
    assert decisions == []


def test_context_prior_is_incremental_train_only_and_missing_context_is_neutral() -> None:
    prior = learn_context_prior(
        [
            {"scene_probabilities": {"sea": 0.9, "forest": 0.1}},
            {"scene_probabilities": {"sea": 0.8, "forest": 0.2}},
        ],
        ("scene",),
    )

    threshold, affinity = context_adjusted_threshold(0.69, {}, prior, 0.05)

    assert prior["source_split"] == "incremental_train_only"
    assert prior["sample_count"] == 2
    assert threshold == 0.69
    assert affinity == 1.0


def test_incompatible_known_context_only_raises_new_box_threshold() -> None:
    prior = {
        "source_split": "incremental_train_only",
        "scene": {"sea": 1.0, "forest": 0.0},
    }
    context = {"scene_probabilities": {"sea": 0.0, "forest": 1.0}}
    records = [
        {"image_id": "old", "class_id": 0, "confidence": 0.20, "xyxy": [0, 0, 5, 5]},
        {"image_id": "old", "class_id": 2, "confidence": 0.72, "xyxy": [0, 0, 5, 5]},
    ]

    kept, rejected = apply_incremental_candidate_gates(
        records,
        {2: 0.69},
        contexts_by_image={"old": context},
        context_prior=prior,
        max_context_penalty=0.05,
    )
    empty_kept, empty_rejected = apply_incremental_candidate_gates(
        [],
        {2: 0.69},
        contexts_by_image={"old": context},
        context_prior=prior,
        max_context_penalty=0.05,
    )

    assert kept == [records[0]]
    assert rejected[0]["effective_activation_threshold"] == 0.74
    assert rejected[0]["context_affinity"] == 0.0
    assert empty_kept == empty_rejected == []


def test_unified_gate_applies_new_class_threshold_and_prototype(tmp_path: Path) -> None:
    train = make_target_image(tmp_path / "train.png", [10, 40, 90, 60])
    prototype = fit_positive_prototype(
        [train], [row(train, [10, 40, 90, 60])], 2, minimum_scale=0.05
    )
    prototype = calibrate_positive_prototype(
        prototype,
        [train],
        [row(train, [10, 40, 90, 60], 0.9)],
        [row(train, [10, 40, 90, 60])],
        target_recall=1.0,
        safety_factor=1.05,
    )
    with Image.open(train) as source:
        image = source.convert("RGB")
    records = [
        {"class_id": 0, "confidence": 0.4, "xyxy": [0, 0, 10, 10]},
        {"class_id": 2, "confidence": 0.6, "xyxy": [10, 40, 90, 60]},
        {"class_id": 2, "confidence": 0.9, "xyxy": [10, 40, 90, 60]},
    ]

    kept, rejected = apply_unified_class_gates(
        image,
        records,
        {
            "activation_thresholds": {2: 0.7},
            "positive_prototypes": {2: prototype},
        },
    )

    assert [(item["class_id"], item["confidence"]) for item in kept] == [
        (0, 0.4),
        (2, 0.9),
    ]
    assert rejected[0]["action"] == "reject_unified_activation_threshold"
