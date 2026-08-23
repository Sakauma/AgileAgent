from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from fair_agent.modules.detection_fusion import (
    apply_incremental_candidate_gates,
    context_adjusted_threshold,
    learn_context_prior,
    suppress_cross_class_overlaps,
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


def test_global_cross_class_suppression_reduces_reported_16_boxes_to_7() -> None:
    tank_boxes = [
        (0.831, [271.3, 269.0, 304.7, 298.1]),
        (0.730, [358.7, 316.4, 389.5, 340.4]),
        (0.708, [494.5, 223.3, 520.2, 244.6]),
        (0.682, [206.9, 387.4, 241.8, 413.9]),
        (0.636, [482.1, 450.2, 517.3, 479.6]),
        (0.573, [318.0, 205.4, 346.3, 228.9]),
        (0.552, [482.3, 452.0, 516.1, 480.5]),
        (0.319, [102.1, 257.7, 134.9, 278.7]),
        (0.258, [114.9, 257.4, 134.3, 272.6]),
        (0.173, [218.9, 387.1, 240.2, 405.2]),
    ]
    armored_boxes = [
        (0.926, [105.9, 257.3, 134.3, 277.8]),
        (0.921, [323.3, 208.4, 345.5, 225.8]),
        (0.912, [208.6, 387.0, 243.5, 414.1]),
        (0.910, [499.4, 221.8, 521.0, 241.4]),
        (0.908, [485.1, 447.3, 518.2, 477.4]),
        (0.894, [362.0, 317.2, 389.1, 339.6]),
    ]
    rows = [
        {
            "image_id": "ir_r2_inc_forest_000001",
            "class_id": 3,
            "confidence": confidence,
            "xyxy": xyxy,
            "source": "frozen_base_model",
        }
        for confidence, xyxy in tank_boxes
    ] + [
        {
            "image_id": "ir_r2_inc_forest_000001",
            "class_id": 5,
            "confidence": confidence,
            "xyxy": xyxy,
            "source": "incremental_model",
        }
        for confidence, xyxy in armored_boxes
    ]

    kept, decisions = suppress_cross_class_overlaps(
        rows,
        iou_threshold=0.50,
        smaller_box_coverage=0.95,
    )

    assert len(kept) == 7
    assert sum(int(row["class_id"]) == 5 for row in kept) == 6
    assert [row for row in kept if int(row["class_id"]) == 3] == [rows[0]]
    assert len(decisions) == 9
    assert {row["suppressed_class_id"] for row in decisions} == {3}
    assert {row["kept_class_id"] for row in decisions} == {5}


def test_global_cross_class_suppression_has_no_pair_whitelist_or_image_leakage() -> None:
    rows = [
        {
            "image_id": "air",
            "class_id": 1,
            "confidence": 0.80,
            "xyxy": [0, 0, 20, 20],
        },
        {
            "image_id": "air",
            "class_id": 4,
            "confidence": 0.90,
            "xyxy": [1, 1, 19, 19],
        },
        {
            "image_id": "sea",
            "class_id": 2,
            "confidence": 0.95,
            "xyxy": [0, 0, 20, 20],
        },
        {
            "image_id": "sea",
            "class_id": 0,
            "confidence": 0.40,
            "xyxy": [1, 1, 19, 19],
        },
        {
            "image_id": "forest",
            "class_id": 3,
            "confidence": 0.70,
            "xyxy": [40, 40, 60, 60],
        },
    ]

    kept, decisions = suppress_cross_class_overlaps(
        rows,
        iou_threshold=0.50,
        smaller_box_coverage=0.95,
    )

    assert {(row["image_id"], int(row["class_id"])) for row in kept} == {
        ("air", 4),
        ("sea", 2),
        ("forest", 3),
    }
    assert len(decisions) == 2


def test_global_cross_class_suppression_leaves_same_class_nms_unchanged() -> None:
    rows = [
        {"image_id": "one", "class_id": 3, "confidence": 0.9, "xyxy": [0, 0, 20, 20]},
        {"image_id": "one", "class_id": 3, "confidence": 0.8, "xyxy": [1, 1, 19, 19]},
    ]

    kept, decisions = suppress_cross_class_overlaps(
        rows,
        iou_threshold=0.50,
        smaller_box_coverage=0.95,
    )

    assert kept == rows
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
