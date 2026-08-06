from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

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
