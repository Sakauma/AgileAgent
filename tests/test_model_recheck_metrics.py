from __future__ import annotations

import yaml

from fair_agent.modules.model_recheck import box_iou, image_outcomes, map50, write_combined_dataset, xywh_to_xyxy


def test_box_conversion_and_iou() -> None:
    box = xywh_to_xyxy([0.5, 0.5, 0.2, 0.2])
    assert box_iou(box, box) == 1.0


def test_custom_map50_perfect_and_false_positive() -> None:
    perfect = [{"gt": [{"class_id": 0, "box": [0.1, 0.1, 0.2, 0.2]}], "pred": [{"class_id": 0, "confidence": 0.9, "box": [0.1, 0.1, 0.2, 0.2]}]}]
    assert map50(perfect, [0], [0]) > 0.99
    missed = [{"gt": [{"class_id": 0, "box": [0.1, 0.1, 0.2, 0.2]}], "pred": [{"class_id": 0, "confidence": 0.9, "box": [0.7, 0.7, 0.8, 0.8]}]}]
    assert map50(missed, [0], [0]) == 0.0


def test_combined_dataset_has_required_train_and_val_keys(tmp_path, monkeypatch) -> None:
    import fair_agent.modules.model_recheck as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    image = tmp_path / "image.png"
    path = write_combined_dataset(tmp_path / "report", [image])
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["train"] == data["val"]


def test_box_matching_reproduces_ultralytics_unique_order() -> None:
    record = {
        "gt": [{"class_id": 0, "box": [0.0, 0.0, 1.0, 1.0]}],
        "pred": [
            {"class_id": 0, "confidence": 0.99, "box": [0.0, 0.0, 0.8, 0.8]},
            {"class_id": 0, "confidence": 0.80, "box": [0.0, 0.0, 1.0, 1.0]},
        ],
    }
    outcomes, _ = image_outcomes(record, 0)
    assert outcomes == [(0.99, 1), (0.8, 0)]


def test_validator_statistics_are_consumed_directly() -> None:
    record = {"gt_classes": [0], "pred": [{"class_id": 0, "confidence": 0.8, "tp50": True}]}
    outcomes, count = image_outcomes(record, 0)
    assert outcomes == [(0.8, 1)]
    assert count == 1
