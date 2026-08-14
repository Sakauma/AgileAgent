from __future__ import annotations

import pytest

from fair_agent.modules.ascend_alignment import compare_api_records, match_detections


def _detection(class_id: int, confidence: float, xyxy: list[float]) -> dict:
    return {"class_id": class_id, "confidence": confidence, "xyxy": xyxy}


def test_detection_alignment_matches_by_class_and_iou_not_response_order() -> None:
    reference = [
        _detection(3, 0.90, [0, 0, 10, 10]),
        _detection(3, 0.80, [100, 100, 110, 110]),
    ]
    candidate = [
        _detection(3, 0.91, [100.2, 100, 110.2, 110]),
        _detection(3, 0.79, [0.2, 0, 10.2, 10]),
    ]

    result = match_detections(reference, candidate)

    assert result["count_equal"] is True
    assert result["class_counts_equal"] is True
    assert result["max_box_abs"] == pytest.approx(0.2)
    assert result["max_confidence_abs"] == pytest.approx(0.11)
    assert [row["candidate_index"] for row in result["matched"]] == [1, 0]


def test_detection_alignment_reports_unmatched_rows_per_class() -> None:
    result = match_detections(
        [_detection(0, 0.8, [0, 0, 1, 1])],
        [
            _detection(0, 0.8, [0, 0, 1, 1]),
            _detection(3, 0.5, [5, 5, 6, 6]),
        ],
    )

    assert result["count_equal"] is False
    assert result["class_counts_equal"] is False
    assert result["unmatched_reference"] == []
    assert result["unmatched_candidate"] == [1]


def test_api_alignment_keeps_threshold_count_and_context_as_hard_gates() -> None:
    reference = {
        "sample": {
            "image_id": "sample",
            "context": {
                "sensor": "ir",
                "scene": "sea",
                "sensor_probabilities": {"ir": 0.9, "sar": 0.1},
                "scene_probabilities": {"sea": 0.6, "urban": 0.4},
            },
            "detections": [_detection(3, 0.5, [1, 2, 3, 4])],
        }
    }
    candidate = {
        "sample": {
            "image_id": "sample",
            "context": {
                "sensor": "ir",
                "scene": "urban",
                "sensor_probabilities": {"ir": 0.89, "sar": 0.11},
                "scene_probabilities": {"sea": 0.49, "urban": 0.51},
            },
            "detections": [
                _detection(3, 0.501, [1, 2, 3, 4]),
                _detection(3, 0.7, [10, 20, 30, 40]),
            ],
        }
    }

    report = compare_api_records(reference, candidate)

    assert report["passed"] is False
    assert report["reason_counts"] == {"class_counts": 1, "detection_count": 1, "scene": 1}
    assert report["mismatches"][0]["detections"]["unmatched_candidate"] == [1]
