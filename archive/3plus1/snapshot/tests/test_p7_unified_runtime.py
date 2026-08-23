from __future__ import annotations

from fair_agent.modules.web_inference import (
    apply_unified_record_ownership,
    unified_logical_protocol_result,
)


def _gates() -> dict:
    return {
        "activation_thresholds": {2: 0.63},
        "protocol_id": "warship-incremental",
        "class_names": {2: "warship"},
        "new_map50": 0.68,
        "krr": 0.97,
        "context_gate": {"enabled": False},
    }


def test_single_detector_preserves_logical_old_and_new_sources() -> None:
    records = [
        {"class_id": 0, "class_name": "soldier", "confidence": 0.8},
        {"class_id": 2, "class_name": "warship", "confidence": 0.7},
    ]

    owned = apply_unified_record_ownership(records, _gates())

    assert owned[0]["source"] == "frozen_base_model"
    assert owned[0]["protocol_id"] is None
    assert owned[1]["source"] == "incremental_model"
    assert owned[1]["protocol_id"] == "warship-incremental"


def test_single_detector_emits_existing_protocol_audit_shape() -> None:
    kept = [{"class_id": 2, "class_name": "warship", "confidence": 0.7}]
    rejected = [
        {
            "class_id": 2,
            "class_name": "warship",
            "confidence": 0.4,
            "action": "reject_unified_activation_threshold",
        },
        {
            "class_id": 2,
            "class_name": "warship",
            "confidence": 0.7,
            "action": "reject_positive_prototype",
        },
    ]

    result = unified_logical_protocol_result(kept, rejected, _gates(), {})

    assert result is not None
    assert result["id"] == "warship-incremental"
    assert result["status"] == "activated"
    assert result["raw_candidate_count"] == 3
    assert result["candidate_count"] == 1
    assert result["activation_rejected_count"] == 1
    assert result["prototype_rejected_count"] == 1
    assert result["activation_threshold"] == 0.63
    assert result["physical_model_shared"] is True
