from __future__ import annotations

from fair_agent.modules.incremental_guardian import (
    assess_incremental_candidate,
    learn_confusion_graph,
)
from fair_agent.modules.web_inference import arbitrate_cross_class_conflicts


GATES = {
    "official_hard": {
        "base_map50_min": 0.80,
        "new_map50_min": 0.60,
        "krr_min": 0.95,
        "old_data_overlap_max": 0,
    },
    "advisory": {
        "cumulative_map50_min": 0.80,
        "lock_precision_min": 0.70,
        "false_activation_rate_max": 0.15,
        "latency_proxy_ms_max": 33.3,
    },
}

GUARDIAN = {
    "recovery_actions": {
        "NEW_KNOWLEDGE_UNDERFIT": ["retry_training_on_dev"],
        "CROSS_CLASS_CONFLICT_RISK": ["learn_confusion_graph", "shadow_compare"],
        "OLD_DATA_LEAKAGE": ["block_training"],
    }
}


def metrics(**overrides: float) -> dict:
    values = {
        "base_map50": 0.828,
        "new_map50": 0.735,
        "krr": 0.9998,
        "combined_map50": 0.7965,
        "lock_precision": 0.82,
        "false_activation_rate": 0.10,
        "mean_inference_ms": 55.69,
    }
    values.update(overrides)
    return values


def compliance(**overrides: object) -> dict:
    values = {
        "compliance": "passed",
        "lineage_evidence": "current",
        "old_raw_image_count": 0,
        "old_raw_label_count": 0,
        "old_cache_count": 0,
        "unverified_cache_count": 0,
    }
    values.update(overrides)
    return values


def test_official_full_score_passes_while_internal_metrics_only_warn() -> None:
    result = assess_incremental_candidate(metrics(), compliance(), GATES, GUARDIAN)
    assert result["accepted"] is True
    assert result["status"] == "accepted_with_warnings"
    assert set(result["warnings"]) == {"cumulative_map50", "latency_proxy_ms"}
    assert all(row["passed"] for row in result["official_hard"].values())
    assert any(
        row["diagnosis"] == "CROSS_CLASS_CONFLICT_RISK" and row["automatic"]
        for row in result["recovery_plan"]
    )


def test_new_map_full_score_failure_blocks_promotion_and_selects_recovery() -> None:
    result = assess_incremental_candidate(
        metrics(new_map50=0.59, combined_map50=0.90, mean_inference_ms=20.0),
        compliance(),
        GATES,
        GUARDIAN,
    )
    assert result["accepted"] is False
    assert result["official_hard"]["new_map50"]["passed"] is False
    diagnosis = next(row for row in result["diagnoses"] if row["metric"] == "new_map50")
    assert diagnosis["code"] == "NEW_KNOWLEDGE_UNDERFIT"
    assert diagnosis["actions"] == ["retry_training_on_dev"]


def test_old_data_overlap_is_a_structural_hard_failure() -> None:
    result = assess_incremental_candidate(
        metrics(mean_inference_ms=20.0),
        compliance(old_raw_label_count=1),
        GATES,
        GUARDIAN,
    )
    assert result["accepted"] is False
    assert result["official_hard"]["old_data_overlap"]["value"] == 1.0
    assert result["official_hard"]["old_data_overlap"]["passed"] is False


def test_confusion_graph_discovers_classes_from_dev_without_fixed_ids() -> None:
    ground_truth = [{"image_id": "sample", "class_id": 7, "xyxy": [0, 0, 20, 20]}]
    base = [{"image_id": "sample", "class_id": 3, "confidence": 0.86, "xyxy": [1, 1, 19, 19]}]
    specialist = [{"image_id": "sample", "class_id": 7, "confidence": 0.80, "xyxy": [0, 0, 20, 20]}]
    graph = learn_confusion_graph(
        base,
        specialist,
        ground_truth,
        [7],
        {
            "match_iou": 0.50,
            "min_support": 1,
            "specialist_deficit_padding": 0.02,
            "specialist_deficit_cap": 0.25,
        },
    )
    assert graph["source_split"] == "incremental_dev_only"
    assert graph["edges"][0]["new_class_id"] == 7
    assert graph["edges"][0]["confused_old_class_id"] == 3
    assert graph["edges"][0]["max_specialist_deficit"] == 0.08


def test_learned_confusion_can_remove_base_false_positive() -> None:
    base = [{"class_id": 3, "confidence": 0.86, "xyxy": [1, 1, 19, 19]}]
    specialist = [{
        "class_id": 7,
        "confidence": 0.80,
        "xyxy": [0, 0, 20, 20],
        "protocol_id": "dynamic-expert",
    }]
    graph = {
        "edges": [{
            "new_class_id": 7,
            "confused_old_class_id": 3,
            "support": 2,
            "iou_threshold": 0.50,
            "max_specialist_deficit": 0.08,
        }]
    }
    base_kept, specialist_kept, decisions = arbitrate_cross_class_conflicts(
        base, specialist, 0.50, 0.50, 0.15, graph
    )
    assert base_kept == []
    assert specialist_kept == specialist
    assert decisions[0]["action"] == "suppress_base"
    assert decisions[0]["reason"] == "learned_cross_class_confusion"


def test_unknown_confusion_pair_keeps_conservative_fallback() -> None:
    base = [{"class_id": 4, "confidence": 0.86, "xyxy": [1, 1, 19, 19]}]
    specialist = [{"class_id": 7, "confidence": 0.80, "xyxy": [0, 0, 20, 20]}]
    base_kept, specialist_kept, decisions = arbitrate_cross_class_conflicts(
        base, specialist, 0.50, 0.50, 0.15, {"edges": []}
    )
    assert base_kept == base
    assert specialist_kept == []
    assert decisions[0]["action"] == "reject_specialist"
