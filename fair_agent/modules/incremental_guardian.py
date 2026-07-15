from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence


COMPETITION_SOURCE = "赛题方案第5-7页（性能指标评分）"


def _check(
    value: float,
    threshold: float,
    operator: str,
    source: str,
    blocking: bool,
) -> Dict[str, Any]:
    if operator == ">=":
        passed = value >= threshold
    elif operator == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"不支持的门禁运算符：{operator}")
    return {
        "value": float(value),
        "threshold": float(threshold),
        "operator": operator,
        "passed": bool(passed),
        "blocking": bool(blocking),
        "source": source,
    }


def data_overlap_count(compliance: Mapping[str, Any]) -> int:
    return sum(
        int(compliance.get(key, 0) or 0)
        for key in (
            "old_raw_image_count",
            "old_raw_label_count",
            "old_cache_count",
            "unverified_cache_count",
        )
    )


def diagnose_assessment(
    official_hard: Mapping[str, Mapping[str, Any]],
    advisory: Mapping[str, Mapping[str, Any]],
    recovery_actions: Mapping[str, Sequence[str]],
) -> list[Dict[str, Any]]:
    diagnoses: list[Dict[str, Any]] = []
    failure_codes = {
        "data_lineage": "DATA_COMPLIANCE_FAILED",
        "old_data_overlap": "OLD_DATA_LEAKAGE",
        "base_map50": "BASE_FULL_SCORE_NOT_REACHED",
        "new_map50": "NEW_KNOWLEDGE_UNDERFIT",
        "krr": "KNOWLEDGE_RETENTION_REGRESSION",
    }
    warning_codes = {
        "cumulative_map50": "CROSS_CLASS_CONFLICT_RISK",
        "lock_precision": "LOW_DEPLOYMENT_PRECISION",
        "false_activation_rate": "FALSE_ACTIVATION_RISK",
        "latency_proxy_ms": "LATENCY_OPTIMIZATION_REQUIRED",
    }
    for name, result in official_hard.items():
        if result["passed"]:
            continue
        code = failure_codes.get(name, name.upper())
        diagnoses.append({
            "code": code,
            "severity": "blocking",
            "metric": name,
            "actions": list(recovery_actions.get(code, [])),
        })
    for name, result in advisory.items():
        if result["passed"]:
            continue
        code = warning_codes.get(name, name.upper())
        diagnoses.append({
            "code": code,
            "severity": "warning",
            "metric": name,
            "actions": list(recovery_actions.get(code, [])),
        })
    return diagnoses


def assess_incremental_candidate(
    metrics: Mapping[str, Any],
    compliance: Mapping[str, Any],
    gates: Mapping[str, Any],
    guardian: Mapping[str, Any],
) -> Dict[str, Any]:
    hard = gates["official_hard"]
    advisory_config = gates["advisory"]
    overlap = data_overlap_count(compliance)
    lineage_current = (
        compliance.get("compliance") == "passed"
        and compliance.get("lineage_evidence") in {"current", "not_required"}
    )
    official_hard = {
        "data_lineage": {
            "value": 1.0 if lineage_current else 0.0,
            "threshold": 1.0,
            "operator": ">=",
            "passed": bool(lineage_current),
            "blocking": True,
            "source": "增量训练数据血缘与可达文件审计",
        },
        "old_data_overlap": _check(
            overlap,
            float(hard["old_data_overlap_max"]),
            "<=",
            "赛题约束：增量学习阶段只能使用增量数据集",
            True,
        ),
        "base_map50": _check(
            float(metrics["base_map50"]),
            float(hard["base_map50_min"]),
            ">=",
            COMPETITION_SOURCE,
            True,
        ),
        "new_map50": _check(
            float(metrics["new_map50"]),
            float(hard["new_map50_min"]),
            ">=",
            COMPETITION_SOURCE,
            True,
        ),
        "krr": _check(
            float(metrics["krr"]),
            float(hard["krr_min"]),
            ">=",
            COMPETITION_SOURCE,
            True,
        ),
    }
    advisory = {
        "cumulative_map50": _check(
            float(metrics["combined_map50"]),
            float(advisory_config["cumulative_map50_min"]),
            ">=",
            "Agent内部累计质量观察指标，非赛题独立评分项",
            False,
        ),
        "lock_precision": _check(
            float(metrics["lock_precision"]),
            float(advisory_config["lock_precision_min"]),
            ">=",
            "Agent部署风险观察指标",
            False,
        ),
        "false_activation_rate": _check(
            float(metrics["false_activation_rate"]),
            float(advisory_config["false_activation_rate_max"]),
            "<=",
            "Agent部署风险观察指标",
            False,
        ),
        "latency_proxy_ms": _check(
            float(metrics["mean_inference_ms"]),
            float(advisory_config["latency_proxy_ms_max"]),
            "<=",
            "x86代理性能观察指标，不能替代310B实测FPS",
            False,
        ),
    }
    accepted = all(item["passed"] for item in official_hard.values())
    diagnoses = diagnose_assessment(
        official_hard,
        advisory,
        guardian.get("recovery_actions", {}),
    )
    return {
        "schema_version": 1,
        "accepted": accepted,
        "status": "accepted_with_warnings" if accepted and any(not row["passed"] for row in advisory.values()) else (
            "accepted" if accepted else "rejected"
        ),
        "official_hard": official_hard,
        "advisory": advisory,
        "warnings": [name for name, row in advisory.items() if not row["passed"]],
        "diagnoses": diagnoses,
        "recovery_plan": [
            {
                "diagnosis": item["code"],
                "severity": item["severity"],
                "actions": item["actions"],
                "automatic": item["code"] == "CROSS_CLASS_CONFLICT_RISK",
                "lock_feedback_allowed": False,
            }
            for item in diagnoses
        ],
        "data_compliance": {
            "lineage_evidence": compliance.get("lineage_evidence"),
            "old_raw_image_count": int(compliance.get("old_raw_image_count", 0) or 0),
            "old_raw_label_count": int(compliance.get("old_raw_label_count", 0) or 0),
            "old_cache_count": int(compliance.get("old_cache_count", 0) or 0),
            "unverified_cache_count": int(compliance.get("unverified_cache_count", 0) or 0),
            "old_data_overlap_count": overlap,
        },
    }


def _iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def learn_confusion_graph(
    base_predictions: Iterable[Mapping[str, Any]],
    specialist_predictions: Iterable[Mapping[str, Any]],
    ground_truth: Iterable[Mapping[str, Any]],
    focus_class_ids: Iterable[int],
    settings: Mapping[str, Any],
) -> Dict[str, Any]:
    focus = {int(value) for value in focus_class_ids}
    match_iou = float(settings["match_iou"])
    min_support = int(settings["min_support"])
    deficit_padding = float(settings["specialist_deficit_padding"])
    deficit_cap = float(settings["specialist_deficit_cap"])
    base_by_image: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    specialist_by_image: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in base_predictions:
        base_by_image[str(row["image_id"])].append(row)
    for row in specialist_predictions:
        specialist_by_image[str(row["image_id"])].append(row)

    evidence: Dict[tuple[int, int], list[Dict[str, float]]] = defaultdict(list)
    for target in ground_truth:
        new_class_id = int(target["class_id"])
        if new_class_id not in focus:
            continue
        image_id = str(target["image_id"])
        matched_specialists = [
            row for row in specialist_by_image[image_id]
            if int(row["class_id"]) == new_class_id
            and _iou(row["xyxy"], target["xyxy"]) >= match_iou
        ]
        if not matched_specialists:
            continue
        specialist = max(matched_specialists, key=lambda row: float(row.get("confidence", 0.0)))
        for base in base_by_image[image_id]:
            old_class_id = int(base["class_id"])
            if old_class_id == new_class_id or _iou(base["xyxy"], target["xyxy"]) < match_iou:
                continue
            pair_iou = _iou(base["xyxy"], specialist["xyxy"])
            if pair_iou < match_iou:
                continue
            base_confidence = float(base.get("confidence", 0.0))
            specialist_confidence = float(specialist.get("confidence", 0.0))
            evidence[(new_class_id, old_class_id)].append({
                "iou": pair_iou,
                "base_confidence": base_confidence,
                "specialist_confidence": specialist_confidence,
                "specialist_deficit": max(0.0, base_confidence - specialist_confidence),
            })

    edges = []
    for (new_class_id, old_class_id), rows in sorted(evidence.items()):
        if len(rows) < min_support:
            continue
        deficits = sorted(row["specialist_deficit"] for row in rows)
        percentile_index = max(0, min(len(deficits) - 1, round(0.95 * (len(deficits) - 1))))
        maximum_deficit = min(deficit_cap, deficits[percentile_index] + deficit_padding)
        edges.append({
            "new_class_id": new_class_id,
            "confused_old_class_id": old_class_id,
            "support": len(rows),
            "iou_threshold": match_iou,
            "max_specialist_deficit": round(maximum_deficit, 6),
            "mean_iou": round(sum(row["iou"] for row in rows) / len(rows), 6),
            "mean_base_confidence": round(sum(row["base_confidence"] for row in rows) / len(rows), 6),
            "mean_specialist_confidence": round(
                sum(row["specialist_confidence"] for row in rows) / len(rows), 6
            ),
            "strategy": "calibrated_specialist_override",
        })
    return {
        "schema_version": 1,
        "source_split": "incremental_dev_only",
        "focus_class_ids": sorted(focus),
        "match_iou": match_iou,
        "minimum_support": min_support,
        "edges": edges,
        "hard_scene_gate": False,
    }


def confusion_edge(
    graph: Mapping[str, Any] | None,
    specialist_class_id: int,
    base_class_id: int,
) -> Mapping[str, Any] | None:
    for edge in (graph or {}).get("edges", []):
        if (
            int(edge["new_class_id"]) == int(specialist_class_id)
            and int(edge["confused_old_class_id"]) == int(base_class_id)
        ):
            return edge
    return None
