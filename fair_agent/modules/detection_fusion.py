from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from fair_agent.modules.incremental_guardian import confusion_edge


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def arbitrate_cross_class_conflicts(
    base_records: Iterable[Mapping[str, Any]],
    incremental_records: Iterable[Mapping[str, Any]],
    conflict_iou: float,
    base_confidence: float,
    incremental_margin: float,
    confusion_graph: Mapping[str, Any] | None = None,
    preserve_base_class_owners: bool = False,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Resolve spatial old/new class conflicts with one deployment/evaluation policy.

    The conservative fallback never removes a frozen owner. A learned confusion edge
    may override the base record only when owner preservation is disabled explicitly.
    Non-overlapping records always coexist, so old and new classes may appear together
    in one image.
    """
    base_rows = [dict(row) for row in base_records]
    kept: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    suppressed_base_indices: set[int] = set()
    for raw_candidate in incremental_records:
        candidate = dict(raw_candidate)
        fallback_conflict = None
        learned_overrides: list[tuple[int, Dict[str, Any]]] = []
        for index, base in enumerate(base_rows):
            if int(base["class_id"]) == int(candidate["class_id"]):
                continue
            overlap = box_iou(candidate["xyxy"], base["xyxy"])
            base_score = float(base.get("confidence", 0.0))
            incremental_score = float(candidate.get("confidence", 0.0))
            edge = confusion_edge(
                confusion_graph,
                int(candidate["class_id"]),
                int(base["class_id"]),
            )
            if edge is not None and (
                overlap >= float(edge["iou_threshold"])
                and incremental_score + float(edge["max_specialist_deficit"])
                >= base_score
            ):
                learned_overrides.append(
                    (
                        index,
                        {
                            "action": "suppress_base",
                            "protocol_id": candidate.get("protocol_id"),
                            "incremental_class_id": int(candidate["class_id"]),
                            # Compatibility field consumed by existing UI/tests.
                            "specialist_class_id": int(candidate["class_id"]),
                            "base_class_id": int(base["class_id"]),
                            "iou": round(overlap, 6),
                            "incremental_confidence": round(incremental_score, 6),
                            "specialist_confidence": round(incremental_score, 6),
                            "base_confidence": round(base_score, 6),
                            "evidence_support": int(edge["support"]),
                            "reason": "learned_cross_class_confusion",
                        },
                    )
                )
                continue
            if (
                overlap >= float(conflict_iou)
                and base_score >= float(base_confidence)
                and incremental_score <= base_score + float(incremental_margin)
            ):
                fallback_conflict = {
                    "action": "reject_specialist",
                    "protocol_id": candidate.get("protocol_id"),
                    "incremental_class_id": int(candidate["class_id"]),
                    "specialist_class_id": int(candidate["class_id"]),
                    "base_class_id": int(base["class_id"]),
                    "iou": round(overlap, 6),
                    "incremental_confidence": round(incremental_score, 6),
                    "specialist_confidence": round(incremental_score, 6),
                    "base_confidence": round(base_score, 6),
                    "reason": "cross_class_conflict",
                }
        if learned_overrides:
            for index, decision in learned_overrides:
                if preserve_base_class_owners:
                    decisions.append(
                        {
                            **decision,
                            "action": "coexist_preserve_base_owner",
                            "reason": "learned_confusion_with_frozen_owner_preserved",
                        }
                    )
                else:
                    suppressed_base_indices.add(index)
                    decisions.append(decision)
            kept.append(candidate)
        elif fallback_conflict is None:
            kept.append(candidate)
        else:
            decisions.append(fallback_conflict)
    base_kept = [
        row for index, row in enumerate(base_rows) if index not in suppressed_base_indices
    ]
    return base_kept, kept, decisions
