from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from fair_agent.modules.incremental_guardian import confusion_edge


def box_overlap_metrics(
    first: Iterable[float], second: Iterable[float]
) -> Dict[str, float]:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return {
        "iou": intersection / union if union > 0 else 0.0,
        "first_coverage": intersection / first_area if first_area > 0 else 0.0,
        "second_coverage": intersection / second_area if second_area > 0 else 0.0,
        "smaller_coverage": (
            intersection / min(first_area, second_area)
            if min(first_area, second_area) > 0
            else 0.0
        ),
    }


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    return box_overlap_metrics(first, second)["iou"]


def pairwise_box_overlap_metrics(
    first: Sequence[Iterable[float]],
    second: Sequence[Iterable[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return IoU and first-box coverage matrices using float64 semantics."""
    first_boxes = np.asarray(first, dtype=np.float64).reshape((-1, 4))
    second_boxes = np.asarray(second, dtype=np.float64).reshape((-1, 4))
    shape = (len(first_boxes), len(second_boxes))
    if not first_boxes.size or not second_boxes.size:
        return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64)

    intersection_width = np.maximum(
        0.0,
        np.minimum(first_boxes[:, None, 2], second_boxes[None, :, 2])
        - np.maximum(first_boxes[:, None, 0], second_boxes[None, :, 0]),
    )
    intersection_height = np.maximum(
        0.0,
        np.minimum(first_boxes[:, None, 3], second_boxes[None, :, 3])
        - np.maximum(first_boxes[:, None, 1], second_boxes[None, :, 1]),
    )
    intersection = intersection_width * intersection_height
    first_area = (
        np.maximum(0.0, first_boxes[:, 2] - first_boxes[:, 0])
        * np.maximum(0.0, first_boxes[:, 3] - first_boxes[:, 1])
    )
    second_area = (
        np.maximum(0.0, second_boxes[:, 2] - second_boxes[:, 0])
        * np.maximum(0.0, second_boxes[:, 3] - second_boxes[:, 1])
    )
    union = first_area[:, None] + second_area[None, :] - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros(shape, dtype=np.float64),
        where=union > 0.0,
    )
    first_coverage = np.divide(
        intersection,
        first_area[:, None],
        out=np.zeros(shape, dtype=np.float64),
        where=first_area[:, None] > 0.0,
    )
    return iou, first_coverage


def context_affinity(
    context: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
    neutral_score: float = 1.0,
) -> float:
    """Measure compatibility with a learned known-context prior.

    Missing context or priors are deliberately neutral. Context therefore can
    raise the evidence required for a new-class box, but can never activate a
    specialist or hard-route an image by itself.
    """
    prior = dict(prior or {})
    components: List[float] = []
    for dimension in ("sensor", "scene"):
        weights = prior.get(dimension)
        probabilities = context.get(f"{dimension}_probabilities")
        if isinstance(weights, Mapping) and isinstance(probabilities, Mapping) and weights:
            score = sum(
                float(probabilities.get(name, 0.0)) * float(weight)
                for name, weight in weights.items()
            )
            components.append(max(0.0, min(1.0, score)))
    return sum(components) / len(components) if components else float(neutral_score)


def learn_context_prior(
    contexts: Sequence[Mapping[str, Any]],
    dimensions: Sequence[str] = ("scene",),
) -> Dict[str, Any]:
    """Learn a portable prior from new-class training images only.

    The prior uses model probabilities instead of filenames or lock labels.
    Each dimension is normalized so its strongest observed state has weight
    one. Empty evidence produces no gate rather than an unsafe hard decision.
    """
    result: Dict[str, Any] = {
        "schema_version": 1,
        "source_split": "incremental_train_only",
        "sample_count": len(contexts),
    }
    for dimension in dimensions:
        totals: Dict[str, float] = {}
        for context in contexts:
            probabilities = context.get(f"{dimension}_probabilities")
            if not isinstance(probabilities, Mapping):
                continue
            for name, value in probabilities.items():
                totals[str(name)] = totals.get(str(name), 0.0) + float(value)
        peak = max(totals.values(), default=0.0)
        if peak > 0:
            result[str(dimension)] = {
                name: value / peak for name, value in sorted(totals.items())
            }
    return result


def context_adjusted_threshold(
    base_threshold: float,
    context: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
    max_penalty: float,
) -> tuple[float, float]:
    affinity = context_affinity(context, prior, neutral_score=1.0)
    threshold = min(1.0, float(base_threshold) + float(max_penalty) * (1.0 - affinity))
    return threshold, affinity


def apply_incremental_candidate_gates(
    records: Iterable[Mapping[str, Any]],
    class_thresholds: Mapping[int, float],
    *,
    contexts_by_image: Mapping[str, Mapping[str, Any]] | None = None,
    context_prior: Mapping[str, Any] | None = None,
    max_context_penalty: float = 0.0,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply frozen dev thresholds and optional soft known-context evidence."""
    contexts = dict(contexts_by_image or {})
    thresholds = {int(key): float(value) for key, value in class_thresholds.items()}
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for raw_row in records:
        row = dict(raw_row)
        class_id = int(row["class_id"])
        base_threshold = thresholds.get(class_id)
        if base_threshold is None:
            kept.append(row)
            continue
        context = contexts.get(str(row.get("image_id")), {})
        effective_threshold, affinity = context_adjusted_threshold(
            base_threshold,
            context,
            context_prior,
            max_context_penalty,
        )
        if float(row.get("confidence", 0.0)) >= effective_threshold:
            kept.append(row)
            continue
        rejected.append(
            {
                **row,
                "action": "reject_incremental_candidate",
                "reason": "below_context_adjusted_activation_threshold",
                "base_activation_threshold": round(base_threshold, 6),
                "effective_activation_threshold": round(effective_threshold, 6),
                "context_affinity": round(affinity, 6),
            }
        )
    return kept, rejected


def arbitrate_cross_class_conflicts(
    base_records: Iterable[Mapping[str, Any]],
    incremental_records: Iterable[Mapping[str, Any]],
    conflict_iou: float,
    base_confidence: float,
    incremental_margin: float,
    confusion_graph: Mapping[str, Any] | None = None,
    preserve_base_class_owners: bool = False,
    incremental_coverage: float | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Resolve spatial old/new class conflicts with one deployment/evaluation policy.

    The conservative fallback never removes a frozen owner. A learned confusion edge
    may override the base record only when owner preservation is disabled explicitly.
    Non-overlapping records always coexist, so old and new classes may appear together
    in one image.
    """
    base_rows = [dict(row) for row in base_records]
    incremental_rows = [dict(row) for row in incremental_records]
    pairwise_iou: np.ndarray | None = None
    pairwise_incremental_coverage: np.ndarray | None = None
    if len(base_rows) * len(incremental_rows) >= 16:
        pairwise_iou, pairwise_incremental_coverage = pairwise_box_overlap_metrics(
            [row["xyxy"] for row in incremental_rows],
            [row["xyxy"] for row in base_rows],
        )
    kept: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    suppressed_base_indices: set[int] = set()
    for candidate_index, candidate in enumerate(incremental_rows):
        fallback_conflict = None
        learned_overrides: list[tuple[int, Dict[str, Any]]] = []
        for index, base in enumerate(base_rows):
            if int(base["class_id"]) == int(candidate["class_id"]):
                continue
            if pairwise_iou is None or pairwise_incremental_coverage is None:
                overlap_metrics = box_overlap_metrics(candidate["xyxy"], base["xyxy"])
                overlap = overlap_metrics["iou"]
                candidate_coverage = overlap_metrics["first_coverage"]
            else:
                overlap = float(pairwise_iou[candidate_index, index])
                candidate_coverage = float(
                    pairwise_incremental_coverage[candidate_index, index]
                )
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
                            "base_class_id": int(base["class_id"]),
                            "iou": round(overlap, 6),
                            "incremental_confidence": round(incremental_score, 6),
                            "base_confidence": round(base_score, 6),
                            "evidence_support": int(edge["support"]),
                            "reason": "learned_cross_class_confusion",
                        },
                    )
                )
                continue
            if (
                (
                    overlap >= float(conflict_iou)
                    or (
                        incremental_coverage is not None
                        and candidate_coverage >= float(incremental_coverage)
                    )
                )
                and base_score >= float(base_confidence)
                and incremental_score <= base_score + float(incremental_margin)
            ):
                fallback_conflict = {
                    "action": "reject_specialist",
                    "protocol_id": candidate.get("protocol_id"),
                    "incremental_class_id": int(candidate["class_id"]),
                    "base_class_id": int(base["class_id"]),
                    "iou": round(overlap, 6),
                    "incremental_coverage": round(candidate_coverage, 6),
                    "incremental_confidence": round(incremental_score, 6),
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
