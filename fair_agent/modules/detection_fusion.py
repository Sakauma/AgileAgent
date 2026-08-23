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


def suppress_cross_class_overlaps(
    records: Iterable[Mapping[str, Any]],
    *,
    iou_threshold: float,
    smaller_box_coverage: float | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Greedily keep the highest-confidence class for one spatial target.

    This is class-agnostic *between* classes only. Same-class duplicate handling
    remains the responsibility of the detector backend or the existing
    class-aware NMS pass. Records are grouped by ``image_id`` when present, so
    the same function is safe for both online single-image inference and
    offline split evaluation.

    A containment threshold complements IoU for a small box that is almost
    entirely inside a larger prediction of the same physical target. The
    original input order is restored after greedy arbitration to keep API
    output stable; confidence sorting is used only to select each winner.
    """
    iou_limit = float(iou_threshold)
    coverage_limit = (
        None if smaller_box_coverage is None else float(smaller_box_coverage)
    )
    if not 0.0 <= iou_limit <= 1.0:
        raise ValueError("cross-class IoU threshold must be within [0, 1]")
    if coverage_limit is not None and not 0.0 <= coverage_limit <= 1.0:
        raise ValueError("smaller-box coverage threshold must be within [0, 1]")

    rows = [dict(row) for row in records]
    grouped_indices: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        image_id = str(row.get("image_id", "__single_image__"))
        grouped_indices.setdefault(image_id, []).append(index)

    suppressed_indices: set[int] = set()
    decisions: List[Dict[str, Any]] = []
    for image_id, indices in grouped_indices.items():
        ordered = sorted(
            indices,
            key=lambda index: (-float(rows[index].get("confidence", 0.0)), index),
        )
        winner_indices: List[int] = []
        for candidate_index in ordered:
            candidate = rows[candidate_index]
            conflict: tuple[int, Dict[str, float]] | None = None
            for winner_index in winner_indices:
                winner = rows[winner_index]
                if int(candidate["class_id"]) == int(winner["class_id"]):
                    continue
                overlap = box_overlap_metrics(candidate["xyxy"], winner["xyxy"])
                if overlap["iou"] >= iou_limit or (
                    coverage_limit is not None
                    and overlap["smaller_coverage"] >= coverage_limit
                ):
                    conflict = winner_index, overlap
                    break
            if conflict is None:
                winner_indices.append(candidate_index)
                continue

            winner_index, overlap = conflict
            winner = rows[winner_index]
            suppressed_indices.add(candidate_index)
            decisions.append(
                {
                    "action": "suppress_cross_class_duplicate",
                    "reason": "global_highest_confidence_overlap",
                    "image_id": image_id,
                    "kept_class_id": int(winner["class_id"]),
                    "suppressed_class_id": int(candidate["class_id"]),
                    "kept_confidence": round(
                        float(winner.get("confidence", 0.0)), 6
                    ),
                    "suppressed_confidence": round(
                        float(candidate.get("confidence", 0.0)), 6
                    ),
                    "iou": round(float(overlap["iou"]), 6),
                    "smaller_box_coverage": round(
                        float(overlap["smaller_coverage"]), 6
                    ),
                    "kept_source": winner.get("source"),
                    "suppressed_source": candidate.get("source"),
                }
            )

    return (
        [row for index, row in enumerate(rows) if index not in suppressed_indices],
        decisions,
    )


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


def context_prior_for_class(
    prior: Mapping[str, Any] | None,
    class_id: int,
) -> Dict[str, Any]:
    """Resolve a class-specific known-context prior with legacy fallback.

    Schema v2 stores priors under ``per_class`` so sea-bound and land-bound
    classes do not dilute each other.  Older single-prior profiles remain
    valid and are returned unchanged.
    """
    normalized = dict(prior or {})
    per_class = normalized.get("per_class")
    if isinstance(per_class, Mapping):
        selected = per_class.get(str(int(class_id)), per_class.get(int(class_id)))
        if isinstance(selected, Mapping):
            return dict(selected)
        return {}
    return normalized


def context_penalty_for_class(
    gate: Mapping[str, Any] | None,
    class_id: int,
) -> float:
    """Resolve a per-class soft-threshold penalty with scalar fallback."""
    normalized = dict(gate or {})
    penalties = normalized.get("max_threshold_penalties")
    if isinstance(penalties, Mapping):
        value = penalties.get(str(int(class_id)), penalties.get(int(class_id)))
        if value is not None:
            return float(value)
    return float(normalized.get("max_threshold_penalty", 0.0))


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
