from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_detections(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    class_key: str = "class_id",
) -> Dict[str, Any]:
    """Match detections by class and IoU before measuring numerical drift.

    API and OM post-processing order is confidence-dependent. Pairing two lists by
    response position can therefore report a large coordinate error for two valid,
    merely reordered boxes. This matcher uses deterministic class-aware greedy IoU
    assignment. Confidence difference and original indices provide stable tie-breaks.
    """

    reference_by_class: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    candidate_by_class: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(reference):
        reference_by_class[int(row[class_key])].append((index, row))
    for index, row in enumerate(candidate):
        candidate_by_class[int(row[class_key])].append((index, row))

    matched: list[Dict[str, Any]] = []
    unmatched_reference: list[int] = []
    unmatched_candidate: list[int] = []
    for class_id in sorted(set(reference_by_class) | set(candidate_by_class)):
        expected = reference_by_class.get(class_id, [])
        actual = candidate_by_class.get(class_id, [])
        pairs = []
        for reference_index, reference_row in expected:
            for candidate_index, candidate_row in actual:
                overlap = box_iou(reference_row["xyxy"], candidate_row["xyxy"])
                confidence_delta = abs(
                    float(reference_row["confidence"])
                    - float(candidate_row["confidence"])
                )
                pairs.append(
                    (
                        -overlap,
                        confidence_delta,
                        reference_index,
                        candidate_index,
                        reference_row,
                        candidate_row,
                    )
                )
        used_reference: set[int] = set()
        used_candidate: set[int] = set()
        for (
            negative_overlap,
            confidence_delta,
            reference_index,
            candidate_index,
            reference_row,
            candidate_row,
        ) in sorted(pairs, key=lambda item: item[:4]):
            if reference_index in used_reference or candidate_index in used_candidate:
                continue
            used_reference.add(reference_index)
            used_candidate.add(candidate_index)
            coordinate_delta = max(
                abs(float(left) - float(right))
                for left, right in zip(reference_row["xyxy"], candidate_row["xyxy"])
            )
            matched.append(
                {
                    "class_id": class_id,
                    "reference_index": reference_index,
                    "candidate_index": candidate_index,
                    "iou": -negative_overlap,
                    "max_box_abs": coordinate_delta,
                    "confidence_abs": confidence_delta,
                }
            )
        unmatched_reference.extend(
            index for index, _row in expected if index not in used_reference
        )
        unmatched_candidate.extend(
            index for index, _row in actual if index not in used_candidate
        )

    reference_classes = Counter(int(row[class_key]) for row in reference)
    candidate_classes = Counter(int(row[class_key]) for row in candidate)
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "count_equal": len(reference) == len(candidate),
        "class_counts_equal": reference_classes == candidate_classes,
        "reference_class_counts": dict(sorted(reference_classes.items())),
        "candidate_class_counts": dict(sorted(candidate_classes.items())),
        "matched": sorted(matched, key=lambda row: row["reference_index"]),
        "unmatched_reference": sorted(unmatched_reference),
        "unmatched_candidate": sorted(unmatched_candidate),
        "max_box_abs": max((row["max_box_abs"] for row in matched), default=0.0),
        "max_confidence_abs": max(
            (row["confidence_abs"] for row in matched), default=0.0
        ),
    }


def _context_difference(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "sensor_equal": reference.get("sensor") == candidate.get("sensor"),
        "scene_equal": reference.get("scene") == candidate.get("scene"),
    }
    for kind in ("sensor", "scene"):
        expected = reference.get(f"{kind}_probabilities") or {}
        actual = candidate.get(f"{kind}_probabilities") or {}
        labels = set(expected) | set(actual)
        result[f"max_{kind}_probability_abs"] = max(
            (
                abs(float(expected.get(label, 0.0)) - float(actual.get(label, 0.0)))
                for label in labels
            ),
            default=0.0,
        )
    return result


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("image_id") or record.get("filename")
    if not value:
        raise ValueError("对齐记录缺少image_id/filename。")
    return Path(str(value)).stem


def read_jsonl(path: Path) -> dict[str, Dict[str, Any]]:
    records: dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL第{line_number}行无效：{path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL第{line_number}行不是对象：{path}")
            image_id = _record_id(value)
            if image_id in records:
                raise ValueError(f"JSONL包含重复图像：{image_id}: {path}")
            records[image_id] = value
    return records


def compare_api_records(
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    max_box_abs: float = 1.0,
    max_confidence_abs: float = 0.02,
) -> Dict[str, Any]:
    reference_ids = set(reference)
    candidate_ids = set(candidate)
    missing_candidate = sorted(reference_ids - candidate_ids)
    unexpected_candidate = sorted(candidate_ids - reference_ids)
    rows: list[Dict[str, Any]] = []
    maxima = {
        "box": 0.0,
        "confidence": 0.0,
        "sensor_probability": 0.0,
        "scene_probability": 0.0,
    }
    reason_counts: Counter[str] = Counter()
    for image_id in sorted(reference_ids & candidate_ids):
        expected = reference[image_id]
        actual = candidate[image_id]
        detections = match_detections(
            expected.get("detections") or [], actual.get("detections") or []
        )
        context = _context_difference(
            expected.get("context") or {}, actual.get("context") or {}
        )
        maxima["box"] = max(maxima["box"], float(detections["max_box_abs"]))
        maxima["confidence"] = max(
            maxima["confidence"], float(detections["max_confidence_abs"])
        )
        maxima["sensor_probability"] = max(
            maxima["sensor_probability"],
            float(context["max_sensor_probability_abs"]),
        )
        maxima["scene_probability"] = max(
            maxima["scene_probability"],
            float(context["max_scene_probability_abs"]),
        )
        reasons = []
        if not detections["count_equal"]:
            reasons.append("detection_count")
        if not detections["class_counts_equal"]:
            reasons.append("class_counts")
        if detections["max_box_abs"] > max_box_abs:
            reasons.append("box_delta")
        if detections["max_confidence_abs"] > max_confidence_abs:
            reasons.append("confidence_delta")
        if not context["sensor_equal"]:
            reasons.append("sensor")
        if not context["scene_equal"]:
            reasons.append("scene")
        if reasons:
            reason_counts.update(reasons)
            rows.append(
                {
                    "image_id": image_id,
                    "filename": str(actual.get("filename") or expected.get("filename") or ""),
                    "reasons": reasons,
                    "detections": detections,
                    "context": context,
                }
            )

    gates = {
        "image_sets_equal": not missing_candidate and not unexpected_candidate,
        "detection_counts_equal": reason_counts["detection_count"] == 0,
        "class_counts_equal": reason_counts["class_counts"] == 0,
        "max_box_abs": maxima["box"] <= max_box_abs,
        "max_confidence_abs": maxima["confidence"] <= max_confidence_abs,
        "sensor_labels_equal": reason_counts["sensor"] == 0,
        "scene_labels_equal": reason_counts["scene"] == 0,
    }
    return {
        "schema_version": 1,
        "reference_image_count": len(reference),
        "candidate_image_count": len(candidate),
        "compared_image_count": len(reference_ids & candidate_ids),
        "missing_candidate": missing_candidate,
        "unexpected_candidate": unexpected_candidate,
        "thresholds": {
            "max_box_abs": max_box_abs,
            "max_confidence_abs": max_confidence_abs,
        },
        "maxima": {
            "max_box_abs": maxima["box"],
            "max_confidence_abs": maxima["confidence"],
            "max_sensor_probability_abs": maxima["sensor_probability"],
            "max_scene_probability_abs": maxima["scene_probability"],
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "mismatch_count": len(rows),
        "mismatches": rows,
        "gates": gates,
        "passed": all(gates.values()),
    }
