from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from PIL import Image

from fair_agent.core.config import ROOT, rel_path, resolve_path
from fair_agent.core.hashes import sha256_file
from fair_agent.modules.detection_fusion import (
    arbitrate_cross_class_conflicts,
    suppress_cross_class_overlaps,
)


CLASS_NAMES = {
    0: "soldier",
    1: "small_aircraft",
    2: "warship",
    3: "tank",
    4: "patrol_boat",
    5: "armored_vehicle",
}


def read_split(path: str | Path) -> List[Path]:
    resolved = resolve_path(path)
    return [resolve_path(line.strip()) for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_label(image: Path) -> Path:
    adjacent = image.with_suffix(".txt")
    if adjacent.exists():
        return adjacent
    sibling = image.parent.parent / "labels" / f"{image.stem}.txt"
    if sibling.exists():
        return sibling
    # Also support the canonical YOLO layout:
    # ``dataset/images/<split>/x.png -> dataset/labels/<split>/x.txt``.
    if image.parent.parent.name == "images":
        standard = (
            image.parent.parent.parent
            / "labels"
            / image.parent.name
            / f"{image.stem}.txt"
        )
        if standard.exists():
            return standard
    raise FileNotFoundError(f"找不到图像标签：{image}")


def read_yolo_labels(path: Path) -> List[tuple[int, float, float, float, float]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        columns = line.split()
        if len(columns) != 5:
            raise ValueError(f"YOLO 标签列数错误：{path}:{line_number}")
        class_id = int(float(columns[0]))
        values = tuple(float(value) for value in columns[1:])
        rows.append((class_id, *values))
    return rows


def image_class_ids(image: Path) -> set[int]:
    return {row[0] for row in read_yolo_labels(source_label(image))}


def yolo_ground_truth(images: Sequence[Path], class_ids: Iterable[int] | None = None) -> List[Dict[str, Any]]:
    allowed = set(class_ids) if class_ids is not None else None
    rows = []
    for image in images:
        with Image.open(image) as source:
            width, height = source.size
        for class_id, x, y, box_width, box_height in read_yolo_labels(source_label(image)):
            if allowed is not None and class_id not in allowed:
                continue
            x1 = (x - box_width / 2) * width
            y1 = (y - box_height / 2) * height
            x2 = (x + box_width / 2) * width
            y2 = (y + box_height / 2) * height
            rows.append({
                "image_id": image.stem,
                "class_id": class_id,
                "xyxy": [x1, y1, x2, y2],
            })
    return rows


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def _class_matches(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    class_id: int,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    class_gt: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in ground_truth:
        if int(row["class_id"]) == class_id:
            class_gt[str(row["image_id"])].append(row)
    ordered = sorted(
        (row for row in predictions if int(row["class_id"]) == class_id),
        key=lambda row: -float(row["confidence"]),
    )
    true_positive = np.zeros(len(ordered), dtype=float)
    false_positive = np.zeros(len(ordered), dtype=float)
    prediction_indices: Dict[str, List[int]] = defaultdict(list)
    for index, prediction in enumerate(ordered):
        prediction_indices[str(prediction["image_id"])].append(index)
    for image_id, indices in prediction_indices.items():
        matches = []
        for gt_index, target in enumerate(class_gt.get(image_id, [])):
            for prediction_index in indices:
                overlap = box_iou(ordered[prediction_index]["xyxy"], target["xyxy"])
                if overlap >= iou_threshold:
                    matches.append((gt_index, prediction_index, overlap))
        best_per_prediction = []
        used_predictions: set[int] = set()
        for match in sorted(matches, key=lambda row: -row[2]):
            if match[1] not in used_predictions:
                best_per_prediction.append(match)
                used_predictions.add(match[1])
        used_targets: set[int] = set()
        for gt_index, prediction_index, _overlap in sorted(
            best_per_prediction, key=lambda row: row[1]
        ):
            if gt_index in used_targets:
                continue
            true_positive[prediction_index] = 1.0
            used_targets.add(gt_index)
    false_positive[true_positive == 0] = 1.0
    return true_positive, false_positive, sum(len(rows) for rows in class_gt.values())


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    final_recall = recall[-1] if len(recall) else 1.0
    mrec = np.concatenate(([0.0], recall, [final_recall], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    values = np.interp(x, mrec, mpre)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, x))
    return float(np.trapz(values, x))


def evaluate_ap50(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    class_ids: Iterable[int],
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    per_class: Dict[int, float] = {}
    for class_id in class_ids:
        true_positive, false_positive, target_count = _class_matches(
            predictions, ground_truth, int(class_id), iou_threshold
        )
        if target_count == 0:
            continue
        tp = np.cumsum(true_positive)
        fp = np.cumsum(false_positive)
        recall = tp / target_count
        precision = tp / np.maximum(tp + fp, 1e-12)
        per_class[int(class_id)] = _compute_ap(recall, precision) if len(precision) else 0.0
    return {
        "map50": mean(per_class.values()) if per_class else 0.0,
        "per_class_ap50": per_class,
    }


def retention_metrics(
    before_predictions: Sequence[Mapping[str, Any]],
    after_predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    old_class_ids: Iterable[int],
) -> Dict[str, Any]:
    """Recompute old-class retention from the actual pre/post prediction records."""
    old_ids = {int(value) for value in old_class_ids}
    before_metrics = evaluate_ap50(before_predictions, ground_truth, old_ids)
    after_metrics = evaluate_ap50(after_predictions, ground_truth, old_ids)

    def canonical(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
        return sorted(
            (
                str(row["image_id"]),
                int(row["class_id"]),
                -float(row["confidence"]),
                tuple(float(value) for value in row["xyxy"]),
            )
            for row in rows
            if int(row["class_id"]) in old_ids
        )

    before = float(before_metrics["map50"])
    after = float(after_metrics["map50"])
    return {
        "old_map50_before": before,
        "old_map50_after": after,
        "krr": after / before if before else 0.0,
        "old_prediction_equivalent": canonical(before_predictions) == canonical(after_predictions),
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
    }


def precision_recall(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    class_id: int,
    threshold: float,
    iou_threshold: float = 0.50,
) -> Dict[str, float | int]:
    selected = [row for row in predictions if float(row["confidence"]) >= threshold]
    true_positive, false_positive, target_count = _class_matches(selected, ground_truth, class_id, iou_threshold)
    tp = int(true_positive.sum())
    fp = int(false_positive.sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / target_count if target_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "targets": target_count}


def calibrate_threshold(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    class_id: int,
    minimum: float = 0.01,
    maximum: float = 0.99,
    step: float = 0.01,
    target_precision: float = 0.90,
) -> Dict[str, Any]:
    count = int(round((maximum - minimum) / step)) + 1
    curve = []
    for index in range(count):
        threshold = round(minimum + index * step, 6)
        metrics = precision_recall(predictions, ground_truth, class_id, threshold)
        curve.append({"threshold": threshold, **metrics})
    qualified = [row for row in curve if float(row["precision"]) >= target_precision and int(row["tp"]) > 0]
    if qualified:
        selected = max(qualified, key=lambda row: (float(row["recall"]), float(row["f1"]), float(row["threshold"])))
        passed = True
        reason = "target_precision_reached"
    else:
        selected = max(curve, key=lambda row: (float(row["f1"]), float(row["precision"]), float(row["threshold"])))
        passed = False
        reason = "fallback_max_f1"
    return {"passed": passed, "reason": reason, "target_precision": target_precision, "selected": selected, "curve": curve}


def class_aware_nms(predictions: Sequence[Mapping[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    output = []
    grouped: Dict[tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(str(row["image_id"]), int(row["class_id"]))].append(row)
    for rows in grouped.values():
        kept = []
        for candidate in sorted(rows, key=lambda row: -float(row["confidence"])):
            if any(box_iou(candidate["xyxy"], existing["xyxy"]) >= iou_threshold for existing in kept):
                continue
            kept.append(candidate)
        output.extend(dict(row) for row in kept)
    return output


def fuse_old_new_predictions(
    old_predictions: Sequence[Mapping[str, Any]],
    new_predictions: Sequence[Mapping[str, Any]],
    *,
    nms_iou: float,
    cross_class: Mapping[str, Any] | None = None,
    cross_class_suppression: Mapping[str, Any] | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    settings = dict(cross_class or {})
    if settings.get("enabled", False):
        old_kept, new_kept, decisions = arbitrate_cross_class_conflicts(
            old_predictions,
            new_predictions,
            float(settings.get("iou", 0.50)),
            float(settings.get("base_confidence", 0.50)),
            float(settings.get("incremental_margin", settings.get("specialist_margin", 0.15))),
            None,
            bool(settings.get("preserve_base_class_owners", True)),
            (
                float(settings["incremental_coverage"])
                if settings.get("incremental_coverage") is not None
                else None
            ),
        )
    else:
        old_kept = [dict(row) for row in old_predictions]
        new_kept = [dict(row) for row in new_predictions]
        decisions = []
    # The base validator has already applied NMS. Re-running NMS on old rows can
    # silently remove an old prediction and lower KRR even though the base model
    # is byte-for-byte frozen. Only the newly introduced class owner needs a
    # second NMS pass before the two disjoint ownership streams are concatenated.
    fused = [dict(row) for row in old_kept]
    fused.extend(class_aware_nms(new_kept, float(nms_iou)))
    suppression = dict(cross_class_suppression or {})
    if suppression.get("enabled") is True:
        fused, suppression_decisions = suppress_cross_class_overlaps(
            fused,
            iou_threshold=float(suppression["iou"]),
            smaller_box_coverage=(
                float(suppression["smaller_box_coverage"])
                if suppression.get("smaller_box_coverage") is not None
                else None
            ),
        )
        decisions.extend(suppression_decisions)
    return fused, decisions


def subset_rows(rows: Sequence[Mapping[str, Any]], image_ids: Iterable[str]) -> List[Dict[str, Any]]:
    allowed = set(image_ids)
    return [dict(row) for row in rows if str(row["image_id"]) in allowed]


def load_experiment_profile(
    profile_id: str,
    profile_root: str | Path | None = None,
) -> Dict[str, Any]:
    root = (
        resolve_path(profile_root)
        if profile_root is not None
        else ROOT / "models" / "profiles"
    )
    active = root / profile_id / "active.json"
    if not active.is_file():
        raise FileNotFoundError(f"增量检测档尚未通过验收：{profile_id}")
    profile = json.loads(active.read_text(encoding="utf-8"))
    if (
        profile.get("profile_id") != profile_id
        or profile.get("competition_accepted") is not True
        or profile.get("deployment_accepted") is not True
        or profile.get("incremental_mode") != "class_incremental"
        or profile.get("evidence_level") != "verified"
        or profile.get("deployment") != "dual_detector"
    ):
        raise ValueError(f"增量检测档无效：{profile_id}")

    try:
        class_names = {
            int(key): str(value)
            for key, value in dict(profile["class_names"]).items()
        }
        base_mapping = {
            int(key): int(value)
            for key, value in dict(profile["base_local_to_global"]).items()
        }
        specialist_mapping = {
            int(key): int(value)
            for key, value in dict(
                profile["specialist_local_to_global"]
            ).items()
        }
        new_global_ids = sorted(
            int(value) for value in profile["new_global_ids"]
        )
        thresholds = {
            int(key): float(value)
            for key, value in dict(profile["activation_thresholds"]).items()
        }
        base_thresholds = {
            int(key): float(value)
            for key, value in dict(
                profile["base_activation_thresholds"]
            ).items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"增量检测档字段无效：{profile_id}") from exc

    if (
        class_names != CLASS_NAMES
        or set(base_mapping) != set(range(len(base_mapping)))
        or set(specialist_mapping) != set(range(len(specialist_mapping)))
        or set(base_mapping.values()) != {0, 1, 2, 3}
        or set(specialist_mapping.values()) != {4, 5}
        or new_global_ids != [4, 5]
        or set(thresholds) != {4, 5}
        or set(base_thresholds) != {0, 1, 2, 3}
        or any(not 0.01 <= value <= 1.0 for value in thresholds.values())
        or any(not 0.01 <= value <= 1.0 for value in base_thresholds.values())
    ):
        raise ValueError(f"增量检测档类别或阈值无效：{profile_id}")

    for path_key, hash_key in (
        ("base_weight", "base_sha256"),
        ("specialist_weight", "specialist_sha256"),
    ):
        weight = resolve_path(profile[path_key])
        expected = str(profile.get(hash_key) or "")
        if (
            not weight.is_file()
            or len(expected) != 64
            or sha256_file(weight) != expected
        ):
            raise ValueError(f"增量检测档权重校验失败：{profile_id}:{path_key}")

    context_gate = profile.get("context_gate")
    context_prior = profile.get("context_prior")
    if (
        not isinstance(context_gate, Mapping)
        or context_gate.get("enabled") is not True
        or context_gate.get("policy") != "soft_threshold_penalty"
        or context_gate.get("hard_routing") is not False
        or context_gate.get("learning_data_scope") != "incremental_train_only"
        or not isinstance(context_prior, Mapping)
        or context_prior.get("source_split") != "incremental_train_only"
    ):
        raise ValueError(f"增量检测档新增类场景门控无效：{profile_id}")
    penalties = {
        int(key): float(value)
        for key, value in dict(
            context_gate.get("max_threshold_penalties") or {}
        ).items()
    }
    if set(penalties) != {4, 5} or any(
        not 0.0 <= value <= 1.0 for value in penalties.values()
    ):
        raise ValueError(f"增量检测档新增类场景惩罚无效：{profile_id}")
    context_prior_path = resolve_path(profile["context_prior_source"])
    context_prior_hash = str(profile.get("context_prior_sha256") or "")
    if (
        not context_prior_path.is_file()
        or len(context_prior_hash) != 64
        or sha256_file(context_prior_path) != context_prior_hash
        or json.loads(context_prior_path.read_text(encoding="utf-8"))
        != dict(context_prior)
    ):
        raise ValueError(f"增量检测档新增类场景先验无效：{profile_id}")

    base_gate = profile.get("base_context_gate")
    base_prior = profile.get("base_context_prior")
    if (
        not isinstance(base_gate, Mapping)
        or base_gate.get("enabled") is not True
        or base_gate.get("policy") != "soft_threshold_penalty"
        or base_gate.get("hard_routing") is not False
        or base_gate.get("learning_data_scope") != "base_train_only"
        or not isinstance(base_prior, Mapping)
        or base_prior.get("source_split") != "base_train_only"
    ):
        raise ValueError(f"增量检测档 Base 场景门控无效：{profile_id}")
    base_penalties = {
        int(key): float(value)
        for key, value in dict(
            base_gate.get("max_threshold_penalties") or {}
        ).items()
    }
    if set(base_penalties) != {0, 1, 2, 3} or any(
        not 0.0 <= value <= 1.0 for value in base_penalties.values()
    ):
        raise ValueError(f"增量检测档 Base 场景惩罚无效：{profile_id}")
    base_prior_path = resolve_path(profile["base_context_prior_source"])
    base_prior_hash = str(profile.get("base_context_prior_sha256") or "")
    if (
        not base_prior_path.is_file()
        or len(base_prior_hash) != 64
        or sha256_file(base_prior_path) != base_prior_hash
        or json.loads(base_prior_path.read_text(encoding="utf-8"))
        != dict(base_prior)
    ):
        raise ValueError(f"增量检测档 Base 场景先验无效：{profile_id}")

    try:
        calibration_sources = {
            int(key): str(value)
            for key, value in dict(profile["calibration_sources"]).items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"增量检测档校准索引无效：{profile_id}") from exc
    if set(calibration_sources) != {4, 5}:
        raise ValueError(f"增量检测档缺少逐类校准证据：{profile_id}")
    calibration_payloads: Dict[Path, Dict[str, Any]] = {}
    for class_id, source in calibration_sources.items():
        calibration = resolve_path(source)
        if not calibration.is_file():
            raise ValueError(
                f"增量检测档缺少校准证据：{profile_id}:{class_id}"
            )
        payload = calibration_payloads.setdefault(
            calibration,
            json.loads(calibration.read_text(encoding="utf-8")),
        )
        try:
            calibrated_threshold = float(
                payload["per_class_thresholds"][str(class_id)]
            )
            selected_threshold = float(
                payload["per_class"][str(class_id)]["selected"]["threshold"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"增量检测档逐类校准证据无效：{profile_id}:{class_id}"
            ) from exc
        data_scope = dict(payload.get("data_scope") or {})
        if (
            int(payload.get("schema_version", 0)) < 4
            or payload.get("phase") != "system_calibration"
            or payload.get("counted_as_incremental_learning") is not False
            or payload.get("detector_weights_updated") is not False
            or payload.get("learning_data_scope") is not None
            or payload.get("source_split") != "mixed_dev_only"
            or payload.get("deployment_policy")
            != "competition_map50_dev_calibrated"
            or data_scope.get("gate_selection") != "mixed_dev_only"
            or data_scope.get("scene_sensor_model_training")
            != "base_and_incremental_train_dev"
            or data_scope.get("scene_sensor_model_recheck")
            != "base_and_incremental_lock_frozen_model_only"
            or data_scope.get("base_context_prior") != "base_train_only"
            or data_scope.get("incremental_context_prior")
            != "incremental_train_only"
            or abs(calibrated_threshold - thresholds[class_id]) > 1e-12
            or abs(selected_threshold - thresholds[class_id]) > 1e-12
        ):
            raise ValueError(
                f"增量检测档逐类校准证据无效：{profile_id}:{class_id}"
            )

    metrics_path = resolve_path(profile["metrics_source"])
    if not metrics_path.is_file():
        raise ValueError(f"增量检测档缺少冻结评测证据：{profile_id}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    score_gates = dict(metrics.get("score_gates") or {})
    lock = dict(metrics.get("lock") or {})
    if (
        int(metrics.get("schema_version", 0)) < 5
        or metrics.get("phase") != "joint_evaluation"
        or metrics.get("counted_as_incremental_learning") is not False
        or metrics.get("detector_weights_updated") is not False
        or metrics.get("model_selection_allowed") is not False
        or metrics.get("incremental_learning_data_scope")
        != "incremental_dataset_only"
        or metrics.get("learning_data_scope") is not None
        or metrics.get("incremental_mode") != "class_incremental"
        or metrics.get("old_raw_image_count") != 0
        or metrics.get("predictions_frozen_before_lock_labels") is not True
        or metrics.get("competition_accepted") is not True
        or metrics.get("deployment_accepted") is not True
        or not score_gates
        or not all(score_gates.values())
    ):
        raise ValueError(f"增量检测档冻结评测证据无效：{profile_id}")

    quality = {
        int(key): dict(value)
        for key, value in dict(
            lock.get("new_class_quality") or {}
        ).items()
    }
    false_activation = {
        int(key): dict(value)
        for key, value in dict(
            lock.get("false_activation") or {}
        ).items()
    }
    if set(quality) != {4, 5} or set(false_activation) != {4, 5}:
        raise ValueError(f"增量检测档逐类诊断证据不完整：{profile_id}")

    profile["class_names"] = class_names
    profile["base_local_to_global"] = base_mapping
    profile["specialist_local_to_global"] = specialist_mapping
    profile["new_global_ids"] = new_global_ids
    profile["new_classes"] = {
        class_id: class_names[class_id] for class_id in new_global_ids
    }
    profile["new_class"] = "、".join(
        profile["new_classes"][class_id] for class_id in new_global_ids
    )
    profile["activation_thresholds"] = thresholds
    profile["base_activation_thresholds"] = base_thresholds
    profile["calibration_sources"] = {
        class_id: rel_path(resolve_path(source))
        for class_id, source in calibration_sources.items()
    }
    profile["metrics_source"] = rel_path(metrics_path)
    profile["new_map50"] = float(lock["new_map50"])
    profile["krr"] = float(lock["krr"])
    profile["full_map50"] = float(lock["full_map50"])
    profile["lock_precision_by_class"] = {
        class_id: float(row["precision"])
        for class_id, row in quality.items()
    }
    profile["lock_recall_by_class"] = {
        class_id: float(row["recall"])
        for class_id, row in quality.items()
    }
    profile["lock_false_activation_rate_by_class"] = {
        class_id: float(row["false_activation_rate"])
        for class_id, row in false_activation.items()
    }
    profile["lock_precision"] = min(
        profile["lock_precision_by_class"].values()
    )
    profile["lock_recall"] = min(
        profile["lock_recall_by_class"].values()
    )
    profile["lock_false_activation_rate"] = max(
        profile["lock_false_activation_rate_by_class"].values()
    )
    return profile


def discover_experiment_profiles(root: str | Path | None = None) -> Dict[str, Any]:
    profile_root = resolve_path(root) if root is not None else ROOT / "models" / "profiles"
    profiles: List[Dict[str, Any]] = []
    errors: List[str] = []
    if profile_root.exists():
        for active in sorted(profile_root.glob("*/active.json")):
            profile_id = active.parent.name
            try:
                profiles.append(load_experiment_profile(profile_id, profile_root))
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                errors.append(f"{profile_id}:{exc}")
    score_profiles = [profile for profile in profiles if profile.get("competition_accepted")]
    return {
        "registry": rel_path(profile_root),
        "core_verified_count": len(profiles),
        "verified_count": len(score_profiles),
        "core_class_incremental_verified": bool(profiles),
        "true_class_incremental_verified": bool(score_profiles),
        "profiles": profiles,
        "errors": errors,
    }
