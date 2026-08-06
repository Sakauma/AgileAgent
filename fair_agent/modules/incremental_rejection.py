from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from PIL import Image

from fair_agent.modules.detection_fusion import box_iou


DESCRIPTOR_VERSION = "positive_crop_v1"


def _quantile_higher(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("原型距离集合不能为空")
    ordered = np.asarray(values, dtype=np.float64)
    try:
        return float(np.quantile(ordered, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(ordered, quantile, interpolation="higher"))


def crop_descriptor(
    image: Image.Image,
    xyxy: Iterable[float],
    grid_size: int = 8,
) -> np.ndarray:
    """Build a compact sensor-agnostic descriptor from one candidate crop."""
    if int(grid_size) < 4:
        raise ValueError("原型描述子 grid_size 不能小于4")
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    width, height = image.size
    left = max(0, min(width - 1, int(math.floor(x1))))
    top = max(0, min(height - 1, int(math.floor(y1))))
    right = max(left + 1, min(width, int(math.ceil(x2))))
    bottom = max(top + 1, min(height, int(math.ceil(y2))))
    if right <= left or bottom <= top:
        raise ValueError("候选框裁剪区域为空")

    gray = image.convert("L")
    crop = gray.crop((left, top, right, bottom)).resize(
        (int(grid_size), int(grid_size)),
        Image.Resampling.BILINEAR,
    )
    raw = np.asarray(crop, dtype=np.float32) / 255.0
    intensity_mean = float(raw.mean())
    intensity_std = float(raw.std())
    normalized = (raw - intensity_mean) / max(intensity_std, 0.05)
    box_width = max(1.0, float(right - left))
    box_height = max(1.0, float(bottom - top))
    shape = np.asarray(
        [
            math.log(box_width / box_height),
            math.log(max(1e-8, (box_width * box_height) / float(width * height))),
            intensity_mean,
            intensity_std,
        ],
        dtype=np.float32,
    )
    return np.concatenate((normalized.reshape(-1), shape)).astype(np.float32)


def _image_lookup(image_paths: Iterable[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for raw_path in image_paths:
        path = Path(raw_path)
        if path.stem in lookup and lookup[path.stem].resolve() != path.resolve():
            raise ValueError(f"原型图像 stem 重复：{path.stem}")
        lookup[path.stem] = path
    return lookup


def _row_descriptors(
    image_paths: Iterable[Path],
    rows: Iterable[Mapping[str, Any]],
    class_id: int,
    grid_size: int,
) -> list[np.ndarray]:
    lookup = _image_lookup(image_paths)
    grouped: Dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if int(row["class_id"]) != int(class_id):
            continue
        grouped.setdefault(str(row["image_id"]), []).append(row)
    descriptors: list[np.ndarray] = []
    for image_id, image_rows in grouped.items():
        path = lookup.get(image_id)
        if path is None:
            raise ValueError(f"原型记录缺少对应图像：{image_id}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        descriptors.extend(
            crop_descriptor(image, row["xyxy"], grid_size) for row in image_rows
        )
    return descriptors


def _distance(descriptor: np.ndarray, center: np.ndarray, scale: np.ndarray) -> float:
    normalized = (descriptor.astype(np.float64) - center) / scale
    return float(np.sqrt(np.mean(np.square(np.clip(normalized, -12.0, 12.0)))))


def fit_positive_prototype(
    image_paths: Sequence[Path],
    ground_truth: Sequence[Mapping[str, Any]],
    class_id: int,
    *,
    grid_size: int = 8,
    minimum_scale: float = 0.10,
) -> Dict[str, Any]:
    descriptors = _row_descriptors(image_paths, ground_truth, class_id, grid_size)
    if not descriptors:
        raise ValueError("增量训练集没有可用于原型拟合的新类框")
    matrix = np.stack(descriptors).astype(np.float64)
    center = np.median(matrix, axis=0)
    absolute_deviation = np.median(np.abs(matrix - center), axis=0)
    scale = np.maximum(1.4826 * absolute_deviation, float(minimum_scale))
    distances = [_distance(row, center, scale) for row in matrix]
    return {
        "schema_version": 1,
        "method": DESCRIPTOR_VERSION,
        "class_id": int(class_id),
        "grid_size": int(grid_size),
        "minimum_scale": float(minimum_scale),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "train_positive_count": len(descriptors),
        "train_distance_p95": _quantile_higher(distances, 0.95),
        "train_distance_max": max(distances),
        "distance_threshold": None,
        "calibrated": False,
        "learning_data_scope": "incremental_dataset_only",
    }


def _matched_true_positive_predictions(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    class_id: int,
    iou_threshold: float,
) -> list[Dict[str, Any]]:
    targets_by_image: Dict[str, list[Mapping[str, Any]]] = {}
    for target in ground_truth:
        if int(target["class_id"]) == int(class_id):
            targets_by_image.setdefault(str(target["image_id"]), []).append(target)
    matched: list[Dict[str, Any]] = []
    for image_id, targets in targets_by_image.items():
        candidates = sorted(
            (
                dict(row)
                for row in predictions
                if str(row["image_id"]) == image_id
                and int(row["class_id"]) == int(class_id)
            ),
            key=lambda row: -float(row.get("confidence", 0.0)),
        )
        used: set[int] = set()
        for candidate in candidates:
            overlaps = [
                (box_iou(candidate["xyxy"], target["xyxy"]), index)
                for index, target in enumerate(targets)
                if index not in used
            ]
            if not overlaps:
                continue
            overlap, index = max(overlaps)
            if overlap >= float(iou_threshold):
                used.add(index)
                matched.append(candidate)
    return matched


def calibrate_positive_prototype(
    prototype: Mapping[str, Any],
    image_paths: Sequence[Path],
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    target_recall: float = 0.95,
    safety_factor: float = 1.10,
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    if not 0.0 < float(target_recall) <= 1.0:
        raise ValueError("原型门控 target_recall 必须位于(0, 1]")
    if float(safety_factor) < 1.0:
        raise ValueError("原型门控 safety_factor 不能小于1")
    class_id = int(prototype["class_id"])
    grid_size = int(prototype["grid_size"])
    matched = _matched_true_positive_predictions(
        predictions, ground_truth, class_id, iou_threshold
    )
    calibration_source = "incremental_dev_true_positive_predictions"
    if not matched:
        matched = [
            dict(row) for row in ground_truth if int(row["class_id"]) == class_id
        ]
        calibration_source = "incremental_dev_ground_truth_fallback"
    descriptors = _row_descriptors(image_paths, matched, class_id, grid_size)
    if not descriptors:
        raise ValueError("增量验证集没有可用于原型校准的新类框")
    center = np.asarray(prototype["center"], dtype=np.float64)
    scale = np.asarray(prototype["scale"], dtype=np.float64)
    distances = [_distance(row, center, scale) for row in descriptors]
    dev_limit = _quantile_higher(distances, float(target_recall))
    train_limit = float(prototype["train_distance_p95"])
    threshold = max(dev_limit, train_limit) * float(safety_factor)
    return {
        **dict(prototype),
        "distance_threshold": float(threshold),
        "calibrated": True,
        "calibration_source": calibration_source,
        "dev_positive_count": len(descriptors),
        "dev_distance_quantile": float(dev_limit),
        "target_recall": float(target_recall),
        "safety_factor": float(safety_factor),
        "iou_threshold": float(iou_threshold),
    }


def apply_positive_prototype_to_image(
    image: Image.Image,
    predictions: Iterable[Mapping[str, Any]],
    prototype: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not prototype.get("calibrated") or prototype.get("distance_threshold") is None:
        raise ValueError("新类正样本原型尚未校准")
    class_id = int(prototype["class_id"])
    grid_size = int(prototype["grid_size"])
    center = np.asarray(prototype["center"], dtype=np.float64)
    scale = np.asarray(prototype["scale"], dtype=np.float64)
    threshold = float(prototype["distance_threshold"])
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for raw_row in predictions:
        row = dict(raw_row)
        if int(row["class_id"]) != class_id:
            kept.append(row)
            continue
        descriptor = crop_descriptor(image, row["xyxy"], grid_size)
        distance = _distance(descriptor, center, scale)
        enriched = {**row, "prototype_distance": round(distance, 6)}
        if distance <= threshold:
            kept.append(enriched)
        else:
            rejected.append(
                {
                    **enriched,
                    "action": "reject_incremental_prototype",
                    "distance_threshold": round(threshold, 6),
                    "reason": "outside_incremental_positive_prototype",
                }
            )
    return kept, rejected


def apply_positive_prototype(
    predictions: Sequence[Mapping[str, Any]],
    image_paths: Sequence[Path],
    prototype: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lookup = _image_lookup(image_paths)
    grouped: Dict[str, list[Mapping[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(str(row["image_id"]), []).append(row)
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for image_id, rows in grouped.items():
        path = lookup.get(image_id)
        if path is None:
            raise ValueError(f"原型推理记录缺少对应图像：{image_id}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        image_kept, image_rejected = apply_positive_prototype_to_image(
            image, rows, prototype
        )
        kept.extend(image_kept)
        rejected.extend(image_rejected)
    return kept, rejected
