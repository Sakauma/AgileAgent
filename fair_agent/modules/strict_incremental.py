from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

from fair_agent.core.config import ROOT, rel_path, resolve_path


GLOBAL_CLASS_NAMES = {0: "soldier", 1: "small_aircraft", 2: "warship", 3: "tank"}


def load_yaml(path: str | Path) -> Dict[str, Any]:
    resolved = resolve_path(path)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是映射：{resolved}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def _write_subset(
    images: Sequence[Path],
    target_root: Path,
    global_to_local: Mapping[int, int],
) -> List[Path]:
    generated = []
    for source in images:
        target_image = target_root / "images" / source.name
        target_label = target_root / "labels" / f"{source.stem}.txt"
        _link_or_copy(source, target_image)
        output_labels = []
        for class_id, x, y, width, height in read_yolo_labels(source_label(source)):
            if class_id in global_to_local:
                output_labels.append(
                    f"{global_to_local[class_id]} {x:.10g} {y:.10g} {width:.10g} {height:.10g}"
                )
        target_label.parent.mkdir(parents=True, exist_ok=True)
        target_label.write_text("\n".join(output_labels) + ("\n" if output_labels else ""), encoding="utf-8")
        generated.append(target_image)
    return generated


def _write_split(path: Path, images: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the generated view path. Resolving a symlink would make Ultralytics
    # pair the source image with the original four-class label.
    path.write_text("\n".join(str(image.absolute()) for image in images) + "\n", encoding="utf-8")


def _write_dataset_yaml(path: Path, split_paths: Mapping[str, Path], names: Mapping[int, str]) -> None:
    data = {
        "path": str(path.parent.resolve()),
        "train": str(split_paths["train"].resolve()),
        "val": str(split_paths["val"].resolve()),
        "test": str(split_paths["test"].resolve()),
        "names": dict(names),
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def validate_protocol_spec(protocol: Mapping[str, Any]) -> None:
    new_id = int(protocol["new_global_id"])
    new_name = str(protocol["new_class"])
    mapping = {int(key): int(value) for key, value in protocol["base_local_to_global"].items()}
    if GLOBAL_CLASS_NAMES.get(new_id) != new_name:
        raise ValueError(f"新增类别映射错误：{protocol['id']}")
    if set(mapping) != {0, 1, 2} or new_id in set(mapping.values()):
        raise ValueError(f"基础类别映射必须是连续三类且与新增类互斥：{protocol['id']}")
    expected = {GLOBAL_CLASS_NAMES[value] for value in mapping.values()}
    if expected != set(protocol["base_classes"]):
        raise ValueError(f"基础类别名称与映射不一致：{protocol['id']}")


def build_protocol_dataset(
    protocol: Mapping[str, Any],
    source_splits: Mapping[str, str | Path],
    output_dir: Path,
    include_lock: bool = False,
) -> Dict[str, Any]:
    validate_protocol_spec(protocol)
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖严格增量数据集：{output_dir}")
    source_rows = {name: read_split(path) for name, path in source_splits.items()}
    if set(source_rows) != {"train", "val", "lock"}:
        raise ValueError("source_splits 必须包含 train、val、lock")
    source_split_stems = {
        name: [image.stem for image in images] for name, images in source_rows.items()
    }
    source_split_intersections = {
        "train_val": sorted(set(source_split_stems["train"]) & set(source_split_stems["val"])),
        "train_lock": sorted(set(source_split_stems["train"]) & set(source_split_stems["lock"])),
        "val_lock": sorted(set(source_split_stems["val"]) & set(source_split_stems["lock"])),
    }
    if any(source_split_intersections.values()):
        raise RuntimeError(f"源划分存在重复 stem：{source_split_intersections}")
    new_id = int(protocol["new_global_id"])
    local_to_global = {int(key): int(value) for key, value in protocol["base_local_to_global"].items()}
    global_to_local = {global_id: local_id for local_id, global_id in local_to_global.items()}
    base_names = {local_id: GLOBAL_CLASS_NAMES[global_id] for local_id, global_id in local_to_global.items()}
    new_names = {0: str(protocol["new_class"])}

    unified_student = bool(protocol.get("build_unified_student", False))
    selected: Dict[str, Dict[str, List[Path]]] = {"base": {}, "incremental": {}}
    if unified_student:
        selected["student"] = {}
    for split_name, rows in source_rows.items():
        if split_name == "lock":
            selected["base"]["test"] = list(rows) if include_lock else []
            selected["incremental"]["test"] = list(rows) if include_lock else []
            if unified_student:
                selected["student"]["test"] = list(rows) if include_lock else []
        else:
            image_classes = {image: image_class_ids(image) for image in rows}
            invalid_ids = {
                image.name: sorted(classes - set(GLOBAL_CLASS_NAMES))
                for image, classes in image_classes.items()
                if not classes <= set(GLOBAL_CLASS_NAMES)
            }
            if invalid_ids:
                raise ValueError(f"严格增量源标签包含未知类别：{invalid_ids}")
            cooccurrence = {
                image.name: sorted(classes)
                for image, classes in image_classes.items()
                if new_id in classes and classes != {new_id}
            }
            if cooccurrence:
                raise ValueError(f"新增类训练/验证图像存在旧类共现：{cooccurrence}")
            contains_new = {image: new_id in image_classes[image] for image in rows}
            target = "train" if split_name == "train" else "val"
            selected["base"][target] = [image for image in rows if not contains_new[image]]
            selected["incremental"][target] = [image for image in rows if contains_new[image]]
            if unified_student:
                selected["student"][target] = list(selected["incremental"][target])

    expected = protocol.get("expected_incremental_counts", {})
    for split_name in ("train", "val"):
        if split_name in expected and len(selected["incremental"][split_name]) != int(expected[split_name]):
            raise ValueError(
                f"{protocol['id']} 新增类 {split_name} 样本数不符："
                f"expected={expected[split_name]} actual={len(selected['incremental'][split_name])}"
            )

    output_dir.mkdir(parents=True)
    generated: Dict[str, Dict[str, List[Path]]] = {phase: {} for phase in selected}
    phase_specs: List[tuple[str, Mapping[int, int], Mapping[int, str]]] = [
        ("base", global_to_local, base_names),
        ("incremental", {new_id: 0}, new_names),
    ]
    if unified_student:
        phase_specs.append(("student", {new_id: new_id}, GLOBAL_CLASS_NAMES))
    for phase, mapping, names in phase_specs:
        split_files = {}
        for split_name in ("train", "val", "test"):
            images = _write_subset(selected[phase][split_name], output_dir / phase / split_name, mapping)
            generated[phase][split_name] = images
            split_file = output_dir / phase / "splits" / f"{split_name}.txt"
            _write_split(split_file, images)
            split_files[split_name] = split_file
        _write_dataset_yaml(output_dir / phase / "dataset.yaml", split_files, names)

    base_train_stems = {path.stem for path in selected["base"]["train"]}
    incremental_train_stems = {path.stem for path in selected["incremental"]["train"]}
    base_val_stems = {path.stem for path in selected["base"]["val"]}
    incremental_val_stems = {path.stem for path in selected["incremental"]["val"]}
    intersections = {
        "base_incremental_train": sorted(base_train_stems & incremental_train_stems),
        "base_incremental_val": sorted(base_val_stems & incremental_val_stems),
        "incremental_train_val": sorted(incremental_train_stems & incremental_val_stems),
    }
    if any(intersections.values()):
        raise RuntimeError(f"严格增量集合存在交集：{intersections}")

    source_hashes = {
        name: sha256_file(resolve_path(path)) for name, path in source_splits.items()
    }
    manifest = {
        "schema_version": 1,
        "protocol": protocol["id"],
        "incremental_mode": "class_incremental",
        "learning_data_scope": "incremental_dataset_only",
        "base_classes": list(protocol["base_classes"]),
        "new_class": protocol["new_class"],
        "new_global_id": new_id,
        "base_local_to_global": local_to_global,
        "specialist_local_to_global": {0: new_id},
        "source_split_sha256": source_hashes,
        "source_split_stems": source_split_stems,
        "source_split_intersections": source_split_intersections,
        "counts": {
            phase: {split: len(rows) for split, rows in split_rows.items()}
            for phase, split_rows in selected.items()
        },
        "source_stems": {
            phase: {split: [path.stem for path in rows] for split, rows in split_rows.items()}
            for phase, split_rows in selected.items()
        },
        "intersections": intersections,
        "base_nc": 3,
        "specialist_nc": 1,
        "student_nc": 4 if unified_student else None,
        "unified_student_enabled": unified_student,
        "old_raw_image_count": 0,
        "original_data_modified": False,
        "lock_materialized_after_freeze": include_lock,
        "base_dataset": rel_path(output_dir / "base" / "dataset.yaml"),
        "incremental_dataset": rel_path(output_dir / "incremental" / "dataset.yaml"),
        "student_dataset": rel_path(output_dir / "student" / "dataset.yaml") if unified_student else None,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def materialize_lock_data(
    protocol: Mapping[str, Any],
    source_lock: str | Path,
    output_dir: Path,
) -> Dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("lock_materialized_after_freeze"):
        raise FileExistsError(f"lock-val 已经物化：{output_dir}")
    lock_path = resolve_path(source_lock)
    if sha256_file(lock_path) != manifest.get("source_split_sha256", {}).get("lock"):
        raise ValueError("lock-val 划分在训练冻结后发生变化")
    lock_images = read_split(source_lock)
    if [image.stem for image in lock_images] != manifest.get("source_split_stems", {}).get("lock"):
        raise ValueError("lock-val 图像清单在训练冻结后发生变化")
    expected_lock = protocol.get("expected_incremental_counts", {}).get("lock_positive")
    if expected_lock is not None:
        actual_lock = sum(int(protocol["new_global_id"]) in image_class_ids(image) for image in lock_images)
        if actual_lock != int(expected_lock):
            raise ValueError(
                f"{protocol['id']} 新增类 lock-val 样本数不符：expected={expected_lock} actual={actual_lock}"
            )
    local_to_global = {int(key): int(value) for key, value in protocol["base_local_to_global"].items()}
    global_to_local = {global_id: local_id for local_id, global_id in local_to_global.items()}
    new_id = int(protocol["new_global_id"])
    phases: List[tuple[str, Mapping[int, int]]] = [
        ("base", global_to_local),
        ("incremental", {new_id: 0}),
    ]
    if manifest.get("unified_student_enabled"):
        phases.append(("student", {class_id: class_id for class_id in GLOBAL_CLASS_NAMES}))
    for phase, mapping in phases:
        target_root = output_dir / phase / "test"
        if any(target_root.rglob("*")):
            raise FileExistsError(f"拒绝覆盖 lock-val 视图：{target_root}")
        images = _write_subset(lock_images, target_root, mapping)
        _write_split(output_dir / phase / "splits" / "test.txt", images)
        manifest["counts"][phase]["test"] = len(images)
        manifest["source_stems"][phase]["test"] = [path.stem for path in lock_images]
    manifest["lock_materialized_after_freeze"] = True
    manifest["lock_materialized_weight_freeze_required"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


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
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


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


def subset_rows(rows: Sequence[Mapping[str, Any]], image_ids: Iterable[str]) -> List[Dict[str, Any]]:
    allowed = set(image_ids)
    return [dict(row) for row in rows if str(row["image_id"]) in allowed]


def bootstrap_metrics(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    image_paths: Sequence[Path],
    new_class_id: int,
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    presence = defaultdict(set)
    for row in ground_truth:
        presence[str(row["image_id"])].add(int(row["class_id"]))
    strata: Dict[tuple[str, bool], List[str]] = defaultdict(list)
    for image in image_paths:
        image_id = image.stem
        sensor = image.stem.split("_", 1)[0]
        strata[(sensor, new_class_id in presence[image_id])].append(image_id)
    rng = random.Random(seed)
    full_values = []
    new_values = []
    predictions_by_image = defaultdict(list)
    gt_by_image = defaultdict(list)
    for row in predictions:
        predictions_by_image[str(row["image_id"])].append(row)
    for row in ground_truth:
        gt_by_image[str(row["image_id"])].append(row)
    for iteration in range(iterations):
        sampled_predictions = []
        sampled_gt = []
        sample_index = 0
        for values in strata.values():
            for original_id in rng.choices(values, k=len(values)):
                boot_id = f"boot:{iteration}:{sample_index}:{original_id}"
                sample_index += 1
                sampled_predictions.extend({**row, "image_id": boot_id} for row in predictions_by_image[original_id])
                sampled_gt.extend({**row, "image_id": boot_id} for row in gt_by_image[original_id])
        full_values.append(evaluate_ap50(sampled_predictions, sampled_gt, GLOBAL_CLASS_NAMES)["map50"])
        new_values.append(evaluate_ap50(sampled_predictions, sampled_gt, [new_class_id])["map50"])

    def summarize(values: Sequence[float]) -> Dict[str, float]:
        return {
            "median": float(np.median(values)),
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
        }

    return {"iterations": iterations, "full_map50": summarize(full_values), "new_map50": summarize(new_values)}


def load_experiment_profile(profile_id: str, profile_root: str | Path | None = None) -> Dict[str, Any]:
    root = resolve_path(profile_root) if profile_root is not None else ROOT / "models" / "experiments" / "strict_3plus1"
    active = root / profile_id / "active.json"
    if not active.exists():
        raise FileNotFoundError(f"严格增量实验档尚未通过验收：{profile_id}")
    profile = json.loads(active.read_text(encoding="utf-8"))
    if (
        profile.get("profile_id") != profile_id
        or profile.get("acceptance") != "passed"
        or profile.get("incremental_mode") != "class_incremental"
        or profile.get("evidence_level") != "verified"
    ):
        raise ValueError(f"严格增量实验档无效：{profile_id}")
    try:
        threshold = float(profile["activation_threshold"])
        new_global_id = int(profile["new_global_id"])
        mapping = {int(key): int(value) for key, value in profile["base_local_to_global"].items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"严格增量实验档字段无效：{profile_id}") from exc
    if not 0.01 <= threshold <= 1.0 or set(mapping) != {0, 1, 2} or new_global_id in mapping.values():
        raise ValueError(f"严格增量实验档类别映射或阈值无效：{profile_id}")
    for path_key, hash_key in (("base_weight", "base_sha256"), ("specialist_weight", "specialist_sha256")):
        weight = resolve_path(profile[path_key])
        if not weight.exists() or sha256_file(weight) != profile[hash_key]:
            raise ValueError(f"严格增量实验档权重校验失败：{profile_id}:{path_key}")
    calibration = resolve_path(profile.get("calibration_source", "__missing_calibration__"))
    if not calibration.exists():
        raise ValueError(f"严格增量实验档缺少校准证据：{profile_id}")
    metrics_path = resolve_path(
        profile.get("metrics_source")
        or (Path(profile["base_weight"]).parent / "metrics.json")
    )
    if not metrics_path.exists():
        raise ValueError(f"严格增量实验档缺少评测证据：{profile_id}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        metrics.get("accepted") is not True
        or metrics.get("incremental_mode") != "class_incremental"
        or metrics.get("learning_data_scope") != "incremental_dataset_only"
        or metrics.get("old_raw_image_count") != 0
        or not all(metrics.get("gates", {}).values())
    ):
        raise ValueError(f"严格增量实验档合规证据无效：{profile_id}")
    profile["metrics_source"] = rel_path(metrics_path)
    profile["lock_precision"] = metrics.get("lock_deployment_metrics", {}).get("precision")
    profile["lock_recall"] = metrics.get("lock_deployment_metrics", {}).get("recall")
    profile["lock_false_activation_rate"] = metrics.get("false_activation", {}).get("false_activation_rate")
    return profile


def discover_experiment_profiles(root: str | Path | None = None) -> Dict[str, Any]:
    profile_root = resolve_path(root) if root is not None else ROOT / "models" / "experiments" / "strict_3plus1"
    profiles: List[Dict[str, Any]] = []
    errors: List[str] = []
    if profile_root.exists():
        for active in sorted(profile_root.glob("*/active.json")):
            profile_id = active.parent.name
            try:
                profiles.append(load_experiment_profile(profile_id, profile_root))
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                errors.append(f"{profile_id}:{exc}")
    return {
        "registry": rel_path(profile_root),
        "verified_count": len(profiles),
        "true_class_incremental_verified": bool(profiles),
        "profiles": profiles,
        "errors": errors,
    }
