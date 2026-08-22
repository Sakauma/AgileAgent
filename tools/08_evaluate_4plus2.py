#!/usr/bin/env python3
"""Freeze and score one registry-defined cumulative incremental round."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.incremental_round_registry import (  # noqa: E402
    DEFAULT_ROUND_REGISTRY,
    load_incremental_round_registry,
    rounds_through,
    select_round,
)


def parse_specialist_weight(value: str) -> tuple[str, Path]:
    round_id, separator, path = value.partition("=")
    if not separator or not round_id.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "--specialist-weight 必须使用 ROUND_ID=/path/to/best.pt"
        )
    return round_id.strip(), Path(path.strip())


def resolve_split(
    data_root: Path, split_reference: str, *, require_labels: bool = True
) -> list[Path]:
    split_path = Path(split_reference)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    if not split_path.is_file():
        raise FileNotFoundError(f"划分不存在：{split_path}")
    images: list[Path] = []
    for line_number, raw in enumerate(
        split_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw.strip()
        if not value:
            continue
        image = Path(value)
        if not image.is_absolute():
            image = data_root / image
        image = image.resolve()
        try:
            image.relative_to(data_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"{split_path}:{line_number} 越出数据根目录：{image}"
            ) from exc
        if not image.is_file() or (
            require_labels and not image.with_suffix(".txt").is_file()
        ):
            raise FileNotFoundError(
                f"{split_path}:{line_number} 图像或标签不存在：{image}"
            )
        images.append(image)
    if not images or len(images) != len(set(images)):
        raise ValueError(f"划分为空或包含重复图像：{split_path}")
    return images


def ensure_cumulative_contract(groups: Mapping[str, Sequence[Path]]) -> list[Path]:
    images = [path for rows in groups.values() for path in rows]
    paths = [path.resolve() for path in images]
    stems = [path.stem for path in paths]
    if len(paths) != len(set(paths)) or len(stems) != len(set(stems)):
        raise ValueError("截至当前轮的累计清单包含重复图像或重复 stem")
    return paths


def predict_records(
    weight: Path,
    images: Sequence[Path],
    local_to_global: Mapping[int, int],
    *,
    device: str,
    imgsz: int,
    batch: int,
    confidence: float,
    source_name: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    from ultralytics import YOLO

    model = YOLO(str(weight))
    results = model.predict(
        source=[str(path) for path in images],
        device=device,
        imgsz=imgsz,
        batch=batch,
        conf=confidence,
        iou=0.70,
        max_det=300,
        rect=True,
        augment=False,
        verbose=False,
    )
    if len(results) != len(images):
        raise RuntimeError(
            f"预测数量不一致：expected={len(images)} actual={len(results)}"
        )
    expected_ids = [path.stem for path in images]
    result_ids = [Path(str(result.path)).stem for result in results]
    if Counter(expected_ids) != Counter(result_ids):
        raise RuntimeError("预测结果与输入图像 stem 不一致")
    mapping = {int(key): int(value) for key, value in local_to_global.items()}
    records: list[dict[str, Any]] = []
    speed_totals: dict[str, float] = {}
    for result, image_id in zip(results, result_ids):
        for key, value in (getattr(result, "speed", None) or {}).items():
            speed_totals[key] = speed_totals.get(key, 0.0) + float(value)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for xyxy, confidence_value, local_value in zip(
            boxes.xyxy.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
            boxes.cls.detach().cpu().tolist(),
        ):
            local_id = int(local_value)
            if local_id not in mapping:
                raise RuntimeError(
                    f"{source_name} 输出未登记的局部类别：{local_id}"
                )
            records.append(
                {
                    "image_id": image_id,
                    "class_id": mapping[local_id],
                    "confidence": float(confidence_value),
                    "xyxy": [float(value) for value in xyxy],
                    "source": source_name,
                }
            )
    speed = {key: value / len(results) for key, value in speed_totals.items()}
    return records, speed


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖冻结预测：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def subset(
    rows: Sequence[Mapping[str, Any]], image_ids: Iterable[str]
) -> list[dict[str, Any]]:
    allowed = set(image_ids)
    return [dict(row) for row in rows if str(row["image_id"]) in allowed]


def apply_thresholds(
    rows: Sequence[Mapping[str, Any]], thresholds: Mapping[int, float]
) -> list[dict[str, Any]]:
    normalized = {int(key): float(value) for key, value in thresholds.items()}
    return [
        dict(row)
        for row in rows
        if float(row["confidence"])
        >= normalized.get(int(row["class_id"]), 0.0)
    ]


def inherited_thresholds(
    path: Path | None,
    required_ids: Sequence[int],
    expected_parent_round: Mapping[str, Any] | None,
) -> tuple[dict[int, float], str | None]:
    if not required_ids:
        if path is not None or expected_parent_round is not None:
            raise ValueError("首轮不应提供父代校准文件")
        return {}, None
    if path is None or expected_parent_round is None:
        raise ValueError("第二轮及以后必须通过 --parent-calibration 冻结旧类阈值")
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    thresholds = {
        int(key): float(value)
        for key, value in dict(payload.get("per_class_thresholds") or {}).items()
    }
    if (
        payload.get("phase") != "system_calibration"
        or payload.get("counted_as_incremental_learning") is not False
        or payload.get("detector_weights_updated") is not False
        or payload.get("round_id") != expected_parent_round["round_id"]
        or payload.get("round_index") != expected_parent_round["round_index"]
        or payload.get("generation_id") != expected_parent_round["generation_id"]
        or set(thresholds) != set(required_ids)
        or any(not 0.01 <= value <= 1.0 for value in thresholds.values())
    ):
        raise ValueError("父代校准文件与上一轮冻结代际不一致")
    return (
        {class_id: thresholds[class_id] for class_id in required_ids},
        resolved.as_posix(),
    )


def calibrate_current_thresholds(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    class_names: Mapping[int, str],
    current_class_ids: Sequence[int],
    inherited: Mapping[int, float],
    minimum: float,
    maximum: float,
    step: float,
    round_spec: Mapping[str, Any],
    parent_calibration: str | None,
) -> dict[str, Any]:
    from fair_agent.modules.strict_incremental import evaluate_ap50, precision_recall

    count = int(round((maximum - minimum) / step)) + 1
    thresholds = {int(key): float(value) for key, value in inherited.items()}
    per_class: dict[str, Any] = {}
    for class_id in current_class_ids:
        curve = []
        for index in range(count):
            threshold = round(minimum + index * step, 6)
            quality = precision_recall(
                predictions, ground_truth, class_id, threshold
            )
            filtered = apply_thresholds(predictions, {class_id: threshold})
            ap50 = float(
                evaluate_ap50(filtered, ground_truth, [class_id])["map50"]
            )
            curve.append({"threshold": threshold, "map50": ap50, **quality})
        selected = max(
            curve,
            key=lambda row: (
                float(row["map50"]),
                float(row["recall"]),
                float(row["f1"]),
                -float(row["threshold"]),
            ),
        )
        thresholds[class_id] = float(selected["threshold"])
        per_class[str(class_id)] = {
            "class_name": class_names[class_id],
            "selected": selected,
            "curve": curve,
        }
    selected_current = apply_thresholds(predictions, thresholds)
    return {
        "schema_version": 5,
        "phase": "system_calibration",
        "counted_as_incremental_learning": False,
        "detector_weights_updated": False,
        "round_id": round_spec["round_id"],
        "round_index": round_spec["round_index"],
        "parent_generation_id": round_spec["parent_generation_id"],
        "generation_id": round_spec["generation_id"],
        "source_split": "cumulative_dev_only",
        "selection_metric": "current_round_per_class_mAP50",
        "parent_calibration": parent_calibration,
        "old_incremental_thresholds_frozen": True,
        "per_class_thresholds": {
            str(key): value for key, value in thresholds.items()
        },
        "per_class": per_class,
        "selected_new_map50": float(
            evaluate_ap50(
                selected_current, ground_truth, current_class_ids
            )["map50"]
        ),
    }


def false_activation_metrics(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    images: Sequence[Path],
    thresholds: Mapping[int, float],
    class_ids: Sequence[int],
    class_names: Mapping[int, str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for class_id in class_ids:
        positive = {
            str(row["image_id"])
            for row in ground_truth
            if int(row["class_id"]) == class_id
        }
        negative = {path.stem for path in images} - positive
        activated = {
            str(row["image_id"])
            for row in predictions
            if int(row["class_id"]) == class_id
            and float(row["confidence"]) >= float(thresholds[class_id])
        }
        false_images = negative & activated
        output[str(class_id)] = {
            "class_name": class_names[class_id],
            "negative_image_count": len(negative),
            "false_activation_image_count": len(false_images),
            "false_activation_rate": (
                len(false_images) / len(negative) if negative else 0.0
            ),
        }
    return output


def score_round(
    base_predictions: Sequence[Mapping[str, Any]],
    previous_predictions: Sequence[Mapping[str, Any]],
    current_predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    base_images: Sequence[Path],
    cumulative_images: Sequence[Path],
    thresholds: Mapping[int, float],
    base_class_ids: Sequence[int],
    old_class_ids: Sequence[int],
    current_class_ids: Sequence[int],
    learned_class_ids: Sequence[int],
    class_names: Mapping[int, str],
) -> dict[str, Any]:
    from fair_agent.modules.strict_incremental import (
        evaluate_ap50,
        precision_recall,
        retention_metrics,
    )

    selected_previous = apply_thresholds(previous_predictions, thresholds)
    selected_current = apply_thresholds(current_predictions, thresholds)
    parent_predictions = [*base_predictions, *selected_previous]
    child_predictions = [*parent_predictions, *selected_current]
    base_ids = {path.stem for path in base_images}
    base_metrics = evaluate_ap50(
        subset(base_predictions, base_ids),
        subset(ground_truth, base_ids),
        base_class_ids,
    )
    retention = retention_metrics(
        parent_predictions,
        child_predictions,
        ground_truth,
        old_class_ids,
    )
    new_metrics = evaluate_ap50(
        child_predictions, ground_truth, current_class_ids
    )
    full_metrics = evaluate_ap50(
        child_predictions, ground_truth, learned_class_ids
    )
    per_class_quality = {
        str(class_id): {
            "class_name": class_names[class_id],
            "map50": float(new_metrics["per_class_ap50"].get(class_id, 0.0)),
            **precision_recall(
                current_predictions,
                ground_truth,
                class_id,
                float(thresholds[class_id]),
            ),
            "threshold": float(thresholds[class_id]),
        }
        for class_id in current_class_ids
    }
    return {
        "image_count": len(cumulative_images),
        "base_image_count": len(base_images),
        "base_map50": float(base_metrics["map50"]),
        "base_per_class_ap50": {
            str(key): value
            for key, value in base_metrics["per_class_ap50"].items()
        },
        "old_map50_before": float(retention["old_map50_before"]),
        "old_map50_after": float(retention["old_map50_after"]),
        "krr": float(retention["krr"]),
        "old_prediction_equivalent": bool(
            retention["old_prediction_equivalent"]
        ),
        "new_map50": float(new_metrics["map50"]),
        "new_per_class_ap50": {
            str(key): value
            for key, value in new_metrics["per_class_ap50"].items()
        },
        "full_map50": float(full_metrics["map50"]),
        "full_per_class_ap50": {
            str(key): value
            for key, value in full_metrics["per_class_ap50"].items()
        },
        "new_class_quality": per_class_quality,
        "false_activation": false_activation_metrics(
            current_predictions,
            ground_truth,
            cumulative_images,
            thresholds,
            current_class_ids,
            class_names,
        ),
        "prediction_counts": {
            "base": len(base_predictions),
            "previous_specialists_before_threshold": len(previous_predictions),
            "previous_specialists_after_threshold": len(selected_previous),
            "current_specialist_before_threshold": len(current_predictions),
            "current_specialist_after_threshold": len(selected_current),
            "parent_fused": len(parent_predictions),
            "child_fused": len(child_predictions),
        },
    }


def freeze_split_predictions(
    split_role: str,
    images: Sequence[Path],
    base_weight: Path,
    specialist_weights: Mapping[str, Path],
    active_rounds: Sequence[Mapping[str, Any]],
    base_mapping: Mapping[int, int],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    base_predictions, base_speed = predict_records(
        base_weight,
        images,
        base_mapping,
        device=args.device,
        imgsz=args.base_imgsz,
        batch=args.batch,
        confidence=args.prediction_confidence,
        source_name="frozen_base_model",
    )
    write_jsonl(
        output_dir / "frozen" / f"base_{split_role}_predictions.jsonl",
        base_predictions,
    )
    specialists: dict[str, list[dict[str, Any]]] = {}
    speeds: dict[str, Any] = {"base": base_speed, "specialists": {}}
    aggregate: list[dict[str, Any]] = []
    for round_spec in active_rounds:
        round_id = str(round_spec["round_id"])
        rows, speed = predict_records(
            specialist_weights[round_id],
            images,
            round_spec["specialist"]["local_to_global"],
            device=args.device,
            imgsz=args.specialist_imgsz,
            batch=args.batch,
            confidence=args.prediction_confidence,
            source_name=str(round_spec["specialist"]["model_id"]),
        )
        specialists[round_id] = rows
        aggregate.extend(rows)
        speeds["specialists"][round_id] = speed
        write_jsonl(
            output_dir
            / "frozen"
            / f"{round_id}_{split_role}_predictions.jsonl",
            rows,
        )
    write_jsonl(
        output_dir / "frozen" / f"specialist_{split_role}_predictions.jsonl",
        aggregate,
    )
    return base_predictions, specialists, speeds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按轮次/类别注册表冻结并复核累计类别增量模型。"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--round-registry", type=Path, default=ROOT / DEFAULT_ROUND_REGISTRY
    )
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--base-weight", type=Path, required=True)
    parser.add_argument(
        "--specialist-weight",
        type=parse_specialist_weight,
        action="append",
        required=True,
        help="可重复：ROUND_ID=/path/to/best.pt",
    )
    parser.add_argument("--parent-calibration", type=Path)
    parser.add_argument(
        "--scene-weight",
        type=Path,
        default=ROOT / "models" / "context" / "scene_sensor_net.pt",
    )
    parser.add_argument("--skip-context-check", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--base-imgsz", type=int, default=1280)
    parser.add_argument("--specialist-imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=18)
    parser.add_argument("--prediction-confidence", type=float, default=0.01)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    args = parser.parse_args()

    if not 0.0 < args.prediction_confidence <= 1.0:
        raise ValueError("--prediction-confidence 必须位于 (0, 1]")
    if (
        not 0.01 <= args.threshold_min <= args.threshold_max <= 1.0
        or args.threshold_step <= 0.0
    ):
        raise ValueError("阈值搜索范围必须满足 0.01 <= min <= max <= 1 且 step > 0")

    data_root = args.data_root.expanduser().resolve()
    base_weight = args.base_weight.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    registry = load_incremental_round_registry(args.round_registry)
    target_round = select_round(registry, args.round_id)
    active_rounds = rounds_through(registry, args.round_id)
    if len(args.specialist_weight) != len(
        {round_id for round_id, _path in args.specialist_weight}
    ):
        raise ValueError("--specialist-weight 不能重复提供同一轮次")
    raw_specialist_weights = dict(args.specialist_weight)
    expected_round_ids = {str(row["round_id"]) for row in active_rounds}
    if set(raw_specialist_weights) != expected_round_ids:
        raise ValueError(
            "必须且只能提供截至当前轮的专家权重："
            f"expected={sorted(expected_round_ids)} "
            f"actual={sorted(raw_specialist_weights)}"
        )
    specialist_weights = {
        round_id: path.expanduser().resolve()
        for round_id, path in raw_specialist_weights.items()
    }
    for path in (base_weight, *specialist_weights.values()):
        if not path.is_file():
            raise FileNotFoundError(f"评测权重不存在：{path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有轮次评测目录：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dev = resolve_split(data_root, registry["base"]["splits"]["dev"])
    base_lock = resolve_split(
        data_root, registry["base"]["splits"]["lock"], require_labels=False
    )
    dev_groups = {"base": base_dev}
    lock_groups = {"base": base_lock}
    for round_spec in active_rounds:
        round_id = str(round_spec["round_id"])
        dev_groups[round_id] = resolve_split(
            data_root, round_spec["splits"]["dev"]
        )
        lock_groups[round_id] = resolve_split(
            data_root, round_spec["splits"]["lock"], require_labels=False
        )
    cumulative_dev = ensure_cumulative_contract(dev_groups)
    cumulative_lock = ensure_cumulative_contract(lock_groups)

    base_mapping = registry["base"]["local_to_global"]
    base_dev_predictions, specialist_dev, dev_speed = freeze_split_predictions(
        "dev",
        cumulative_dev,
        base_weight,
        specialist_weights,
        active_rounds,
        base_mapping,
        output_dir,
        args,
    )
    from fair_agent.modules.strict_incremental import yolo_ground_truth

    class_names = {
        class_id: registry["class_names"][class_id]
        for class_id in target_round["learned_class_ids"]
    }
    dev_ground_truth = yolo_ground_truth(cumulative_dev, class_names)
    previous_rounds = active_rounds[:-1]
    previous_incremental_ids = [
        class_id
        for row in previous_rounds
        for class_id in row["new_class_ids"]
    ]
    inherited, parent_calibration = inherited_thresholds(
        args.parent_calibration,
        previous_incremental_ids,
        previous_rounds[-1] if previous_rounds else None,
    )
    all_dev_specialists = [
        prediction
        for row in active_rounds
        for prediction in specialist_dev[str(row["round_id"])]
    ]
    calibration = calibrate_current_thresholds(
        all_dev_specialists,
        dev_ground_truth,
        class_names,
        target_round["new_class_ids"],
        inherited,
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
        target_round,
        parent_calibration,
    )
    thresholds = {
        int(key): float(value)
        for key, value in calibration["per_class_thresholds"].items()
    }
    previous_dev_predictions = [
        prediction
        for row in previous_rounds
        for prediction in specialist_dev[str(row["round_id"])]
    ]
    current_dev_predictions = specialist_dev[str(target_round["round_id"])]
    dev_metrics = score_round(
        base_dev_predictions,
        previous_dev_predictions,
        current_dev_predictions,
        dev_ground_truth,
        base_dev,
        cumulative_dev,
        thresholds,
        registry["base"]["class_ids"],
        target_round["old_class_ids"],
        target_round["new_class_ids"],
        target_round["learned_class_ids"],
        class_names,
    )
    atomic_json(output_dir / "calibration.json", calibration)

    # All detector predictions are persisted before any cumulative lock label is read.
    base_lock_predictions, specialist_lock, lock_speed = freeze_split_predictions(
        "lock",
        cumulative_lock,
        base_weight,
        specialist_weights,
        active_rounds,
        base_mapping,
        output_dir,
        args,
    )
    lock_ground_truth = yolo_ground_truth(cumulative_lock, class_names)
    previous_lock_predictions = [
        prediction
        for row in previous_rounds
        for prediction in specialist_lock[str(row["round_id"])]
    ]
    current_lock_predictions = specialist_lock[str(target_round["round_id"])]
    lock_metrics = score_round(
        base_lock_predictions,
        previous_lock_predictions,
        current_lock_predictions,
        lock_ground_truth,
        base_lock,
        cumulative_lock,
        thresholds,
        registry["base"]["class_ids"],
        target_round["old_class_ids"],
        target_round["new_class_ids"],
        target_round["learned_class_ids"],
        class_names,
    )

    context_metrics: dict[str, Any] = {"status": "not_evaluated"}
    context_checks: dict[str, bool] = {}
    if not args.skip_context_check:
        scene_weight = args.scene_weight.expanduser().resolve()
        if not scene_weight.is_file():
            raise FileNotFoundError(f"Scene-SensorNet 权重不存在：{scene_weight}")
        from fair_agent.models.context import evaluate_context_paths, load_context_model

        context_model, context_checkpoint = load_context_model(
            scene_weight, f"cuda:{args.device}"
        )
        context_metrics = evaluate_context_paths(
            context_model,
            context_checkpoint,
            cumulative_lock,
            f"cuda:{args.device}",
            batch_size=max(1, args.batch),
        )
        context_checks = {
            "sensor_accuracy": float(context_metrics["sensor_accuracy"]) >= 0.90,
            "scene_accuracy": float(context_metrics["scene_accuracy"]) >= 0.70,
            "joint_accuracy": float(context_metrics["joint_accuracy"]) >= 0.65,
        }
    score_gates = {
        "base_map50": float(lock_metrics["base_map50"]) >= 0.80,
        "new_map50": float(lock_metrics["new_map50"]) >= 0.60,
        "krr": float(lock_metrics["krr"]) >= 0.95,
    }
    now = datetime.now().astimezone()
    metrics = {
        "schema_version": 6,
        "run_id": (
            f"{registry['protocol_id']}-{target_round['round_id']}-"
            f"{now.strftime('%Y%m%d-%H%M%S')}"
        ),
        "created_at": now.isoformat(),
        "protocol": registry["protocol_id"],
        "round_registry": Path(registry["path"]).as_posix(),
        "phase": "joint_evaluation",
        "counted_as_incremental_learning": False,
        "detector_weights_updated": False,
        "model_selection_allowed": False,
        "incremental_mode": "class_incremental",
        "incremental_learning_data_scope": "incremental_dataset_only",
        "old_raw_image_count": 0,
        "old_raw_label_count": 0,
        "lineage": {
            "round_id": target_round["round_id"],
            "round_index": target_round["round_index"],
            "parent_generation_id": target_round["parent_generation_id"],
            "generation_id": target_round["generation_id"],
            "old_class_ids": target_round["old_class_ids"],
            "new_class_ids": target_round["new_class_ids"],
            "learned_class_ids": target_round["learned_class_ids"],
        },
        "class_map": {str(key): value for key, value in class_names.items()},
        "base_local_to_global": {
            str(key): value for key, value in base_mapping.items()
        },
        "specialists": [
            {
                "round_id": row["round_id"],
                "model_id": row["specialist"]["model_id"],
                "local_to_global": {
                    str(key): value
                    for key, value in row["specialist"]["local_to_global"].items()
                },
                "weight": specialist_weights[str(row["round_id"])].as_posix(),
            }
            for row in active_rounds
        ],
        "weights": {
            "base": base_weight.as_posix(),
            "scene_sensor": (
                None
                if args.skip_context_check
                else args.scene_weight.expanduser().resolve().as_posix()
            ),
        },
        "inference": {
            "base_imgsz": args.base_imgsz,
            "specialist_imgsz": args.specialist_imgsz,
            "prediction_confidence": args.prediction_confidence,
            "iou": 0.70,
            "max_det": 300,
            "fusion": "registry_owned_disjoint_class_concatenation",
            "context_gate_enabled": False,
            "per_class_thresholds": {
                str(key): value for key, value in thresholds.items()
            },
        },
        "splits": {
            "dev": {key: len(value) for key, value in dev_groups.items()},
            "lock": {key: len(value) for key, value in lock_groups.items()},
            "cumulative_dev": len(cumulative_dev),
            "cumulative_lock": len(cumulative_lock),
        },
        "round_metrics": {
            "new_map50": lock_metrics["new_map50"],
            "krr": lock_metrics["krr"],
            "full_map50": lock_metrics["full_map50"],
        },
        "dev": dev_metrics,
        "lock": lock_metrics,
        "context_lock": context_metrics,
        "speed_ms_per_image": {"dev": dev_speed, "lock": lock_speed},
        "score_gates": score_gates,
        "context_checks": context_checks,
        "competition_accepted": all(score_gates.values()),
        "context_accepted": (
            all(context_checks.values()) if context_checks else None
        ),
        "accepted": all(score_gates.values()),
        "predictions_frozen_before_lock_labels": True,
        "scene_sensor_is_incremental_learner": False,
    }
    atomic_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
