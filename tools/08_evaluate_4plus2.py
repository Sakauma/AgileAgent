#!/usr/bin/env python3
"""Freeze 4+2 predictions, calibrate on dev, and score the untouched lock split."""

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


GLOBAL_CLASS_NAMES = {
    0: "soldier",
    1: "small_aircraft",
    2: "warship",
    3: "tank",
    4: "patrol_boat",
    5: "armored_vehicle",
}
BASE_LOCAL_TO_GLOBAL = {0: 0, 1: 1, 2: 2, 3: 3}
SPECIALIST_LOCAL_TO_GLOBAL = {0: 4, 1: 5}
BASE_CLASS_IDS = tuple(BASE_LOCAL_TO_GLOBAL.values())
NEW_CLASS_IDS = tuple(SPECIALIST_LOCAL_TO_GLOBAL.values())


def resolve_split(
    data_root: Path, split_name: str, *, require_labels: bool = True
) -> list[Path]:
    split_path = data_root / "splits" / "strict_4plus2" / split_name
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
            image.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(
                f"{split_path}:{line_number} 越出数据根目录：{image}"
            ) from exc
        if not image.is_file() or (
            require_labels and not image.with_suffix(".txt").is_file()
        ):
            raise FileNotFoundError(f"划分图像或标签不存在：{image}")
        images.append(image)
    if not images or len(images) != len(set(images)):
        raise ValueError(f"划分为空或包含重复图像：{split_path}")
    return images


def ensure_mixed_contract(
    base_images: Sequence[Path],
    incremental_images: Sequence[Path],
    mixed_images: Sequence[Path],
    split_name: str,
) -> None:
    expected = [path.resolve() for path in [*base_images, *incremental_images]]
    actual = [path.resolve() for path in mixed_images]
    if Counter(expected) != Counter(actual) or len(expected) != len(actual):
        raise ValueError(f"{split_name} 不等于对应 Base + Increment 清单")


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
                raise RuntimeError(f"模型输出未登记的局部类别：{local_id}")
            records.append(
                {
                    "image_id": image_id,
                    "class_id": mapping[local_id],
                    "confidence": float(confidence_value),
                    "xyxy": [float(value) for value in xyxy],
                    "source": source_name,
                }
            )
    speed = {
        key: value / len(results) for key, value in speed_totals.items()
    }
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


def subset(rows: Sequence[Mapping[str, Any]], image_ids: Iterable[str]) -> list[dict[str, Any]]:
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


def calibrate_map50_thresholds(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    minimum: float,
    maximum: float,
    step: float,
) -> dict[str, Any]:
    from fair_agent.modules.strict_incremental import evaluate_ap50, precision_recall

    count = int(round((maximum - minimum) / step)) + 1
    per_class: dict[str, Any] = {}
    thresholds: dict[int, float] = {}
    for class_id in NEW_CLASS_IDS:
        curve = []
        for index in range(count):
            threshold = round(minimum + index * step, 6)
            filtered = [
                dict(row)
                for row in predictions
                if int(row["class_id"]) != class_id
                or float(row["confidence"]) >= threshold
            ]
            ap50 = float(
                evaluate_ap50(filtered, ground_truth, [class_id])["map50"]
            )
            quality = precision_recall(
                predictions, ground_truth, class_id, threshold
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
            "class_name": GLOBAL_CLASS_NAMES[class_id],
            "selected": selected,
            "curve": curve,
        }
    filtered = apply_thresholds(predictions, thresholds)
    return {
        "schema_version": 2,
        "learning_data_scope": "incremental_dataset_only",
        "source_split": "mixed_dev_only",
        "selection_metric": "per_class_mAP50",
        "deployment_policy": "competition_map50_dev_calibrated",
        "per_class_thresholds": {str(key): value for key, value in thresholds.items()},
        "per_class": per_class,
        "selected_new_map50": float(
            evaluate_ap50(filtered, ground_truth, NEW_CLASS_IDS)["map50"]
        ),
    }


def false_activation_metrics(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    images: Sequence[Path],
    thresholds: Mapping[int, float],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for class_id in NEW_CLASS_IDS:
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
            "class_name": GLOBAL_CLASS_NAMES[class_id],
            "negative_image_count": len(negative),
            "false_activation_image_count": len(false_images),
            "false_activation_rate": (
                len(false_images) / len(negative) if negative else 0.0
            ),
        }
    return output


def score_split(
    base_predictions: Sequence[Mapping[str, Any]],
    specialist_predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    base_images: Sequence[Path],
    mixed_images: Sequence[Path],
    thresholds: Mapping[int, float],
) -> dict[str, Any]:
    from fair_agent.modules.strict_incremental import (
        evaluate_ap50,
        precision_recall,
        retention_metrics,
    )

    selected_new = apply_thresholds(specialist_predictions, thresholds)
    fused = [dict(row) for row in base_predictions] + selected_new
    base_ids = {path.stem for path in base_images}
    base_ground_truth = subset(ground_truth, base_ids)
    base_scoped_predictions = subset(base_predictions, base_ids)
    base_metrics = evaluate_ap50(
        base_scoped_predictions, base_ground_truth, BASE_CLASS_IDS
    )
    retention = retention_metrics(
        base_predictions,
        fused,
        ground_truth,
        BASE_CLASS_IDS,
    )
    new_metrics = evaluate_ap50(fused, ground_truth, NEW_CLASS_IDS)
    full_metrics = evaluate_ap50(fused, ground_truth, GLOBAL_CLASS_NAMES)
    per_class_quality = {
        str(class_id): {
            "class_name": GLOBAL_CLASS_NAMES[class_id],
            "map50": float(new_metrics["per_class_ap50"].get(class_id, 0.0)),
            **precision_recall(
                specialist_predictions,
                ground_truth,
                class_id,
                float(thresholds[class_id]),
            ),
            "threshold": float(thresholds[class_id]),
        }
        for class_id in NEW_CLASS_IDS
    }
    return {
        "image_count": len(mixed_images),
        "base_image_count": len(base_images),
        "base_map50": float(base_metrics["map50"]),
        "base_per_class_ap50": {
            str(key): value for key, value in base_metrics["per_class_ap50"].items()
        },
        "old_map50_before": float(retention["old_map50_before"]),
        "old_map50_after": float(retention["old_map50_after"]),
        "krr": float(retention["krr"]),
        "old_prediction_equivalent": bool(retention["old_prediction_equivalent"]),
        "new_map50": float(new_metrics["map50"]),
        "new_per_class_ap50": {
            str(key): value for key, value in new_metrics["per_class_ap50"].items()
        },
        "full_map50": float(full_metrics["map50"]),
        "full_per_class_ap50": {
            str(key): value for key, value in full_metrics["per_class_ap50"].items()
        },
        "new_class_quality": per_class_quality,
        "false_activation": false_activation_metrics(
            specialist_predictions, ground_truth, mixed_images, thresholds
        ),
        "prediction_counts": {
            "base": len(base_predictions),
            "specialist_before_threshold": len(specialist_predictions),
            "specialist_after_threshold": len(selected_new),
            "fused": len(fused),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按 mAP50 冻结、校准并复核正式 4+2 模型组合。"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-weight", type=Path, required=True)
    parser.add_argument("--specialist-weight", type=Path, required=True)
    parser.add_argument("--scene-weight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--base-imgsz", type=int, default=1280)
    parser.add_argument("--specialist-imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=18)
    parser.add_argument("--prediction-confidence", type=float, default=0.01)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.50)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    base_weight = args.base_weight.expanduser().resolve()
    specialist_weight = args.specialist_weight.expanduser().resolve()
    scene_weight = args.scene_weight.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (base_weight, specialist_weight, scene_weight):
        if not path.is_file():
            raise FileNotFoundError(f"评测权重不存在：{path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有 4+2 评测目录：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dev = resolve_split(data_root, "base_dev.txt")
    increment_dev = resolve_split(data_root, "increment_dev.txt")
    mixed_dev = resolve_split(data_root, "mixed_dev.txt")
    base_lock = resolve_split(data_root, "base_lock.txt", require_labels=False)
    increment_lock = resolve_split(
        data_root, "increment_lock.txt", require_labels=False
    )
    mixed_lock = resolve_split(data_root, "mixed_lock.txt", require_labels=False)
    ensure_mixed_contract(base_dev, increment_dev, mixed_dev, "mixed_dev")
    ensure_mixed_contract(base_lock, increment_lock, mixed_lock, "mixed_lock")

    # Development predictions and calibration may use dev labels.
    base_dev_predictions, base_dev_speed = predict_records(
        base_weight,
        mixed_dev,
        BASE_LOCAL_TO_GLOBAL,
        device=args.device,
        imgsz=args.base_imgsz,
        batch=args.batch,
        confidence=args.prediction_confidence,
        source_name="frozen_base_model",
    )
    specialist_dev_predictions, specialist_dev_speed = predict_records(
        specialist_weight,
        mixed_dev,
        SPECIALIST_LOCAL_TO_GLOBAL,
        device=args.device,
        imgsz=args.specialist_imgsz,
        batch=args.batch,
        confidence=args.prediction_confidence,
        source_name="incremental_model",
    )
    write_jsonl(output_dir / "frozen" / "base_dev_predictions.jsonl", base_dev_predictions)
    write_jsonl(
        output_dir / "frozen" / "specialist_dev_predictions.jsonl",
        specialist_dev_predictions,
    )
    from fair_agent.modules.strict_incremental import yolo_ground_truth

    dev_ground_truth = yolo_ground_truth(mixed_dev, GLOBAL_CLASS_NAMES)
    calibration = calibrate_map50_thresholds(
        specialist_dev_predictions,
        dev_ground_truth,
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )
    thresholds = {
        int(key): float(value)
        for key, value in calibration["per_class_thresholds"].items()
    }
    dev_metrics = score_split(
        base_dev_predictions,
        specialist_dev_predictions,
        dev_ground_truth,
        base_dev,
        mixed_dev,
        thresholds,
    )
    atomic_json(output_dir / "calibration.json", calibration)

    # Lock predictions are frozen before this process reads any lock labels.
    base_lock_predictions, base_lock_speed = predict_records(
        base_weight,
        mixed_lock,
        BASE_LOCAL_TO_GLOBAL,
        device=args.device,
        imgsz=args.base_imgsz,
        batch=args.batch,
        confidence=args.prediction_confidence,
        source_name="frozen_base_model",
    )
    specialist_lock_predictions, specialist_lock_speed = predict_records(
        specialist_weight,
        mixed_lock,
        SPECIALIST_LOCAL_TO_GLOBAL,
        device=args.device,
        imgsz=args.specialist_imgsz,
        batch=args.batch,
        confidence=args.prediction_confidence,
        source_name="incremental_model",
    )
    write_jsonl(
        output_dir / "frozen" / "base_lock_predictions.jsonl",
        base_lock_predictions,
    )
    write_jsonl(
        output_dir / "frozen" / "specialist_lock_predictions.jsonl",
        specialist_lock_predictions,
    )
    lock_ground_truth = yolo_ground_truth(mixed_lock, GLOBAL_CLASS_NAMES)
    lock_metrics = score_split(
        base_lock_predictions,
        specialist_lock_predictions,
        lock_ground_truth,
        base_lock,
        mixed_lock,
        thresholds,
    )

    from fair_agent.models.context import evaluate_context_paths, load_context_model

    context_model, context_checkpoint = load_context_model(
        scene_weight, f"cuda:{args.device}"
    )
    context_metrics = evaluate_context_paths(
        context_model,
        context_checkpoint,
        mixed_lock,
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
    metrics = {
        "schema_version": 3,
        "run_id": f"strict-4plus2-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}",
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": "strict-4plus2-parallel-specialist",
        "incremental_mode": "class_incremental",
        "learning_data_scope": "incremental_dataset_only",
        "old_raw_image_count": 0,
        "old_raw_label_count": 0,
        "class_map": {str(key): value for key, value in GLOBAL_CLASS_NAMES.items()},
        "base_local_to_global": {
            str(key): value for key, value in BASE_LOCAL_TO_GLOBAL.items()
        },
        "specialist_local_to_global": {
            str(key): value for key, value in SPECIALIST_LOCAL_TO_GLOBAL.items()
        },
        "weights": {
            "base": base_weight.as_posix(),
            "specialist": specialist_weight.as_posix(),
            "scene_sensor": scene_weight.as_posix(),
        },
        "inference": {
            "base_imgsz": args.base_imgsz,
            "specialist_imgsz": args.specialist_imgsz,
            "prediction_confidence": args.prediction_confidence,
            "iou": 0.70,
            "max_det": 300,
            "fusion": "disjoint_class_owner_concatenation",
            "context_gate_enabled": False,
            "per_class_thresholds": {
                str(key): value for key, value in thresholds.items()
            },
        },
        "splits": {
            "base_dev": len(base_dev),
            "increment_dev": len(increment_dev),
            "mixed_dev": len(mixed_dev),
            "base_lock": len(base_lock),
            "increment_lock": len(increment_lock),
            "mixed_lock": len(mixed_lock),
        },
        "dev": dev_metrics,
        "lock": lock_metrics,
        "context_lock": context_metrics,
        "speed_ms_per_image": {
            "dev_base": base_dev_speed,
            "dev_specialist": specialist_dev_speed,
            "lock_base": base_lock_speed,
            "lock_specialist": specialist_lock_speed,
        },
        "score_gates": score_gates,
        "context_checks": context_checks,
        "competition_accepted": all(score_gates.values()),
        "context_accepted": all(context_checks.values()),
        "accepted": all(score_gates.values()) and all(context_checks.values()),
        "predictions_frozen_before_lock_labels": True,
    }
    atomic_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
