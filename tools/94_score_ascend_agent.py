#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from fair_agent.modules.strict_incremental import (
    evaluate_ap50,
    precision_recall,
    read_split,
    retention_metrics,
    subset_rows,
    yolo_ground_truth,
)


ROOT = Path(__file__).resolve().parents[1]
ACCURACY_GATE_KEYS = {
    "base_map50": "base_map50_min",
    "new_map50": "new_map50_min",
    "krr": "krr_min",
}


def parse_class_ids(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("class ids必须是逗号分隔的非负整数") from exc
    if not values or len(values) != len(set(values)) or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("class ids必须是互异的非负整数")
    return values


def load_accuracy_gates(path: Path) -> dict[str, float]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("kind") != "ascend310b_full_score_method"
    ):
        raise ValueError(f"满分方法配置非法：{path}")
    raw = (payload.get("competition") or {}).get("accuracy_gates") or {}
    try:
        gates = {
            metric: float(raw[config_key])
            for metric, config_key in ACCURACY_GATE_KEYS.items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"满分方法配置缺少有效精度门禁：{path}") from exc
    if any(not 0.0 <= value <= 1.0 for value in gates.values()):
        raise ValueError(f"满分方法精度门禁必须位于[0,1]：{path}")
    return gates


def _false_activation_rate(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    image_ids: set[str],
    class_id: int,
) -> float:
    positive = {
        str(row["image_id"])
        for row in ground_truth
        if int(row["class_id"]) == int(class_id)
    }
    negative = image_ids - positive
    activated = {
        str(row["image_id"])
        for row in predictions
        if int(row["class_id"]) == int(class_id)
    }
    return len(negative & activated) / len(negative) if negative else 0.0


def _false_activation_rate_for_classes(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    image_ids: set[str],
    class_ids: Sequence[int],
) -> float:
    selected = {int(value) for value in class_ids}
    positive = {
        str(row["image_id"])
        for row in ground_truth
        if int(row["class_id"]) in selected
    }
    negative = image_ids - positive
    activated = {
        str(row["image_id"])
        for row in predictions
        if int(row["class_id"]) in selected
    }
    return len(negative & activated) / len(negative) if negative else 0.0


def _read_predictions(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base: list[dict[str, Any]] = []
    combined: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        image_id = str(record["image_id"])
        for detection in record.get("detections", []):
            row = {
                "image_id": image_id,
                "class_id": int(detection["class_id"]),
                "confidence": float(detection["confidence"]),
                "xyxy": [float(value) for value in detection["xyxy"]],
            }
            combined.append(row)
            if detection.get("source") == "frozen_base_model":
                base.append(row)
    return base, combined


def main() -> int:
    parser = argparse.ArgumentParser(
        description="冻结Ascend Agent无标签预测后，在本机读取标签并评分。"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--mixed-split", type=Path, default=Path("splits/strict_4plus2/mixed_lock.txt")
    )
    parser.add_argument(
        "--base-split", type=Path, default=Path("splits/strict_4plus2/base_lock.txt")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method-config",
        type=Path,
        default=ROOT / "configs/ascend310b/full_score_method.yaml",
    )
    parser.add_argument("--expected-images", type=int, default=89)
    parser.add_argument("--old-class-ids", type=parse_class_ids, default=[0, 1, 2, 3])
    parser.add_argument(
        "--new-class-ids",
        type=parse_class_ids,
        default=[4, 5],
        help="逗号分隔的新增类ID。",
    )
    args = parser.parse_args()

    accuracy_gates = load_accuracy_gates(args.method_config.resolve())

    if args.expected_images <= 0:
        raise ValueError("expected-images必须为正整数。")
    old_ids = [int(value) for value in args.old_class_ids]
    new_ids = [int(value) for value in args.new_class_ids]
    if set(new_ids) & set(old_ids):
        raise ValueError("new-class-ids不能与old-class-ids重叠。")

    mixed = read_split(args.mixed_split.resolve())
    base_images = read_split(args.base_split.resolve())
    mixed_ids = {path.stem for path in mixed}
    if len(mixed) != args.expected_images or len(mixed_ids) != len(mixed):
        raise RuntimeError(
            f"mixed_test应包含{args.expected_images}张stem唯一图像"
        )

    base_predictions, combined = _read_predictions(args.predictions.resolve())
    if {str(row["image_id"]) for row in combined} - mixed_ids:
        raise RuntimeError("冻结预测包含mixed_test之外的图像")

    # This is intentionally the first point where labels are opened.
    ground_truth = yolo_ground_truth(mixed)
    base_ids = {path.stem for path in base_images}
    base_metrics = evaluate_ap50(
        subset_rows(base_predictions, base_ids),
        subset_rows(ground_truth, base_ids),
        old_ids,
    )
    retention = retention_metrics(base_predictions, combined, ground_truth, old_ids)
    new_metrics = evaluate_ap50(combined, ground_truth, new_ids)
    full_metrics = evaluate_ap50(combined, ground_truth, sorted([*old_ids, *new_ids]))
    lock_pr_by_class = {
        class_id: precision_recall(combined, ground_truth, class_id, 0.63)
        for class_id in new_ids
    }
    lock_precision = sum(
        float(row["precision"]) for row in lock_pr_by_class.values()
    ) / len(lock_pr_by_class)
    lock_recall = sum(
        float(row["recall"]) for row in lock_pr_by_class.values()
    ) / len(lock_pr_by_class)
    false_activation_by_class = {
        class_id: _false_activation_rate(
            combined, ground_truth, mixed_ids, class_id
        )
        for class_id in new_ids
    }
    false_activation = _false_activation_rate_for_classes(
        combined, ground_truth, mixed_ids, new_ids
    )
    metrics = {
        "base_map50": float(base_metrics["map50"]),
        "new_map50": float(new_metrics["map50"]),
        "krr": float(retention["krr"]),
        "old_map50_before": float(retention["old_map50_before"]),
        "old_map50_after": float(retention["old_map50_after"]),
        "old_prediction_equivalent": bool(retention["old_prediction_equivalent"]),
        "full_map50": float(full_metrics["map50"]),
        "per_class_ap50": {
            str(key): float(value)
            for key, value in full_metrics["per_class_ap50"].items()
        },
        "lock_precision": float(lock_precision),
        "lock_recall": float(lock_recall),
        "lock_precision_recall_by_class": {
            str(class_id): {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
            }
            for class_id, row in lock_pr_by_class.items()
        },
        "false_activation_rate": float(false_activation),
        "false_activation_rate_by_class": {
            str(class_id): float(value)
            for class_id, value in false_activation_by_class.items()
        },
    }
    competition_gates = {
        name: metrics[name] >= minimum
        for name, minimum in accuracy_gates.items()
    }
    diagnostic_checks = {
        "lock_precision": metrics["lock_precision"] >= 0.90,
        "false_activation_rate": metrics["false_activation_rate"] <= 0.05,
    }
    result = {
        "schema_version": 2,
        "provider": "Ascend ACL",
        "unlabeled_predictions_frozen_before_labels": True,
        "image_count": len(mixed),
        "base_image_count": len(base_images),
        "prediction_count": len(combined),
        "class_contract": {
            "old_class_ids": old_ids,
            "new_class_ids": new_ids,
        },
        "metrics": metrics,
        # Base mAP50、New-mAP50 与 KRR 是赛题唯一三项精度计分门槛。
        # precision/误激活率继续保留为部署诊断，但不得再淘汰一个计分
        # 满分候选。
        "competition_gates": competition_gates,
        "competition_gate_thresholds": accuracy_gates,
        "diagnostic_checks": diagnostic_checks,
        "diagnostic_warnings": [
            name for name, passed in diagnostic_checks.items() if not passed
        ],
        "score_passed": all(competition_gates.values()),
        "passed": all(competition_gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
