#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fair_agent.modules.strict_incremental import (
    GLOBAL_CLASS_NAMES,
    evaluate_ap50,
    precision_recall,
    read_split,
    retention_metrics,
    subset_rows,
    yolo_ground_truth,
)


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
        "--mixed-split", type=Path, default=Path("splits/strict_3plus1/mixed_test.txt")
    )
    parser.add_argument(
        "--base-split", type=Path, default=Path("splits/strict_3plus1/base_test.txt")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mixed = read_split(args.mixed_split.resolve())
    base_images = read_split(args.base_split.resolve())
    mixed_ids = {path.stem for path in mixed}
    if len(mixed) != 89 or len(mixed_ids) != len(mixed):
        raise RuntimeError("mixed_test应包含89张stem唯一图像")

    base_predictions, combined = _read_predictions(args.predictions.resolve())
    if {str(row["image_id"]) for row in combined} - mixed_ids:
        raise RuntimeError("冻结预测包含mixed_test之外的图像")

    # This is intentionally the first point where labels are opened.
    ground_truth = yolo_ground_truth(mixed)
    base_ids = {path.stem for path in base_images}
    old_ids = [0, 1, 3]
    new_id = 2
    base_metrics = evaluate_ap50(
        subset_rows(base_predictions, base_ids),
        subset_rows(ground_truth, base_ids),
        old_ids,
    )
    retention = retention_metrics(base_predictions, combined, ground_truth, old_ids)
    new_metrics = evaluate_ap50(combined, ground_truth, [new_id])
    full_metrics = evaluate_ap50(combined, ground_truth, GLOBAL_CLASS_NAMES)
    lock_pr = precision_recall(combined, ground_truth, new_id, 0.63)
    false_activation = _false_activation_rate(combined, ground_truth, mixed_ids, new_id)
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
        "lock_precision": float(lock_pr["precision"]),
        "lock_recall": float(lock_pr["recall"]),
        "false_activation_rate": float(false_activation),
    }
    competition_gates = {
        "base_map50": metrics["base_map50"] >= 0.80,
        "new_map50": metrics["new_map50"] >= 0.60,
        "krr": metrics["krr"] >= 0.95,
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
        "metrics": metrics,
        # Base mAP50、New-mAP50 与 KRR 是赛题唯一三项精度计分门槛。
        # precision/误激活率继续保留为部署诊断，但不得再淘汰一个计分
        # 满分候选。
        "competition_gates": competition_gates,
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
