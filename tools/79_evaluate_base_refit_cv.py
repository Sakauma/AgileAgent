#!/usr/bin/env python3
"""Evaluate a base-detector family with strictly out-of-fold predictions."""

from __future__ import annotations

import argparse
import json
import runpy
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import (
    evaluate_ap50,
    read_split,
    sha256_file,
    subset_rows,
    yolo_ground_truth,
)


FORBIDDEN_MARKERS = ("mixed_test", "base_test", "lock")
BASE_LOCAL_TO_GLOBAL = {0: 0, 1: 1, 2: 3}
BASE_GLOBAL_CLASS_IDS = [0, 1, 3]


def reject_test_reference(path: Path, role: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"OOF {role} 不得引用 test/lock：{path}")


def parse_model(value: str) -> tuple[str, Path, int]:
    if "=" not in value or "@" not in value:
        raise argparse.ArgumentTypeError("--model 使用 FOLD=WEIGHT@IMGSZ")
    name, payload = value.split("=", 1)
    path_text, imgsz_text = payload.rsplit("@", 1)
    if not name or not imgsz_text.isdigit():
        raise argparse.ArgumentTypeError("--model 使用 FOLD=WEIGHT@IMGSZ")
    return name, Path(path_text), int(imgsz_text)


def parse_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--training-report 使用 FOLD=REPORT")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--training-report 使用 FOLD=REPORT")
    return name, Path(path)


def parse_tile_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("--tile-size 使用 WIDTHxHEIGHT") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--tile-size 必须为正整数")
    return width, height


def validate_training_report(
    fold_name: str,
    report_path: Path,
    weight_path: Path,
    expected_val_count: int,
    expected_epochs: int,
) -> dict[str, Any]:
    reject_test_reference(report_path, "training report")
    reject_test_reference(weight_path, "weight")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    training = dict(report.get("training", {}))
    dataset = dict(report.get("dataset_audit", {}))
    reported_val_count = int(dataset.get("dev_count", dataset.get("val_count", -1)))
    if (
        str(report.get("candidate")) != fold_name
        or report.get("selection_scope") != "base_train_and_dev_only"
        or bool(report.get("lock_data_access", True))
        or int(training.get("requested_epochs", -1)) != expected_epochs
        or int(training.get("completed_epochs", -1)) != expected_epochs
        or bool(training.get("stopped_early", True))
        or reported_val_count != expected_val_count
        or bool(dataset.get("source_declared_test_split", True))
        or bool(dataset.get("training_declared_test_split", True))
    ):
        raise ValueError(f"{fold_name} 训练报告不满足完整非测试 OOF 约束")
    expected_hash = str(report.get("best_weight_sha256", ""))
    actual_hash = sha256_file(weight_path)
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(f"{fold_name} best.pt 与训练报告不一致")
    return {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "weight": str(weight_path),
        "weight_sha256": actual_hash,
        "best_epoch": int(training["best_epoch"]),
        "best_metric_value": float(training["best_metric_value"]),
        "completed_epochs": int(training["completed_epochs"]),
    }


def subgroup_metrics(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    image_ids: Sequence[str],
) -> dict[str, Any]:
    sequences = sorted({image_id.rsplit("_", 1)[0] for image_id in image_ids})
    output = {}
    for sequence in sequences:
        selected = {image_id for image_id in image_ids if image_id.startswith(f"{sequence}_")}
        output[sequence] = evaluate_ap50(
            subset_rows(predictions, selected),
            subset_rows(targets, selected),
            BASE_GLOBAL_CLASS_IDS,
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--model", type=parse_model, action="append", required=True)
    parser.add_argument(
        "--training-report", type=parse_report, action="append", required=True
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--inference-mode",
        choices=("full_frame", "sliding_window"),
        default="full_frame",
    )
    parser.add_argument("--tile-size", type=parse_tile_size)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--focus-class-local", type=int, default=0)
    args = parser.parse_args()
    if not str(args.device).isdigit():
        raise ValueError("device 必须是明确的单卡编号")
    if args.inference_mode == "sliding_window" and args.tile_size is None:
        raise ValueError("sliding_window 必须显式提供 --tile-size")
    if not 0.0 <= float(args.tile_overlap) < 1.0:
        raise ValueError("tile overlap 必须在 [0, 1) 内")
    if int(args.focus_class_local) not in BASE_LOCAL_TO_GLOBAL:
        raise ValueError("focus class local 未注册到基础类别")

    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    reject_test_reference(manifest_path, "manifest")
    reject_test_reference(output, "output")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有 OOF 报告：{output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("selection_scope") != "base_train_and_dev_only"
        or bool(manifest.get("lock_data_access", True))
        or int(manifest.get("validation_coverage_count", -1))
        != int(manifest.get("combined_non_test_count", -2))
    ):
        raise ValueError("fold manifest 不是完整的非测试交叉验证边界")
    folds = {f"fold_{int(row['fold'])}": dict(row) for row in manifest["folds"]}
    models = {name: (path.resolve(), imgsz) for name, path, imgsz in args.model}
    reports = {name: path.resolve() for name, path in args.training_report}
    if set(models) != set(folds) or set(reports) != set(folds):
        raise ValueError(
            f"model/report 必须恰好覆盖全部 folds：expected={sorted(folds)} "
            f"models={sorted(models)} reports={sorted(reports)}"
        )

    runner = runpy.run_path(str(ROOT / "tools" / "70_run_strict_3plus1.py"))
    small_object = runpy.run_path(
        str(ROOT / "tools" / "76_evaluate_small_object_inference.py")
    )
    from ultralytics import YOLO

    predict_config = {
        "common": {"imgsz": 0},
        "predict": {
            "conf": float(args.conf),
            "iou": float(args.iou),
            "max_det": int(args.max_det),
            "batch": int(args.batch),
            "rect": True,
            "augment": False,
        },
    }
    all_predictions = []
    all_targets = []
    all_image_ids = []
    fold_metrics = {}
    model_audit = {}
    seen: set[str] = set()
    output.mkdir(parents=True)
    prediction_dir = output / "predictions"
    for fold_name in sorted(folds, key=lambda name: int(name.rsplit("_", 1)[1])):
        fold = folds[fold_name]
        images = read_split(Path(fold["val_split"]))
        image_ids = [image.stem for image in images]
        reject_test_reference(Path(fold["val_split"]), f"{fold_name} validation")
        if len(images) != int(fold["val_count"]) or seen & set(image_ids):
            raise ValueError(f"{fold_name} 验证集数量错误或与前折重复")
        seen.update(image_ids)
        weight, imgsz = models[fold_name]
        training_audit = validate_training_report(
            fold_name,
            reports[fold_name],
            weight,
            len(images),
            int(args.epochs),
        )
        detector = YOLO(str(weight))
        tile_audit = None
        if args.inference_mode == "full_frame":
            predictions, inference_ms = runner["predict_records"](
                detector,
                images,
                BASE_LOCAL_TO_GLOBAL,
                predict_config,
                str(args.device),
                f"{args.family}:{fold_name}",
                int(imgsz),
            )
        else:
            local_predictions, inference_ms, tile_audit = small_object["predict_tiles"](
                detector,
                images,
                args.tile_size,
                float(args.tile_overlap),
                int(imgsz),
                int(args.batch),
                str(args.device),
                int(args.focus_class_local),
                float(args.conf),
                float(args.iou),
                int(args.max_det),
                runner["evaluation_predictor_class"](),
                f"{args.family}:{fold_name}",
            )
            predictions = [
                {
                    **dict(row),
                    "class_id": BASE_LOCAL_TO_GLOBAL[int(row["class_id"])],
                }
                for row in local_predictions
            ]
        # Labels are read only from this fold's non-test validation list.
        targets = yolo_ground_truth(images, BASE_GLOBAL_CLASS_IDS)
        metrics = evaluate_ap50(predictions, targets, BASE_GLOBAL_CLASS_IDS)
        fold_metrics[fold_name] = {
            "image_count": len(images),
            "map50": float(metrics["map50"]),
            "per_class_ap50": metrics["per_class_ap50"],
        }
        model_audit[fold_name] = {
            **training_audit,
            "imgsz": int(imgsz),
            "prediction_count": len(predictions),
            "inference_ms": float(inference_ms),
            "inference_mode": args.inference_mode,
            "tile": tile_audit,
        }
        runner["write_jsonl_artifact"](
            prediction_dir / f"{fold_name}.jsonl", predictions
        )
        all_predictions.extend(predictions)
        all_targets.extend(targets)
        all_image_ids.extend(image_ids)

    if len(seen) != int(manifest["combined_non_test_count"]):
        raise ValueError("OOF 预测未覆盖全部非测试图像")
    aggregate = evaluate_ap50(all_predictions, all_targets, BASE_GLOBAL_CLASS_IDS)
    fold_scores = [float(item["map50"]) for item in fold_metrics.values()]
    report = {
        "schema_version": 1,
        "family": args.family,
        "selection_scope": "base_train_and_dev_oof_only",
        "lock_data_access": False,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "image_count": len(seen),
        "class_ids": BASE_GLOBAL_CLASS_IDS,
        "inference_mode": args.inference_mode,
        "tile_size": (
            {"width": args.tile_size[0], "height": args.tile_size[1]}
            if args.tile_size
            else None
        ),
        "tile_overlap": float(args.tile_overlap),
        "models": model_audit,
        "folds": fold_metrics,
        "fold_summary": {
            "minimum_map50": min(fold_scores),
            "mean_map50": statistics.fmean(fold_scores),
            "median_map50": statistics.median(fold_scores),
            "maximum_map50": max(fold_scores),
        },
        "oof": {
            "map50": float(aggregate["map50"]),
            "per_class_ap50": aggregate["per_class_ap50"],
            "subgroups": subgroup_metrics(all_predictions, all_targets, all_image_ids),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
