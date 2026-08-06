#!/usr/bin/env python3
"""Evaluate full-frame and sliding-window base inference on a non-lock split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


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


def parse_model(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("model 必须使用 name=/absolute/path/to/best.pt")
    if not all(character.isalnum() or character in "._-" for character in name):
        raise argparse.ArgumentTypeError(f"模型名包含非法字符：{name}")
    return name, Path(raw_path).expanduser().resolve()


def parse_sized_model(value: str) -> tuple[str, Path, int]:
    model_value, separator, raw_size = value.rpartition("@")
    if not separator or not raw_size.isdigit():
        raise argparse.ArgumentTypeError(
            "passthrough model 必须使用 name=/absolute/path/to/best.pt@imgsz"
        )
    name, path = parse_model(model_value)
    return name, path, int(raw_size)


def parse_crop_size(value: str) -> tuple[int, int]:
    pieces = value.lower().split("x")
    if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
        raise argparse.ArgumentTypeError("--crop-size 必须使用 WIDTHxHEIGHT")
    width, height = map(int, pieces)
    if width < 32 or height < 32:
        raise argparse.ArgumentTypeError("裁剪宽高必须至少为 32")
    return width, height


def float_grid(value: str) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values:
        raise argparse.ArgumentTypeError("数值网格不得为空")
    return values


def reject_lock_reference(path: Path, purpose: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"{purpose} 不得引用 test/lock：{path}")


def sliding_starts(length: int, tile: int, requested_overlap: float) -> list[int]:
    if tile >= length:
        return [0]
    if not 0.0 <= requested_overlap < 1.0:
        raise ValueError("tile overlap 必须在 [0, 1) 内")
    maximum_step = max(1.0, tile * (1.0 - requested_overlap))
    interval_count = max(1, math.ceil((length - tile) / maximum_step))
    return [
        int(round(index * (length - tile) / interval_count))
        for index in range(interval_count + 1)
    ]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def records_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    return hashlib.sha256((payload + ("\n" if rows else "")).encode("utf-8")).hexdigest()


def predict_tiles(
    model: Any,
    images: Sequence[Path],
    crop_size: tuple[int, int],
    overlap: float,
    imgsz: int,
    batch: int,
    device: str,
    focus_class: int,
    conf: float,
    iou: float,
    max_det: int,
    predictor_class: type,
    source_name: str,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inference_ms = 0.0
    tile_count = 0
    windows_per_image: dict[str, int] = {}
    crop_width, crop_height = crop_size
    for image_path in images:
        with Image.open(image_path) as source:
            source.load()
            image_width, image_height = source.size
            if crop_width > image_width or crop_height > image_height:
                raise ValueError(
                    f"裁剪 {crop_width}x{crop_height} 大于原图 "
                    f"{image_width}x{image_height}：{image_path.name}"
                )
            windows = [
                (left, top, left + crop_width, top + crop_height)
                for top in sliding_starts(image_height, crop_height, overlap)
                for left in sliding_starts(image_width, crop_width, overlap)
            ]
            crops = [source.crop(window).convert("RGB") for window in windows]
        windows_per_image[image_path.stem] = len(windows)
        tile_count += len(windows)
        for start in range(0, len(crops), batch):
            batch_crops = crops[start : start + batch]
            batch_windows = windows[start : start + batch]
            results = model.predict(
                source=batch_crops,
                imgsz=int(imgsz),
                batch=len(batch_crops),
                conf=float(conf),
                iou=float(iou),
                max_det=int(max_det),
                rect=False,
                classes=[int(focus_class)],
                device=str(device),
                verbose=False,
                predictor=predictor_class,
            )
            if len(results) != len(batch_windows):
                raise RuntimeError(
                    f"切片预测数量不一致：expected={len(batch_windows)} actual={len(results)}"
                )
            for result, window in zip(results, batch_windows):
                inference_ms += float((getattr(result, "speed", None) or {}).get("inference", 0.0))
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                left, top, _right, _bottom = window
                for xyxy, confidence, class_id in zip(
                    boxes.xyxy.detach().cpu().tolist(),
                    boxes.conf.detach().cpu().tolist(),
                    boxes.cls.detach().cpu().tolist(),
                ):
                    if int(class_id) != int(focus_class):
                        raise RuntimeError(f"切片模型输出了非 focus 类别：{class_id}")
                    rows.append(
                        {
                            "image_id": image_path.stem,
                            "class_id": int(class_id),
                            "confidence": float(confidence),
                            "xyxy": [
                                float(xyxy[0]) + left,
                                float(xyxy[1]) + top,
                                float(xyxy[2]) + left,
                                float(xyxy[3]) + top,
                            ],
                            "source": source_name,
                        }
                    )
    return rows, inference_ms, {
        "tile_count": tile_count,
        "windows_per_image": windows_per_image,
        "crop_size": {"width": crop_width, "height": crop_height},
        "requested_overlap": overlap,
    }


def subgroup_metrics(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    images: Sequence[Path],
    class_ids: Sequence[int],
) -> dict[str, Any]:
    groups: dict[str, set[str]] = {}
    for image in images:
        pieces = image.stem.rsplit("_", 1)
        sequence = pieces[0] if len(pieces) == 2 and pieces[1].isdigit() else image.stem
        groups.setdefault(sequence, set()).add(image.stem)
    output = {}
    for sequence, image_ids in sorted(groups.items()):
        output[sequence] = evaluate_ap50(
            subset_rows(predictions, image_ids),
            subset_rows(targets, image_ids),
            class_ids,
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--primary", type=parse_model, required=True)
    parser.add_argument("--secondary", type=parse_model, required=True)
    parser.add_argument(
        "--passthrough-model",
        action="append",
        type=parse_sized_model,
        default=[],
        help="额外冻结 owner，格式 name=weights@imgsz；预测写入选模证据但不参与 soldier 网格搜索。",
    )
    parser.add_argument("--crop-size", type=parse_crop_size, required=True)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-imgsz", type=int, default=896)
    parser.add_argument(
        "--secondary-full-imgsz",
        type=int,
        help="secondary 完整帧推理尺度；默认与 primary 的 --full-imgsz 相同。",
    )
    parser.add_argument("--tile-imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--model-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--focus-class", type=int, default=0)
    parser.add_argument("--class-ids", default="0,1,2")
    parser.add_argument("--fusion-ious", type=float_grid, default=float_grid("0.35,0.45,0.55,0.65"))
    parser.add_argument(
        "--secondary-scales",
        type=float_grid,
        default=float_grid("0.35,0.50,0.65,0.80,1.00"),
    )
    parser.add_argument(
        "--agreement-bonuses",
        type=float_grid,
        default=float_grid("0.00,0.05,0.10,0.15"),
    )
    parser.add_argument(
        "--selection-scope",
        choices=(
            "auto",
            "base_dev_only",
            "base_train_internal_temporal_holdout",
            "final_refit_checkpoint_holdout",
        ),
        default="auto",
        help="显式标记非测试评估用途；final refit holdout 不得作为独立性能证据。",
    )
    args = parser.parse_args()
    if not str(args.device).isdigit():
        raise ValueError("device 必须是明确的单卡编号")

    split = args.split.expanduser().resolve()
    output = args.output.expanduser().resolve()
    reject_lock_reference(split, "selection split")
    reject_lock_reference(output, "selection output")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有评估：{output}")
    images = read_split(split)
    if not images or len({path.stem for path in images}) != len(images):
        raise ValueError("评估清单必须非空且 stem 唯一")
    manifest_path = split.parent.parent / "manifest.json"
    source_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    if source_manifest and bool(source_manifest.get("lock_data_access", True)):
        raise ValueError("数据集 manifest 未证明 lock_data_access=false")
    primary_name, primary_weight = args.primary
    secondary_name, secondary_weight = args.secondary
    passthrough_models = list(args.passthrough_model)
    secondary_full_name = f"{secondary_name}_full"
    secondary_tile_name = f"{secondary_name}_tile"
    owner_names = [primary_name, secondary_full_name, secondary_tile_name] + [
        row[0] for row in passthrough_models
    ]
    if len(owner_names) != len(set(owner_names)):
        raise ValueError("所有 owner 名称必须唯一")
    for weight in (primary_weight, secondary_weight, *(row[1] for row in passthrough_models)):
        if not weight.is_file():
            raise FileNotFoundError(weight)
    class_ids = [int(value) for value in args.class_ids.split(",")]
    if args.focus_class not in class_ids:
        raise ValueError("focus class 必须属于 class ids")

    output.mkdir(parents=True)
    runner = runpy.run_path(str(ROOT / "tools" / "70_run_strict_3plus1.py"))
    ensemble = runpy.run_path(str(ROOT / "tools" / "72_select_base_ensemble.py"))
    from ultralytics import YOLO

    predict_config = {
        "common": {"imgsz": int(args.full_imgsz)},
        "predict": {
            "conf": float(args.conf),
            "iou": float(args.model_iou),
            "max_det": int(args.max_det),
            "rect": True,
            "augment": False,
            "batch": int(args.batch),
            "evaluation_batch": 1,
        },
    }
    identity = {class_id: class_id for class_id in class_ids}
    secondary_full_imgsz = int(args.secondary_full_imgsz or args.full_imgsz)
    primary_model = YOLO(str(primary_weight))
    primary_rows, primary_ms = runner["predict_records"](
        primary_model,
        images,
        identity,
        predict_config,
        str(args.device),
        primary_name,
        int(args.full_imgsz),
    )
    secondary_model = YOLO(str(secondary_weight))
    secondary_full_rows, secondary_full_ms = runner["predict_records"](
        secondary_model,
        images,
        identity,
        predict_config,
        str(args.device),
        secondary_full_name,
        secondary_full_imgsz,
    )
    secondary_tile_rows, secondary_tile_ms, tile_audit = predict_tiles(
        secondary_model,
        images,
        args.crop_size,
        float(args.tile_overlap),
        int(args.tile_imgsz),
        int(args.batch),
        str(args.device),
        int(args.focus_class),
        float(args.conf),
        float(args.model_iou),
        int(args.max_det),
        runner["evaluation_predictor_class"](),
        secondary_tile_name,
    )
    passthrough_rows: dict[str, list[dict[str, Any]]] = {}
    passthrough_audit = []
    for name, weight, model_imgsz in passthrough_models:
        model = YOLO(str(weight))
        rows, inference_ms = runner["predict_records"](
            model,
            images,
            identity,
            predict_config,
            str(args.device),
            name,
            int(model_imgsz),
        )
        passthrough_rows[name] = rows
        passthrough_audit.append(
            {
                "name": name,
                "weight": str(weight),
                "weight_sha256": sha256_file(weight),
                "imgsz": int(model_imgsz),
                "inference_mode": "full_frame",
                "inference_ms": float(inference_ms),
                "prediction_count": len(rows),
                "predictions_sha256": records_sha256(rows),
            }
        )
    write_jsonl(output / "predictions" / f"{primary_name}.jsonl", primary_rows)
    write_jsonl(output / "predictions" / f"{secondary_full_name}.jsonl", secondary_full_rows)
    write_jsonl(output / "predictions" / f"{secondary_tile_name}.jsonl", secondary_tile_rows)
    for name, rows in passthrough_rows.items():
        write_jsonl(output / "predictions" / f"{name}.jsonl", rows)

    targets = yolo_ground_truth(images, class_ids)
    predictions_by_model = {
        primary_name: primary_rows,
        secondary_full_name: secondary_full_rows,
        secondary_tile_name: secondary_tile_rows,
        **passthrough_rows,
    }
    variants: dict[str, Sequence[Mapping[str, Any]]] = {
        "primary_full": primary_rows,
        "secondary_full": secondary_full_rows,
    }
    secondary_sets = {
        "full": [secondary_full_name],
        "tile": [secondary_tile_name],
        "full_tile": [secondary_full_name, secondary_tile_name],
    }
    for set_name, secondary_names in secondary_sets.items():
        for fusion_iou in args.fusion_ious:
            for scale in args.secondary_scales:
                for bonus in args.agreement_bonuses:
                    for weighted in (False, True):
                        name = (
                            f"{set_name}_iou{fusion_iou:.2f}_scale{scale:.2f}_"
                            f"bonus{bonus:.2f}_wbf{int(weighted)}"
                        )
                        variants[name] = ensemble["fuse_focus_class"](
                            predictions_by_model,
                            primary_name,
                            secondary_names,
                            int(args.focus_class),
                            float(fusion_iou),
                            float(scale),
                            float(bonus),
                            bool(weighted),
                        )

    evaluated = {}
    for name, rows in variants.items():
        metrics = evaluate_ap50(rows, targets, class_ids)
        evaluated[name] = {
            **metrics,
            "prediction_count": len(rows),
            "subgroups": subgroup_metrics(rows, targets, images, class_ids),
        }
    primary_subgroups = evaluated["primary_full"]["subgroups"]
    for metrics in evaluated.values():
        deltas = {
            subgroup: float(item["map50"])
            - float(primary_subgroups[subgroup]["map50"])
            for subgroup, item in metrics["subgroups"].items()
        }
        metrics["subgroup_delta_vs_primary"] = deltas
        metrics["worst_subgroup_delta_vs_primary"] = min(deltas.values()) if deltas else 0.0
        metrics["degraded_subgroup_count"] = sum(delta < -0.01 for delta in deltas.values())
    ranking = sorted(
        evaluated.items(),
        key=lambda item: (
            float(item[1]["map50"]),
            float(item[1]["per_class_ap50"].get(int(args.focus_class), 0.0)),
        ),
        reverse=True,
    )
    selection_scope = args.selection_scope
    if selection_scope == "auto":
        selection_scope = (
            "base_dev_only"
            if source_manifest and source_manifest.get("split_mode") == "external"
            else "base_train_internal_temporal_holdout"
        )
    model_audit = [
        {
            "name": primary_name,
            "weight": str(primary_weight),
            "weight_sha256": sha256_file(primary_weight),
            "imgsz": int(args.full_imgsz),
            "inference_mode": "full_frame",
            "prediction_count": len(primary_rows),
            "predictions_sha256": records_sha256(primary_rows),
        },
        {
            "name": secondary_full_name,
            "weight": str(secondary_weight),
            "weight_sha256": sha256_file(secondary_weight),
            "imgsz": secondary_full_imgsz,
            "inference_mode": "full_frame",
            "prediction_count": len(secondary_full_rows),
            "predictions_sha256": records_sha256(secondary_full_rows),
        },
        {
            "name": secondary_tile_name,
            "weight": str(secondary_weight),
            "weight_sha256": sha256_file(secondary_weight),
            "imgsz": int(args.tile_imgsz),
            "inference_mode": "sliding_window",
            "tile": {**tile_audit, "focus_class_local": int(args.focus_class)},
            "prediction_count": len(secondary_tile_rows),
            "predictions_sha256": records_sha256(secondary_tile_rows),
        },
        *passthrough_audit,
    ]
    report = {
        "schema_version": 1,
        "selection_scope": selection_scope,
        "independent_test_evidence": False,
        "performance_evidence": selection_scope != "final_refit_checkpoint_holdout",
        "lock_data_access": False,
        "split": str(split),
        "split_sha256": sha256_file(split),
        "source_manifest": str(manifest_path) if source_manifest else None,
        "source_split_mode": source_manifest.get("split_mode") if source_manifest else None,
        "image_count": len(images),
        "primary": {
            "name": primary_name,
            "weight": str(primary_weight),
            "weight_sha256": sha256_file(primary_weight),
            "full_inference_ms": primary_ms,
        },
        "secondary": {
            "name": secondary_name,
            "weight": str(secondary_weight),
            "weight_sha256": sha256_file(secondary_weight),
            "full_inference_ms": secondary_full_ms,
            "tile_inference_ms": secondary_tile_ms,
        },
        "models": model_audit,
        "passthrough_models": passthrough_audit,
        "tile_audit": tile_audit,
        "parameters": {
            "full_imgsz": int(args.full_imgsz),
            "secondary_full_imgsz": secondary_full_imgsz,
            "tile_imgsz": int(args.tile_imgsz),
            "batch": int(args.batch),
            "conf": float(args.conf),
            "model_iou": float(args.model_iou),
            "focus_class": int(args.focus_class),
            "class_ids": class_ids,
        },
        "top_variants": [
            {"name": name, **metrics} for name, metrics in ranking[:30]
        ],
        "variants": evaluated,
        "created_unix_time": time.time(),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["top_variants"][:10], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
