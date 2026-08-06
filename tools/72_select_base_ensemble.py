#!/usr/bin/env python3
"""Select a conservative base-detector ensemble using base-dev only.

The script deliberately accepts an explicit dev image list and never opens a
test/lock split.  The first ``--model`` is the primary detector.  Predictions
from additional detectors are considered only for ``--focus-class`` (soldier
in the current 3+1 protocol); all other classes remain byte-for-byte owned by
the primary detector.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import runpy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import (
    evaluate_ap50,
    read_split,
    sha256_file,
    yolo_ground_truth,
)


def parse_model(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("--model 必须使用 name=/absolute/path/to/best.pt")
    if not all(character.isalnum() or character in "._-" for character in name):
        raise argparse.ArgumentTypeError(f"模型名称包含非法字符：{name}")
    return name, Path(raw_path).expanduser().resolve()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def records_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    return hashlib.sha256((payload + ("\n" if rows else "")).encode("utf-8")).hexdigest()


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def cluster_rows(rows: Sequence[Mapping[str, Any]], iou_threshold: float) -> list[list[Dict[str, Any]]]:
    """Greedily cluster same-image, same-class boxes across detector owners."""
    grouped: dict[tuple[str, int], list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["image_id"]), int(row["class_id"]))].append(dict(row))
    clusters: list[list[Dict[str, Any]]] = []
    for key in sorted(grouped):
        class_rows = sorted(grouped[key], key=lambda row: -float(row["confidence"]))
        class_clusters: list[list[Dict[str, Any]]] = []
        for candidate in class_rows:
            best_index = None
            best_iou = -1.0
            for index, cluster in enumerate(class_clusters):
                overlap = max(box_iou(candidate["xyxy"], member["xyxy"]) for member in cluster)
                if overlap >= iou_threshold and overlap > best_iou:
                    best_index = index
                    best_iou = overlap
            if best_index is None:
                class_clusters.append([candidate])
            else:
                class_clusters[best_index].append(candidate)
        clusters.extend(class_clusters)
    return clusters


def fuse_focus_class(
    predictions_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    primary: str,
    secondary_names: Sequence[str],
    focus_class: int,
    iou_threshold: float,
    secondary_scale: float,
    agreement_bonus: float,
    weighted_boxes: bool,
) -> list[Dict[str, Any]]:
    """Keep non-focus classes from primary and conservatively fuse the focus class."""
    primary_rows = [dict(row) for row in predictions_by_model[primary]]
    output = [row for row in primary_rows if int(row["class_id"]) != int(focus_class)]
    focus_rows: list[Dict[str, Any]] = []
    for row in primary_rows:
        if int(row["class_id"]) == int(focus_class):
            focus_rows.append({**row, "owner": primary, "model_scale": 1.0})
    for name in secondary_names:
        for row in predictions_by_model[name]:
            if int(row["class_id"]) == int(focus_class):
                focus_rows.append({**row, "owner": name, "model_scale": float(secondary_scale)})

    for cluster in cluster_rows(focus_rows, float(iou_threshold)):
        owners = {str(row["owner"]) for row in cluster}
        scaled_scores = [
            float(row["confidence"]) * float(row["model_scale"]) for row in cluster
        ]
        best_index = max(range(len(cluster)), key=lambda index: scaled_scores[index])
        best = dict(cluster[best_index])
        if weighted_boxes and len(owners) > 1:
            weights = [
                max(1e-9, float(row["confidence"]) * float(row["model_scale"]))
                for row in cluster
            ]
            weight_sum = sum(weights)
            best["xyxy"] = [
                sum(float(row["xyxy"][axis]) * weight for row, weight in zip(cluster, weights))
                / weight_sum
                for axis in range(4)
            ]
        best["confidence"] = min(
            1.0,
            max(scaled_scores) + float(agreement_bonus) * max(0, len(owners) - 1),
        )
        best["source"] = "base_dev_ensemble"
        best["ensemble_owners"] = sorted(owners)
        best.pop("owner", None)
        best.pop("model_scale", None)
        output.append(best)
    output.sort(
        key=lambda row: (
            str(row["image_id"]),
            int(row["class_id"]),
            -float(row["confidence"]),
        )
    )
    return output


def float_grid(raw: str) -> list[float]:
    values = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not values:
        raise argparse.ArgumentTypeError("数值网格不能为空")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True, help="仅含 base_dev 图像的 txt 清单")
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--model-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="对尚未命中缓存的模型启用 Ultralytics 增强推理；缓存预测保持原样。",
    )
    parser.add_argument("--focus-class", type=int, default=0)
    parser.add_argument("--class-ids", default="0,1,2")
    parser.add_argument("--max-secondary", type=int, default=2)
    parser.add_argument("--fusion-ious", type=float_grid, default=float_grid("0.45,0.55,0.65"))
    parser.add_argument(
        "--secondary-scales", type=float_grid, default=float_grid("0.35,0.50,0.65,0.80,1.00")
    )
    parser.add_argument(
        "--agreement-bonuses", type=float_grid, default=float_grid("0.00,0.05,0.10,0.15")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    split = args.split.expanduser().resolve()
    lowered = str(split).lower()
    if "mixed_test" in lowered or "lock" in lowered or "base_test" in lowered:
        raise ValueError("集成选择只允许读取 base_dev，拒绝 test/lock 清单")
    images = read_split(split)
    if not images or len({path.stem for path in images}) != len(images):
        raise ValueError("base_dev 清单必须非空且图像 stem 唯一")
    models = list(args.model)
    names = [name for name, _path in models]
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("至少需要两个名称唯一的基础检测器")
    for _name, path in models:
        if not path.is_file():
            raise FileNotFoundError(path)

    class_ids = [int(value) for value in args.class_ids.split(",")]
    if int(args.focus_class) not in class_ids:
        raise ValueError("focus-class 必须属于 class-ids")
    ground_truth = yolo_ground_truth(images)
    output = args.output.expanduser().resolve()
    cache_dir = output / "predictions"
    runner = runpy.run_path(str(ROOT / "tools" / "70_run_strict_3plus1.py"))
    from ultralytics import YOLO

    predict_config = {
        "common": {"imgsz": int(args.imgsz)},
        "predict": {
            "conf": float(args.conf),
            "iou": float(args.model_iou),
            "max_det": int(args.max_det),
            "rect": True,
            "augment": bool(args.augment),
            "batch": int(args.batch),
            "evaluation_batch": int(args.batch),
        },
    }
    mapping = {class_id: class_id for class_id in class_ids}
    predictions_by_model: dict[str, list[Dict[str, Any]]] = {}
    model_audit = []
    for name, weight in models:
        cache_path = cache_dir / f"{name}.jsonl"
        if args.reuse_cache and cache_path.is_file():
            rows = read_jsonl(cache_path)
            inference_ms = None
        else:
            detector = YOLO(str(weight))
            rows, inference_ms = runner["predict_records"](
                detector,
                images,
                mapping,
                predict_config,
                str(args.device),
                name,
                int(args.imgsz),
            )
            write_jsonl(cache_path, rows)
        predictions_by_model[name] = rows
        metrics = evaluate_ap50(rows, ground_truth, class_ids)
        model_audit.append(
            {
                "name": name,
                "weight": str(weight),
                "weight_sha256": sha256_file(weight),
                "prediction_count": len(rows),
                "predictions_sha256": records_sha256(rows),
                "inference_ms": inference_ms,
                "map50": float(metrics["map50"]),
                "per_class_ap50": metrics["per_class_ap50"],
            }
        )

    primary = names[0]
    secondaries = names[1:]
    candidates = []
    maximum = min(max(1, int(args.max_secondary)), len(secondaries))
    for secondary_count in range(1, maximum + 1):
        for selected in itertools.combinations(secondaries, secondary_count):
            for iou_threshold, scale, bonus, weighted in itertools.product(
                args.fusion_ious,
                args.secondary_scales,
                args.agreement_bonuses,
                (False, True),
            ):
                rows = fuse_focus_class(
                    predictions_by_model,
                    primary,
                    selected,
                    int(args.focus_class),
                    iou_threshold,
                    scale,
                    bonus,
                    weighted,
                )
                metrics = evaluate_ap50(rows, ground_truth, class_ids)
                candidates.append(
                    {
                        "primary": primary,
                        "secondary": list(selected),
                        "focus_class": int(args.focus_class),
                        "fusion_iou": float(iou_threshold),
                        "secondary_scale": float(scale),
                        "agreement_bonus": float(bonus),
                        "weighted_boxes": bool(weighted),
                        "map50": float(metrics["map50"]),
                        "per_class_ap50": metrics["per_class_ap50"],
                        "prediction_count": len(rows),
                        "predictions_sha256": records_sha256(rows),
                    }
                )
    candidates.sort(
        key=lambda row: (
            -float(row["map50"]),
            len(row["secondary"]),
            float(row["agreement_bonus"]),
            -float(row["fusion_iou"]),
            str(row["secondary"]),
        )
    )
    best = candidates[0]
    best_rows = fuse_focus_class(
        predictions_by_model,
        primary,
        best["secondary"],
        int(args.focus_class),
        float(best["fusion_iou"]),
        float(best["secondary_scale"]),
        float(best["agreement_bonus"]),
        bool(best["weighted_boxes"]),
    )
    write_jsonl(output / "best_dev_predictions.jsonl", best_rows)
    report = {
        "schema_version": 1,
        "selection_scope": "base_dev_only",
        "lock_data_access": False,
        "split": str(split),
        "split_sha256": sha256_file(split),
        "image_count": len(images),
        "imgsz": int(args.imgsz),
        "augment_for_uncached_models": bool(args.augment),
        "primary": primary,
        "focus_class": int(args.focus_class),
        "models": model_audit,
        "search_space": {
            "max_secondary": maximum,
            "fusion_ious": args.fusion_ious,
            "secondary_scales": args.secondary_scales,
            "agreement_bonuses": args.agreement_bonuses,
            "weighted_boxes": [False, True],
            "candidate_count": len(candidates),
        },
        "best": best,
        "top_candidates": candidates[:50],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
