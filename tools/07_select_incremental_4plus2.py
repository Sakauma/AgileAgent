#!/usr/bin/env python3
"""Re-evaluate 4+2 specialist candidates on dev and select by mAP50."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


CLASS_NAMES = {0: "patrol_boat", 1: "armored_vehicle"}


def parse_csv(value: str) -> list[str]:
    rows = [item.strip() for item in value.split(",") if item.strip()]
    if not rows:
        raise argparse.ArgumentTypeError("列表不能为空")
    if len(rows) != len(set(rows)):
        raise argparse.ArgumentTypeError("列表不能包含重复值")
    return rows


def parse_seeds(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_candidate(
    weight: Path,
    model_tag: str,
    seed: int,
    dataset_yaml: Path,
    output_dir: Path,
    device: str,
    imgsz: int,
    batch: int,
    workers: int,
) -> dict[str, Any]:
    from ultralytics import YOLO

    name = f"{model_tag}_seed{seed}"
    print(
        json.dumps(
            {
                "event": "incremental_candidate_validation_start",
                "candidate": name,
                "weight": weight.as_posix(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    result = YOLO(str(weight)).val(
        data=str(dataset_yaml),
        split="val",
        device=device,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        conf=0.001,
        iou=0.70,
        max_det=300,
        rect=True,
        augment=False,
        plots=False,
        save_json=False,
        project=str(output_dir / "validation"),
        name=name,
        exist_ok=True,
        verbose=True,
    )
    ap50 = [float(value) for value in result.box.ap50]
    per_class = {
        CLASS_NAMES[index]: ap50[index]
        for index in sorted(CLASS_NAMES)
        if index < len(ap50)
    }
    if set(per_class) != set(CLASS_NAMES.values()):
        raise RuntimeError(f"候选没有返回完整的二类 AP50：{per_class}")
    row = {
        "model_tag": model_tag,
        "seed": seed,
        "weight": weight.as_posix(),
        "map50": float(result.box.map50),
        "minimum_class_ap50": min(per_class.values()),
        "map50_95": float(result.box.map),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "per_class_ap50": per_class,
        "speed_ms": {key: float(value) for key, value in result.speed.items()},
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def write_markdown(path: Path, rows: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    lines = [
        "# 4+2 二类增量专家复评排名",
        "",
        "选择口径：固定 Increment dev 的 mAP50 主排序，最弱新增类 AP50 次排序；未读取 lock。",
        "",
        "| 排名 | 初始化 | seed | mAP50 | 最弱类 AP50 | mAP50-95 | Precision | Recall |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | {row['model_tag']} | {row['seed']} | "
            f"{row['map50']:.6f} | {row['minimum_class_ap50']:.6f} | "
            f"{row['map50_95']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"最终选择：`{selected['model_tag']}`，seed `{selected['seed']}`，"
            f"Increment dev mAP50 `{selected['map50']:.6f}`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="复评并选择 4+2 二类增量专家。")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--model-tag", default="yolo26s_generic")
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=18)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    project = args.project.expanduser().resolve()
    dataset_yaml = args.dataset_yaml.expanduser().resolve()
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"二类派生数据配置不存在：{dataset_yaml}")

    candidates: list[tuple[int, Path]] = []
    for seed in args.seeds:
        weight = project / f"{args.model_tag}_seed{seed}" / "weights" / "best.pt"
        results = project / f"{args.model_tag}_seed{seed}" / "results.csv"
        if not weight.is_file() or not results.is_file():
            raise FileNotFoundError(f"增量候选尚未完成：{args.model_tag}/seed={seed}")
        candidates.append((seed, weight))

    output_dir = project / "selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_primary": "Increment dev mAP50",
        "selection_secondary": "minimum per-class Increment dev AP50",
        "lock_used": False,
        "candidates": [],
    }
    progress_path = output_dir / "incremental_selection.json"
    atomic_json(progress_path, payload)
    for seed, weight in candidates:
        payload["candidates"].append(
            validate_candidate(
                weight,
                args.model_tag,
                seed,
                dataset_yaml,
                output_dir,
                args.device,
                args.imgsz,
                args.batch,
                args.workers,
            )
        )
        atomic_json(progress_path, payload)

    ranked = sorted(
        payload["candidates"],
        key=lambda row: (
            row["map50"],
            row["minimum_class_ap50"],
            row["map50_95"],
            row["recall"],
        ),
        reverse=True,
    )
    selected = ranked[0]
    selected_dir = output_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_weight = selected_dir / "best_incremental.pt"
    shutil.copy2(selected["weight"], selected_weight)
    source_run = Path(selected["weight"]).parents[1]
    for filename in ("args.yaml", "results.csv"):
        source = source_run / filename
        if source.is_file():
            shutil.copy2(source, selected_dir / filename)

    payload["ranking"] = ranked
    payload["selected"] = {**selected, "promoted_weight": selected_weight.as_posix()}
    atomic_json(progress_path, payload)
    write_markdown(output_dir / "incremental_selection.md", ranked, selected)
    print(json.dumps(payload["selected"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
