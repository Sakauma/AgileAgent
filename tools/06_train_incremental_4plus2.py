#!/usr/bin/env python3
"""Train one multi-seed two-class specialist queue for the formal 4+2 split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any

import yaml


GLOBAL_NEW_TO_LOCAL = {4: 0, 5: 1}
LOCAL_CLASS_NAMES = {0: "patrol_boat", 1: "armored_vehicle"}


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("至少需要一个训练随机种子")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("训练随机种子不能重复")
    return seeds


def parse_batch(value: str) -> int | float:
    try:
        batch = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch 必须是正整数或 0 到 1 的显存占比") from exc
    if 0.0 < batch < 1.0:
        return batch
    if batch >= 1.0 and batch.is_integer():
        return int(batch)
    raise argparse.ArgumentTypeError("batch 必须是正整数或 0 到 1 的显存占比")


def resolve_split(data_root: Path, split_name: str) -> list[Path]:
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
        if not image.is_file():
            raise FileNotFoundError(f"划分图像不存在：{image}")
        if not image.with_suffix(".txt").is_file():
            raise FileNotFoundError(f"原始标签不存在：{image.with_suffix('.txt')}")
        images.append(image)
    if not images:
        raise ValueError(f"划分为空：{split_path}")
    if len(images) != len(set(images)):
        raise ValueError(f"划分包含重复图像：{split_path}")
    return images


def projected_labels(source: Path) -> tuple[list[str], dict[int, int]]:
    output: list[str] = []
    counts = {local_id: 0 for local_id in LOCAL_CLASS_NAMES}
    for line_number, raw in enumerate(
        source.with_suffix(".txt").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = raw.strip()
        if not value:
            continue
        parts = value.split()
        if len(parts) != 5:
            raise ValueError(
                f"{source.with_suffix('.txt')}:{line_number} 不是五列 YOLO 标签"
            )
        global_id = int(parts[0])
        coordinates = [float(item) for item in parts[1:]]
        if not all(0.0 <= item <= 1.0 for item in coordinates):
            raise ValueError(
                f"{source.with_suffix('.txt')}:{line_number} 坐标越界"
            )
        if global_id not in GLOBAL_NEW_TO_LOCAL:
            continue
        local_id = GLOBAL_NEW_TO_LOCAL[global_id]
        output.append(f"{local_id} {' '.join(parts[1:])}")
        counts[local_id] += 1
    if not output:
        raise ValueError(f"增量图像没有全局类 4/5 标注：{source}")
    return output, counts


def link_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        raise FileExistsError(f"拒绝覆盖派生图像：{target}")
    target.symlink_to(source.resolve())


def materialize_dataset(data_root: Path, project: Path, queue_tag: str) -> Path:
    control = project / "_control" / queue_tag
    dataset_yaml = control / "incremental_4plus2.yaml"
    manifest_path = control / "dataset_manifest.json"
    if dataset_yaml.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("data_root") != data_root.as_posix()
            or manifest.get("global_to_local") != {"4": 0, "5": 1}
        ):
            raise ValueError(f"已有派生数据视图与本次参数不一致：{control}")
        return dataset_yaml
    if control.exists() and any(control.iterdir()):
        raise FileExistsError(f"拒绝覆盖不完整的派生数据视图：{control}")

    split_specs = {"train": "increment_train.txt", "val": "increment_dev.txt"}
    split_counts: dict[str, Any] = {}
    for target_split, source_split in split_specs.items():
        images = resolve_split(data_root, source_split)
        class_counts = {local_id: 0 for local_id in LOCAL_CLASS_NAMES}
        for source in images:
            target_image = control / "dataset" / "images" / target_split / source.name
            target_label = (
                control / "dataset" / "labels" / target_split / f"{source.stem}.txt"
            )
            link_image(source, target_image)
            rows, counts = projected_labels(source)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text("\n".join(rows) + "\n", encoding="utf-8")
            for class_id, count in counts.items():
                class_counts[class_id] += count
        if any(count == 0 for count in class_counts.values()):
            raise ValueError(f"{source_split} 未覆盖全部两个新增类别：{class_counts}")
        split_counts[target_split] = {
            "images": len(images),
            "objects": {
                LOCAL_CLASS_NAMES[class_id]: count
                for class_id, count in class_counts.items()
            },
        }

    control.mkdir(parents=True, exist_ok=True)
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": (control / "dataset").resolve().as_posix(),
                "train": "images/train",
                "val": "images/val",
                "names": LOCAL_CLASS_NAMES,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "data_root": data_root.as_posix(),
        "source_scope": "incremental_dataset_only",
        "original_labels_modified": False,
        "global_to_local": {"4": 0, "5": 1},
        "splits": split_counts,
        "lock_used": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset_yaml


def configure_map50_checkpointing(trainer: Any) -> None:
    """Make best.pt and EarlyStopping follow the competition mAP50 metric."""
    metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
    box = getattr(metrics, "box", None)
    if box is None:
        raise RuntimeError("Ultralytics validator 尚未初始化，无法按 mAP50 选择权重")

    def map50_fitness(metric: Any) -> float:
        return float(metric.map50)

    box.fitness = MethodType(map50_fitness, box)
    trainer._checkpoint_metric = "metrics/mAP50(B)"


def metric_dict(result: Any) -> dict[str, float]:
    values = getattr(result, "results_dict", {}) or {}
    return {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, (int, float))
    }


def history_summary(results_csv: Path) -> dict[str, Any]:
    if not results_csv.is_file():
        return {"completed_epochs": 0, "best_epoch": None, "best_map50": None}
    with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"completed_epochs": 0, "best_epoch": None, "best_map50": None}
    key = next((name for name in rows[0] if "mAP50(B)" in name), None)
    if key is None:
        return {"completed_epochs": len(rows), "best_epoch": None, "best_map50": None}
    best = max(rows, key=lambda row: float(row[key]))
    return {
        "completed_epochs": len(rows),
        "best_epoch": int(float(best["epoch"])),
        "best_map50": float(best[key]),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="训练正式 4+2 数据的二类增量专家多随机种子队列。"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-tag", default="yolo26s_generic")
    parser.add_argument("--queue-tag", required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=parse_batch, default=18)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    project = args.project.expanduser().resolve()
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"通用预训练权重不存在：{model}")
    dataset_yaml = materialize_dataset(data_root, project, args.queue_tag)
    if args.prepare_only:
        print(dataset_yaml, flush=True)
        return 0

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "每条队列必须通过 CUDA_VISIBLE_DEVICES 只暴露一张物理 GPU，"
            f"当前可见 {torch.cuda.device_count()} 张"
        )

    summary_path = project / f"{args.model_tag}_{args.queue_tag}_summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "model": model.as_posix(),
        "model_tag": args.model_tag,
        "queue_tag": args.queue_tag,
        "visible_gpu": torch.cuda.get_device_name(0),
        "seeds": args.seeds,
        "settings": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "epochs": args.epochs,
            "patience": args.patience,
            "workers": args.workers,
            "checkpoint_metric": "incremental dev mAP50",
            "training_data_scope": "incremental_dataset_only",
            "lock_used": False,
        },
        "runs": [],
    }
    atomic_json(summary_path, summary)

    for seed in args.seeds:
        name = f"{args.model_tag}_seed{seed}"
        run_dir = project / name
        best_weight = run_dir / "weights" / "best.pt"
        results_csv = run_dir / "results.csv"
        if best_weight.is_file() and results_csv.is_file():
            summary["runs"].append(
                {
                    "seed": seed,
                    "name": name,
                    "status": "already_complete",
                    "best_weight": best_weight.as_posix(),
                    **history_summary(results_csv),
                }
            )
            atomic_json(summary_path, summary)
            continue

        started = time.monotonic()
        print(
            json.dumps(
                {
                    "event": "incremental_training_start",
                    "model": args.model_tag,
                    "seed": seed,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "checkpoint_metric": "mAP50",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            last_weight = run_dir / "weights" / "last.pt"
            source = last_weight if last_weight.is_file() else model
            specialist = YOLO(str(source))
            specialist.add_callback(
                "on_pretrain_routine_end", configure_map50_checkpointing
            )
            if last_weight.is_file():
                result = specialist.train(resume=True)
                resumed = True
            else:
                result = specialist.train(
                    data=str(dataset_yaml),
                    project=str(project),
                    name=name,
                    exist_ok=False,
                    device=args.device,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    epochs=args.epochs,
                    workers=args.workers,
                    optimizer="AdamW",
                    lr0=0.001,
                    lrf=0.01,
                    cos_lr=True,
                    warmup_epochs=3.0,
                    weight_decay=0.0005,
                    patience=args.patience,
                    cache=False,
                    amp=True,
                    deterministic=True,
                    seed=seed,
                    close_mosaic=20,
                    mosaic=0.80,
                    mixup=0.0,
                    copy_paste=0.0,
                    degrees=3.0,
                    translate=0.10,
                    scale=0.35,
                    shear=0.0,
                    perspective=0.0,
                    fliplr=0.50,
                    flipud=0.0,
                    hsv_h=0.0,
                    hsv_s=0.0,
                    hsv_v=0.25,
                    multi_scale=0.0,
                    plots=True,
                    save=True,
                    save_period=25,
                    val=True,
                    verbose=True,
                )
                resumed = False
            row = {
                "seed": seed,
                "name": name,
                "status": "complete",
                "resumed": resumed,
                "seconds": time.monotonic() - started,
                "best_weight": best_weight.as_posix(),
                "metrics": metric_dict(result),
                **history_summary(results_csv),
            }
        except Exception as exc:
            row = {
                "seed": seed,
                "name": name,
                "status": "failed",
                "seconds": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
            summary["runs"].append(row)
            atomic_json(summary_path, summary)
            raise
        summary["runs"].append(row)
        atomic_json(summary_path, summary)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print(summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
