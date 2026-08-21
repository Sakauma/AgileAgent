#!/usr/bin/env python3
"""Run one reproducible Base-model seed queue for the formal 4+2 dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


CLASS_NAMES = {
    0: "soldier",
    1: "small_aircraft",
    2: "warship",
    3: "tank",
}


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(seeds) < 2:
        raise argparse.ArgumentTypeError("至少需要两个训练随机种子")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("训练随机种子不能重复")
    return seeds


def parse_batch(value: str) -> int | float:
    """Accept either an explicit batch size or an AutoBatch memory fraction."""
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
        label = image.with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(f"训练标签不存在：{label}")
        images.append(image)

    if not images:
        raise ValueError(f"划分为空：{split_path}")
    if len(images) != len(set(images)):
        raise ValueError(f"划分包含重复图像：{split_path}")
    return images


def validate_labels(images: list[Path]) -> dict[str, int]:
    objects = 0
    per_class = {name: 0 for name in CLASS_NAMES.values()}
    for image in images:
        for line_number, raw in enumerate(
            image.with_suffix(".txt").read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            value = raw.strip()
            if not value:
                continue
            parts = value.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{image.with_suffix('.txt')}:{line_number} 不是五列 YOLO 标签"
                )
            class_id = int(parts[0])
            if class_id not in CLASS_NAMES:
                raise ValueError(
                    f"{image.with_suffix('.txt')}:{line_number} 类别越界：{class_id}"
                )
            coords = [float(item) for item in parts[1:]]
            if not all(0.0 <= item <= 1.0 for item in coords):
                raise ValueError(
                    f"{image.with_suffix('.txt')}:{line_number} 坐标越界"
                )
            objects += 1
            per_class[CLASS_NAMES[class_id]] += 1
    return {"objects": objects, **per_class}


def materialize_dataset(data_root: Path, project: Path, model_tag: str) -> Path:
    train_images = resolve_split(data_root, "base_train.txt")
    dev_images = resolve_split(data_root, "base_dev.txt")
    overlap = set(train_images) & set(dev_images)
    if overlap:
        raise ValueError(f"Base train/dev 重复 {len(overlap)} 张图像")

    control = project / "_control" / model_tag
    control.mkdir(parents=True, exist_ok=True)
    train_txt = control / "base_train_absolute.txt"
    dev_txt = control / "base_dev_absolute.txt"
    train_txt.write_text(
        "\n".join(path.as_posix() for path in train_images) + "\n",
        encoding="utf-8",
    )
    dev_txt.write_text(
        "\n".join(path.as_posix() for path in dev_images) + "\n",
        encoding="utf-8",
    )

    dataset_yaml = control / "base_4plus2.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": data_root.as_posix(),
                "train": train_txt.as_posix(),
                "val": dev_txt.as_posix(),
                "names": CLASS_NAMES,
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
        "train_images": len(train_images),
        "dev_images": len(dev_images),
        "train_labels": validate_labels(train_images),
        "dev_labels": validate_labels(dev_images),
        "lock_used": False,
    }
    (control / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset_yaml


def metric_dict(result: Any) -> dict[str, float]:
    values = getattr(result, "results_dict", {}) or {}
    return {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, (int, float))
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="训练一个模型规格的多随机种子 4+2 Base 队列。"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("3407,20260821,8675309"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=parse_batch, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    project = args.project.expanduser().resolve()
    model_path = Path(args.model).expanduser()
    if model_path.suffix == ".pt" and model_path.parent != Path("."):
        model_value = str(model_path.resolve())
        if not Path(model_value).is_file():
            raise FileNotFoundError(f"初始化权重不存在：{model_value}")
    else:
        model_value = args.model

    dataset_yaml = materialize_dataset(data_root, project, args.model_tag)
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

    summary_path = project / f"{args.model_tag}_queue_summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "model": model_value,
        "model_tag": args.model_tag,
        "visible_gpu": torch.cuda.get_device_name(0),
        "seeds": args.seeds,
        "settings": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "epochs": args.epochs,
            "patience": args.patience,
            "workers": args.workers,
            "optimizer": "AdamW",
            "selection_primary": "Base dev metrics/mAP50(B)",
            "selection_secondary": "Base dev metrics/mAP50-95(B)",
            "lock_used": False,
        },
        "runs": [],
    }
    write_summary(summary_path, summary)

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
                }
            )
            write_summary(summary_path, summary)
            continue

        started = time.monotonic()
        print(
            json.dumps(
                {
                    "event": "base_training_start",
                    "model": args.model_tag,
                    "seed": seed,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "epochs": args.epochs,
                    "patience": args.patience,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            last_weight = run_dir / "weights" / "last.pt"
            if last_weight.is_file():
                result = YOLO(str(last_weight)).train(resume=True)
                resumed = True
            else:
                result = YOLO(model_value).train(
                    data=str(dataset_yaml),
                    project=str(project),
                    name=name,
                    exist_ok=True,
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
                    cache="ram",
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
            write_summary(summary_path, summary)
            raise
        summary["runs"].append(row)
        write_summary(summary_path, summary)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print(summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
