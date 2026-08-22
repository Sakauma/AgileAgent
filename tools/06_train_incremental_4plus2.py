#!/usr/bin/env python3
"""Train one registry-selected class-incremental round without Base replay."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.incremental_round_registry import (  # noqa: E402
    DEFAULT_ROUND_REGISTRY,
    introduced_class_names,
    load_incremental_round_registry,
    select_round,
)


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


def resolve_split(data_root: Path, split_reference: str) -> list[Path]:
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
        reference = Path(value)
        image = reference if reference.is_absolute() else data_root / reference
        image = image.resolve()
        try:
            image.relative_to(data_root.resolve())
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


def projected_labels(
    source: Path,
    global_to_local: dict[int, int],
    local_class_names: dict[int, str],
) -> tuple[list[str], dict[int, int]]:
    output: list[str] = []
    counts = {local_id: 0 for local_id in local_class_names}
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
        if global_id not in global_to_local:
            continue
        local_id = global_to_local[global_id]
        output.append(f"{local_id} {' '.join(parts[1:])}")
        counts[local_id] += 1
    return output, counts


def link_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        raise FileExistsError(f"拒绝覆盖派生图像：{target}")
    target.symlink_to(source.resolve())


def materialize_dataset(
    data_root: Path,
    project: Path,
    queue_tag: str,
    registry: dict[str, Any],
    round_spec: dict[str, Any],
) -> Path:
    round_id = str(round_spec["round_id"])
    control = project / "_control" / queue_tag / round_id
    dataset_yaml = control / "incremental_round.yaml"
    manifest_path = control / "dataset_manifest.json"
    local_to_global = {
        int(key): int(value)
        for key, value in round_spec["specialist"]["local_to_global"].items()
    }
    global_to_local = {
        global_id: local_id for local_id, global_id in local_to_global.items()
    }
    global_names = introduced_class_names(registry, round_spec["new_class_ids"])
    local_class_names = {
        local_id: global_names[global_id]
        for local_id, global_id in local_to_global.items()
    }
    serialized_mapping = {
        str(global_id): local_id for global_id, local_id in global_to_local.items()
    }
    if dataset_yaml.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("data_root") != data_root.as_posix()
            or manifest.get("round_id") != round_id
            or manifest.get("global_to_local") != serialized_mapping
        ):
            raise ValueError(f"已有派生数据视图与本次参数不一致：{control}")
        return dataset_yaml
    if control.exists() and any(control.iterdir()):
        raise FileExistsError(f"拒绝覆盖不完整的派生数据视图：{control}")

    split_specs = {
        "train": str(round_spec["splits"]["train"]),
        "val": str(round_spec["splits"]["dev"]),
    }
    split_counts: dict[str, Any] = {}
    for target_split, source_split in split_specs.items():
        images = resolve_split(data_root, source_split)
        class_counts = {local_id: 0 for local_id in local_class_names}
        selected_images = 0
        for source in images:
            rows, counts = projected_labels(
                source, global_to_local, local_class_names
            )
            if not rows:
                raise ValueError(
                    f"轮次清单包含不属于 {round_id} 的图像：{source}"
                )
            target_image = control / "dataset" / "images" / target_split / source.name
            target_label = (
                control / "dataset" / "labels" / target_split / f"{source.stem}.txt"
            )
            link_image(source, target_image)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text("\n".join(rows) + "\n", encoding="utf-8")
            selected_images += 1
            for class_id, count in counts.items():
                class_counts[class_id] += count
        if any(count == 0 for count in class_counts.values()):
            raise ValueError(f"{source_split} 未覆盖本轮全部新增类别：{class_counts}")
        split_counts[target_split] = {
            "source_images": len(images),
            "selected_images": selected_images,
            "objects": {
                local_class_names[class_id]: count
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
                "names": local_class_names,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "phase": "incremental_learning",
        "counted_as_incremental_learning": True,
        "detector_weights_updated": False,
        "data_root": data_root.as_posix(),
        "round_registry": Path(registry["path"]).as_posix(),
        "round_id": round_id,
        "round_index": int(round_spec["round_index"]),
        "parent_generation_id": round_spec["parent_generation_id"],
        "generation_id": round_spec["generation_id"],
        "new_class_ids": list(round_spec["new_class_ids"]),
        "old_class_ids": list(round_spec["old_class_ids"]),
        "source_scope": "incremental_dataset_only",
        "old_raw_image_count": 0,
        "old_raw_label_count": 0,
        "original_labels_modified": False,
        "image_selector": "contains_current_round_class",
        "label_projection": "current_round_classes_only",
        "global_to_local": serialized_mapping,
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
        description="按轮次注册表训练正式类别增量专家多随机种子队列。"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--round-registry", type=Path, default=ROOT / DEFAULT_ROUND_REGISTRY
    )
    parser.add_argument("--round-id", required=True)
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
    registry = load_incremental_round_registry(args.round_registry)
    round_spec = select_round(registry, args.round_id)
    dataset_yaml = materialize_dataset(
        data_root, project, args.queue_tag, registry, round_spec
    )
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

    summary_path = (
        project
        / f"{round_spec['round_id']}_{args.model_tag}_{args.queue_tag}_summary.json"
    )
    summary: dict[str, Any] = {
        "schema_version": 2,
        "phase": "incremental_learning",
        "counted_as_incremental_learning": True,
        "detector_weights_updated": [round_spec["specialist"]["model_id"]],
        "round_registry": Path(registry["path"]).as_posix(),
        "round_id": round_spec["round_id"],
        "round_index": round_spec["round_index"],
        "parent_generation_id": round_spec["parent_generation_id"],
        "generation_id": round_spec["generation_id"],
        "new_class_ids": round_spec["new_class_ids"],
        "old_class_ids": round_spec["old_class_ids"],
        "learned_class_ids": round_spec["learned_class_ids"],
        "specialist_model_id": round_spec["specialist"]["model_id"],
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
            "validation_data_scope": "incremental_dataset_only",
            "old_raw_image_count": 0,
            "old_raw_label_count": 0,
            "base_detector_weights_frozen": True,
            "old_expert_weights_frozen": True,
            "scene_sensor_is_incremental_learner": False,
            "lock_used": False,
        },
        "runs": [],
    }
    atomic_json(summary_path, summary)

    for seed in args.seeds:
        name = f"{round_spec['round_id']}_{args.model_tag}_seed{seed}"
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
                    "round_id": round_spec["round_id"],
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
