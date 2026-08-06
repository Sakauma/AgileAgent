#!/usr/bin/env python3
"""Build a leak-free base dataset with deterministic small-object crops.

The source split may contain both base and incremental images.  Only images
whose labels are a non-empty subset of the configured base classes are
materialized.  Crop images are generated from the training portion only;
validation always contains untouched full frames so that model selection
measures deployment behaviour rather than crop recognition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import (
    GLOBAL_CLASS_NAMES,
    read_split,
    read_yolo_labels,
    sha256_file,
    source_label,
)


BASE_LOCAL_TO_GLOBAL = {0: 0, 1: 1, 2: 3}
GLOBAL_TO_BASE_LOCAL = {value: key for key, value in BASE_LOCAL_TO_GLOBAL.items()}
FORBIDDEN_SPLIT_MARKERS = ("mixed_test", "base_test", "lock")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


Label = tuple[int, float, float, float, float]


def reject_lock_path(path: Path, purpose: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_SPLIT_MARKERS):
        raise ValueError(f"{purpose} 不得引用 test/lock 路径：{path}")


def parse_crop_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)[xX](\d+)", value.strip())
    if not match:
        raise argparse.ArgumentTypeError("--crop-size 必须使用 WIDTHxHEIGHT，例如 320x256")
    width, height = map(int, match.groups())
    if width < 32 or height < 32:
        raise argparse.ArgumentTypeError("裁剪宽高必须至少为 32")
    return width, height


def sequence_position(path: Path) -> tuple[str, int, str]:
    match = re.fullmatch(r"(.+?)_(\d+)", path.stem)
    if not match:
        raise ValueError(f"图像名缺少可审计的序列帧号：{path.name}")
    return match.group(1), int(match.group(2)), path.name


def temporal_holdout(
    images: Sequence[Path], fraction: float
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    if not 0.0 < fraction < 0.5:
        raise ValueError("temporal holdout fraction 必须在 (0, 0.5) 内")
    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        key, _position, _name = sequence_position(image)
        groups[key].append(image)

    train: list[Path] = []
    val: list[Path] = []
    audit: dict[str, Any] = {}
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda path: sequence_position(path)[1:])
        if len(ordered) < 2:
            raise ValueError(f"序列 {key} 只有 {len(ordered)} 帧，无法做时序留出")
        holdout_count = min(len(ordered) - 1, max(1, math.ceil(len(ordered) * fraction)))
        split_at = len(ordered) - holdout_count
        train.extend(ordered[:split_at])
        val.extend(ordered[split_at:])
        audit[key] = {
            "source_count": len(ordered),
            "train_count": split_at,
            "val_count": holdout_count,
            "train_last_frame": sequence_position(ordered[split_at - 1])[1],
            "val_first_frame": sequence_position(ordered[split_at])[1],
        }
    return sorted(train), sorted(val), audit


def recent_sequence_tail(images: Sequence[Path], fraction: float) -> list[Path]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("recent fraction 必须在 (0, 1] 内")
    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        key, _position, _name = sequence_position(image)
        groups[key].append(image)
    recent: list[Path] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda path: sequence_position(path)[1:])
        count = max(1, math.ceil(len(ordered) * fraction))
        recent.extend(ordered[-count:])
    return sorted(recent)


def select_base_images(images: Sequence[Path]) -> tuple[list[Path], dict[str, Any]]:
    known = set(GLOBAL_CLASS_NAMES)
    base_global = set(GLOBAL_TO_BASE_LOCAL)
    selected: list[Path] = []
    excluded_incremental: list[str] = []
    class_image_counts: Counter[int] = Counter()
    seen_stems: set[str] = set()
    for image in images:
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"不支持的图像格式：{image}")
        if not image.is_file():
            raise FileNotFoundError(image)
        if image.stem in seen_stems:
            raise ValueError(f"源清单包含重复 stem：{image.stem}")
        seen_stems.add(image.stem)
        labels = read_yolo_labels(source_label(image))
        classes = {row[0] for row in labels}
        unknown = classes - known
        if unknown:
            raise ValueError(f"源标签包含未知类别：{image.name} -> {sorted(unknown)}")
        if not classes:
            raise ValueError(f"基础数据不允许空标签：{image.name}")
        if not classes <= base_global:
            excluded_incremental.append(image.stem)
            continue
        selected.append(image)
        class_image_counts.update(classes)
    return selected, {
        "source_count": len(images),
        "base_selected_count": len(selected),
        "incremental_excluded_count": len(excluded_incremental),
        "incremental_excluded_stems": sorted(excluded_incremental),
        "base_global_class_image_counts": dict(sorted(class_image_counts.items())),
    }


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source.resolve())
        return "symlink"
    except OSError:
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            shutil.copy2(source, target)
            return "copy"


def remap_labels(labels: Iterable[Label]) -> list[Label]:
    remapped = []
    for class_id, cx, cy, width, height in labels:
        if class_id not in GLOBAL_TO_BASE_LOCAL:
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"YOLO 标签越界：{(class_id, cx, cy, width, height)}")
        remapped.append((GLOBAL_TO_BASE_LOCAL[class_id], cx, cy, width, height))
    return remapped


def write_labels(path: Path, labels: Sequence[Label]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"{class_id} {cx:.10g} {cy:.10g} {width:.10g} {height:.10g}"
        for class_id, cx, cy, width, height in labels
    ]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def deterministic_offset(stem: str, seed: int, span: float) -> tuple[float, float]:
    if span == 0.0:
        return 0.0, 0.0
    digest = hashlib.sha256(f"{seed}:{stem}".encode("utf-8")).digest()
    x_unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    y_unit = int.from_bytes(digest[8:16], "big") / float(2**64 - 1)
    return (2.0 * x_unit - 1.0) * span, (2.0 * y_unit - 1.0) * span


def sliding_starts(length: int, tile: int, requested_overlap: float) -> list[int]:
    if tile >= length:
        return [0]
    if not 0.0 <= requested_overlap < 1.0:
        raise ValueError("crop overlap 必须在 [0, 1) 内")
    maximum_step = max(1.0, tile * (1.0 - requested_overlap))
    interval_count = max(1, math.ceil((length - tile) / maximum_step))
    return [
        int(round(index * (length - tile) / interval_count))
        for index in range(interval_count + 1)
    ]


def crop_window(
    image_size: tuple[int, int],
    labels: Sequence[Label],
    crop_size: tuple[int, int],
    stem: str,
    seed: int,
    jitter_fraction: float,
) -> tuple[int, int, int, int, Label]:
    image_width, image_height = image_size
    crop_width, crop_height = crop_size
    if crop_width > image_width or crop_height > image_height:
        raise ValueError(
            f"裁剪 {crop_width}x{crop_height} 大于原图 {image_width}x{image_height}"
        )
    if not labels:
        raise ValueError("无标签图像不能生成小目标裁剪")
    focus = min(labels, key=lambda row: (row[3] * row[4], row[0], row[1], row[2]))
    _class_id, cx, cy, _width, _height = focus
    dx, dy = deterministic_offset(stem, seed, jitter_fraction)
    desired_x = cx * image_width + dx * crop_width
    desired_y = cy * image_height + dy * crop_height
    left = int(round(desired_x - crop_width / 2.0))
    top = int(round(desired_y - crop_height / 2.0))
    left = min(max(0, left), image_width - crop_width)
    top = min(max(0, top), image_height - crop_height)
    return left, top, left + crop_width, top + crop_height, focus


def crop_labels(
    labels: Sequence[Label],
    image_size: tuple[int, int],
    window: tuple[int, int, int, int],
    min_visible_fraction: float,
) -> list[Label]:
    image_width, image_height = image_size
    left, top, right, bottom = window
    crop_width = right - left
    crop_height = bottom - top
    output: list[Label] = []
    for class_id, cx, cy, width, height in labels:
        x1 = (cx - width / 2.0) * image_width
        y1 = (cy - height / 2.0) * image_height
        x2 = (cx + width / 2.0) * image_width
        y2 = (cy + height / 2.0) * image_height
        clipped_x1 = max(float(left), x1)
        clipped_y1 = max(float(top), y1)
        clipped_x2 = min(float(right), x2)
        clipped_y2 = min(float(bottom), y2)
        visible_width = max(0.0, clipped_x2 - clipped_x1)
        visible_height = max(0.0, clipped_y2 - clipped_y1)
        original_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        visible_area = visible_width * visible_height
        center_inside = left <= cx * image_width <= right and top <= cy * image_height <= bottom
        if (
            not center_inside
            or original_area <= 0.0
            or visible_area / original_area < min_visible_fraction
            or visible_width < 1.0
            or visible_height < 1.0
        ):
            continue
        output.append(
            (
                class_id,
                ((clipped_x1 + clipped_x2) / 2.0 - left) / crop_width,
                ((clipped_y1 + clipped_y2) / 2.0 - top) / crop_height,
                visible_width / crop_width,
                visible_height / crop_height,
            )
        )
    return output


def materialize_full(
    source: Path,
    output_root: Path,
    split: str,
    alias_suffix: str = "",
) -> tuple[Path, list[Label], str]:
    target_name = f"{source.stem}{alias_suffix}{source.suffix.lower()}"
    target_image = output_root / "images" / split / target_name
    target_label = output_root / "labels" / split / f"{Path(target_name).stem}.txt"
    mode = _link_or_copy(source, target_image)
    labels = remap_labels(read_yolo_labels(source_label(source)))
    if not labels:
        raise ValueError(f"基础图像重映射后无标签：{source.name}")
    write_labels(target_label, labels)
    return target_image, labels, mode


def materialize_crop(
    source: Path,
    labels: Sequence[Label],
    output_root: Path,
    crop_size: tuple[int, int],
    seed: int,
    jitter_fraction: float,
    min_visible_fraction: float,
) -> tuple[Path, list[Label], dict[str, Any]]:
    with Image.open(source) as image:
        image.load()
        window_with_focus = crop_window(
            image.size,
            labels,
            crop_size,
            source.stem,
            seed,
            jitter_fraction,
        )
        window = window_with_focus[:4]
        focus = window_with_focus[4]
        labels_out = crop_labels(labels, image.size, window, min_visible_fraction)
        if not labels_out:
            raise RuntimeError(f"裁剪未保留任何目标：{source.name}")
        focus_class = int(focus[0])
        if focus_class not in {row[0] for row in labels_out}:
            raise RuntimeError(f"裁剪丢失最小目标：{source.name}")
        width, height = crop_size
        crop_name = f"{source.stem}__small_{width}x{height}{source.suffix.lower()}"
        target_image = output_root / "images" / "train" / crop_name
        target_label = output_root / "labels" / "train" / f"{Path(crop_name).stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        image.crop(window).save(target_image)
    write_labels(target_label, labels_out)
    return target_image, labels_out, {
        "source_stem": source.stem,
        "crop_stem": target_image.stem,
        "window_xyxy": list(window),
        "focus_class": focus_class,
        "focus_area_fraction": float(focus[3] * focus[4]),
        "label_count": len(labels_out),
    }


def materialize_sliding_crops(
    source: Path,
    labels: Sequence[Label],
    output_root: Path,
    crop_size: tuple[int, int],
    overlap: float,
    min_visible_fraction: float,
) -> tuple[list[Path], list[dict[str, Any]]]:
    target_images: list[Path] = []
    audit: list[dict[str, Any]] = []
    crop_width, crop_height = crop_size
    with Image.open(source) as image:
        image.load()
        image_width, image_height = image.size
        if crop_width > image_width or crop_height > image_height:
            raise ValueError(
                f"裁剪 {crop_width}x{crop_height} 大于原图 "
                f"{image_width}x{image_height}：{source.name}"
            )
        windows = [
            (left, top, left + crop_width, top + crop_height)
            for top in sliding_starts(image_height, crop_height, overlap)
            for left in sliding_starts(image_width, crop_width, overlap)
        ]
        for left, top, right, bottom in windows:
            window = (left, top, right, bottom)
            labels_out = crop_labels(labels, image.size, window, min_visible_fraction)
            crop_name = (
                f"{source.stem}__grid_{crop_width}x{crop_height}_"
                f"x{left:04d}_y{top:04d}{source.suffix.lower()}"
            )
            target_image = output_root / "images" / "train" / crop_name
            target_label = output_root / "labels" / "train" / f"{Path(crop_name).stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            image.crop(window).save(target_image)
            write_labels(target_label, labels_out)
            target_images.append(target_image)
            audit.append(
                {
                    "source_stem": source.stem,
                    "crop_stem": target_image.stem,
                    "window_xyxy": list(window),
                    "focus_class": None,
                    "focus_area_fraction": None,
                    "label_count": len(labels_out),
                }
            )
    return target_images, audit


def write_split(path: Path, images: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(str(image.absolute()) for image in images) + "\n",
        encoding="utf-8",
    )


def count_labels(images: Sequence[Path], output_root: Path) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for image in images:
        relative = image.relative_to(output_root / "images")
        label = output_root / "labels" / relative.with_suffix(".txt")
        counts.update(row[0] for row in read_yolo_labels(label))
    return dict(sorted(counts.items()))


def build_dataset(
    source_split: Path,
    output_root: Path,
    crop_size: tuple[int, int],
    split_mode: str,
    val_source_split: Path | None = None,
    holdout_fraction: float = 0.20,
    seed: int = 20260705,
    jitter_fraction: float = 0.10,
    min_visible_fraction: float = 0.50,
    include_crops: bool = True,
    crop_strategy: str = "smallest",
    crop_overlap: float = 0.20,
    recent_fraction: float = 0.25,
    recent_full_repeats: int = 0,
) -> dict[str, Any]:
    source_split = source_split.resolve()
    output_root = output_root.resolve()
    reject_lock_path(source_split, "source split")
    if val_source_split is not None:
        val_source_split = val_source_split.resolve()
        reject_lock_path(val_source_split, "validation split")
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有小目标数据集：{output_root}")
    if not 0.0 <= jitter_fraction <= 0.25:
        raise ValueError("jitter_fraction 必须在 [0, 0.25] 内")
    if not 0.0 < min_visible_fraction <= 1.0:
        raise ValueError("min_visible_fraction 必须在 (0, 1] 内")
    if crop_strategy not in {"smallest", "sliding"}:
        raise ValueError(f"未知 crop strategy：{crop_strategy}")
    if recent_full_repeats < 0 or recent_full_repeats > 4:
        raise ValueError("recent_full_repeats 必须在 [0, 4] 内")

    source_rows = read_split(source_split)
    base_rows, source_audit = select_base_images(source_rows)
    temporal_audit: dict[str, Any] | None = None
    val_source_audit: dict[str, Any] | None = None
    if split_mode == "temporal":
        if val_source_split is not None:
            raise ValueError("temporal 模式不得传入 --val-source-split")
        train_sources, val_sources, temporal_audit = temporal_holdout(
            base_rows, holdout_fraction
        )
    elif split_mode == "external":
        if val_source_split is None:
            raise ValueError("external 模式必须传入 --val-source-split")
        train_sources = list(base_rows)
        val_rows = read_split(val_source_split)
        val_sources, val_source_audit = select_base_images(val_rows)
    else:
        raise ValueError(f"未知 split mode：{split_mode}")

    train_stems = {path.stem for path in train_sources}
    val_stems = {path.stem for path in val_sources}
    overlap = sorted(train_stems & val_stems)
    if not train_sources or not val_sources or overlap:
        raise ValueError(
            f"基础 train/val 必须非空且 stem 互斥："
            f"train={len(train_sources)} val={len(val_sources)} overlap={overlap}"
        )

    output_root.mkdir(parents=True)
    train_images: list[Path] = []
    val_images: list[Path] = []
    crop_audit: list[dict[str, Any]] = []
    link_modes: Counter[str] = Counter()
    recent_sources = (
        recent_sequence_tail(train_sources, recent_fraction)
        if recent_full_repeats
        else []
    )
    try:
        for source in train_sources:
            full_image, labels, link_mode = materialize_full(source, output_root, "train")
            train_images.append(full_image)
            link_modes[link_mode] += 1
            if include_crops:
                if crop_strategy == "smallest":
                    crop_image, _crop_rows, item = materialize_crop(
                        source,
                        labels,
                        output_root,
                        crop_size,
                        seed,
                        jitter_fraction,
                        min_visible_fraction,
                    )
                    train_images.append(crop_image)
                    crop_audit.append(item)
                else:
                    crop_images, items = materialize_sliding_crops(
                        source,
                        labels,
                        output_root,
                        crop_size,
                        crop_overlap,
                        min_visible_fraction,
                    )
                    train_images.extend(crop_images)
                    crop_audit.extend(items)
        for source in recent_sources:
            for repeat_index in range(1, recent_full_repeats + 1):
                full_image, _labels, link_mode = materialize_full(
                    source,
                    output_root,
                    "train",
                    alias_suffix=f"__recent_r{repeat_index}",
                )
                train_images.append(full_image)
                link_modes[link_mode] += 1
        for source in val_sources:
            full_image, _labels, link_mode = materialize_full(source, output_root, "val")
            val_images.append(full_image)
            link_modes[link_mode] += 1

        train_split = output_root / "splits" / "train.txt"
        val_split = output_root / "splits" / "val.txt"
        write_split(train_split, train_images)
        write_split(val_split, val_images)
        dataset = {
            "path": str(output_root),
            "train": "splits/train.txt",
            "val": "splits/val.txt",
            "nc": 3,
            "names": {
                local_id: GLOBAL_CLASS_NAMES[global_id]
                for local_id, global_id in BASE_LOCAL_TO_GLOBAL.items()
            },
        }
        dataset_path = output_root / "dataset.yaml"
        dataset_path.write_text(
            yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "purpose": "leak_free_category_agnostic_small_object_training",
            "selection_scope": "base_train_and_base_dev_only",
            "lock_data_access": False,
            "source_split": str(source_split),
            "source_split_sha256": sha256_file(source_split),
            "val_source_split": str(val_source_split) if val_source_split else None,
            "val_source_split_sha256": sha256_file(val_source_split) if val_source_split else None,
            "source_audit": source_audit,
            "val_source_audit": val_source_audit,
            "split_mode": split_mode,
            "temporal_holdout_fraction": holdout_fraction if split_mode == "temporal" else None,
            "temporal_sequences": temporal_audit,
            "seed": seed,
            "crop_enabled": include_crops,
            "crop_strategy": crop_strategy,
            "crop_size": {"width": crop_size[0], "height": crop_size[1]},
            "crop_overlap": crop_overlap if crop_strategy == "sliding" else None,
            "recent_fraction": recent_fraction if recent_full_repeats else None,
            "recent_full_repeats": recent_full_repeats,
            "recent_source_count": len(recent_sources),
            "recent_materialized_count": len(recent_sources) * recent_full_repeats,
            "recent_source_stems": [path.stem for path in recent_sources],
            "jitter_fraction": jitter_fraction,
            "min_visible_fraction": min_visible_fraction,
            "train_source_count": len(train_sources),
            "train_full_count": len(train_sources),
            "train_crop_count": len(crop_audit),
            "train_materialized_count": len(train_images),
            "val_source_count": len(val_sources),
            "val_full_count": len(val_images),
            "val_crop_count": 0,
            "source_train_val_overlap": overlap,
            "base_local_to_global": BASE_LOCAL_TO_GLOBAL,
            "train_label_counts": count_labels(train_images, output_root),
            "val_label_counts": count_labels(val_images, output_root),
            "full_image_materialization": dict(sorted(link_modes.items())),
            "crop_records": crop_audit,
            "dataset_yaml": str(dataset_path),
        }
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--crop-size", type=parse_crop_size, required=True)
    parser.add_argument("--split-mode", choices=("temporal", "external"), required=True)
    parser.add_argument("--val-source-split", type=Path)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--jitter-fraction", type=float, default=0.10)
    parser.add_argument("--min-visible-fraction", type=float, default=0.50)
    parser.add_argument("--crop-strategy", choices=("smallest", "sliding"), default="smallest")
    parser.add_argument("--crop-overlap", type=float, default=0.20)
    parser.add_argument("--recent-fraction", type=float, default=0.25)
    parser.add_argument("--recent-full-repeats", type=int, default=0)
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="只生成完整原图，用作与裁剪训练同划分的对照组。",
    )
    args = parser.parse_args()
    manifest = build_dataset(
        args.source_split,
        args.output_root,
        args.crop_size,
        args.split_mode,
        args.val_source_split,
        args.holdout_fraction,
        args.seed,
        args.jitter_fraction,
        args.min_visible_fraction,
        not args.no_crops,
        args.crop_strategy,
        args.crop_overlap,
        args.recent_fraction,
        args.recent_full_repeats,
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
