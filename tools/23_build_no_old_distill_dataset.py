#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import yaml

from fair_agent.modules.incremental_compliance import verify_class_incremental_learning_scope


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["soldier", "small_aircraft", "warship", "tank"]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    return path.absolute().relative_to(ROOT.absolute()).as_posix()


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def read_split(path: Path) -> list[Path]:
    values = [resolve(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [str(value) for value in values if not value.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split images: {missing[:5]}")
    return values


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        parts[-2] = "labels"
    except IndexError as exc:
        raise ValueError(f"Image path has no images directory: {image}") from exc
    return Path(*parts).with_suffix(".txt")


def read_ground_truth(path: Path, new_ids: set[int]) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        columns = raw.split()
        if len(columns) != 5:
            raise ValueError(f"Invalid YOLO label in {path}: {raw}")
        class_id = int(float(columns[0]))
        if class_id not in new_ids:
            raise ValueError(f"Non-new-class ground truth found in incremental label {path}: {class_id}")
        lines.append(" ".join(columns))
    return lines


def symlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def pseudo_lines(result: Any, old_ids: set[int], max_per_image: int) -> list[str]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    values: list[tuple[float, str]] = []
    for cls_value, conf_value, xywhn in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xywhn.tolist()):
        class_id = int(cls_value)
        if class_id not in old_ids:
            continue
        line = f"{class_id} " + " ".join(f"{float(value):.8f}" for value in xywhn)
        values.append((float(conf_value), line))
    values.sort(key=lambda item: item[0], reverse=True)
    return [line for _, line in values[:max_per_image]]


def write_list(path: Path, values: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rel(value) for value in values) + "\n", encoding="utf-8")


def build_subset(
    model: Any,
    source_images: Sequence[Path],
    subset_dir: Path,
    new_ids: set[int],
    old_ids: set[int],
    predict_args: Mapping[str, Any],
    max_pseudo_per_image: int,
    include_old_pseudo: bool = True,
) -> Dict[str, Any]:
    results = list(model.predict(source=[str(path) for path in source_images], save=False, verbose=False, **predict_args))
    if len(results) != len(source_images):
        raise RuntimeError("Teacher prediction count does not match incremental image count")
    output_images: list[Path] = []
    ground_truth_objects = 0
    pseudo_objects = 0
    images_with_pseudo = 0
    for image, result in zip(source_images, results):
        target_image = subset_dir / "images" / image.name
        target_label = subset_dir / "labels" / f"{image.stem}.txt"
        gt = read_ground_truth(label_for_image(image), new_ids)
        pseudo = pseudo_lines(result, old_ids, max_pseudo_per_image) if include_old_pseudo else []
        symlink_or_copy(image, target_image)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        target_label.write_text("\n".join(gt + pseudo) + "\n", encoding="utf-8")
        output_images.append(target_image)
        ground_truth_objects += len(gt)
        pseudo_objects += len(pseudo)
        images_with_pseudo += int(bool(pseudo))
    return {
        "images": output_images,
        "image_count": len(output_images),
        "new_ground_truth_objects": ground_truth_objects,
        "old_pseudo_objects": pseudo_objects,
        "images_with_old_pseudo": images_with_pseudo,
    }


def write_dataset_yaml(path: Path, train_list: Path, val_list: Path) -> None:
    data = {
        "path": ".",
        "train": rel(train_list),
        "val": rel(val_list),
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_protocol(config: Mapping[str, Any], protocol: Mapping[str, Any], device: str | None) -> Path:
    from ultralytics import YOLO

    name = str(protocol["name"])
    output_dir = resolve(config["output_root"]) / name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite generated protocol: {output_dir}")
    output_dir.mkdir(parents=True)
    new_ids = {CLASS_NAMES.index(value) for value in protocol["new_classes"]}
    old_ids = {CLASS_NAMES.index(value) for value in protocol["base_classes"]}
    train_source = read_split(resolve(protocol["new_train_split"]))
    val_source = read_split(resolve(protocol["new_val_split"]))
    teacher = resolve(protocol["teacher_weight"])
    if not teacher.exists():
        raise FileNotFoundError(f"Teacher weight does not exist: {teacher}")

    predict_args = dict(config.get("teacher_prediction", {}))
    max_pseudo = int(predict_args.pop("max_pseudo_per_image", 100))
    if device is not None:
        predict_args["device"] = device
    model = YOLO(str(teacher))
    train = build_subset(model, train_source, output_dir / "train", new_ids, old_ids, predict_args, max_pseudo)
    val = build_subset(
        model,
        val_source,
        output_dir / "val",
        new_ids,
        old_ids,
        predict_args,
        max_pseudo,
        include_old_pseudo=False,
    )

    train_list = output_dir / "splits" / "train.txt"
    val_list = output_dir / "splits" / "val.txt"
    write_list(train_list, train["images"])
    write_list(val_list, val["images"])
    compliance = verify_class_incremental_learning_scope(
        train["images"],
        val["images"],
        train_source,
        val_source,
        protocol["base_classes"],
        protocol["new_classes"],
        verify_content=True,
    )
    if not compliance["compliant"]:
        raise RuntimeError(f"New-only compliance check failed: {compliance}")
    dataset_yaml = output_dir / "learning_dataset.yaml"
    write_dataset_yaml(dataset_yaml, train_list, val_list)
    manifest = {
        "protocol": name,
        "task_type": "class_incremental_object_detection",
        "method": "new_images_only_teacher_pseudolabel_distillation",
        "teacher_weight": rel(teacher),
        "base_classes": list(protocol["base_classes"]),
        "new_classes": list(protocol["new_classes"]),
        "training_source_policy": "new_incremental_images_only",
        "validation_source_policy": "new_incremental_images_and_ground_truth_only",
        "learning_data_scope": "incremental_dataset_only",
        "old_dataset_access_during_learning": "forbidden",
        "teacher_inference_scope": "incremental_images_only",
        "train": {key: value for key, value in train.items() if key != "images"},
        "val": {key: value for key, value in val.items() if key != "images"},
        "compliance": compliance,
        "learning_dataset_yaml": rel(dataset_yaml),
        "post_freeze_evaluation_split": rel(resolve(protocol["full_test_split"])),
        "post_freeze_evaluation_affects_training": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "incremental_no_old_distill_yolo11s.yaml")
    parser.add_argument("--protocol", action="append")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_yaml(args.config)
    selected = set(args.protocol or [])
    built = 0
    for protocol in config["protocols"]:
        if selected and protocol["name"] not in selected:
            continue
        output = build_protocol(config, protocol, args.device)
        print(rel(output / "manifest.json"))
        built += 1
    if not built:
        raise ValueError("No protocol selected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
