#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

from fair_agent.modules.incremental_compliance import verify_new_images_only


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["soldier", "small_aircraft", "warship", "tank"]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    return path.absolute().relative_to(ROOT.absolute()).as_posix()


def read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def read_split(path: Path) -> list[Path]:
    return [resolve(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_label(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def link_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def build_subset(images: Iterable[Path], output: Path, global_new_id: int, keep_backgrounds: bool) -> list[Path]:
    generated = []
    for image in images:
        labels = []
        for line in source_label(image).read_text(encoding="utf-8").splitlines():
            columns = line.split()
            if columns and int(float(columns[0])) == global_new_id:
                labels.append("0 " + " ".join(columns[1:]))
        if not labels and not keep_backgrounds:
            continue
        target_image = output / "images" / image.name
        target_label = output / "labels" / f"{image.stem}.txt"
        link_image(image, target_image)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        target_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        generated.append(target_image)
    return generated


def write_split(path: Path, images: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rel(image) for image in images) + "\n", encoding="utf-8")


def build_protocol(config: Mapping[str, Any], protocol: Mapping[str, Any]) -> Path:
    name = str(protocol["name"])
    new_name = str(protocol["new_classes"][0])
    new_id = CLASS_NAMES.index(new_name)
    output = resolve(config["specialist_output_root"]) / name
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite specialist dataset: {output}")
    output.mkdir(parents=True)
    train_source = read_split(resolve(protocol["new_train_split"]))
    val_source = read_split(resolve(protocol["new_val_split"]))
    test_source = read_split(resolve(protocol["full_test_split"]))
    train = build_subset(train_source, output / "train", new_id, keep_backgrounds=False)
    val = build_subset(val_source, output / "val", new_id, keep_backgrounds=False)
    test = build_subset(test_source, output / "test", new_id, keep_backgrounds=True)
    train_list = output / "splits" / "train.txt"
    val_list = output / "splits" / "val.txt"
    test_list = output / "splits" / "test.txt"
    write_split(train_list, train)
    write_split(val_list, val)
    write_split(test_list, test)
    dataset = {"path": ".", "train": rel(train_list), "val": rel(val_list), "test": rel(test_list), "names": {0: new_name}}
    dataset_path = output / "specialist_dataset.yaml"
    dataset_path.write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True), encoding="utf-8")
    compliance = verify_new_images_only(train, train_source)
    if not compliance["compliant"]:
        raise RuntimeError(f"Specialist dataset compliance failed: {compliance}")
    manifest = {
        "protocol": name,
        "new_class": new_name,
        "global_class_id": new_id,
        "specialist_class_id": 0,
        "training_source_policy": "new_incremental_images_only",
        "train_images": len(train),
        "val_images": len(val),
        "test_images": len(test),
        "compliance": compliance,
        "dataset_yaml": rel(dataset_path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dataset_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "incremental_no_old_distill_yolo11s.yaml")
    parser.add_argument("--protocol", action="append")
    args = parser.parse_args()
    config = read_yaml(args.config)
    selected = set(args.protocol or [])
    count = 0
    for protocol in config["protocols"]:
        if selected and protocol["name"] not in selected:
            continue
        print(rel(build_protocol(config, protocol)))
        count += 1
    if not count:
        raise ValueError("No protocol selected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
