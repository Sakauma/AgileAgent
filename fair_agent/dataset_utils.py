from __future__ import annotations

import csv
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets_r1_base_train"
REPORTS_DIR = ROOT / "reports"
SPLITS_DIR = ROOT / "splits"
CLASSES_FILE = DATASET_DIR / "classes.txt"
IMAGE_EXT = ".png"
EXPECTED_SENSORS = {"ir", "sar"}
EXPECTED_SCENES = {"air", "forest", "sea", "urban"}

METADATA_FIELDS = [
    "image_path",
    "label_path",
    "image_id",
    "sensor",
    "dataset_round",
    "scene",
    "width",
    "height",
    "num_objects",
    "classes_present",
    "class_ids",
    "object_area_min",
    "object_area_mean",
    "object_area_max",
]


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_classes() -> List[str]:
    return [line.strip() for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_png_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG file: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def parse_image_stem(stem: str) -> Dict[str, str]:
    parts = stem.split("_")
    if len(parts) != 5:
        raise ValueError(f"Unexpected dataset filename pattern: {stem}")
    sensor, round_part, base_part, scene, image_id = parts
    dataset_round = f"{round_part}_{base_part}"
    if sensor not in EXPECTED_SENSORS:
        raise ValueError(f"Unexpected sensor '{sensor}' in {stem}")
    if scene not in EXPECTED_SCENES:
        raise ValueError(f"Unexpected scene '{scene}' in {stem}")
    return {
        "sensor": sensor,
        "dataset_round": dataset_round,
        "scene": scene,
        "image_id": image_id,
    }


def validate_label_file(label_path: Path, class_count: int) -> Tuple[List[Dict[str, float]], List[Dict[str, str]]]:
    objects: List[Dict[str, float]] = []
    errors: List[Dict[str, str]] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append({"file": label_path.name, "line": str(line_number), "type": "bad_column_count", "value": line})
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            errors.append({"file": label_path.name, "line": str(line_number), "type": "bad_class_id", "value": line})
            continue
        try:
            x_center, y_center, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append({"file": label_path.name, "line": str(line_number), "type": "bad_float", "value": line})
            continue
        if not 0 <= class_id < class_count:
            errors.append({"file": label_path.name, "line": str(line_number), "type": "class_id_out_of_range", "value": line})
        if any(value < 0 or value > 1 for value in [x_center, y_center, width, height]):
            errors.append({"file": label_path.name, "line": str(line_number), "type": "bbox_value_out_of_range", "value": line})
        objects.append(
            {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "area": width * height,
            }
        )
    return objects, errors


def scan_dataset() -> Dict[str, object]:
    classes = read_classes()
    images = sorted(DATASET_DIR.glob(f"*{IMAGE_EXT}"))
    labels = sorted(path for path in DATASET_DIR.glob("*.txt") if path.name != "classes.txt")
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []
    distribution = {
        "sensor": Counter(),
        "scene": Counter(),
        "sensor_scene": Counter(),
        "class_objects": Counter(),
        "class_images": Counter(),
        "objects_per_image": Counter(),
        "class_by_sensor": defaultdict(Counter),
        "class_by_scene": defaultdict(Counter),
    }

    for stem in sorted(image_stems - label_stems):
        errors.append({"file": f"{stem}{IMAGE_EXT}", "line": "", "type": "missing_label", "value": stem})
    for stem in sorted(label_stems - image_stems):
        errors.append({"file": f"{stem}.txt", "line": "", "type": "missing_image", "value": stem})

    for image_path in images:
        label_path = DATASET_DIR / f"{image_path.stem}.txt"
        try:
            parsed = parse_image_stem(image_path.stem)
        except ValueError as exc:
            errors.append({"file": image_path.name, "line": "", "type": "bad_filename", "value": str(exc)})
            continue
        try:
            width, height = read_png_size(image_path)
        except ValueError as exc:
            errors.append({"file": image_path.name, "line": "", "type": "bad_png", "value": str(exc)})
            width, height = 0, 0
        objects, label_errors = validate_label_file(label_path, len(classes)) if label_path.exists() else ([], [])
        errors.extend(label_errors)

        class_ids = sorted({int(obj["class_id"]) for obj in objects})
        class_names = [classes[class_id] for class_id in class_ids if 0 <= class_id < len(classes)]
        areas = [float(obj["area"]) for obj in objects]
        sensor = parsed["sensor"]
        scene = parsed["scene"]

        distribution["sensor"][sensor] += 1
        distribution["scene"][scene] += 1
        distribution["sensor_scene"][f"{sensor}/{scene}"] += 1
        distribution["objects_per_image"][len(objects)] += 1
        for class_name in class_names:
            distribution["class_images"][class_name] += 1
        for obj in objects:
            class_id = int(obj["class_id"])
            if 0 <= class_id < len(classes):
                class_name = classes[class_id]
                distribution["class_objects"][class_name] += 1
                distribution["class_by_sensor"][sensor][class_name] += 1
                distribution["class_by_scene"][scene][class_name] += 1

        rows.append(
            {
                "image_path": relpath(image_path),
                "label_path": relpath(label_path),
                "image_id": parsed["image_id"],
                "sensor": sensor,
                "dataset_round": parsed["dataset_round"],
                "scene": scene,
                "width": width,
                "height": height,
                "num_objects": len(objects),
                "classes_present": ";".join(class_names),
                "class_ids": ";".join(str(class_id) for class_id in class_ids),
                "object_area_min": f"{min(areas):.8f}" if areas else "",
                "object_area_mean": f"{sum(areas) / len(areas):.8f}" if areas else "",
                "object_area_max": f"{max(areas):.8f}" if areas else "",
            }
        )

    summary = {
        "total_images": len(images),
        "total_labels": len(labels),
        "total_objects": sum(int(row["num_objects"]) for row in rows),
        "missing_labels": len(image_stems - label_stems),
        "missing_images": len(label_stems - image_stems),
        "invalid_errors": len([err for err in errors if err["type"] not in {"missing_label", "missing_image"}]),
        "classes": classes,
        "distributions": counter_to_plain_dict(distribution),
        "errors": errors,
        "rows": rows,
    }
    return summary


def counter_to_plain_dict(value):
    if isinstance(value, Counter):
        return dict(sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, defaultdict):
        return {key: counter_to_plain_dict(counter) for key, counter in sorted(value.items())}
    if isinstance(value, dict):
        return {key: counter_to_plain_dict(item) for key, item in value.items()}
    return value


def write_metadata(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METADATA_FIELDS})


def read_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def count_values(rows: Iterable[Dict[str, str]], field: str) -> Counter:
    counter = Counter()
    for row in rows:
        counter[row[field]] += 1
    return counter


def count_class_presence(rows: Iterable[Dict[str, str]]) -> Counter:
    counter = Counter()
    for row in rows:
        for class_name in row["classes_present"].split(";"):
            if class_name:
                counter[class_name] += 1
    return counter
