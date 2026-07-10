#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_gpu_device(value: Any) -> str:
    device = str("0" if value is None else value)
    if not device.isdigit():
        raise ValueError("推理设备必须是非负 GPU 编号")
    return device


def collect_images(source: Path, exts: Iterable[str]) -> list[Path]:
    suffixes = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in exts}
    if source.is_file() and source.suffix.lower() == ".txt":
        images = [resolve_path(line.strip()) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif source.is_file():
        images = [source]
    elif source.is_dir():
        images = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    else:
        raise FileNotFoundError(f"Source does not exist: {source}")
    if not images:
        raise FileNotFoundError(f"No images found from source: {source}")
    missing = [path for path in images if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image paths, first one: {missing[0]}")
    return images


def validate_unique_stems(images: Iterable[Path]) -> None:
    seen = set()
    for image in images:
        if image.stem in seen:
            raise ValueError(f"Duplicate image stem would overwrite label output: {image.stem}")
        seen.add(image.stem)


def format_float(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def format_optional(value: Any) -> str:
    return "null" if value is None else str(value)


def prepare_output_dir(output_dir: Path, allowed_root: Path) -> None:
    resolved = output_dir.resolve()
    root = allowed_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Output must be a new child directory of {root}: {resolved}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    for child in ["labels", "labels_conf"]:
        target = output_dir / child
        target.mkdir(parents=True, exist_ok=True)


def write_txt(path: Path, lines: list[str], write_empty: bool) -> None:
    if lines or write_empty:
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    fields = [
        "image_name",
        "image_path",
        "class_id",
        "class_name",
        "confidence",
        "x_center",
        "y_center",
        "width",
        "height",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def write_readme(
    path: Path,
    config: Mapping[str, Any],
    image_count: int,
    prediction_count: int,
    raw_prediction_count: int,
) -> None:
    min_export_conf = config["output"].get("min_export_conf")
    text = [
        "# 提交预测包",
        "",
        "由 `tools/42_predict_submission.py` 生成。",
        "",
        f"- 模型：`{config['model']}`",
        f"- 输入尺寸：`{config['predict']['imgsz']}`",
        f"- 置信度阈值：`{config['predict']['conf']}`",
        f"- IoU 阈值：`{config['predict']['iou']}`",
        f"- 最大检测数：`{config['predict']['max_det']}`",
        f"- 最小导出置信度：`{format_optional(min_export_conf)}`",
        f"- 图像数：`{image_count}`",
        f"- 导出预测数：`{prediction_count}`",
        f"- 原始预测数：`{raw_prediction_count}`",
        "",
        "## 文件说明",
        "",
        "- `labels/`：每张图像对应一个 YOLO 文本，格式为 `class x_center y_center width height`。",
        "- `labels_conf/`：每张图像对应一个 YOLO 文本，格式为 `class x_center y_center width height confidence`。",
        "- `predictions.csv`：包含归一化框和绝对坐标框的扁平预测表。",
        "- `predictions.json`：使用结构化 JSON 保存相同预测结果。",
        "- `manifest.json`：记录运行元数据和输出文件哈希。",
        "",
        "提交时请采用官方评测器要求的格式。保留 CSV 和 JSON 文件，以便提交格式变化时进行转换。",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def make_zip(zip_path: Path, output_dir: Path, include_names: Iterable[str]) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in include_names:
            path = output_dir / name
            if path.is_dir():
                for item in sorted(path.rglob("*")):
                    if item.is_file():
                        zf.write(item, item.relative_to(output_dir).as_posix())
            elif path.is_file():
                zf.write(path, path.relative_to(output_dir).as_posix())


def run_prediction(config: Mapping[str, Any], source_override: str | None, output_override: str | None, limit: int | None) -> Path:
    source_cfg = dict(config["source"])
    output_cfg = dict(config["output"])
    predict_cfg = dict(config["predict"])
    predict_cfg["device"] = validate_gpu_device(predict_cfg.get("device"))
    source = resolve_path(source_override or source_cfg["path"])
    output_root = resolve_path(output_cfg.get("root", "runs/submission"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{output_cfg.get('package_name', 'submission')}_{run_id}"
    output_dir = resolve_path(output_override) if output_override else output_root / default_name
    images = collect_images(source, source_cfg.get("image_exts", [".png", ".jpg", ".jpeg"]))
    if limit is not None:
        images = images[:limit]
    validate_unique_stems(images)
    prepare_output_dir(output_dir, output_root)

    names = {int(key): str(value) for key, value in dict(config["names"]).items()}
    model_path = resolve_path(config["model"])
    expected_hash = config.get("expected_sha256")
    actual_hash = sha256_file(model_path)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(f"Model SHA256 mismatch: expected {expected_hash}, got {actual_hash}")
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = list(model.predict(
        source=[str(path) for path in images],
        save=False,
        verbose=False,
        **predict_cfg,
    ))
    if len(results) != len(images):
        raise RuntimeError(f"Prediction result count mismatch: images={len(images)} results={len(results)}")

    all_rows: list[Dict[str, Any]] = []
    json_rows: list[Dict[str, Any]] = []
    labels_dir = output_dir / "labels"
    labels_conf_dir = output_dir / "labels_conf"
    write_empty = bool(output_cfg.get("write_empty_txt", True))
    min_export_conf = output_cfg.get("min_export_conf")
    if min_export_conf is not None:
        min_export_conf = float(min_export_conf)
    raw_prediction_count = 0

    for image_path, result in zip(images, results):
        plain_lines: list[str] = []
        conf_lines: list[str] = []
        image_rows: list[Dict[str, Any]] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn.cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()
            cls_values = boxes.cls.cpu().tolist()
            conf_values = boxes.conf.cpu().tolist()
            for norm_box, abs_box, cls_value, conf_value in zip(xywhn, xyxy, cls_values, conf_values):
                raw_prediction_count += 1
                cls_id = int(cls_value)
                x_center, y_center, width, height = [float(value) for value in norm_box]
                conf = float(conf_value)
                if min_export_conf is not None and conf < min_export_conf:
                    continue
                plain_lines.append(
                    " ".join([str(cls_id), *[format_float(v) for v in [x_center, y_center, width, height]]])
                )
                conf_lines.append(
                    " ".join([str(cls_id), *[format_float(v) for v in [x_center, y_center, width, height]], format_float(conf)])
                )
                row = {
                    "image_name": image_path.name,
                    "image_path": rel(image_path),
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": conf,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "x1": float(abs_box[0]),
                    "y1": float(abs_box[1]),
                    "x2": float(abs_box[2]),
                    "y2": float(abs_box[3]),
                }
                image_rows.append(row)
                all_rows.append(row)
        if output_cfg.get("write_plain_yolo_txt", True):
            write_txt(labels_dir / f"{image_path.stem}.txt", plain_lines, write_empty)
        if output_cfg.get("write_conf_yolo_txt", True):
            write_txt(labels_conf_dir / f"{image_path.stem}.txt", conf_lines, write_empty)
        json_rows.append({"image_name": image_path.name, "image_path": rel(image_path), "predictions": image_rows})

    if output_cfg.get("write_csv", True):
        write_csv(output_dir / "predictions.csv", all_rows)
    if output_cfg.get("write_json", True):
        (output_dir / "predictions.json").write_text(json.dumps(json_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(output_dir / "README.md", config, len(images), len(all_rows), raw_prediction_count)

    manifest = {
        "config": config,
        "source": rel(source),
        "output_dir": rel(output_dir),
        "image_count": len(images),
        "prediction_count": len(all_rows),
        "raw_prediction_count": raw_prediction_count,
        "min_export_conf": min_export_conf,
        "model_sha256": sha256_file(resolve_path(config["model"])),
        "outputs": {},
    }
    if output_cfg.get("make_zip", True):
        package_name = output_cfg.get("package_name", "submission")
        package_zip = output_dir / f"{package_name}.zip"
        labels_zip = output_dir / f"{package_name}_labels_only.zip"
        labels_conf_zip = output_dir / f"{package_name}_labels_conf_only.zip"
        make_zip(package_zip, output_dir, ["labels", "labels_conf", "predictions.csv", "predictions.json", "README.md"])
        make_zip(labels_zip, output_dir, ["labels"])
        make_zip(labels_conf_zip, output_dir, ["labels_conf"])
    if output_cfg.get("write_plain_yolo_txt", True) and len(list(labels_dir.glob("*.txt"))) != len(images):
        raise RuntimeError("Plain label file count does not match image count")
    if output_cfg.get("write_conf_yolo_txt", True) and len(list(labels_conf_dir.glob("*.txt"))) != len(images):
        raise RuntimeError("Confidence label file count does not match image count")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][rel(path)] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "submission_infer_yolo11s_imgsz640.yaml")
    parser.add_argument("--source", help="Override source directory, image file, or txt list.")
    parser.add_argument("--output", help="Override output directory.")
    parser.add_argument("--limit", type=int, help="Optional smoke-test limit.")
    args = parser.parse_args()
    config = load_yaml(args.config)
    output_dir = run_prediction(config, args.source, args.output, args.limit)
    print(output_dir.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
