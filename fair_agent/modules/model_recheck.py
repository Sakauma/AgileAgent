from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from fair_agent.core.config import ROOT, rel_path, resolve_path
from fair_agent.core.hashes import sha256_file
from fair_agent.modules.freeze import refresh_frozen_assets

CLASS_NAMES = ["soldier", "small_aircraft", "warship", "tank"]


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    x, y, w, h = box
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def ap_101(recall: Sequence[float], precision: Sequence[float]) -> float:
    import numpy as np

    if not recall:
        return 0.0
    mrec = np.concatenate(([0.0], np.asarray(recall), [1.0]))
    mpre = np.concatenate(([1.0], np.asarray(precision), [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def image_outcomes(record: Mapping[str, Any], class_id: int) -> tuple[List[tuple[float, int]], int]:
    import numpy as np

    if "gt_classes" in record:
        ground_truth_count = sum(int(value) == class_id for value in record["gt_classes"])
        outcomes = [
            (float(item["confidence"]), int(bool(item["tp50"])))
            for item in record["pred"] if int(item["class_id"]) == class_id
        ]
        return outcomes, ground_truth_count
    ground_truth = [item["box"] for item in record["gt"] if int(item["class_id"]) == class_id]
    predictions = [item for item in record["pred"] if int(item["class_id"]) == class_id]
    matches = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            iou = box_iou(pred["box"], gt)
            if iou >= 0.5:
                matches.append((iou, gt_index, pred_index))
    matched_pred: set[int] = set()
    if matches:
        matched = np.asarray([[gt_index, pred_index, iou] for iou, gt_index, pred_index in matches], dtype=float)
        if len(matched) > 1:
            matched = matched[matched[:, 2].argsort()[::-1]]
            matched = matched[np.unique(matched[:, 1], return_index=True)[1]]
            matched = matched[np.unique(matched[:, 0], return_index=True)[1]]
        matched_pred = {int(value) for value in matched[:, 1]}
    outcomes = [(float(pred["confidence"]), int(index in matched_pred)) for index, pred in enumerate(predictions)]
    return outcomes, len(ground_truth)


def map50(records: Sequence[Mapping[str, Any]], indices: Sequence[int], class_ids: Sequence[int]) -> float:
    aps: List[float] = []
    for class_id in class_ids:
        detections: List[tuple[float, int]] = []
        gt_count = 0
        for index in indices:
            outcomes, count = image_outcomes(records[index], class_id)
            detections.extend(outcomes)
            gt_count += count
        if gt_count == 0:
            continue
        detections.sort(key=lambda item: item[0], reverse=True)
        tp = 0
        fp = 0
        recall: List[float] = []
        precision: List[float] = []
        for _confidence, is_tp in detections:
            tp += is_tp
            fp += 1 - is_tp
            recall.append(tp / gt_count)
            precision.append(tp / max(tp + fp, 1))
        aps.append(ap_101(recall, precision))
    return sum(aps) / len(aps) if aps else 0.0


def load_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = Path(row["image_path"]).as_posix()
        result[key] = row
        result[Path(key).name] = row
    return result


def load_images(split_paths: Iterable[str]) -> List[Path]:
    images: List[Path] = []
    for split_path in split_paths:
        for line in resolve_path(split_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                images.append(resolve_path(line.strip()))
    if len(images) != len(set(path.resolve() for path in images)):
        raise ValueError("Combined evaluation splits contain duplicate images")
    return images


def load_ground_truth(image: Path) -> List[Dict[str, Any]]:
    labels = []
    for line in image.with_suffix(".txt").read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO label: {image.with_suffix('.txt')}")
        labels.append({"class_id": int(values[0]), "box": xywh_to_xyxy([float(value) for value in values[1:]])})
    return labels


def predict_records(model: Any, images: Sequence[Path], metadata: Mapping[str, Dict[str, str]], imgsz: int, config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    results = list(model.predict(
        source=[str(path) for path in images], imgsz=imgsz, conf=float(config["conf"]), iou=float(config["iou"]),
        max_det=int(config["max_det"]), device=str(config["device"]), batch=int(config["batch"]), verbose=False, save=False,
    ))
    if len(results) != len(images):
        raise RuntimeError(f"Prediction count mismatch for imgsz={imgsz}")
    records = []
    for image, result in zip(images, results):
        row = metadata.get(rel_path(image)) or metadata.get(image.name)
        if row is None:
            raise KeyError(f"Metadata missing for {image}")
        predictions = []
        if result.boxes is not None:
            for box, cls_id, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                height, width = result.orig_shape
                predictions.append({
                    "class_id": int(cls_id), "confidence": float(conf),
                    "box": [float(box[0]) / width, float(box[1]) / height, float(box[2]) / width, float(box[3]) / height],
                })
        records.append({
            "image_path": rel_path(image), "sensor": row["sensor"], "scene": row["scene"],
            "classes_present": row["classes_present"], "gt": load_ground_truth(image), "pred": predictions,
        })
    return records


def bootstrap_deltas(records_by_size: Mapping[int, Sequence[Mapping[str, Any]]], sizes: Sequence[int], replicates: int, seed: int) -> List[float]:
    import numpy as np

    reference, candidate = sizes
    records = records_by_size[reference]
    strata: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(records):
        strata[f"{row['sensor']}|{row['scene']}|{row['classes_present']}"].append(index)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        sample: List[int] = []
        for indices in strata.values():
            sample.extend(int(value) for value in rng.choice(indices, size=len(indices), replace=True))
        deltas.append(map50(records_by_size[candidate], sample, [0, 1, 2, 3]) - map50(records_by_size[reference], sample, [0, 1, 2, 3]))
    return deltas


def group_metrics(records_by_size: Mapping[int, Sequence[Mapping[str, Any]]], sizes: Sequence[int], common_classes: Sequence[int]) -> List[Dict[str, Any]]:
    reference_records = records_by_size[sizes[0]]
    groups: Dict[str, List[int]] = {"overall": list(range(len(reference_records)))}
    for index, row in enumerate(reference_records):
        groups.setdefault(f"sensor:{row['sensor']}", []).append(index)
        groups.setdefault(f"scene:{row['scene']}", []).append(index)
        groups.setdefault(f"sensor_scene:{row['sensor']}/{row['scene']}", []).append(index)
    output = []
    for name, indices in sorted(groups.items()):
        item: Dict[str, Any] = {"group": name, "image_count": len(indices)}
        for size in sizes:
            item[f"map50_{size}"] = map50(records_by_size[size], indices, [0, 1, 2, 3])
            item[f"common_map50_{size}"] = map50(records_by_size[size], indices, common_classes)
            for class_id, class_name in enumerate(CLASS_NAMES):
                item[f"{class_name}_map50_{size}"] = map50(records_by_size[size], indices, [class_id])
        item["delta"] = item[f"map50_{sizes[1]}"] - item[f"map50_{sizes[0]}"]
        output.append(item)
    return output


def write_combined_dataset(report_dir: Path, images: Sequence[Path]) -> Path:
    import yaml

    report_dir.mkdir(parents=True, exist_ok=True)
    split_path = report_dir / "combined_eval.txt"
    split_path.write_text("\n".join(rel_path(path) for path in images) + "\n", encoding="utf-8")
    data_path = report_dir / "combined_dataset.yaml"
    data_path.write_text(yaml.safe_dump({"path": str(ROOT), "train": rel_path(split_path), "val": rel_path(split_path), "names": {index: name for index, name in enumerate(CLASS_NAMES)}}, sort_keys=False), encoding="utf-8")
    return data_path


def validate_and_collect(model: Any, data_path: Path, imgsz: int, metadata: Mapping[str, Dict[str, str]], config: Mapping[str, Any]) -> tuple[Any, List[Dict[str, Any]]]:
    from ultralytics.models.yolo.detect import DetectionValidator

    class CollectingValidator(DetectionValidator):
        last_instance = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.collected: List[Dict[str, Any]] = []
            CollectingValidator.last_instance = self

        def update_metrics(self, preds, batch):
            stats = self.metrics.stats
            before = len(stats["target_cls"])
            result = super().update_metrics(preds, batch)
            stats = self.metrics.stats
            appended = len(stats["target_cls"]) - before
            if appended != len(batch["im_file"]):
                raise RuntimeError("Validator did not append one statistics record per image")
            for offset, image_name in enumerate(batch["im_file"]):
                index = before + offset
                self.collected.append({
                    "image_name": str(image_name),
                    "conf": stats["conf"][index].tolist(),
                    "pred_cls": stats["pred_cls"][index].tolist(),
                    "tp": stats["tp"][index].tolist(),
                    "target_cls": stats["target_cls"][index].tolist(),
                })
            return result

    metrics = model.val(
        validator=CollectingValidator, data=str(data_path), split="val", imgsz=imgsz,
        conf=float(config["conf"]), iou=float(config["iou"]), max_det=int(config["max_det"]),
        device=str(config["device"]), batch=int(config["batch"]), workers=int(config["workers"]),
        plots=False, save_json=False, project=config["project"], name=f"validation_{imgsz}",
    )
    validator = CollectingValidator.last_instance
    if validator is None or not validator.collected:
        raise RuntimeError("Unable to collect one validator statistics record per image")
    records = []
    for collected in validator.collected:
        path = Path(collected["image_name"])
        row = metadata.get(rel_path(path)) or metadata.get(path.name)
        if row is None:
            raise KeyError(f"Metadata missing for validator image {path}")
        conf = collected["conf"]
        pred_cls = collected["pred_cls"]
        tp = collected["tp"]
        target_cls = collected["target_cls"]
        records.append({
            "image_path": rel_path(path), "sensor": row["sensor"], "scene": row["scene"],
            "classes_present": row["classes_present"], "gt_classes": [int(value) for value in target_cls],
            "pred": [{"class_id": int(cls_id), "confidence": float(score), "tp50": bool(correct[0])} for cls_id, score, correct in zip(pred_cls, conf, tp)],
        })
    return metrics, records


def freeze_candidate(config: Mapping[str, Any], selected_imgsz: int, report_dir: Path) -> None:
    card = resolve_path("reports/final_candidate_card.md")
    card.write_text(f"# 最终候选方案卡片\n\n稳定性复核选择：`YOLO11s + imgsz={selected_imgsz}`。\n\n- 权重：`final_submission_assets/best.pt`\n- 复核报告：`{rel_path(report_dir / 'stability_report.md')}`\n- 权重 SHA256：`{config['expected_sha256']}`\n", encoding="utf-8")
    result = json.loads((report_dir / "stability_metrics.json").read_text(encoding="utf-8"))
    overall = next(row for row in result["groups"] if row["group"] == "overall")
    sar = next(row for row in result["groups"] if row["group"] == "sensor:sar")
    metrics = {
        "combined_all_map50": overall[f"map50_{selected_imgsz}"],
        "combined_sar_map50": sar[f"map50_{selected_imgsz}"],
        "combined_soldier_map50": overall[f"soldier_map50_{selected_imgsz}"],
        "bootstrap_delta_ci95": [result["bootstrap"]["ci95_lower"], result["bootstrap"]["ci95_upper"]],
        "selection_rule": result["recommendation"],
    }
    refresh_frozen_assets(selected_imgsz, "stability_rechecked", report_dir / "stability_report.md", metrics)


def run_model_recheck(config_path: Path, reuse_predictions_dir: Path | None = None) -> Path:
    import numpy as np
    import yaml
    from ultralytics import YOLO

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_path = resolve_path(config["model"])
    actual_hash = sha256_file(model_path)
    if actual_hash != config["expected_sha256"]:
        raise ValueError("Frozen model SHA256 mismatch")
    report_dir = resolve_path(config["report_root"]) / datetime.now().strftime("recheck_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=False)
    images = load_images(config["splits"])
    metadata = load_metadata(resolve_path(config["metadata"]))
    model = YOLO(str(model_path))
    sizes = [int(value) for value in config["imgsz"]]
    if reuse_predictions_dir is not None:
        reuse_dir = resolve_path(reuse_predictions_dir)
        records_by_size = {size: json.loads((reuse_dir / f"predictions_{size}.json").read_text(encoding="utf-8")) for size in sizes}
        expected_paths = [rel_path(path) for path in images]
        for size, records in records_by_size.items():
            if [row.get("image_path") for row in records] != expected_paths:
                raise ValueError(f"Reused prediction image order mismatch for imgsz={size}")
    else:
        records_by_size = {size: predict_records(model, images, metadata, size, config) for size in sizes}
    for size, records in records_by_size.items():
        (report_dir / f"predictions_{size}.json").write_text(json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")
    data_path = write_combined_dataset(report_dir, images)
    validation = {}
    validator_records_by_size = {}
    for size in sizes:
        metrics, validator_records = validate_and_collect(model, data_path, size, metadata, config)
        validator_records_by_size[size] = validator_records
        (report_dir / f"validator_stats_{size}.json").write_text(json.dumps(validator_records, ensure_ascii=False) + "\n", encoding="utf-8")
        custom = map50(validator_records, list(range(len(images))), [0, 1, 2, 3])
        official = float(metrics.box.map50)
        validation[str(size)] = {"custom_map50": custom, "ultralytics_map50": official, "absolute_error": abs(custom - official)}
    tolerance = float(config["ap_validation_tolerance"])
    if any(item["absolute_error"] > tolerance for item in validation.values()):
        result = {"status": "failed_ap_validation", "validation": validation, "selected_imgsz": None}
    else:
        deltas = bootstrap_deltas(validator_records_by_size, sizes, int(config["bootstrap_replicates"]), int(config["seed"]))
        groups = group_metrics(validator_records_by_size, sizes, [int(value) for value in config["common_class_ids"]])
        lower, median, upper = [float(value) for value in np.percentile(deltas, [2.5, 50, 97.5])]
        decision = config["decision"]
        overall_delta = next(row["delta"] for row in groups if row["group"] == "overall")
        subgroup_failures = [row for row in groups if row["image_count"] >= int(decision["subgroup_min_images"]) and row["delta"] < -float(decision["max_subgroup_drop"])]
        keep_768 = overall_delta > 0 and lower >= float(decision["min_ci_lower"]) and not subgroup_failures
        selected = int(decision["candidate_imgsz"] if keep_768 else decision["reference_imgsz"])
        result = {"status": "passed", "validation": validation, "bootstrap": {"replicates": len(deltas), "median_delta": median, "ci95_lower": lower, "ci95_upper": upper}, "groups": groups, "subgroup_failures": subgroup_failures, "selected_imgsz": selected, "recommendation": "keep_768" if keep_768 else "use_640"}
        with (report_dir / "bootstrap_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["replicate", "delta_map50"]); writer.writerows(enumerate(deltas, 1))
    (report_dir / "stability_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 640/768 推理尺寸稳定性复核", "", f"- status: `{result['status']}`", f"- images: `{len(images)}`", f"- model_sha256: `{actual_hash}`", ""]
    if result["status"] == "passed":
        lines.extend([f"- recommendation: `{result['recommendation']}`", f"- selected_imgsz: `{result['selected_imgsz']}`", f"- bootstrap 95% CI: `[{result['bootstrap']['ci95_lower']:.6f}, {result['bootstrap']['ci95_upper']:.6f}]`", "", "## Groups", "", "| group | images | mAP50 640 | mAP50 768 | delta |", "|---|---:|---:|---:|---:|"])
        for row in result["groups"]:
            lines.append(f"| {row['group']} | {row['image_count']} | {row['map50_640']:.5f} | {row['map50_768']:.5f} | {row['delta']:+.5f} |")
        overall = next(row for row in result["groups"] if row["group"] == "overall")
        lines.extend(["", "## Per Class", "", "| class | mAP50 640 | mAP50 768 |", "|---|---:|---:|"])
        for class_name in CLASS_NAMES:
            lines.append(f"| {class_name} | {overall[f'{class_name}_map50_640']:.5f} | {overall[f'{class_name}_map50_768']:.5f} |")
    (report_dir / "stability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"created_at": datetime.now().isoformat(timespec="seconds"), "config": rel_path(config_path), "model_sha256": actual_hash, "files": {path.name: sha256_file(path) for path in report_dir.iterdir() if path.is_file()}}
    (report_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result.get("status") == "passed" and config["decision"].get("freeze_after_success", False):
        freeze_candidate(config, int(result["selected_imgsz"]), report_dir)
    return report_dir
