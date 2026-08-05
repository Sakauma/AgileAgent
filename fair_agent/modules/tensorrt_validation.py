from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml
from PIL import Image

from fair_agent.core.config import config_sha256, rel_path, resolve_path, write_config
from fair_agent.core.hashes import sha256_file
from fair_agent.modules.api_benchmark import benchmark_api
from fair_agent.modules.generation_management import active_generation_registry
from fair_agent.modules.model_generations import load_generation_registry
from fair_agent.modules.strict_incremental import (
    calibrate_threshold,
    evaluate_ap50,
    precision_recall,
    retention_metrics,
    yolo_ground_truth,
)


def _rows(results: list[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows = []
    for result in results:
        for detection in result["detections"]:
            rows.append({
                "image_id": Path(str(result["filename"])).stem,
                "class_id": int(detection["class_id"]),
                "confidence": float(detection["confidence"]),
                "xyxy": list(detection["xyxy"]),
            })
    return rows


def _apply_protocol_thresholds(
    protocols: Mapping[str, Dict[str, Any]],
    thresholds: Mapping[int, float],
) -> None:
    for protocol in protocols.values():
        owned = [int(value) for value in protocol.get("global_class_ids", [])]
        activation = {
            int(key): float(value)
            for key, value in dict(protocol.get("activation_thresholds") or {}).items()
        }
        for class_id in owned:
            if class_id in thresholds:
                activation[class_id] = float(thresholds[class_id])
        protocol["activation_thresholds"] = activation
        if len(owned) == 1 and owned[0] in activation:
            protocol["activation_threshold"] = activation[owned[0]]


def _predict(
    config: Mapping[str, Any],
    backend: str,
    paths: list[Path],
    threshold_overrides: Mapping[int, float] | None = None,
    generation_id: str | None = None,
) -> tuple[list[Dict[str, Any]], float]:
    from fair_agent.modules.web_inference import WebInferenceEngine
    from fair_agent.web.app import build_web_settings

    effective = copy.deepcopy(dict(config))
    effective["inference"]["backend"] = backend
    if backend == "tensorrt_engine":
        effective["tensorrt_backend"]["validated"] = True
    settings = build_web_settings(effective, generation_id=generation_id)
    if threshold_overrides:
        _apply_protocol_thresholds(settings["protocols"], threshold_overrides)
    engine = WebInferenceEngine(
        settings["detector_path"], settings["context_path"], settings["device_index"],
        settings["predict"], settings["protocols"], settings["class_names"],
        settings["base_class_ids"], settings["base_local_to_global"], settings["routing"],
        settings["generation_id"], settings["base_model_id"], settings["class_owners"],
        backend, settings["native_backend"],
    )
    results = []
    batch_size = int(config["tensorrt_backend"]["validation"]["evaluation_batch_size"])
    for offset in range(0, len(paths), batch_size):
        items = []
        for path in paths[offset: offset + batch_size]:
            with Image.open(path) as source:
                source.load()
                items.append((source.convert("RGB"), path.name))
        results.extend(engine.predict_batch(items, float(config["inference"]["confidence_min"]), "auto"))
    mean_ms = sum(float(item["inference_ms"]) for item in results) / max(1, len(results))
    return _rows(results), mean_ms


def _image_classes(path: Path) -> set[int]:
    return {
        int(fields[0])
        for line in path.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
        if len(fields := line.split()) == 5
    }


def _competition_accuracy_gates(
    metrics: Mapping[str, float],
    official_hard: Mapping[str, float],
    thresholds_calibrated: bool,
) -> Dict[str, bool]:
    return {
        "quantized_thresholds_calibrated": bool(thresholds_calibrated),
        "base_map50": float(metrics["base_map50"]) >= float(official_hard["base_map50_min"]),
        "new_map50": float(metrics["new_map50"]) >= float(official_hard["new_map50_min"]),
        "krr": float(metrics["krr"]) >= float(official_hard["krr_min"]),
    }


def _calibrate_quantized_thresholds(config: Mapping[str, Any]) -> Dict[str, Any]:
    from fair_agent.backends.inference import TensorRTEngineBackend
    from fair_agent.modules.web_inference import remap_specialist_records_dynamic, result_records

    backend_options = copy.deepcopy(dict(config["tensorrt_backend"]))
    backend_options["validated"] = True
    settings = backend_options["int8_calibration"]
    split = resolve_path(settings["threshold_split"])
    paths = [
        resolve_path(line.strip())
        for line in split.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not paths or any(not path.is_file() or not path.with_suffix(".txt").is_file() for path in paths):
        raise ValueError("INT8部署阈值校准划分为空或包含缺失图像/标签。")
    registry = load_generation_registry(active_generation_registry(config))
    production_id = str(registry["channels"]["production"])
    generation = registry["generations_by_id"][production_id]
    model_ids = set(generation.get("model_members") or generation["class_owners"].values())
    threshold_min = float(config["incremental_workbench"]["lifecycle"]["threshold_min"])
    threshold_max = float(config["incremental_workbench"]["lifecycle"]["threshold_max"])
    threshold_step = float(config["incremental_workbench"]["lifecycle"]["threshold_step"])
    target_precision = float(config["incremental_workbench"]["lifecycle"]["calibration_target_precision"])
    thresholds: Dict[int, float] = {}
    per_model: Dict[str, Any] = {}
    for model_id in sorted(model_ids):
        model = registry["models_by_id"][model_id]
        if model["role"] not in {"class_incremental_expert", "target_incremental_expert"}:
            continue
        owned = {int(value) for value in model["owns_classes"]}
        images = [path for path in paths if _image_classes(path) and _image_classes(path) <= owned]
        if not images:
            raise ValueError(f"INT8专家{model_id}在threshold_split中没有严格增量dev样本。")
        predictor = TensorRTEngineBackend(
            model["resolved_path"],
            str(config["runtime"]["default_device"]),
            backend_options,
        )
        predictions = []
        batch_size = int(backend_options["validation"]["evaluation_batch_size"])
        for offset in range(0, len(images), batch_size):
            chunk = images[offset:offset + batch_size]
            sources = []
            for path in chunk:
                with Image.open(path) as source:
                    source.load()
                    sources.append(source.convert("RGB"))
            results = predictor.predict_batch(
                sources,
                imgsz=int(config["inference"]["specialist_imgsz"]),
                conf=min(0.001, threshold_min),
                iou=float(config["inference"]["iou"]),
                max_det=int(config["inference"]["max_det"]),
            )
            for path, result in zip(chunk, results):
                rows = remap_specialist_records_dynamic(
                    result_records(result), model["local_to_global"], {}, model_id
                )
                predictions.extend({**row, "image_id": path.stem} for row in rows)
        ground_truth = yolo_ground_truth(images)
        calibrated_classes = {}
        for class_id in sorted(owned):
            calibration = calibrate_threshold(
                predictions,
                ground_truth,
                class_id,
                threshold_min,
                threshold_max,
                threshold_step,
                target_precision,
            )
            calibrated_classes[str(class_id)] = calibration
            thresholds[class_id] = float(calibration["selected"]["threshold"])
        per_model[model_id] = {
            "image_count": len(images),
            "image_stems": [path.stem for path in images],
            "classes": calibrated_classes,
        }
    target_precision_reached = bool(thresholds) and all(
        calibration["passed"]
        for model in per_model.values()
        for calibration in model["classes"].values()
    )
    calibrated_class_count = sum(
        len(model["classes"]) for model in per_model.values()
    )
    completed = bool(thresholds) and len(thresholds) == calibrated_class_count
    return {
        "split": rel_path(split),
        "split_sha256": sha256_file(split),
        "lock_content_read": False,
        "target_precision": target_precision,
        "thresholds": {str(key): value for key, value in thresholds.items()},
        "models": per_model,
        "target_precision_reached": target_precision_reached,
        "passed": completed,
    }


def validate_tensorrt(config: Mapping[str, Any], activate: bool = False) -> Dict[str, Any]:
    backend = config["tensorrt_backend"]
    if any(not entry.get("sha256") for entry in [*backend["engines"].values(), backend["context_engine"]]):
        raise ValueError("TensorRT engine哈希尚未全部登记。")
    split = resolve_path(config["performance"]["benchmark_split"])
    paths = [resolve_path(line.strip()) for line in split.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("TensorRT验收划分为空或包含缺失图像。")
    registry = load_generation_registry(active_generation_registry(config))
    production_id = str(registry["channels"]["production"])
    production = registry["generations_by_id"][production_id]
    parent_id = str(production.get("parent") or "")
    if parent_id not in registry["generations_by_id"]:
        raise ValueError("当前production缺少可复核的增量前父代际。")
    class_ids = sorted(int(value) for value in production["classes"])
    old_class_ids = sorted(int(value) for value in production["old_class_ids"])
    new_class_ids = sorted(int(value) for value in production["new_class_ids"])
    if not old_class_ids or not new_class_ids:
        raise ValueError("当前production缺少增量学习新旧类别集合。")
    ground_truth = yolo_ground_truth(paths)
    threshold_calibration = _calibrate_quantized_thresholds(config)
    calibrated_thresholds = {
        int(key): float(value) for key, value in threshold_calibration["thresholds"].items()
    }
    cuda_predictions, cuda_ms = _predict(config, "ultralytics_cuda", paths)
    trt_predictions, trt_ms = _predict(
        config, "tensorrt_engine", paths, calibrated_thresholds
    )
    trt_before_predictions, trt_before_ms = _predict(
        config, "tensorrt_engine", paths, generation_id=parent_id
    )
    cuda_overall = float(evaluate_ap50(cuda_predictions, ground_truth, class_ids)["map50"])
    trt_overall = float(evaluate_ap50(trt_predictions, ground_truth, class_ids)["map50"])
    retention = retention_metrics(
        trt_before_predictions, trt_predictions, ground_truth, old_class_ids
    )
    base_map50 = float(retention["old_map50_before"])
    old_after_map50 = float(retention["old_map50_after"])
    new_map50 = float(evaluate_ap50(trt_predictions, ground_truth, new_class_ids)["map50"])
    per_class = {}
    for class_id in class_ids:
        cuda_value = float(evaluate_ap50(cuda_predictions, ground_truth, [class_id])["map50"])
        trt_value = float(evaluate_ap50(trt_predictions, ground_truth, [class_id])["map50"])
        per_class[str(class_id)] = {"cuda_map50": cuda_value, "tensorrt_map50": trt_value, "delta": trt_value - cuda_value}
    official_hard = config["gates"]["official_hard"]
    advisory = config["gates"]["advisory"]
    competition_metrics = {
        "base_map50": base_map50,
        "old_map50_after": old_after_map50,
        "new_map50": new_map50,
        "krr": float(retention["krr"]),
        "combined_map50": trt_overall,
    }
    competition_gates = _competition_accuracy_gates(
        competition_metrics, official_hard, bool(threshold_calibration["passed"])
    )
    validation = backend["validation"]
    consistency_checks = {
        "overall_map50_delta": abs(trt_overall - cuda_overall)
        <= float(validation["max_overall_map50_delta_warning"]),
        "per_class_map50_delta": all(
            abs(item["delta"]) <= float(validation["max_per_class_map50_delta_warning"])
            for item in per_class.values()
        ),
        "old_prediction_equivalent": bool(retention["old_prediction_equivalent"]),
    }
    per_new_class = {}
    false_activation_rates = []
    lock_precisions = []
    image_ids = {path.stem for path in paths}
    for class_id in new_class_ids:
        threshold = calibrated_thresholds[class_id]
        deployment = precision_recall(trt_predictions, ground_truth, class_id, threshold)
        positive_ids = {
            str(row["image_id"])
            for row in ground_truth
            if int(row["class_id"]) == class_id
        }
        negative_ids = image_ids - positive_ids
        false_image_ids = {
            str(row["image_id"])
            for row in trt_predictions
            if int(row["class_id"]) == class_id
            and str(row["image_id"]) in negative_ids
            and float(row["confidence"]) >= threshold
        }
        false_activation_rate = (
            len(false_image_ids) / len(negative_ids) if negative_ids else 0.0
        )
        per_new_class[str(class_id)] = {
            **deployment,
            "threshold": threshold,
            "false_activation_rate": false_activation_rate,
            "false_activation_image_count": len(false_image_ids),
            "negative_image_count": len(negative_ids),
        }
        lock_precisions.append(float(deployment["precision"]))
        false_activation_rates.append(false_activation_rate)
    deployment_diagnostics = {
        "lock_precision": min(lock_precisions),
        "false_activation_rate": max(false_activation_rates),
        "per_new_class": per_new_class,
    }
    diagnostic_checks = {
        **consistency_checks,
        "calibration_target_precision": bool(
            threshold_calibration["target_precision_reached"]
        ),
        "lock_precision": deployment_diagnostics["lock_precision"]
        >= float(advisory["lock_precision_min"]),
        "false_activation_rate": deployment_diagnostics["false_activation_rate"]
        <= float(advisory["false_activation_rate_max"]),
    }
    warnings = [name for name, passed in diagnostic_checks.items() if not passed]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = resolve_path("reports/tensorrt_validation") / run_id
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "validation.json"
    preliminary = {
        "schema_version": 2, "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_sha256": config_sha256({key: value for key, value in config.items() if not str(key).startswith("_")}),
        "split": rel_path(split), "split_sha256": sha256_file(split),
        "evaluation_batch_size": int(backend["validation"]["evaluation_batch_size"]),
        "accuracy": {
            "cuda_map50": cuda_overall, "tensorrt_map50": trt_overall,
            "delta": trt_overall - cuda_overall, "per_class": per_class,
            "cuda_mean_inference_ms": cuda_ms, "tensorrt_mean_inference_ms": trt_ms,
            "tensorrt_base_mean_inference_ms": trt_before_ms,
        },
        "competition_metrics": competition_metrics,
        "threshold_calibration": threshold_calibration,
        "competition_gates": competition_gates,
        "deployment_diagnostics": deployment_diagnostics,
        "diagnostic_checks": diagnostic_checks,
        "warnings": warnings,
        "validation_status": (
            "accuracy_passed_performance_pending"
            if all(competition_gates.values())
            else "accuracy_rejected"
        ),
        "accepted": all(competition_gates.values()),
    }
    report_path.write_text(json.dumps(preliminary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if all(competition_gates.values()):
        performance_config = copy.deepcopy(dict(config))
        performance_config["inference"]["backend"] = "tensorrt_engine"
        performance_config["tensorrt_backend"]["validated"] = True
        performance_config["tensorrt_backend"]["validation_report"] = rel_path(report_path)
        performance_config["_config_overrides"] = [
            *list(config.get("_config_overrides", [])),
            "inference.backend=tensorrt_engine", "tensorrt_backend.validated=true",
            f"tensorrt_backend.validation_report={rel_path(report_path)}",
        ]
        performance = benchmark_api(performance_config)
        final = {
            **preliminary,
            "performance": performance,
            "accepted": bool(performance["accepted"]),
            "validation_status": (
                "accepted" if performance["accepted"] else "performance_rejected"
            ),
        }
    else:
        final = {
            **preliminary,
            "performance": None,
            "accepted": False,
            "validation_status": "accuracy_rejected",
        }
    report_path.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {**final, "report": rel_path(report_path), "report_sha256": sha256_file(report_path)}
    if activate:
        if not result["accepted"]:
            raise RuntimeError("TensorRT赛题精度或FPS门禁未通过，拒绝启用。")
        config_path = resolve_path(config["_config_path"])
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["inference"]["backend"] = "tensorrt_engine"
        payload["tensorrt_backend"]["validated"] = True
        payload["tensorrt_backend"]["validation_report"] = rel_path(report_path)
        write_config(config_path, payload, "tensorrt-validation:activate")
        result["activated_config"] = rel_path(config_path)
    return result
