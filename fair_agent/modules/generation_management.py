from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from PIL import Image

from fair_agent.core.config import config_sha256, inference_backend_options, rel_path, resolve_path
from fair_agent.core.hashes import sha256_file
from fair_agent.core.runtime_log import event_log_from_config
from fair_agent.modules.model_generations import load_generation_registry
from fair_agent.modules.strict_incremental import (
    evaluate_ap50,
    precision_recall,
    retention_metrics,
    yolo_ground_truth,
)


def _raw_registry(path: str | Path) -> Dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def _atomic_registry_write(path: Path, registry: Mapping[str, Any], operation: str) -> None:
    audit_root = resolve_path("reports/generation_audit")
    audit_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = audit_root / "backups" / stamp / path.name
    backup.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    load_generation_registry(path)
    event = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "operation": operation,
        "registry": rel_path(path),
        "registry_sha256": sha256_file(path),
        "backup": rel_path(backup),
    }
    with (audit_root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def ensure_recheck_candidate(config: Mapping[str, Any], candidate_id: str) -> Dict[str, Any]:
    generation_cfg = config["generation"]
    registry_path = resolve_path(generation_cfg["registry"])
    registry = _raw_registry(registry_path)
    generations = {str(item["id"]): item for item in registry["generations"]}
    models = {str(item["id"]): item for item in registry["models"]}
    threshold = float(generation_cfg["calibrated_threshold"])
    if candidate_id in generations:
        owner = str(generations[candidate_id]["class_owners"]["2"])
        if abs(float(models[owner]["activation_threshold"]) - threshold) > 1e-9:
            raise ValueError("候选代际已存在，但激活阈值与配置不一致。")
        event_log_from_config(config).append(
            "generation.registration.reused", component="generation", generation_id=candidate_id,
            details={"registry": rel_path(registry_path), "registry_sha256": sha256_file(registry_path)},
        )
        return load_generation_registry(registry_path)
    if candidate_id != str(generation_cfg["candidate_id"]):
        raise ValueError(f"未知候选代际：{candidate_id}")
    source_model = dict(models["strict_p02_warship_v1"])
    source_generation = dict(generations["generation-1"])
    model_id = "strict_p02_warship_recheck_v2"
    source_model.update({
        "id": model_id,
        "activation_threshold": threshold,
        "status": "pending_deployment_recheck",
        "deployment_metrics": {},
        "acceptance": {
            "min_lock_precision": float(generation_cfg["acceptance"]["min_lock_precision"]),
            "max_false_activation_rate": float(generation_cfg["acceptance"]["max_false_activation_rate"]),
            "passed": False,
        },
    })
    source_generation.update({
        "id": candidate_id,
        "parent": "generation-0",
        "class_owners": {**source_generation["class_owners"], "2": model_id},
        "status": "pending_deployment_recheck",
        "metrics": {},
        "acceptance": {"core_metrics_passed": True, "deployment_recheck_passed": False},
    })
    registry["models"].append(source_model)
    registry["generations"].append(source_generation)
    registry["channels"]["candidate"] = candidate_id
    previous_hash = sha256_file(registry_path)
    _atomic_registry_write(registry_path, registry, f"register:{candidate_id}")
    event_log_from_config(config).append(
        "generation.registered", component="generation", generation_id=candidate_id,
        details={
            "parent_generation": source_generation["parent"],
            "model_id": model_id,
            "activation_threshold": threshold,
            "calibration_source": source_model.get("calibration_source"),
            "registry_sha256_before": previous_hash,
            "registry_sha256_after": sha256_file(registry_path),
        },
    )
    return load_generation_registry(registry_path)


def _candidate_settings(registry: Mapping[str, Any], candidate_id: str) -> Dict[str, Any]:
    generation = registry["generations_by_id"][candidate_id]
    models = registry["models_by_id"]
    base_id = next(
        model_id for model_id in set(generation["class_owners"].values())
        if models[model_id]["role"] == "frozen_base"
    )
    expert_id = str(generation["class_owners"][2])
    base = models[base_id]
    expert = models[expert_id]
    if not base["hash_valid"] or not expert["hash_valid"]:
        raise ValueError("候选代际权重缺失或SHA256不匹配。")
    return {
        "base": base,
        "expert": expert,
        "class_names": dict(registry["class_map"]),
        "class_owners": dict(generation["class_owners"]),
        "generation": generation,
    }


def _prediction_rows(results: Iterable[Mapping[str, Any]]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    before: list[Dict[str, Any]] = []
    after: list[Dict[str, Any]] = []
    for result in results:
        for detection in result["detections"]:
            row = {
                "image_id": Path(str(result["filename"])).stem,
                "class_id": int(detection["class_id"]),
                "confidence": float(detection["confidence"]),
                "xyxy": list(detection["xyxy"]),
                "source": detection.get("source"),
            }
            after.append(row)
            if detection.get("source") == "frozen_base_model":
                before.append(row)
    return before, after


def _recheck_generation(config: Mapping[str, Any], candidate_id: str) -> Dict[str, Any]:
    from fair_agent.modules.web_inference import WebInferenceEngine

    registry = ensure_recheck_candidate(config, candidate_id)
    settings = _candidate_settings(registry, candidate_id)
    split = resolve_path(config["generation"]["recheck_lock_split"])
    image_paths = [resolve_path(line.strip()) for line in split.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not image_paths or any(not path.is_file() for path in image_paths):
        raise ValueError("lock-val划分为空或包含缺失图像。")
    web_cfg = config["web"]
    functional = json.loads("{}")
    import yaml
    functional = yaml.safe_load(resolve_path(web_cfg["functional_registry"]).read_text(encoding="utf-8"))
    context_entry = next(item for item in functional["models"] if item["function"] == "context_perception")
    context_path = resolve_path(context_entry["artifacts"][0]["path"])
    expert = settings["expert"]
    generation = settings["generation"]
    protocol = {
        "id": expert["id"],
        "class_name": settings["class_names"][2],
        "new_class": settings["class_names"][2],
        "global_class_id": 2,
        "incremental_mode": "class_incremental",
        "weights": expert["resolved_path"],
        "new_map50": float(expert["metrics"]["new_map50"]),
        "krr": float(registry["generations_by_id"]["generation-1"]["metrics"]["krr"]),
        "available": True,
        "activation_threshold": float(expert["activation_threshold"]),
        "calibration_source": expert["calibration_source"],
        "routing_prior": 1.0,
        "context_prior": {},
    }
    event_log_from_config(config).append(
        "incremental.dev_calibration.consumed",
        component="incremental",
        generation_id=candidate_id,
        protocol_id=str(expert["id"]),
        details={
            "threshold": float(expert["activation_threshold"]),
            "calibration_source": rel_path(resolve_path(expert["calibration_source"])),
            "calibration_source_sha256": sha256_file(resolve_path(expert["calibration_source"])),
        },
    )
    inference = config["inference"]
    engine = WebInferenceEngine(
        settings["base"]["resolved_path"],
        context_path,
        device_index=str(config["runtime"]["default_device"]),
        predict_options=inference,
        incremental_protocols={expert["id"]: protocol},
        class_names=settings["class_names"],
        base_class_ids=settings["base"]["owns_classes"],
        base_local_to_global=settings["base"]["local_to_global"],
        routing_options=config["routing"],
        generation_id=candidate_id,
        base_model_id=settings["base"]["id"],
        class_owners=settings["class_owners"],
        backend_name=str(inference["backend"]),
        native_options=inference_backend_options(config),
    )
    batch_size = int(inference["batch_size"])
    all_results = []
    started = datetime.now()
    event_log_from_config(config).append(
        "incremental.lock.unsealed",
        component="incremental",
        generation_id=candidate_id,
        protocol_id=str(expert["id"]),
        details={
            "lock_split": rel_path(split),
            "lock_split_sha256": sha256_file(split),
            "image_count": len(image_paths),
        },
    )
    for offset in range(0, len(image_paths), batch_size):
        rows = []
        for path in image_paths[offset : offset + batch_size]:
            with Image.open(path) as source:
                source.load()
                rows.append((source.convert("RGB"), path.name, None))
        all_results.extend(engine.predict_batch(rows, float(inference["confidence_min"]), "auto"))
    before, after = _prediction_rows(all_results)
    ground_truth = yolo_ground_truth(image_paths)
    old_ids = [0, 1, 3]
    retention = retention_metrics(before, after, ground_truth, old_ids)
    base_metrics = evaluate_ap50(before, ground_truth, old_ids)
    new_metrics = evaluate_ap50(after, ground_truth, [2])
    combined_metrics = evaluate_ap50(after, ground_truth, [0, 1, 2, 3])
    deployment = precision_recall(after, ground_truth, 2, float(expert["activation_threshold"]))
    positives = {row["image_id"] for row in ground_truth if int(row["class_id"]) == 2}
    negatives = {path.stem for path in image_paths if path.stem not in positives}
    false_positive_images = {
        row["image_id"] for row in after
        if int(row["class_id"]) == 2 and row["image_id"] in negatives
    }
    false_activation_rate = len(false_positive_images) / len(negatives) if negatives else 0.0
    metrics = {
        "base_map50": float(base_metrics["map50"]),
        "new_map50": float(new_metrics["map50"]),
        "krr": float(retention["krr"]),
        "combined_map50": float(combined_metrics["map50"]),
        "lock_precision": float(deployment["precision"]),
        "lock_recall": float(deployment["recall"]),
        "false_activation_rate": float(false_activation_rate),
        "old_prediction_equivalent": bool(retention["old_prediction_equivalent"]),
        "image_count": len(image_paths),
        "negative_image_count": len(negatives),
        "false_positive_image_count": len(false_positive_images),
        "mean_inference_ms": sum(float(row["inference_ms"]) for row in all_results) / len(all_results),
    }
    gates_cfg = config["generation"]["acceptance"]
    gates = {
        "base_map50": metrics["base_map50"] >= float(gates_cfg["min_base_map50"]),
        "new_map50": metrics["new_map50"] >= float(gates_cfg["min_new_map50"]),
        "krr": metrics["krr"] >= float(gates_cfg["min_krr"]),
        "combined_map50": metrics["combined_map50"] >= float(gates_cfg["min_combined_map50"]),
        "lock_precision": metrics["lock_precision"] >= float(gates_cfg["min_lock_precision"]),
        "false_activation_rate": metrics["false_activation_rate"] <= float(gates_cfg["max_false_activation_rate"]),
        "old_prediction_equivalent": metrics["old_prediction_equivalent"],
    }
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = resolve_path(config["generation"]["report_root"]) / f"{candidate_id}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in after:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    config_clean = {key: value for key, value in config.items() if not str(key).startswith("_")}
    manifest = {
        "schema_version": 1,
        "candidate": candidate_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": (datetime.now() - started).total_seconds(),
        "config_sha256": config_sha256(config_clean),
        "registry_sha256": sha256_file(resolve_path(config["generation"]["registry"])),
        "weights": {
            settings["base"]["id"]: settings["base"]["sha256"],
            expert["id"]: expert["sha256"],
        },
        "threshold": float(expert["activation_threshold"]),
        "threshold_source": rel_path(resolve_path(expert["calibration_source"])),
        "lock_split": rel_path(split),
        "lock_split_sha256": sha256_file(split),
        "metrics": metrics,
        "gates": gates,
        "accepted": all(gates.values()),
        "predictions": rel_path(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "manifest": rel_path(manifest_path), "manifest_sha256": sha256_file(manifest_path)}


def _promote_generation(config: Mapping[str, Any], candidate_id: str, manifest_path: str | Path) -> Dict[str, Any]:
    path = resolve_path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("candidate") != candidate_id or manifest.get("accepted") is not True:
        raise ValueError("复核manifest未通过或不属于指定候选代际。")
    if not all(manifest.get("gates", {}).values()):
        raise ValueError("复核manifest存在未通过门禁。")
    clean_config = {key: value for key, value in config.items() if not str(key).startswith("_")}
    if manifest.get("config_sha256") != config_sha256(clean_config):
        raise ValueError("当前有效配置与复核配置不一致。")
    registry_path = resolve_path(config["generation"]["registry"])
    if manifest.get("registry_sha256") != sha256_file(registry_path):
        raise ValueError("代际注册表在复核后发生变化。")
    registry = _raw_registry(registry_path)
    generations = {str(item["id"]): item for item in registry["generations"]}
    models = {str(item["id"]): item for item in registry["models"]}
    if candidate_id not in generations:
        raise ValueError("候选代际未注册。")
    generation = generations[candidate_id]
    expert_id = str(generation["class_owners"]["2"])
    for model_id, expected in manifest["weights"].items():
        model = models[model_id]
        if model["sha256"] != expected or sha256_file(resolve_path(model["path"])) != expected:
            raise ValueError(f"模型权重哈希不一致：{model_id}")
    metrics = dict(manifest["metrics"])
    generation["metrics"] = metrics
    generation["acceptance"] = {"core_metrics_passed": True, "deployment_recheck_passed": True}
    generation["status"] = "active"
    models[expert_id]["deployment_metrics"] = {
        "lock_precision": metrics["lock_precision"],
        "false_activation_rate": metrics["false_activation_rate"],
    }
    models[expert_id]["acceptance"]["passed"] = True
    models[expert_id]["status"] = "active"
    registry["channels"]["production"] = candidate_id
    registry["channels"]["candidate"] = candidate_id
    registry["external_blockers"] = [
        item for item in registry.get("external_blockers", [])
        if item != "deployment_threshold_recheck_pending"
    ]
    _atomic_registry_write(registry_path, registry, f"promote:{candidate_id}:{sha256_file(path)}")
    return {"production": candidate_id, "manifest": rel_path(path), "registry_sha256": sha256_file(registry_path)}


def _rollback_generation(config: Mapping[str, Any], target_id: str) -> Dict[str, Any]:
    registry_path = resolve_path(config["generation"]["registry"])
    registry = _raw_registry(registry_path)
    generations = {str(item["id"]): item for item in registry["generations"]}
    if target_id not in generations or generations[target_id].get("status") != "active":
        raise ValueError("只能回滚到已验证的active代际。")
    previous = str(registry["channels"]["production"])
    registry["channels"]["production"] = target_id
    _atomic_registry_write(registry_path, registry, f"rollback:{previous}->{target_id}")
    return {"previous": previous, "production": target_id, "registry_sha256": sha256_file(registry_path)}


def _audited_generation_action(
    config: Mapping[str, Any],
    action: str,
    generation_id: str,
    callback: Any,
) -> Dict[str, Any]:
    logger = event_log_from_config(config)
    registry_path = resolve_path(config["generation"]["registry"])
    before_registry_hash = sha256_file(registry_path) if registry_path.is_file() else None
    try:
        before_registry = _raw_registry(registry_path)
        production_before = str(before_registry.get("channels", {}).get("production"))
    except (OSError, ValueError, json.JSONDecodeError):
        production_before = None
    event_prefix = {
        "recheck": "generation.lock_recheck",
        "promote": "generation.production_switch",
        "rollback": "generation.rollback",
    }[action]
    trace_id = f"generation_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    logger.append(
        f"{event_prefix}.started", component="generation", trace_id=trace_id,
        generation_id=generation_id,
        details={
            "production_before": production_before,
            "registry_sha256_before": before_registry_hash,
        },
    )
    started = time.perf_counter()
    try:
        result = callback()
    except Exception as exc:
        logger.append(
            f"{event_prefix}.failed", level="error", component="generation", trace_id=trace_id,
            generation_id=generation_id, duration_ms=(time.perf_counter() - started) * 1000,
            message=str(exc),
            details={
                "error_type": type(exc).__name__,
                "production_before": production_before,
                "registry_sha256_before": before_registry_hash,
            },
        )
        if action == "recheck":
            logger.append(
                "incremental.lock_recheck.failed", level="error", component="incremental",
                trace_id=trace_id, generation_id=generation_id,
                duration_ms=(time.perf_counter() - started) * 1000,
                message=str(exc), details={"error_type": type(exc).__name__},
            )
        raise
    after_registry_hash = sha256_file(registry_path) if registry_path.is_file() else None
    try:
        after_registry = _raw_registry(registry_path)
        production_after = str(after_registry.get("channels", {}).get("production"))
    except (OSError, ValueError, json.JSONDecodeError):
        production_after = None
    logger.append(
        f"{event_prefix}.completed", component="generation", trace_id=trace_id,
        generation_id=generation_id, duration_ms=(time.perf_counter() - started) * 1000,
        details={
            "production_before": production_before,
            "production_after": production_after,
            "registry_sha256_before": before_registry_hash,
            "registry_sha256_after": after_registry_hash,
            "result": result,
        },
    )
    if action == "recheck":
        logger.append(
            "incremental.lock_recheck.completed",
            level="info" if result.get("accepted") else "warning",
            component="incremental",
            trace_id=trace_id,
            generation_id=generation_id,
            duration_ms=(time.perf_counter() - started) * 1000,
            details={
                "accepted": bool(result.get("accepted")),
                "threshold": result.get("threshold"),
                "metrics": result.get("metrics"),
                "gates": result.get("gates"),
                "manifest": result.get("manifest"),
                "manifest_sha256": result.get("manifest_sha256"),
            },
        )
    return result


def recheck_generation(config: Mapping[str, Any], candidate_id: str) -> Dict[str, Any]:
    return _audited_generation_action(
        config, "recheck", candidate_id, lambda: _recheck_generation(config, candidate_id)
    )


def promote_generation(
    config: Mapping[str, Any], candidate_id: str, manifest_path: str | Path
) -> Dict[str, Any]:
    return _audited_generation_action(
        config,
        "promote",
        candidate_id,
        lambda: _promote_generation(config, candidate_id, manifest_path),
    )


def rollback_generation(config: Mapping[str, Any], target_id: str) -> Dict[str, Any]:
    return _audited_generation_action(
        config, "rollback", target_id, lambda: _rollback_generation(config, target_id)
    )
