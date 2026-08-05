from __future__ import annotations

import copy
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import yaml
from PIL import Image

from fair_agent.core.config import config_sha256, inference_backend_options, rel_path, resolve_path
from fair_agent.core.hashes import sha256_file
from fair_agent.core.runtime_log import event_log_from_config
from fair_agent.modules.incremental_guardian import (
    assess_incremental_candidate,
    data_overlap_count,
    learn_confusion_graph,
)
from fair_agent.modules.model_generations import generation_settings, load_generation_registry
from fair_agent.modules.strict_incremental import evaluate_ap50, precision_recall, retention_metrics


def _configured_registry_path(config: Mapping[str, Any]) -> Path:
    generation = config["generation"]
    source = resolve_path(generation["registry"])
    default_source = resolve_path("models/generations.json")
    if source != default_source:
        return source
    runtime = resolve_path(generation["runtime_registry"])
    if not runtime.exists():
        runtime.parent.mkdir(parents=True, exist_ok=True)
        temporary = runtime.with_suffix(runtime.suffix + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, runtime)
    return runtime


def active_generation_registry(config: Mapping[str, Any]) -> Path:
    path = _configured_registry_path(config)
    load_generation_registry(path)
    return path


def _raw_registry(path: str | Path) -> Dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def _atomic_registry_write(path: Path, registry: Mapping[str, Any], operation: str) -> None:
    audit_root = resolve_path("reports/generation_audit")
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
    audit_root.mkdir(parents=True, exist_ok=True)
    with (audit_root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def ensure_recheck_candidate(config: Mapping[str, Any], candidate_id: str) -> Dict[str, Any]:
    path = active_generation_registry(config)
    registry = load_generation_registry(path)
    if candidate_id not in registry["generations_by_id"]:
        raise ValueError("候选代际尚未由增量生命周期注册。")
    return registry


def register_trained_candidate(
    config: Mapping[str, Any],
    batch_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    registry_path = active_generation_registry(config)
    registry = _raw_registry(registry_path)
    parent_id = str(registry["channels"]["production"])
    generations = {str(item["id"]): item for item in registry["generations"]}
    models = {str(item["id"]): item for item in registry["models"]}
    parent = generations[parent_id]
    bindings = list(batch_manifest.get("audit", {}).get("class_bindings") or [])
    if not bindings:
        raise ValueError("增量批次缺少冻结的类别绑定。")
    audit = dict(batch_manifest.get("audit") or {})
    data_compliance = {
        "compliance": audit.get("compliance"),
        "lineage_evidence": audit.get("lineage_evidence"),
        "lineage_catalog_hashes": list(audit.get("lineage_catalog_hashes") or []),
        "old_raw_image_count": int(audit.get("old_raw_image_count", 0) or 0),
        "old_raw_label_count": int(audit.get("old_raw_label_count", 0) or 0),
        "old_cache_count": int(audit.get("old_cache_count", 0) or 0),
        "unverified_cache_count": int(audit.get("unverified_cache_count", 0) or 0),
    }
    overlap_limit = int(config["gates"]["official_hard"]["old_data_overlap_max"])
    if (
        data_compliance["compliance"] != "passed"
        or data_compliance["lineage_evidence"] not in {"current", "not_required"}
        or data_overlap_count(data_compliance) > overlap_limit
    ):
        raise ValueError("候选批次未通过增量数据血缘硬门禁。")
    owned_ids = sorted({int(item["global_class_id"]) for item in bindings})
    mode = str(batch_manifest["audit"]["incremental_mode"])
    parent_ids = {int(value) for value in parent["classes"]}
    if mode == "class_incremental" and parent_ids & set(owned_ids):
        raise ValueError("类别增量候选与当前production类别ID重叠。")
    thresholds = {int(key): float(value) for key, value in calibration["per_class_thresholds"].items()}
    sources = {int(key): str(value) for key, value in calibration["calibration_sources"].items()}
    if set(owned_ids) != set(thresholds) or set(owned_ids) != set(sources):
        raise ValueError("逐类阈值、校准证据与候选类别集合不一致。")
    if str(candidate_manifest.get("parent_generation_id") or "") != parent_id:
        raise ValueError("候选训练记录的父代际与当前production不一致。")
    if candidate_manifest.get("frozen_source_unchanged") is not True or (
        candidate_manifest.get("initial_weight_sha256")
        != candidate_manifest.get("initial_weight_sha256_after")
    ):
        raise ValueError("候选训练未证明初始化权重保持冻结。")
    best = resolve_path(candidate_manifest["best_weight"])
    if not best.is_file() or sha256_file(best) != str(candidate_manifest["best_weight_sha256"]):
        raise ValueError("候选权重缺失或哈希不一致。")

    batch_id = str(batch_manifest["batch_id"])
    suffix = "".join(character if character.isalnum() else "_" for character in batch_id).strip("_")
    model_id = f"incremental_expert_{suffix}"
    generation_id = f"incremental_generation_{suffix}"
    if any(str(item["id"]) in {model_id, generation_id} for item in registry["models"] + registry["generations"]):
        raise ValueError("当前批次已经注册过候选代际。")
    local_to_global = {
        str(item["training_class_id"]): int(item["global_class_id"]) for item in bindings
    }
    class_names = {
        int(item["global_class_id"]): str(item["display_name"]) for item in bindings
    }
    registry["class_map"].update({str(key): value for key, value in class_names.items()})
    per_class_metrics = {
        str(key): dict(value) for key, value in calibration.get("per_class_metrics", {}).items()
    }
    model = {
        "id": model_id,
        "display_name": f"增量检测器（{batch_manifest.get('name', batch_id)}）",
        "role": "class_incremental_expert" if mode == "class_incremental" else "target_incremental_expert",
        "incremental_mode": mode,
        "backend": "ultralytics",
        "path": rel_path(best),
        "sha256": sha256_file(best),
        "owns_classes": owned_ids,
        "local_to_global": local_to_global,
        "per_class_thresholds": {str(key): value for key, value in thresholds.items()},
        "calibration_sources": {str(key): rel_path(resolve_path(value)) for key, value in sources.items()},
        "metrics": {
            "new_map50": float(calibration["new_map50"]),
            "per_class": per_class_metrics,
        },
        "dataset_fingerprint": str(batch_manifest["injection"]["dataset_fingerprint"]),
        "data_compliance": data_compliance,
        "parent_model_id": candidate_manifest.get("parent_model_id"),
        "initial_weight": candidate_manifest.get("initial_weight"),
        "initial_weight_sha256": candidate_manifest.get("initial_weight_sha256"),
        "deployment_metrics": {},
        "acceptance": {"passed": False},
        "status": "registered_candidate",
    }
    if len(owned_ids) == 1:
        only = owned_ids[0]
        model["activation_threshold"] = thresholds[only]
        model["calibration_source"] = rel_path(resolve_path(sources[only]))
    owners = {str(key): value for key, value in parent["class_owners"].items()}
    superseded_model_ids = {
        str(owners[str(class_id)]) for class_id in owned_ids if str(class_id) in owners
    }
    independent_class_ids = []
    for class_id in owned_ids:
        parent_owner = models.get(str(owners.get(str(class_id))))
        if mode == "class_incremental" or not parent_owner or parent_owner.get("role") != "frozen_base":
            independent_class_ids.append(class_id)
        owners[str(class_id)] = model_id
    model["independent_class_ids"] = independent_class_ids
    model["supersedes_model_ids"] = sorted(superseded_model_ids)
    active_owner_ids = set(owners.values())
    members = [
        str(value)
        for value in (parent.get("model_members") or parent["class_owners"].values())
        if str(value) not in superseded_model_ids or str(value) in active_owner_ids
    ]
    members = list(dict.fromkeys([*members, model_id]))
    classes = sorted(parent_ids | set(owned_ids))
    generation = {
        "id": generation_id,
        "display_name": f"增量代际（{batch_manifest.get('name', batch_id)}）",
        "parent": parent_id,
        "classes": classes,
        "old_class_ids": sorted(parent_ids),
        "new_class_ids": owned_ids if mode == "class_incremental" else [],
        "updated_class_ids": owned_ids if mode == "target_incremental" else [],
        "class_owners": owners,
        "model_members": members,
        "superseded_model_ids": sorted(superseded_model_ids - active_owner_ids),
        "status": "registered_candidate",
        "incremental_mode": mode,
        "dataset_fingerprint": str(batch_manifest["injection"]["dataset_fingerprint"]),
        "data_compliance": data_compliance,
        "lineage_batch_id": batch_id,
        "evaluation_lock": {
            "manifest": rel_path(resolve_path(batch_manifest["injection"]["sealed_lock_manifest"])),
            "local_to_global": batch_manifest["audit"]["local_to_global"],
        },
        "metrics": {"krr": 0.0},
        "acceptance": {"core_metrics_passed": False, "deployment_recheck_passed": False},
    }
    registry["models"].append(model)
    registry["generations"].append(generation)
    registry["channels"]["candidate"] = generation_id
    _atomic_registry_write(registry_path, registry, f"register:{generation_id}")
    return {
        "generation_id": generation_id,
        "parent_generation_id": parent_id,
        "model_id": model_id,
        "registry": rel_path(registry_path),
    }


def register_generation_deployment(
    config: Mapping[str, Any],
    generation_id: str,
    deployment_id: str,
    deployment: Mapping[str, Any],
) -> Dict[str, Any]:
    registry_path = active_generation_registry(config)
    registry = _raw_registry(registry_path)
    generations = {str(item["id"]): item for item in registry["generations"]}
    models = {str(item["id"]): item for item in registry["models"]}
    generation = generations.get(generation_id)
    if generation is None or not generation.get("parent"):
        raise ValueError("只能为已注册的增量候选登记部署资产。")
    parent = generations[str(generation["parent"])]
    candidate_model_ids = set(generation.get("model_members") or generation["class_owners"].values()) - set(
        parent.get("model_members") or parent["class_owners"].values()
    )
    if len(candidate_model_ids) != 1:
        raise ValueError("一个增量批次必须对应一个可部署的多类专家。")
    model_id = next(iter(candidate_model_ids))
    artifact = resolve_path(deployment["path"])
    expected = str(deployment["sha256"])
    if not artifact.is_file() or sha256_file(artifact) != expected:
        raise ValueError("待登记部署engine缺失或哈希不一致。")
    calibration_manifest = resolve_path(deployment["calibration_manifest"])
    if not calibration_manifest.is_file() or sha256_file(calibration_manifest) != str(
        deployment["calibration_manifest_sha256"]
    ):
        raise ValueError("INT8校准manifest缺失或哈希不一致。")
    model = models[model_id]
    deployments = model.setdefault("deployments", {})
    if deployment_id in deployments:
        raise ValueError(f"模型已登记同名部署资产：{deployment_id}")
    deployments[deployment_id] = dict(deployment)
    _atomic_registry_write(
        registry_path,
        registry,
        f"register-deployment:{generation_id}:{model_id}:{deployment_id}",
    )
    return {
        "generation_id": generation_id,
        "model_id": model_id,
        "deployment_id": deployment_id,
        "engine": rel_path(artifact),
        "engine_sha256": expected,
        "registry": rel_path(registry_path),
    }


def _context_path(config: Mapping[str, Any]) -> Path:
    functional = yaml.safe_load(resolve_path(config["web"]["functional_registry"]).read_text(encoding="utf-8"))
    entry = next(item for item in functional["models"] if item["function"] == "context_perception")
    return resolve_path(entry["artifacts"][0]["path"])


def _engine(config: Mapping[str, Any], registry: Mapping[str, Any], generation_id: str):
    from fair_agent.modules.web_inference import WebInferenceEngine

    settings = generation_settings(registry, generation_id)
    inference = config["inference"]
    backend_options = inference_backend_options(config)
    if str(inference["backend"]) == "tensorrt_engine" and backend_options.get("precision") == "int8":
        backend_options["engines"] = {
            **dict(backend_options.get("engines") or {}),
            **dict(settings.get("engine_deployments") or {}),
        }
    return WebInferenceEngine(
        settings["detector_path"], _context_path(config),
        device_index=str(config["runtime"]["default_device"]),
        predict_options=inference,
        incremental_protocols=settings["protocols"],
        class_names=settings["class_names"],
        base_class_ids=settings["base_class_ids"],
        base_local_to_global=settings["base_local_to_global"],
        routing_options=config["routing"],
        generation_id=generation_id,
        base_model_id=settings["base_model_id"],
        class_owners=settings["class_owners"],
        backend_name=str(inference["backend"]),
        native_options=backend_options,
    )


def _prediction_rows(results: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows = []
    for result in results:
        for detection in result["detections"]:
            rows.append({
                "image_id": Path(str(result["filename"])).stem,
                "class_id": int(detection["class_id"]),
                "confidence": float(detection["confidence"]),
                "xyxy": list(detection["xyxy"]),
                "source": detection.get("source"),
                "protocol_id": detection.get("protocol_id"),
            })
    return rows


def _read_ground_truth(images: Sequence[Path], local_to_global: Mapping[int, int] | None = None) -> list[Dict[str, Any]]:
    mapping = {int(key): int(value) for key, value in (local_to_global or {}).items()}
    rows = []
    for image in images:
        with Image.open(image) as source:
            width, height = source.size
        label = image.with_suffix(".txt")
        if not label.is_file() and "images" in image.parts:
            parts = list(image.parts)
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            label = Path(*parts).with_suffix(".txt")
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            local_id = int(fields[0])
            class_id = mapping.get(local_id, local_id)
            x, y, w, h = [float(value) for value in fields[1:]]
            rows.append({
                "image_id": image.stem,
                "class_id": class_id,
                "xyxy": [(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height],
            })
    return rows


def _candidate_lock(generation: Mapping[str, Any]) -> tuple[list[Path], Dict[int, int]]:
    lock = generation.get("evaluation_lock")
    if not isinstance(lock, Mapping):
        return [], {}
    mapping = {int(key): int(value) for key, value in lock.get("local_to_global", {}).items()}
    if lock.get("split"):
        split = resolve_path(lock["split"])
        images = [
            resolve_path(line.strip())
            for line in split.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not images or any(not path.is_file() for path in images):
            raise FileNotFoundError("代际复核split为空或包含缺失图像。")
        return images, mapping
    manifest_path = resolve_path(lock["manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    images = []
    for row in payload.get("files", []):
        stem = str(row["stem"])
        match = next((path for path in (root / "images" / "lock").glob(stem + ".*") if path.is_file()), None)
        if match is None:
            raise FileNotFoundError(f"封存lock图像不存在：{stem}")
        images.append(match)
    return images, mapping


def _class_deployment_metrics(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    images: Sequence[Path],
    class_id: int,
    threshold: float,
) -> Dict[str, Any]:
    deployment = precision_recall(predictions, ground_truth, class_id, threshold)
    positives = {
        str(row["image_id"])
        for row in ground_truth
        if int(row["class_id"]) == class_id
    }
    negatives = {path.stem for path in images} - positives
    false_images = {
        str(row["image_id"])
        for row in predictions
        if int(row["class_id"]) == class_id
        and str(row["image_id"]) in negatives
        and float(row["confidence"]) >= threshold
    }
    return {
        "precision": float(deployment["precision"]),
        "recall": float(deployment["recall"]),
        "negative_image_count": len(negatives),
        "false_positive_image_count": len(false_images),
        "false_activation_rate": len(false_images) / len(negatives) if negatives else 0.0,
    }


def _generation_model_ids(generation: Mapping[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        str(value) for value in (generation.get("model_members") or generation["class_owners"].values())
    ))


def _root_generation_id(registry: Mapping[str, Any], generation_id: str) -> str:
    generations = registry["generations_by_id"]
    cursor = generation_id
    visited: set[str] = set()
    while True:
        if cursor in visited:
            raise ValueError("代际父链存在循环。")
        visited.add(cursor)
        generation = generations[cursor]
        parent = generation.get("parent")
        if not parent:
            return cursor
        cursor = str(parent)


def _incremental_lock_chain(
    registry: Mapping[str, Any], generation_id: str,
) -> list[Dict[str, Any]]:
    generations = registry["generations_by_id"]
    chain = []
    cursor: str | None = generation_id
    while cursor:
        generation = generations[cursor]
        if isinstance(generation.get("evaluation_lock"), Mapping):
            images, mapping = _candidate_lock(generation)
            chain.append({
                "generation_id": cursor,
                "images": images,
                "local_to_global": mapping,
            })
        cursor = str(generation.get("parent")) if generation.get("parent") else None
    chain.reverse()
    return chain


def _run_engine(engine: Any, image_paths: Sequence[Path], config: Mapping[str, Any]) -> list[Dict[str, Any]]:
    batch_size = int(config["inference"]["batch_size"])
    results = []
    for offset in range(0, len(image_paths), batch_size):
        items = []
        for path in image_paths[offset: offset + batch_size]:
            with Image.open(path) as source:
                source.load()
                items.append((source.convert("RGB"), path.name))
        results.extend(engine.predict_batch(items, float(config["inference"]["confidence_min"]), "auto"))
    return results


def build_candidate_confusion_graph(
    config: Mapping[str, Any],
    batch_dir: Path,
    candidate_id: str,
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    guardian = config["incremental_guardian"]
    settings = guardian["dynamic_confusion"]
    registry_path = active_generation_registry(config)
    registry = load_generation_registry(registry_path)
    generation = registry["generations_by_id"][candidate_id]
    parent_id = str(generation["parent"])
    focus_ids = sorted(
        int(value)
        for value in (generation.get("new_class_ids") or generation.get("updated_class_ids") or [])
    )
    images = sorted(path for path in (batch_dir / "prepared" / "images" / "val").glob("*") if path.is_file())
    if not images or not focus_ids:
        raise ValueError("候选dev或增量类别为空，无法生成动态混淆图。")
    candidate_model_ids = set(_generation_model_ids(generation)) - set(
        _generation_model_ids(registry["generations_by_id"][parent_id])
    )
    if len(candidate_model_ids) != 1:
        raise ValueError("动态混淆图要求每批次恰好一个多类专家。")
    model_id = next(iter(candidate_model_ids))
    local_to_global = registry["models_by_id"][model_id]["local_to_global"]
    specialist_path = resolve_path(calibration["predictions"])
    if not specialist_path.is_file() or sha256_file(specialist_path) != calibration["predictions_sha256"]:
        raise ValueError("增量dev逐框预测缺失或哈希不一致。")
    specialist_predictions = [
        json.loads(line)
        for line in specialist_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    parent_results = _run_engine(_engine(config, registry, parent_id), images, config)
    base_predictions = _prediction_rows(parent_results)
    ground_truth = _read_ground_truth(images, local_to_global)
    if bool(settings["enabled"]):
        graph = learn_confusion_graph(
            base_predictions,
            specialist_predictions,
            ground_truth,
            focus_ids,
            settings,
        )
    else:
        graph = {
            "schema_version": 1,
            "source_split": "incremental_dev_only",
            "focus_class_ids": focus_ids,
            "edges": [],
            "hard_scene_gate": False,
            "disabled": True,
        }
    graph.update({
        "candidate_generation_id": candidate_id,
        "parent_generation_id": parent_id,
        "dev_image_count": len(images),
        "base_prediction_count": len(base_predictions),
        "specialist_prediction_count": len(specialist_predictions),
        "ground_truth_count": len(ground_truth),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    graph_path = batch_dir / "calibration" / f"{candidate_id}-confusion_graph.json"
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    graph_record = {
        "path": rel_path(graph_path),
        "sha256": sha256_file(graph_path),
        "source_split": "incremental_dev_only",
        "edge_count": len(graph["edges"]),
    }
    raw = _raw_registry(registry_path)
    raw_models = {str(item["id"]): item for item in raw["models"]}
    raw_generations = {str(item["id"]): item for item in raw["generations"]}
    raw_models[model_id]["confusion_graph"] = graph_record
    raw_generations[candidate_id]["confusion_graph"] = graph_record
    _atomic_registry_write(registry_path, raw, f"confusion-graph:{candidate_id}:{graph_record['sha256']}")
    event_log_from_config(config).append(
        "incremental.dev_confusion_graph.completed",
        component="incremental",
        generation_id=candidate_id,
        details=graph_record,
    )
    return {**graph, **graph_record}


def shadow_load_generation(config: Mapping[str, Any], candidate_id: str) -> tuple[Any, Dict[str, Any]]:
    registry = ensure_recheck_candidate(config, candidate_id)
    started = time.perf_counter()
    engine = _engine(config, registry, candidate_id)
    smoke_count = int(config["generation"]["shadow_smoke_images"])
    image = Image.new(
        "RGB",
        (int(config["inference"]["warmup_width"]), int(config["inference"]["warmup_height"])),
    )
    results = [
        engine.predict(image, f"shadow-smoke-{index}.png", float(config["inference"]["confidence_default"]), "auto")
        for index in range(smoke_count)
    ]
    summary = {
        "generation_id": candidate_id,
        "smoke_images": smoke_count,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "detections": sum(int(item["detection_count"]) for item in results),
    }
    return engine, summary


def _unseal_lock_once(config: Mapping[str, Any], candidate_id: str) -> Dict[str, Any]:
    registry_path = active_generation_registry(config)
    raw = _raw_registry(registry_path)
    generations = {str(item["id"]): item for item in raw["generations"]}
    generation = generations.get(candidate_id)
    if generation is None:
        raise ValueError("候选代际不存在。")
    if isinstance(generation.get("lock_recheck"), Mapping):
        raise ValueError("候选lock已经解封过；禁止使用同一lock重复复核或调参。")
    ledger_root = resolve_path(config["generation"]["report_root"]) / "_lock_ledger"
    ledger_root.mkdir(parents=True, exist_ok=True)
    marker = ledger_root / f"{candidate_id}.json"
    clean_config = {key: value for key, value in config.items() if not str(key).startswith("_")}
    record = {
        "candidate": candidate_id,
        "status": "unsealed",
        "unsealed_at": datetime.now().isoformat(timespec="seconds"),
        "config_sha256": config_sha256(clean_config),
    }
    try:
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as exc:
        raise ValueError("候选lock解封凭据已存在；禁止重复复核。") from exc
    record["marker"] = rel_path(marker)
    record["marker_sha256"] = sha256_file(marker)
    generation["lock_recheck"] = record
    _atomic_registry_write(registry_path, raw, f"lock-unsealed:{candidate_id}:{record['marker_sha256']}")
    return record


def _run_production_recheck(
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    parent_id: str,
    candidate_id: str,
    root_generation_id: str,
    base_images: Sequence[Path],
    historical_incremental_images: Sequence[Path],
    current_incremental_images: Sequence[Path],
) -> Dict[str, list[Dict[str, Any]]]:
    base_before = _run_engine(_engine(config, registry, root_generation_id), base_images, config)
    base_after = _run_engine(_engine(config, registry, candidate_id), base_images, config)
    historical_before = (
        _run_engine(_engine(config, registry, parent_id), historical_incremental_images, config)
        if historical_incremental_images else []
    )
    historical_after = (
        _run_engine(_engine(config, registry, candidate_id), historical_incremental_images, config)
        if historical_incremental_images else []
    )
    current_after = _run_engine(
        _engine(config, registry, candidate_id), current_incremental_images, config
    )
    return {
        "base_before": base_before,
        "base_after": base_after,
        "historical_before": historical_before,
        "historical_after": historical_after,
        "current_after": current_after,
    }


def _recheck_generation(config: Mapping[str, Any], candidate_id: str) -> Dict[str, Any]:
    registry = ensure_recheck_candidate(config, candidate_id)
    generation = registry["generations_by_id"][candidate_id]
    parent_id = str(generation["parent"])
    old_ids = sorted(int(value) for value in generation["old_class_ids"])
    focus_ids = sorted(int(value) for value in generation["new_class_ids"] or generation.get("updated_class_ids", []))
    if not old_ids or not focus_ids:
        raise ValueError("候选代际缺少动态新旧类别集合。")

    old_split = resolve_path(config["generation"]["recheck_lock_split"])
    old_images = [resolve_path(line.strip()) for line in old_split.read_text(encoding="utf-8").splitlines() if line.strip()]
    lock_chain = _incremental_lock_chain(registry, candidate_id)
    if not lock_chain or lock_chain[-1]["generation_id"] != candidate_id:
        raise ValueError("候选代际缺少当前轮封存lock。")
    historical_groups = lock_chain[:-1]
    current_group = lock_chain[-1]
    historical_incremental_images = list(dict.fromkeys(
        path for group in historical_groups for path in group["images"]
    ))
    historical_images = list(dict.fromkeys([*old_images, *historical_incremental_images]))
    current_images = list(current_group["images"])
    image_paths = list(dict.fromkeys([*historical_images, *current_images]))
    stems = [path.stem for path in image_paths]
    if len(stems) != len(set(stems)):
        raise ValueError("累计lock存在重复图像stem。")
    if not historical_images or not current_images or any(not path.is_file() for path in image_paths):
        raise ValueError("复核lock为空或包含缺失图像。")
    lock_record = _unseal_lock_once(config, candidate_id)
    historical_ground_truth = _read_ground_truth(old_images)
    for group in historical_groups:
        historical_ground_truth.extend(_read_ground_truth(group["images"], group["local_to_global"]))
    current_ground_truth = _read_ground_truth(current_images, current_group["local_to_global"])
    ground_truth = historical_ground_truth + current_ground_truth
    started = time.perf_counter()
    event_log_from_config(config).append(
        "incremental.lock.unsealed", component="incremental", generation_id=candidate_id,
        details={
            "base_lock_count": len(old_images),
            "historical_incremental_lock_count": sum(len(group["images"]) for group in historical_groups),
            "current_incremental_lock_count": len(current_images),
            "cumulative_lock_count": len(image_paths),
            "one_time_lock_record": lock_record,
        },
    )
    metric_registry = copy.deepcopy(registry)
    metric_floor = float(config["inference"]["confidence_min"])
    for model in metric_registry["models_by_id"].values():
        if model["role"] in {"class_incremental_expert", "target_incremental_expert"}:
            model["per_class_thresholds"] = {
                class_id: metric_floor for class_id in model["per_class_thresholds"]
            }
    root_generation_id = _root_generation_id(registry, candidate_id)
    scoped_results = _run_production_recheck(
        config,
        metric_registry,
        parent_id,
        candidate_id,
        root_generation_id,
        old_images,
        historical_incremental_images,
        current_images,
    )
    before_base_results = scoped_results["base_before"]
    after_base_results = scoped_results["base_after"]
    before_results = [*before_base_results, *scoped_results["historical_before"]]
    after_results = [
        *after_base_results,
        *scoped_results["historical_after"],
        *scoped_results["current_after"],
    ]
    before = _prediction_rows(before_results)
    after = _prediction_rows(after_results)
    historical_stems = {path.stem for path in historical_images}
    current_stems = {path.stem for path in current_images}
    after_historical = [row for row in after if row["image_id"] in historical_stems]
    after_current = [row for row in after if row["image_id"] in current_stems]
    retention = retention_metrics(before, after_historical, historical_ground_truth, old_ids)
    base_model_ids = [
        model_id for model_id in _generation_model_ids(generation)
        if registry["models_by_id"][model_id]["role"] == "frozen_base"
    ]
    if len(base_model_ids) != 1:
        raise ValueError("候选代际必须有且只有一个冻结基础模型。")
    base_class_ids = sorted(registry["models_by_id"][base_model_ids[0]]["owns_classes"])
    base_predictions = _prediction_rows(after_base_results)
    base_stems = {path.stem for path in old_images}
    base_ground_truth = _read_ground_truth(old_images)
    base_metrics = evaluate_ap50(
        [row for row in base_predictions if row["image_id"] in base_stems],
        base_ground_truth,
        base_class_ids,
    )
    historical_after_metrics = evaluate_ap50(after_historical, historical_ground_truth, old_ids)
    new_metrics = evaluate_ap50(after_current, current_ground_truth, focus_ids)
    combined_metrics = evaluate_ap50(after, ground_truth, sorted(int(value) for value in generation["classes"]))

    parent = registry["generations_by_id"][parent_id]
    candidate_model_ids = set(_generation_model_ids(generation)) - set(_generation_model_ids(parent))
    models = registry["models_by_id"]
    thresholds: Dict[int, float] = {}
    for model_id in candidate_model_ids:
        thresholds.update(models[model_id]["per_class_thresholds"])
    if set(focus_ids) - set(thresholds):
        raise ValueError("当前轮候选专家缺少逐类阈值。")
    per_class = {}
    false_activation_rates = []
    precisions = []
    for class_id in focus_ids:
        deployment = _class_deployment_metrics(
            after, ground_truth, image_paths, class_id, thresholds[class_id]
        )
        false_rate = float(deployment["false_activation_rate"])
        class_map = evaluate_ap50(after_current, current_ground_truth, [class_id])["map50"]
        per_class[str(class_id)] = {
            "map50": float(class_map), "precision": float(deployment["precision"]),
            "recall": float(deployment["recall"]), "false_activation_rate": float(false_rate),
            "negative_image_count": int(deployment["negative_image_count"]),
            "false_positive_image_count": int(deployment["false_positive_image_count"]),
            "threshold": thresholds[class_id],
        }
        false_activation_rates.append(false_rate)
        precisions.append(float(deployment["precision"]))
    cumulative_per_class = {
        str(class_id): float(evaluate_ap50(after, ground_truth, [class_id])["map50"])
        for class_id in sorted(int(value) for value in generation["classes"])
    }
    metrics = {
        "base_map50": float(base_metrics["map50"]),
        "old_map50_before": float(retention["old_map50_before"]),
        "old_map50_after": float(retention["old_map50_after"]),
        "historical_old_map50_after": float(historical_after_metrics["map50"]),
        "new_map50": float(new_metrics["map50"]),
        "krr": float(retention["krr"]),
        "combined_map50": float(combined_metrics["map50"]),
        "lock_precision": min(precisions),
        "false_activation_rate": max(false_activation_rates),
        "old_prediction_equivalent": bool(retention["old_prediction_equivalent"]),
        "per_class": per_class,
        "cumulative_per_class": cumulative_per_class,
        "image_count": len(image_paths),
        "historical_image_count": len(historical_images),
        "current_image_count": len(current_images),
        "specialist_count": sum(
            1 for model_id in _generation_model_ids(generation)
            if models[model_id]["role"] in {"class_incremental_expert", "target_incremental_expert"}
        ),
        "mean_inference_ms": (
            sum(float(item.get("inference_ms", 0.0)) for item in after_results) / len(after_results)
        ),
    }
    assessment = assess_incremental_candidate(
        metrics,
        generation.get("data_compliance") or {},
        config["gates"],
        config["incremental_guardian"],
    )
    gates = {
        name: bool(result["passed"])
        for name, result in assessment["official_hard"].items()
    }
    diagnostic_checks = {
        **{
            name: bool(result["passed"])
            for name, result in assessment["advisory"].items()
        },
        "old_prediction_equivalent": metrics["old_prediction_equivalent"],
    }
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = resolve_path(config["generation"]["report_root"]) / f"{candidate_id}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    before_path = run_dir / "predictions_before.jsonl"
    after_path = run_dir / "predictions_after.jsonl"
    for path, rows in ((before_path, before), (after_path, after)):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    clean_config = {key: value for key, value in config.items() if not str(key).startswith("_")}
    registry_path = active_generation_registry(config)
    manifest = {
        "schema_version": 2, "candidate": candidate_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.perf_counter() - started,
        "config_sha256": config_sha256(clean_config),
        "registry_sha256": sha256_file(registry_path),
        "weights": {
            model_id: registry["models_by_id"][model_id]["sha256"]
            for model_id in _generation_model_ids(generation)
        },
        "deployments": {
            model_id: {
                deployment_id: {
                    "path": deployment["path"],
                    "sha256": deployment["sha256"],
                    "calibration_manifest": deployment.get("calibration_manifest"),
                    "calibration_manifest_sha256": deployment.get("calibration_manifest_sha256"),
                }
                for deployment_id, deployment in registry["models_by_id"][model_id].get("deployments", {}).items()
            }
            for model_id in _generation_model_ids(generation)
        },
        "model_members": _generation_model_ids(generation),
        "metric_confidence_floor": metric_floor,
        "old_class_ids": old_ids, "focus_class_ids": focus_ids,
        "base_class_ids": base_class_ids,
        "base_metric_generation_id": root_generation_id,
        "lock_groups": {
            "base": [rel_path(path) for path in old_images],
            "historical_incremental": [
                {
                    "generation_id": group["generation_id"],
                    "images": [rel_path(path) for path in group["images"]],
                }
                for group in historical_groups
            ],
            "current": {
                "generation_id": current_group["generation_id"],
                "images": [rel_path(path) for path in current_images],
            },
        },
        "evaluation_semantics": "unlabeled_production_all_owners_v2",
        "inference_generations": {
            "base_before": root_generation_id,
            "base_after": candidate_id,
            "historical_before": parent_id,
            "historical_after": candidate_id,
            "current_after": candidate_id,
        },
        "one_time_lock_record": lock_record,
        "thresholds": {str(key): value for key, value in thresholds.items()},
        "metrics": metrics,
        "gates": gates,
        "diagnostic_checks": diagnostic_checks,
        "guardian_assessment": assessment,
        "warnings": list(dict.fromkeys([
            *assessment["warnings"],
            *(["old_prediction_equivalent"] if not metrics["old_prediction_equivalent"] else []),
        ])),
        "accepted": bool(assessment["accepted"]),
        "predictions_before": rel_path(before_path), "predictions_before_sha256": sha256_file(before_path),
        "predictions_after": rel_path(after_path), "predictions_after_sha256": sha256_file(after_path),
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
    registry_path = active_generation_registry(config)
    if manifest.get("registry_sha256") != sha256_file(registry_path):
        raise ValueError("代际注册表在复核后发生变化。")
    registry = _raw_registry(registry_path)
    generations = {str(item["id"]): item for item in registry["generations"]}
    models = {str(item["id"]): item for item in registry["models"]}
    generation = generations.get(candidate_id)
    if generation is None:
        raise ValueError("候选代际未注册。")
    for model_id, expected in manifest["weights"].items():
        model = models[model_id]
        if model["sha256"] != expected or sha256_file(resolve_path(model["path"])) != expected:
            raise ValueError(f"模型权重哈希不一致：{model_id}")
    for model_id, deployments in manifest.get("deployments", {}).items():
        registered = models[model_id].get("deployments", {})
        for deployment_id, expected in deployments.items():
            deployment = registered.get(deployment_id)
            if not isinstance(deployment, Mapping):
                raise ValueError(f"部署资产未注册：{model_id}:{deployment_id}")
            if deployment.get("sha256") != expected.get("sha256") or sha256_file(
                resolve_path(deployment["path"])
            ) != expected.get("sha256"):
                raise ValueError(f"部署engine哈希不一致：{model_id}:{deployment_id}")
            calibration_path = expected.get("calibration_manifest")
            calibration_hash = expected.get("calibration_manifest_sha256")
            if calibration_path and (
                not resolve_path(calibration_path).is_file()
                or sha256_file(resolve_path(calibration_path)) != calibration_hash
            ):
                raise ValueError(f"部署校准证据哈希不一致：{model_id}:{deployment_id}")
    parent = generations[str(generation["parent"])]
    candidate_models = set(_generation_model_ids(generation)) - set(_generation_model_ids(parent))
    generation["metrics"] = dict(manifest["metrics"])
    generation["lock_recheck"] = {
        **dict(generation.get("lock_recheck") or {}),
        "status": "completed",
        "manifest": rel_path(path),
        "manifest_sha256": sha256_file(path),
    }
    generation["acceptance"] = {"core_metrics_passed": True, "deployment_recheck_passed": True}
    generation["status"] = "active"
    for model_id in candidate_models:
        models[model_id]["deployment_metrics"] = {
            "lock_precision": manifest["metrics"]["lock_precision"],
            "false_activation_rate": manifest["metrics"]["false_activation_rate"],
        }
        models[model_id]["acceptance"]["passed"] = True
        models[model_id]["status"] = "active"
    registry["channels"]["production"] = candidate_id
    registry["channels"]["candidate"] = candidate_id
    _atomic_registry_write(registry_path, registry, f"promote:{candidate_id}:{sha256_file(path)}")
    return {"production": candidate_id, "manifest": rel_path(path), "registry_sha256": sha256_file(registry_path)}


def _rollback_generation(config: Mapping[str, Any], target_id: str) -> Dict[str, Any]:
    registry_path = active_generation_registry(config)
    registry = _raw_registry(registry_path)
    generations = {str(item["id"]): item for item in registry["generations"]}
    if target_id not in generations or generations[target_id].get("status") != "active":
        raise ValueError("只能回滚到已验证的active代际。")
    previous = str(registry["channels"]["production"])
    registry["channels"]["production"] = target_id
    _atomic_registry_write(registry_path, registry, f"rollback:{previous}->{target_id}")
    return {"previous": previous, "production": target_id, "registry_sha256": sha256_file(registry_path)}


def _audited_generation_action(config: Mapping[str, Any], action: str, generation_id: str, callback: Any) -> Dict[str, Any]:
    logger = event_log_from_config(config)
    registry_path = active_generation_registry(config)
    before_hash = sha256_file(registry_path)
    before = _raw_registry(registry_path)
    production_before = str(before.get("channels", {}).get("production"))
    prefix = {"recheck": "generation.lock_recheck", "promote": "generation.production_switch", "rollback": "generation.rollback"}[action]
    trace_id = f"generation_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    logger.append(f"{prefix}.started", component="generation", trace_id=trace_id, generation_id=generation_id,
                  details={"production_before": production_before, "registry_sha256_before": before_hash})
    started = time.perf_counter()
    try:
        result = callback()
    except Exception as exc:
        logger.append(f"{prefix}.failed", level="error", component="generation", trace_id=trace_id,
                      generation_id=generation_id, duration_ms=(time.perf_counter() - started) * 1000,
                      message=str(exc), details={"error_type": type(exc).__name__, "production_before": production_before,
                                                 "registry_sha256_before": before_hash})
        raise
    after = _raw_registry(registry_path)
    logger.append(f"{prefix}.completed", component="generation", trace_id=trace_id, generation_id=generation_id,
                  duration_ms=(time.perf_counter() - started) * 1000,
                  details={"production_before": production_before,
                           "production_after": str(after.get("channels", {}).get("production")),
                           "registry_sha256_before": before_hash,
                           "registry_sha256_after": sha256_file(registry_path), **dict(result)})
    return result


def recheck_generation(config: Mapping[str, Any], candidate_id: str) -> Dict[str, Any]:
    return _audited_generation_action(config, "recheck", candidate_id, lambda: _recheck_generation(config, candidate_id))


def promote_generation(config: Mapping[str, Any], candidate_id: str, manifest_path: str | Path) -> Dict[str, Any]:
    return _audited_generation_action(config, "promote", candidate_id, lambda: _promote_generation(config, candidate_id, manifest_path))


def rollback_generation(config: Mapping[str, Any], target_id: str) -> Dict[str, Any]:
    return _audited_generation_action(config, "rollback", target_id, lambda: _rollback_generation(config, target_id))
