from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.hashes import sha256_file


def load_generation_registry(path: str | Path) -> Dict[str, Any]:
    resolved = resolve_path(path)
    registry = json.loads(resolved.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 2:
        raise ValueError("模型代际注册表版本不受支持")
    classes = {int(key): str(value) for key, value in registry.get("class_map", {}).items()}
    models = {str(item["id"]): dict(item) for item in registry.get("models", [])}
    generations = {str(item["id"]): dict(item) for item in registry.get("generations", [])}
    if len(models) != len(registry.get("models", [])) or len(generations) != len(registry.get("generations", [])):
        raise ValueError("模型或代际ID重复")
    for model in models.values():
        model["owns_classes"] = {int(value) for value in model.get("owns_classes", [])}
        model["local_to_global"] = {
            int(key): int(value) for key, value in model.get("local_to_global", {}).items()
        }
        if set(model["local_to_global"].values()) != model["owns_classes"]:
            raise ValueError(f"模型本地映射与类别所有权不一致：{model['id']}")
        raw_thresholds = model.get("per_class_thresholds")
        if not isinstance(raw_thresholds, Mapping) and len(model["owns_classes"]) == 1 and model.get("activation_threshold") is not None:
            raw_thresholds = {str(next(iter(model["owns_classes"]))): model["activation_threshold"]}
        model["per_class_thresholds"] = {
            int(key): float(value) for key, value in (raw_thresholds or {}).items()
        }
        raw_sources = model.get("calibration_sources")
        if not isinstance(raw_sources, Mapping) and len(model["owns_classes"]) == 1 and model.get("calibration_source"):
            raw_sources = {str(next(iter(model["owns_classes"]))): model["calibration_source"]}
        model["calibration_sources"] = {
            int(key): str(value) for key, value in (raw_sources or {}).items()
        }
        deployments = {}
        for deployment_id, raw_deployment in (model.get("deployments") or {}).items():
            if not isinstance(raw_deployment, Mapping):
                raise ValueError(f"模型部署资产格式非法：{model['id']}:{deployment_id}")
            deployment = dict(raw_deployment)
            artifact_path = resolve_path(deployment.get("path", ""))
            expected = str(deployment.get("sha256") or "")
            deployment["resolved_path"] = artifact_path
            deployment["hash_valid"] = artifact_path.is_file() and len(expected) == 64 and sha256_file(artifact_path) == expected
            deployments[str(deployment_id)] = deployment
        model["deployments"] = deployments
        artifact = resolve_path(model["path"])
        model["resolved_path"] = artifact
        model["hash_valid"] = artifact.is_file() and sha256_file(artifact) == model["sha256"]
    for generation in generations.values():
        owners = {int(key): str(value) for key, value in generation["class_owners"].items()}
        if set(owners) != set(int(value) for value in generation["classes"]):
            raise ValueError(f"代际类别所有权不完整：{generation['id']}")
        if any(owner not in models for owner in owners.values()):
            raise ValueError(f"代际引用未知模型：{generation['id']}")
        for class_id, owner in owners.items():
            if class_id not in models[owner]["owns_classes"]:
                raise ValueError(f"代际所有者未登记对应类别：{generation['id']}:{class_id}")
            if models[owner]["role"] == "benchmark_only":
                raise ValueError(f"benchmark_only模型不得进入运行代际：{generation['id']}")
            if models[owner]["role"] in {"class_incremental_expert", "target_incremental_expert"} and generation.get("status") == "active":
                acceptance = models[owner].get("acceptance", {})
                if acceptance.get("passed") is not True:
                    raise ValueError(f"未通过部署门禁的增量专家不得进入active代际：{generation['id']}")
                thresholds = models[owner]["per_class_thresholds"]
                missing = models[owner]["owns_classes"] - set(thresholds)
                invalid = [value for value in thresholds.values() if not 0.01 <= float(value) <= 1.0]
                if missing or invalid:
                    raise ValueError(f"active增量专家缺少有效逐类激活阈值：{owner}")
                missing_sources = models[owner]["owns_classes"] - set(models[owner]["calibration_sources"])
                if missing_sources:
                    raise ValueError(f"active增量专家缺少逐类dev校准证据：{owner}")
        generation["class_owners"] = owners
        generation["old_class_ids"] = {int(value) for value in generation.get("old_class_ids", [])}
        generation["new_class_ids"] = {int(value) for value in generation.get("new_class_ids", [])}
        generation["updated_class_ids"] = {int(value) for value in generation.get("updated_class_ids", [])}
        if generation["old_class_ids"] & generation["new_class_ids"]:
            raise ValueError(f"代际新旧类别集合重叠：{generation['id']}")
        if generation.get("parent") and generation["old_class_ids"] | generation["new_class_ids"] | generation["updated_class_ids"] != set(owners):
            raise ValueError(f"代际新旧类别集合与所有权不一致：{generation['id']}")
    channels = registry.get("channels", {})
    for channel in ("production", "candidate"):
        if str(channels.get(channel)) not in generations:
            raise ValueError(f"{channel}频道未引用有效代际")
    production = generations[str(channels["production"])]
    if production.get("status") != "active":
        raise ValueError("production频道只能引用active代际")
    benchmark_id = str(channels.get("benchmark") or "")
    if benchmark_id not in models or models[benchmark_id].get("role") != "benchmark_only":
        raise ValueError("benchmark频道必须引用benchmark_only模型")
    registry["class_map"] = classes
    registry["models_by_id"] = models
    registry["generations_by_id"] = generations
    return registry


def generation_settings(registry: Mapping[str, Any], generation_id: str) -> Dict[str, Any]:
    generation_id = str(generation_id)
    generation = registry["generations_by_id"][generation_id]
    model_ids = list(dict.fromkeys(generation["class_owners"].values()))
    models = registry["models_by_id"]
    if any(not models[model_id]["hash_valid"] for model_id in model_ids):
        raise ValueError(f"活动代际存在缺失或哈希错误的权重：{generation_id}")
    base_candidates = [models[model_id] for model_id in model_ids if models[model_id]["role"] == "frozen_base"]
    if len(base_candidates) != 1:
        raise ValueError(f"活动代际必须有且只有一个冻结基础模型：{generation_id}")
    base = base_candidates[0]
    protocols = {}
    engine_deployments: Dict[str, Dict[str, Any]] = {}
    for model_id in model_ids:
        model = models[model_id]
        deployment = model.get("deployments", {}).get("tensorrt_int8")
        if deployment and deployment.get("hash_valid"):
            engine_deployments[rel_path(model["resolved_path"])] = {
                key: value
                for key, value in deployment.items()
                if key in {"path", "sha256", "imgsz", "min_batch_size", "opt_batch_size", "batch_size"}
            }
        if model["role"] not in {"class_incremental_expert", "target_incremental_expert"}:
            continue
        owned = sorted(class_id for class_id, owner in generation["class_owners"].items() if owner == model_id)
        if not owned:
            raise ValueError(f"增量专家没有类别所有权：{model_id}")
        thresholds = {class_id: float(model["per_class_thresholds"][class_id]) for class_id in owned}
        calibration_sources = {
            class_id: str(model["calibration_sources"][class_id]) for class_id in owned
        }
        metrics_by_class = {
            int(key): dict(value) for key, value in model.get("metrics", {}).get("per_class", {}).items()
        }
        first_class = owned[0]
        protocols[model_id] = {
            "id": model_id,
            "display_name": str(model.get("display_name") or model_id),
            "class_names": {class_id: registry["class_map"][class_id] for class_id in owned},
            "global_class_ids": owned,
            "local_to_global": dict(model["local_to_global"]),
            "class_name": registry["class_map"][first_class] if len(owned) == 1 else "、".join(registry["class_map"][value] for value in owned),
            "new_class": registry["class_map"][first_class] if len(owned) == 1 else "、".join(registry["class_map"][value] for value in owned),
            "global_class_id": first_class if len(owned) == 1 else None,
            "incremental_mode": str(model.get("incremental_mode") or ("class_incremental" if model["role"] == "class_incremental_expert" else "target_incremental")),
            "weights": model["resolved_path"],
            "new_map50": float(model.get("metrics", {}).get("new_map50", 0.0)),
            "per_class_metrics": metrics_by_class,
            "krr": float(generation["metrics"]["krr"]),
            "available": generation["status"] in {"active", "registered_candidate", "pending_deployment_recheck"},
            "activation_thresholds": thresholds,
            "calibration_sources": calibration_sources,
            "activation_threshold": thresholds[first_class] if len(owned) == 1 else None,
            "calibration_source": calibration_sources[first_class] if len(owned) == 1 else None,
            "routing_prior": 1.0,
            "context_prior": {},
            "evidence_level": "verified",
            "acceptance": "passed" if generation["status"] == "active" else "pending",
        }
    return {
        "generation_id": generation_id,
        "generation_name": str(generation.get("display_name") or generation_id),
        "generation_status": generation["status"],
        "base_model_id": str(base["id"]),
        "base_model_name": str(base.get("display_name") or base["id"]),
        "detector_path": base["resolved_path"],
        "class_names": dict(registry["class_map"]),
        "active_class_ids": sorted(int(value) for value in generation["classes"]),
        "class_owners": dict(generation["class_owners"]),
        "base_class_ids": sorted(base["owns_classes"]),
        "base_local_to_global": dict(base["local_to_global"]),
        "protocols": protocols,
        "engine_deployments": engine_deployments,
    }


def generation_web_settings(registry: Mapping[str, Any], channel: str = "production") -> Dict[str, Any]:
    return generation_settings(registry, str(registry["channels"][channel]))
