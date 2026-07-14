from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from fair_agent.core.config import resolve_path
from fair_agent.core.hashes import sha256_file


def load_generation_registry(path: str | Path) -> Dict[str, Any]:
    resolved = resolve_path(path)
    registry = json.loads(resolved.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
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
            if models[owner]["role"] == "class_incremental_expert" and generation.get("status") == "active":
                acceptance = models[owner].get("acceptance", {})
                if acceptance.get("passed") is not True:
                    raise ValueError(f"未通过部署门禁的增量专家不得进入active代际：{generation['id']}")
                threshold = models[owner].get("activation_threshold")
                if threshold is None or not 0.01 <= float(threshold) <= 1.0:
                    raise ValueError(f"active增量专家缺少有效激活阈值：{owner}")
        generation["class_owners"] = owners
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


def generation_web_settings(registry: Mapping[str, Any], channel: str = "production") -> Dict[str, Any]:
    generation_id = str(registry["channels"][channel])
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
    for model_id in model_ids:
        model = models[model_id]
        if model["role"] != "class_incremental_expert":
            continue
        owned = [class_id for class_id, owner in generation["class_owners"].items() if owner == model_id]
        if len(owned) != 1:
            raise ValueError(f"当前运行时要求每个类别专家只拥有一个类别：{model_id}")
        class_id = owned[0]
        protocols[model_id] = {
            "id": model_id,
            "class_name": registry["class_map"][class_id],
            "new_class": registry["class_map"][class_id],
            "global_class_id": class_id,
            "incremental_mode": "class_incremental",
            "weights": model["resolved_path"],
            "new_map50": float(model["metrics"]["new_map50"]),
            "krr": float(generation["metrics"]["krr"]),
            "available": generation["status"] == "active",
            "activation_threshold": model["activation_threshold"],
            "calibration_source": model["calibration_source"],
            "routing_prior": 1.0,
            "context_prior": {},
            "evidence_level": "verified",
            "acceptance": "passed" if generation["status"] == "active" else "pending",
        }
    return {
        "generation_id": generation_id,
        "generation_status": generation["status"],
        "base_model_id": str(base["id"]),
        "detector_path": base["resolved_path"],
        "class_names": dict(registry["class_map"]),
        "active_class_ids": sorted(int(value) for value in generation["classes"]),
        "class_owners": dict(generation["class_owners"]),
        "base_class_ids": sorted(base["owns_classes"]),
        "base_local_to_global": dict(base["local_to_global"]),
        "protocols": protocols,
    }
