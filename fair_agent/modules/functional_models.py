from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.hashes import hash_if_exists
from fair_agent.modules.model_generations import load_generation_registry
from fair_agent.modules.strict_incremental import discover_experiment_profiles


def _load_registry(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("功能模型注册表顶层必须是映射。")
    return data


def _incremental_protocol_contract_valid(protocol: Any) -> bool:
    if not isinstance(protocol, dict):
        return False
    if (
        protocol.get("task_type") != "incremental_object_detection"
        or protocol.get("incremental_mode")
        not in {"class_incremental", "target_incremental"}
        or protocol.get("learning_data_scope") != "incremental_dataset_only"
        or protocol.get("evidence_level")
        not in {"unavailable", "rehearsal_only", "verified"}
        or (bool(protocol.get("available")) and not protocol.get("path"))
    ):
        return False
    raw_ids = protocol.get("global_class_ids")
    if isinstance(raw_ids, list):
        try:
            class_ids = [int(value) for value in raw_ids]
            names = {
                int(key): str(value)
                for key, value in dict(protocol.get("class_names") or {}).items()
            }
            thresholds = {
                int(key): float(value)
                for key, value in dict(
                    protocol.get("activation_thresholds") or {}
                ).items()
            }
            sources = {
                int(key): str(value)
                for key, value in dict(
                    protocol.get("calibration_sources") or {}
                ).items()
            }
        except (TypeError, ValueError):
            return False
        return bool(class_ids) and len(class_ids) == len(set(class_ids)) and (
            set(class_ids) == set(names) == set(thresholds) == set(sources)
            and all(names.values())
            and all(0.01 <= value <= 1.0 for value in thresholds.values())
        )
    try:
        int(protocol.get("global_class_id"))
        threshold = float(protocol.get("activation_threshold"))
    except (TypeError, ValueError):
        return False
    return bool(protocol.get("class_name")) and 0.01 <= threshold <= 1.0


def _load_production_ascend_models(
    registry: Mapping[str, Any], errors: list[str]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    deployment_cfg = registry.get("deployment_manifest")
    if not isinstance(deployment_cfg, Mapping) or not deployment_cfg.get("path"):
        errors.append("functional_deployment_manifest_missing")
        return {}, {}

    manifest_path = resolve_path(str(deployment_cfg["path"]))
    status = hash_if_exists(manifest_path)
    expected_sha256 = str(deployment_cfg.get("sha256") or "")
    if not status.get("exists"):
        errors.append("functional_deployment_manifest_missing")
        return {}, {
            "path": rel_path(manifest_path),
            **status,
            "expected_sha256": expected_sha256,
        }
    if not expected_sha256 or status.get("sha256") != expected_sha256:
        errors.append("functional_deployment_manifest_hash_mismatch")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"functional_deployment_manifest_invalid:{exc}")
        return {}, {
            "path": rel_path(manifest_path),
            **status,
            "expected_sha256": expected_sha256,
        }
    if not isinstance(manifest, dict):
        errors.append("functional_deployment_manifest_invalid:top_level_not_mapping")
        return {}, {
            "path": rel_path(manifest_path),
            **status,
            "expected_sha256": expected_sha256,
        }

    releases = manifest.get("ascend_releases")
    production = [
        row
        for row in releases
        if isinstance(row, dict) and row.get("status") == "production"
    ] if isinstance(releases, list) else []
    if len(production) != 1:
        errors.append("functional_ascend_production_release_invalid")
        return {}, {
            "path": rel_path(manifest_path),
            **status,
            "expected_sha256": expected_sha256,
        }
    release = production[0]
    models = release.get("models")
    if (
        release.get("target") != "Ascend310B1"
        or release.get("ready_without_training") is not True
        or not isinstance(models, dict)
        or not models
    ):
        errors.append("functional_ascend_production_release_invalid")
        models = {}
    return dict(models), {
        "path": rel_path(manifest_path),
        **status,
        "expected_sha256": expected_sha256,
        "release_id": release.get("id"),
        "target": release.get("target"),
        "model_keys": sorted(str(key) for key in models),
    }


def validate_functional_models(path: str | Path) -> Dict[str, Any]:
    registry_path = resolve_path(path)
    errors: list[str] = []
    if not registry_path.exists():
        return {"valid": False, "errors": ["functional_registry_missing"], "model_count": 0, "models": [], "collaboration": []}
    try:
        registry = _load_registry(registry_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return {"valid": False, "errors": [f"functional_registry_invalid:{exc}"], "model_count": 0, "models": [], "collaboration": []}

    entries = registry.get("models", [])
    if not isinstance(entries, list) or len(entries) < 3:
        errors.append("functional_model_count_below_three")
        entries = entries if isinstance(entries, list) else []
    ids = [str(item.get("id") or "") for item in entries if isinstance(item, dict)]
    functions = [str(item.get("function") or "") for item in entries if isinstance(item, dict)]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        errors.append("functional_model_ids_invalid")
    if len(functions) != len(set(functions)) or any(not value for value in functions):
        errors.append("functional_model_functions_not_distinct")

    production_ascend_models, deployment_manifest = _load_production_ascend_models(
        registry, errors
    )

    strict_profiles = discover_experiment_profiles()
    if strict_profiles["errors"]:
        errors.extend(f"strict_class_incremental_profile_invalid:{item}" for item in strict_profiles["errors"])
    production_incremental: Dict[str, Any] | None = None
    generation_registry_path = registry.get("generation_registry")
    if not generation_registry_path:
        errors.append("functional_generation_registry_missing")
    else:
        try:
            generations = load_generation_registry(generation_registry_path)
            production_id = str(generations["channels"]["production"])
            production = generations["generations_by_id"][production_id]
            if (
                production.get("status") == "active"
                and production.get("acceptance", {}).get("deployment_recheck_passed") is True
            ):
                owner_ids = set(
                    production.get("model_members") or production.get("class_owners", {}).values()
                )
                experts = [
                    generations["models_by_id"][str(model_id)]
                    for model_id in owner_ids
                    if generations["models_by_id"][str(model_id)].get("role") == "class_incremental_expert"
                ]
                if experts:
                    expert = experts[0]
                    global_class_ids = sorted(
                        int(value) for value in expert["owns_classes"]
                    )
                    activation_thresholds = {
                        class_id: float(expert["per_class_thresholds"][class_id])
                        for class_id in global_class_ids
                    }
                    production_incremental = {
                        "generation_id": production_id,
                        "model_id": expert["id"],
                        "class_names": {
                            class_id: generations["class_map"][class_id]
                            for class_id in global_class_ids
                        },
                        "global_class_ids": global_class_ids,
                        "activation_thresholds": activation_thresholds,
                        "metrics": dict(production.get("metrics", {})),
                        "deployment_accepted": True,
                    }
                    if len(global_class_ids) == 1:
                        global_class_id = global_class_ids[0]
                        production_incremental.update(
                            {
                                "class_name": generations["class_map"][global_class_id],
                                "global_class_id": global_class_id,
                                "activation_threshold": activation_thresholds[
                                    global_class_id
                                ],
                            }
                        )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"functional_generation_registry_invalid:{exc}")
    summaries = []
    for item in entries:
        if not isinstance(item, dict):
            errors.append("functional_model_entry_not_mapping")
            continue
        model_id = str(item.get("id") or "missing")
        if not item.get("inputs") or not item.get("outputs"):
            errors.append(f"functional_contract_missing:{model_id}")
        artifacts = []
        artifact_paths: list[str] = []
        for artifact in item.get("artifacts", []):
            if not isinstance(artifact, dict):
                errors.append(f"functional_artifact_entry_invalid:{model_id}")
                continue
            artifact_path = resolve_path(artifact.get("path", ""))
            relative_artifact_path = rel_path(artifact_path)
            artifact_paths.append(relative_artifact_path)
            status = hash_if_exists(artifact_path)
            expected = str(artifact.get("sha256") or "")
            artifact_runtime = str(artifact.get("runtime") or "")
            if artifact_runtime not in {"x86_gpu", "ascend_310b"}:
                errors.append(
                    f"functional_artifact_runtime_invalid:{model_id}:"
                    f"{relative_artifact_path}"
                )
            matches = bool(status.get("exists")) and status.get("sha256") == expected
            artifacts.append(
                {
                    "path": relative_artifact_path,
                    "runtime": artifact_runtime,
                    **status,
                    "expected_sha256": expected,
                    "matches_expected": matches,
                }
            )
            if not matches:
                errors.append(
                    f"functional_artifact_invalid:{model_id}:"
                    f"{relative_artifact_path}"
                )
        if not artifacts:
            errors.append(f"functional_artifacts_missing:{model_id}")

        runtime = item.get("runtime", {})
        x86_artifacts = [
            artifact for artifact in artifacts if artifact["runtime"] == "x86_gpu"
        ]
        ascend_artifacts = [
            artifact
            for artifact in artifacts
            if artifact["runtime"] == "ascend_310b"
        ]
        if runtime.get("x86_gpu") is True and not x86_artifacts:
            errors.append(f"functional_x86_artifact_missing:{model_id}")
        if runtime.get("ascend_310b") is True:
            ascend_model_key = str(runtime.get("ascend_model_key") or "")
            production_model = production_ascend_models.get(ascend_model_key)
            if not isinstance(production_model, Mapping):
                errors.append(f"functional_ascend_model_not_production:{model_id}")
            else:
                expected_path = str(production_model.get("path") or "")
                expected_hash = str(production_model.get("sha256") or "")
                matching_artifacts = [
                    artifact
                    for artifact in ascend_artifacts
                    if artifact["path"] == expected_path
                    and artifact["expected_sha256"] == expected_hash
                ]
                if len(ascend_artifacts) != 1 or len(matching_artifacts) != 1:
                    errors.append(
                        f"functional_ascend_artifact_not_production:{model_id}"
                    )
        elif ascend_artifacts:
            errors.append(f"functional_ascend_runtime_mismatch:{model_id}")

        evidence_artifacts = x86_artifacts or artifacts
        evidence_artifact_paths = [
            str(artifact["path"]) for artifact in evidence_artifacts
        ]
        evidence_artifact_hashes = {
            str(artifact["path"]): str(artifact["expected_sha256"])
            for artifact in evidence_artifacts
        }

        evidence_cfg = item.get("evidence", {})
        evidence_path = resolve_path(evidence_cfg.get("path", ""))
        evidence = hash_if_exists(evidence_path)
        expected_evidence = str(evidence_cfg.get("sha256") or "")
        if not evidence.get("exists"):
            errors.append(f"functional_evidence_missing:{model_id}")
        elif expected_evidence and evidence.get("sha256") != expected_evidence:
            errors.append(f"functional_evidence_hash_mismatch:{model_id}")
        acceptance = None
        evidence_summary: Dict[str, Any] = {}
        if evidence_path.suffix.lower() == ".json" and evidence_path.exists():
            try:
                evidence_data = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"functional_evidence_invalid:{model_id}:{exc}")
                evidence_data = {}
            if not isinstance(evidence_data, dict):
                errors.append(f"functional_evidence_invalid:{model_id}:top_level_not_mapping")
                evidence_data = {}
            acceptance = evidence_data.get("acceptance", {}).get("passed")
            if evidence_cfg.get("require_acceptance_passed") and acceptance is not True:
                errors.append(f"functional_evidence_not_accepted:{model_id}")

            function_name = item.get("function")
            if function_name == "context_perception":
                lock = evidence_data.get("lock", {})
                evidence_weights = str(evidence_data.get("weights") or "")
                if (
                    evidence_data.get("model_id") != model_id
                    or evidence_weights not in evidence_artifact_paths
                    or evidence_data.get("weights_sha256")
                    != evidence_artifact_hashes.get(evidence_weights)
                ):
                    errors.append(f"functional_evidence_model_mismatch:{model_id}")
                evidence_summary = {
                    "lock_sensor_accuracy": lock.get("sensor_accuracy"),
                    "lock_scene_accuracy": lock.get("scene_accuracy"),
                    "lock_joint_accuracy": lock.get("joint_accuracy"),
                }
            elif function_name == "multimodal_target_detection":
                base = evidence_data.get("base_model", {})
                manifest_entry = next(
                    (
                        entry
                        for entry in evidence_data.get("functional_models", [])
                        if isinstance(entry, dict) and entry.get("id") == model_id
                    ),
                    {},
                )
                base_path = str(base.get("path") or "")
                if (
                    base_path not in evidence_artifact_paths
                    or base.get("sha256")
                    != evidence_artifact_hashes.get(base_path)
                    or manifest_entry.get("function") != function_name
                    or manifest_entry.get("status") != item.get("status")
                ):
                    errors.append(f"functional_evidence_model_mismatch:{model_id}")
                if not isinstance(base.get("base_test_map50"), (int, float)):
                    errors.append(f"functional_evidence_metric_missing:{model_id}")
                evidence_summary = {
                    "base_test_map50": base.get("base_test_map50"),
                    "evaluation_split": base.get("evaluation_split"),
                }
            elif function_name == "incremental_object_detection":
                protocols = evidence_data.get("incremental_models", [])
                manifest_entry = next(
                    (
                        entry
                        for entry in evidence_data.get("functional_models", [])
                        if isinstance(entry, dict) and entry.get("id") == model_id
                    ),
                    {},
                )
                protocol_paths = {
                    str(protocol["path"])
                    for protocol in protocols
                    if isinstance(protocol, dict) and protocol.get("path")
                }
                if (
                    protocol_paths != set(evidence_artifact_paths)
                    or manifest_entry.get("function") != function_name
                    or manifest_entry.get("status") != item.get("status")
                    or manifest_entry.get("model_count")
                    != len(evidence_artifact_paths)
                    or manifest_entry.get("protocol_count") != len(protocols)
                    or manifest_entry.get("task_type") != "incremental_object_detection"
                    or set(manifest_entry.get("supported_modes", [])) != {"class_incremental", "target_incremental"}
                    or manifest_entry.get("current_evidence_mode") not in {"class_incremental", "target_incremental"}
                    or not isinstance(manifest_entry.get("true_class_incremental_verified"), bool)
                    or manifest_entry.get("learning_data_scope") != "incremental_dataset_only"
                    or manifest_entry.get("old_raw_image_count") != 0
                    or any(
                        not _incremental_protocol_contract_valid(protocol)
                        for protocol in protocols
                    )
                ):
                    errors.append(f"functional_evidence_model_mismatch:{model_id}")
                passed_count = sum(protocol.get("acceptance") == "passed" for protocol in protocols if isinstance(protocol, dict))
                if passed_count == len(protocols) and item.get("status") != "verified":
                    errors.append(f"functional_status_mismatch:{model_id}")
                if passed_count < len(protocols) and item.get("status") != "partially_verified":
                    errors.append(f"functional_status_mismatch:{model_id}")
                evidence_summary = {
                    "task_type": manifest_entry.get("task_type"),
                    "primary_mode": manifest_entry.get("primary_mode"),
                    "current_evidence_mode": manifest_entry.get("current_evidence_mode"),
                    "true_class_incremental_verified": bool(
                        manifest_entry.get("true_class_incremental_verified")
                        or strict_profiles["true_class_incremental_verified"]
                        or production_incremental
                    ),
                    "supported_modes": manifest_entry.get("supported_modes"),
                    "learning_data_scope": manifest_entry.get("learning_data_scope"),
                    "old_raw_image_count": manifest_entry.get("old_raw_image_count"),
                    "protocol_count": len(protocols),
                    "passed_protocol_count": passed_count,
                    "strict_verified_count": strict_profiles["verified_count"],
                    "strict_core_verified_count": strict_profiles["core_verified_count"],
                    "production_class_incremental": production_incremental,
                    "strict_verified_profiles": [
                        {
                            "profile_id": profile["profile_id"],
                            "new_class": profile["new_class"],
                            "new_class_ids": profile.get("new_global_ids"),
                            "new_map50": profile["new_map50"],
                            "krr": profile["krr"],
                            "activation_threshold": profile.get("activation_threshold"),
                            "activation_thresholds": profile.get(
                                "activation_thresholds"
                            ),
                            "lock_false_activation_rate": profile.get("lock_false_activation_rate"),
                            "deployment_accepted": profile.get("deployment_accepted"),
                        }
                        for profile in strict_profiles["profiles"]
                    ],
                }
        summaries.append(
            {
                "id": model_id,
                "function": item.get("function"),
                "display_name": item.get("display_name"),
                "implementation": item.get("implementation"),
                "status": item.get("status"),
                "inputs": list(item.get("inputs", [])),
                "outputs": list(item.get("outputs", [])),
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
                "evidence": {
                    "path": rel_path(evidence_path),
                    **evidence,
                    "acceptance_passed": acceptance,
                    "summary": evidence_summary,
                },
                "x86_gpu": bool(runtime.get("x86_gpu")),
                "ascend_310b": bool(runtime.get("ascend_310b")),
                "runtime": dict(runtime),
            }
        )

    collaboration = registry.get("collaboration", [])
    known_ids = set(ids)
    for edge in collaboration:
        if edge.get("from") not in known_ids or edge.get("to") not in known_ids or not edge.get("payload"):
            errors.append("functional_collaboration_invalid")
    if len(collaboration) < 2:
        errors.append("functional_collaboration_incomplete")

    effective_strict_profiles = {
        **strict_profiles,
        "true_class_incremental_verified": bool(
            strict_profiles["true_class_incremental_verified"] or production_incremental
        ),
        "production_profile": production_incremental,
    }
    return {
        "valid": not errors,
        "errors": errors,
        "registry": rel_path(registry_path),
        "schema_version": registry.get("schema_version"),
        "model_count": len(summaries),
        "distinct_function_count": len(set(functions)),
        "all_x86_gpu_ready": bool(summaries) and all(item["x86_gpu"] for item in summaries),
        "all_ascend_310b_ready": bool(summaries) and all(item["ascend_310b"] for item in summaries),
        "deployment_manifest": deployment_manifest,
        "models": summaries,
        "collaboration": collaboration,
        "strict_class_incremental": effective_strict_profiles,
    }
