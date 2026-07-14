from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.hashes import hash_if_exists
from fair_agent.modules.strict_incremental import discover_experiment_profiles


def _load_registry(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("功能模型注册表顶层必须是映射。")
    return data


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

    strict_profiles = discover_experiment_profiles()
    if strict_profiles["errors"]:
        errors.extend(f"strict_class_incremental_profile_invalid:{item}" for item in strict_profiles["errors"])
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
            artifact_paths.append(rel_path(artifact_path))
            status = hash_if_exists(artifact_path)
            expected = str(artifact.get("sha256") or "")
            matches = bool(status.get("exists")) and status.get("sha256") == expected
            artifacts.append({"path": rel_path(artifact_path), **status, "expected_sha256": expected, "matches_expected": matches})
            if not matches:
                errors.append(f"functional_artifact_invalid:{model_id}:{rel_path(artifact_path)}")
        if not artifacts:
            errors.append(f"functional_artifacts_missing:{model_id}")

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
                expected_artifact_hash = artifacts[0].get("expected_sha256") if artifacts else None
                if (
                    evidence_data.get("model_id") != model_id
                    or evidence_data.get("weights") not in artifact_paths
                    or evidence_data.get("weights_sha256") != expected_artifact_hash
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
                expected_artifact_hash = artifacts[0].get("expected_sha256") if artifacts else None
                if (
                    base.get("path") not in artifact_paths
                    or base.get("sha256") != expected_artifact_hash
                    or manifest_entry.get("function") != function_name
                    or manifest_entry.get("status") != item.get("status")
                ):
                    errors.append(f"functional_evidence_model_mismatch:{model_id}")
                if not isinstance(base.get("lock_all_map50"), (int, float)):
                    errors.append(f"functional_evidence_metric_missing:{model_id}")
                evidence_summary = {"lock_all_map50": base.get("lock_all_map50")}
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
                protocol_paths = {str(protocol.get("path") or "") for protocol in protocols if isinstance(protocol, dict)}
                if (
                    protocol_paths != set(artifact_paths)
                    or manifest_entry.get("function") != function_name
                    or manifest_entry.get("status") != item.get("status")
                    or manifest_entry.get("model_count") != len(protocols)
                    or manifest_entry.get("task_type") != "incremental_object_detection"
                    or set(manifest_entry.get("supported_modes", [])) != {"class_incremental", "target_incremental"}
                    or manifest_entry.get("current_evidence_mode") != "target_incremental"
                    or manifest_entry.get("true_class_incremental_verified") is not False
                    or manifest_entry.get("learning_data_scope") != "incremental_dataset_only"
                    or manifest_entry.get("old_raw_image_count") != 0
                    or any(
                        not isinstance(protocol, dict)
                        or protocol.get("task_type") != "incremental_object_detection"
                        or protocol.get("incremental_mode") not in {"class_incremental", "target_incremental"}
                        or protocol.get("learning_data_scope") != "incremental_dataset_only"
                        or not isinstance(protocol.get("global_class_id"), int)
                        or not protocol.get("class_name")
                        or not isinstance(protocol.get("activation_threshold"), (int, float))
                        or protocol.get("evidence_level") not in {"unavailable", "rehearsal_only", "verified"}
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
                    ),
                    "supported_modes": manifest_entry.get("supported_modes"),
                    "learning_data_scope": manifest_entry.get("learning_data_scope"),
                    "old_raw_image_count": manifest_entry.get("old_raw_image_count"),
                    "protocol_count": len(protocols),
                    "passed_protocol_count": passed_count,
                    "strict_verified_count": strict_profiles["verified_count"],
                    "strict_verified_profiles": [
                        {
                            "profile_id": profile["profile_id"],
                            "new_class": profile["new_class"],
                            "new_map50": profile["new_map50"],
                            "krr": profile["krr"],
                            "activation_threshold": profile["activation_threshold"],
                            "lock_false_activation_rate": profile.get("lock_false_activation_rate"),
                        }
                        for profile in strict_profiles["profiles"]
                    ],
                }

        runtime = item.get("runtime", {})
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
            }
        )

    collaboration = registry.get("collaboration", [])
    known_ids = set(ids)
    for edge in collaboration:
        if edge.get("from") not in known_ids or edge.get("to") not in known_ids or not edge.get("payload"):
            errors.append("functional_collaboration_invalid")
    if len(collaboration) < 2:
        errors.append("functional_collaboration_incomplete")

    return {
        "valid": not errors,
        "errors": errors,
        "registry": rel_path(registry_path),
        "schema_version": registry.get("schema_version"),
        "model_count": len(summaries),
        "distinct_function_count": len(set(functions)),
        "all_x86_gpu_ready": bool(summaries) and all(item["x86_gpu"] for item in summaries),
        "all_ascend_310b_ready": bool(summaries) and all(item["ascend_310b"] for item in summaries),
        "models": summaries,
        "collaboration": collaboration,
        "strict_class_incremental": strict_profiles,
    }
