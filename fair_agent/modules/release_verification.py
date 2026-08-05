from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from fair_agent.core.blackboard import build_blackboard
from fair_agent.core.config import ROOT, load_config, rel_path, resolve_path
from fair_agent.core.hashes import hash_if_exists, verify_sha256s
from fair_agent.modules.functional_models import validate_functional_models
from fair_agent.modules.model_generations import load_generation_registry


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是映射：{path}")
    return data


def _validate_model_manifest(manifest: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if int(manifest.get("schema_version") or 0) < 2:
        errors.append("manifest_schema_version_invalid")
    raw_class_map = manifest.get("base_model", {}).get("class_map")
    if not isinstance(raw_class_map, dict) or not raw_class_map:
        return errors + ["manifest_base_class_map_invalid"]
    try:
        base_class_map = {int(class_id): str(name) for class_id, name in raw_class_map.items()}
    except (TypeError, ValueError):
        return errors + ["manifest_base_class_map_invalid"]
    if len(base_class_map) != len(set(base_class_map.values())) or any(not name for name in base_class_map.values()):
        errors.append("manifest_base_class_map_invalid")

    protocols = manifest.get("incremental_models")
    if not isinstance(protocols, list) or not protocols:
        return errors + ["manifest_incremental_models_missing"]
    protocol_ids: List[str] = []
    for item in protocols:
        if not isinstance(item, dict):
            errors.append("manifest_incremental_model_invalid")
            continue
        protocol_id = str(item.get("protocol") or "")
        protocol_ids.append(protocol_id)
        mode = item.get("incremental_mode")
        class_name = str(item.get("class_name") or "")
        try:
            global_class_id = int(item.get("global_class_id"))
            threshold = float(item.get("activation_threshold"))
        except (TypeError, ValueError):
            errors.append(f"manifest_incremental_fields_invalid:{protocol_id}")
            continue
        if mode not in {"class_incremental", "target_incremental"}:
            errors.append(f"manifest_incremental_mode_invalid:{protocol_id}")
        if not class_name or not 0.01 <= threshold <= 1.0 or item.get("evidence_level") not in {
            "unavailable", "rehearsal_only", "verified"
        }:
            errors.append(f"manifest_incremental_fields_invalid:{protocol_id}")
        if mode == "target_incremental" and base_class_map.get(global_class_id) != class_name:
            errors.append(f"manifest_target_class_mapping_invalid:{protocol_id}")
        if mode == "class_incremental":
            raw_base_ids = item.get("base_class_ids")
            if not isinstance(raw_base_ids, list) or not raw_base_ids:
                errors.append(f"manifest_new_class_base_scope_missing:{protocol_id}")
                base_ids = set(base_class_map)
            else:
                try:
                    base_ids = {int(class_id) for class_id in raw_base_ids}
                except (TypeError, ValueError):
                    errors.append(f"manifest_new_class_base_scope_invalid:{protocol_id}")
                    base_ids = set(base_class_map)
            if global_class_id in base_ids:
                errors.append(f"manifest_new_class_overlaps_base:{protocol_id}")
            if item.get("acceptance") == "passed" and (
                not item.get("calibration_source") or item.get("evidence_level") != "verified"
            ):
                errors.append(f"manifest_new_class_calibration_missing:{protocol_id}")
        if item.get("available") and item.get("acceptance") != "passed":
            errors.append(f"manifest_unaccepted_protocol_available:{protocol_id}")
        if item.get("available") and not item.get("path"):
            errors.append(f"manifest_available_protocol_artifact_missing:{protocol_id}")
    if any(not value for value in protocol_ids) or len(protocol_ids) != len(set(protocol_ids)):
        errors.append("manifest_incremental_protocol_ids_invalid")
    return errors


def verify_release(config_path: str | Path = "configs/agent_pipeline.yaml") -> Dict[str, Any]:
    config = load_config(config_path)
    errors: List[str] = []
    assets = config["assets"]
    checksums = verify_sha256s(resolve_path(assets["checksums"]))
    if not checksums["valid"]:
        errors.extend(f"model_checksum:{item}" for item in checksums["errors"])

    required = {}
    for name in assets["required"]:
        status = hash_if_exists(resolve_path(name))
        required[name] = status
        if not status["exists"]:
            errors.append(f"missing_required_asset:{name}")

    model = config["model"]
    model_status = hash_if_exists(resolve_path(model["weights"]))
    if model_status.get("sha256") != model["expected_sha256"]:
        errors.append("base_model_sha256_mismatch")
    manifest_path = resolve_path(assets["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if manifest.get("base_model", {}).get("sha256") != model["expected_sha256"]:
        errors.append("manifest_base_hash_mismatch")
    incremental_models = manifest.get("incremental_models", [])
    errors.extend(_validate_model_manifest(manifest))
    generation_summary: Dict[str, Any] = {}
    try:
        generation_registry = load_generation_registry(assets["generation_registry"])
        production_id = str(generation_registry["channels"]["production"])
        candidate_id = str(generation_registry["channels"]["candidate"])
        production = generation_registry["generations_by_id"][production_id]
        candidate = generation_registry["generations_by_id"][candidate_id]
        production_models = set(
            production.get("model_members") or production["class_owners"].values()
        )
        production_experts = [
            generation_registry["models_by_id"][model_id]
            for model_id in production_models
            if generation_registry["models_by_id"][model_id]["role"] == "class_incremental_expert"
        ]
        if any(model.get("acceptance", {}).get("passed") is not True for model in production_experts):
            errors.append("unverified_incremental_expert_in_production")
        if candidate.get("status") == "active" and candidate_id != production_id:
            errors.append("candidate_generation_prematurely_active")
        if any(
            not generation_registry["models_by_id"][model_id]["hash_valid"]
            for model_id in set(
                candidate.get("model_members") or candidate["class_owners"].values()
            ) | production_models
        ):
            errors.append("generation_model_hash_mismatch")
        generation_summary = {
            "path": rel_path(resolve_path(assets["generation_registry"])),
            "production": production_id,
            "production_classes": sorted(production["classes"]),
            "candidate": candidate_id,
            "candidate_status": candidate.get("status"),
            "benchmark": generation_registry["channels"]["benchmark"],
            "production_metrics": production.get("metrics", {}),
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        errors.append(f"generation_registry_invalid:{exc}")
    functional = validate_functional_models(config["functional_models"]["registry"])
    if not functional["valid"]:
        errors.extend(f"functional_models:{item}" for item in functional["errors"])
    if functional["model_count"] < int(config["functional_models"]["required_count"]):
        errors.append("functional_model_count_invalid")
    if not functional["all_x86_gpu_ready"]:
        errors.append("functional_models_not_x86_gpu_ready")
    incremental_policy_path = resolve_path(config["incremental"]["policy"])
    incremental_policy = _load_yaml(incremental_policy_path) if incremental_policy_path.exists() else {}
    if not incremental_policy:
        errors.append("incremental_detection_policy_missing")
    else:
        if incremental_policy.get("task_type") != "incremental_object_detection":
            errors.append("incremental_detection_task_type_invalid")
        if incremental_policy.get("primary_mode") != "class_incremental":
            errors.append("incremental_detection_primary_mode_invalid")
        if set(incremental_policy.get("supported_modes", [])) != {"class_incremental", "target_incremental"}:
            errors.append("incremental_detection_supported_modes_invalid")
        learning_phase = incremental_policy.get("learning_phase", {})
        if learning_phase.get("training_data_scope") != "incremental_dataset_only":
            errors.append("incremental_detection_training_scope_invalid")
        if learning_phase.get("validation_data_scope") != "incremental_dataset_only":
            errors.append("incremental_detection_validation_scope_invalid")
        if "old_sample_replay" not in learning_phase.get("forbidden_inputs", []):
            errors.append("incremental_detection_replay_gate_missing")

    inference_configs = {}
    for name in ["configs/local_infer_gpu.yaml", config["detector"]["config"]]:
        path = resolve_path(name)
        data = _load_yaml(path)
        predict = data.get("predict", {})
        details = {
            "model": data.get("model"),
            "device": str(predict.get("device")),
            "imgsz": predict.get("imgsz"),
            "batch": predict.get("batch"),
        }
        inference_configs[rel_path(path)] = details
        if data.get("model") != model["weights"]:
            errors.append(f"inference_model_mismatch:{name}")
        if data.get("expected_sha256") != model["expected_sha256"]:
            errors.append(f"inference_hash_mismatch:{name}")
        if not details["device"].isdigit():
            errors.append(f"inference_device_not_gpu:{name}")
        if int(details["imgsz"] or 0) != 640:
            errors.append(f"inference_imgsz_not_640:{name}")
        if int(details["batch"] or 0) != 32:
            errors.append(f"inference_batch_not_32:{name}")

    demo_path = resolve_path(config["blackboard"]["demo_evidence"])
    demo_text = demo_path.read_text(encoding="utf-8") if demo_path.exists() else ""
    demo = json.loads(demo_text) if demo_text else {}
    if not demo:
        errors.append("demo_evidence_missing")
    for forbidden in ["datasets_r1_base_train", ".png", ".jpg", "visualizations/"]:
        if forbidden in demo_text:
            errors.append(f"demo_contains_private_reference:{forbidden}")
    start_script = ROOT / "scripts" / "start_agent.sh"
    start_text = start_script.read_text(encoding="utf-8") if start_script.exists() else ""
    for forbidden in ["pip install", "-m venv", "torch==", "bootstrap_x86.sh\nexec"]:
        if forbidden in start_text:
            errors.append(f"start_script_mutates_environment:{forbidden}")
    runtime_sources = {
        "fair_agent/web/app.py": ("MAX_FILE_BYTES", "MAX_BATCH_FILES", "annotated_base64"),
        "fair_agent/modules/web_inference.py": ("MAX_FILE_BYTES", "MAX_BATCH_FILES", "MAX_BATCH_BYTES"),
        "fair_agent/web/static/assets/app.js": (
            "20 * 1024 * 1024", "200 * 1024 * 1024", "RESULT_CACHE_LIMIT", "annotated_base64"
        ),
    }
    for source_name, forbidden_literals in runtime_sources.items():
        source_text = (ROOT / source_name).read_text(encoding="utf-8")
        for literal in forbidden_literals:
            if literal in source_text:
                errors.append(f"runtime_config_literal:{source_name}:{literal}")
    bootstrap_script = ROOT / "scripts" / "bootstrap_x86.sh"
    bootstrap_text = bootstrap_script.read_text(encoding="utf-8") if bootstrap_script.exists() else ""
    for required_marker in ["python3.12 python3.11 python3.10", 'uv venv --python "${UV_PYTHON:-3.12}" --seed', "2.5.1+cu124", "nvidia-smi", "uname -m"]:
        if required_marker not in bootstrap_text:
            errors.append(f"bootstrap_gate_missing:{required_marker}")

    state = build_blackboard(config)
    if int(state.get("dataset", {}).get("image_count") or 0) != 750:
        errors.append("blackboard_dataset_evidence_missing")
    if len(state.get("incremental_learning", {}).get("protocols", [])) != len(incremental_models):
        errors.append("blackboard_incremental_evidence_missing")
    if config["runtime"]["server_host"] not in {"127.0.0.1", "localhost", "::1"}:
        errors.append("server_not_loopback")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "config": rel_path(resolve_path(config_path)),
        "checksums": checksums,
        "required_assets": required,
        "inference_configs": inference_configs,
        "functional_models": functional,
        "model_generations": generation_summary,
        "incremental_detection_policy": {
            "path": rel_path(incremental_policy_path),
            "valid": not any(error.startswith("incremental_detection_") for error in errors),
            "task_type": incremental_policy.get("task_type"),
        },
        "evidence_mode": state.get("evidence", {}).get("mode"),
        "blockers": state.get("current_blockers", []),
    }
