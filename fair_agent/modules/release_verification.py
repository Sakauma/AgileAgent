from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

from fair_agent.core.blackboard import build_blackboard, resolve_demo_evidence
from fair_agent.core.config import ROOT, load_config, rel_path, resolve_path
from fair_agent.core.hashes import hash_if_exists, verify_sha256s
from fair_agent.modules.functional_models import validate_functional_models
from fair_agent.modules.incremental_round_registry import (
    load_incremental_round_registry,
)
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
        multi_class = isinstance(item.get("global_class_ids"), list)
        try:
            if multi_class:
                global_class_ids = [
                    int(value) for value in item["global_class_ids"]
                ]
                class_names = {
                    int(key): str(value)
                    for key, value in dict(item.get("class_names") or {}).items()
                }
                thresholds = {
                    int(key): float(value)
                    for key, value in dict(
                        item.get("activation_thresholds") or {}
                    ).items()
                }
                calibration_sources = {
                    int(key): str(value)
                    for key, value in dict(
                        item.get("calibration_sources") or {}
                    ).items()
                }
            else:
                global_class_id = int(item.get("global_class_id"))
                global_class_ids = [global_class_id]
                class_names = {
                    global_class_id: str(item.get("class_name") or "")
                }
                thresholds = {
                    global_class_id: float(item.get("activation_threshold"))
                }
                calibration_sources = (
                    {global_class_id: str(item.get("calibration_source"))}
                    if item.get("calibration_source")
                    else {}
                )
        except (TypeError, ValueError):
            errors.append(f"manifest_incremental_fields_invalid:{protocol_id}")
            continue
        if mode not in {"class_incremental", "target_incremental"}:
            errors.append(f"manifest_incremental_mode_invalid:{protocol_id}")
        if (
            not global_class_ids
            or len(global_class_ids) != len(set(global_class_ids))
            or set(global_class_ids) != set(class_names)
            or set(global_class_ids) != set(thresholds)
            or not all(class_names.values())
            or not all(0.01 <= value <= 1.0 for value in thresholds.values())
            or item.get("evidence_level")
            not in {"unavailable", "rehearsal_only", "verified"}
        ):
            errors.append(f"manifest_incremental_fields_invalid:{protocol_id}")
        if mode == "target_incremental" and (
            len(global_class_ids) != 1
            or base_class_map.get(global_class_ids[0])
            != class_names.get(global_class_ids[0])
        ):
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
            if set(global_class_ids) & base_ids:
                errors.append(f"manifest_new_class_overlaps_base:{protocol_id}")
            if item.get("available") and (
                set(calibration_sources) != set(global_class_ids)
                or not all(calibration_sources.values())
                or item.get("evidence_level") != "verified"
            ):
                errors.append(f"manifest_new_class_calibration_missing:{protocol_id}")
        if item.get("available") and item.get("competition_accepted") is not True:
            errors.append(f"manifest_competition_gates_missing:{protocol_id}")
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
    production_expert_paths: set[str] = set()
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
        production_expert_paths = {
            rel_path(model["resolved_path"]) for model in production_experts
        }
        if any(
            model.get("acceptance", {}).get("competition_gates_passed") is not True
            for model in production_experts
        ):
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
    if not functional["all_ascend_310b_ready"]:
        errors.append("functional_models_not_ascend_310b_ready")
    incremental_functional = next(
        (
            row
            for row in functional.get("models", [])
            if row.get("id") == "incremental_model_bank_v1"
        ),
        None,
    )
    functional_expert_paths = {
        str(item.get("path"))
        for item in (
            incremental_functional.get("artifacts", [])
            if isinstance(incremental_functional, Mapping)
            else []
        )
        if item.get("runtime") == "x86_gpu"
    }
    if production_expert_paths != functional_expert_paths:
        errors.append("functional_incremental_assets_not_production_generation")
    round_registry_summary: Dict[str, Any] = {}
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
        scope_definition = incremental_policy.get("scope_definition", {})
        incremental_scope = scope_definition.get("incremental_learning", {})
        if (
            incremental_scope.get("counted_as_incremental_learning") is not True
            or incremental_scope.get("training_data_scope")
            != "incremental_dataset_only"
            or incremental_scope.get("validation_data_scope")
            != "incremental_dataset_only"
            or incremental_scope.get("base_detector_weights_frozen") is not True
            or "scene_sensor_model_training"
            not in incremental_scope.get("excludes", [])
            or "scene_gate_search" not in incremental_scope.get("excludes", [])
        ):
            errors.append("incremental_detection_scope_definition_invalid")
        system_calibration = scope_definition.get("system_calibration", {})
        if (
            system_calibration.get("counted_as_incremental_learning") is not False
            or system_calibration.get("detector_weights_updated") is not False
            or system_calibration.get("base_detector_weights_frozen") is not True
            or system_calibration.get("incremental_detector_weights_frozen")
            is not True
            or "base_incremental_lock_for_functional_model_recheck_only"
            not in system_calibration.get("allowed_data_scopes", [])
        ):
            errors.append("incremental_detection_system_calibration_scope_invalid")
        system_calibration_phase = incremental_policy.get(
            "system_calibration_phase", {}
        )
        calibration_data_roles = system_calibration_phase.get("data_roles", {})
        if (
            system_calibration_phase.get("phase") != "system_calibration"
            or system_calibration_phase.get("counted_as_incremental_learning")
            is not False
            or system_calibration_phase.get("detector_weights_updated") is not False
            or calibration_data_roles.get("base_context_prior")
            != "base_train_only"
            or calibration_data_roles.get("incremental_context_prior")
            != "incremental_train_only"
            or calibration_data_roles.get("gate_selection") != "mixed_dev_only"
        ):
            errors.append("incremental_detection_system_calibration_phase_invalid")
        learning_phase = incremental_policy.get("learning_phase", {})
        if (
            learning_phase.get("phase") != "incremental_learning"
            or learning_phase.get("training_data_scope")
            != "incremental_dataset_only"
        ):
            errors.append("incremental_detection_training_scope_invalid")
        if learning_phase.get("validation_data_scope") != "incremental_dataset_only":
            errors.append("incremental_detection_validation_scope_invalid")
        if "old_sample_replay" not in learning_phase.get("forbidden_inputs", []):
            errors.append("incremental_detection_replay_gate_missing")
        evaluation_phase = incremental_policy.get("evaluation_phase", {})
        if (
            evaluation_phase.get("phase") != "joint_evaluation"
            or evaluation_phase.get("detector_weights_updated") is not False
            or evaluation_phase.get("model_selection_allowed") is not False
        ):
            errors.append("incremental_detection_evaluation_scope_invalid")

        round_execution = incremental_policy.get("round_execution", {})
        configured_round_registry = str(
            config["incremental"].get("round_registry") or ""
        )
        if (
            not configured_round_registry
            or round_execution.get("registry") != configured_round_registry
            or round_execution.get("class_ids_from_registry_only") is not True
            or round_execution.get("fixed_new_class_ids_forbidden") is not True
            or round_execution.get("require_parent_child_generation_chain")
            is not True
            or round_execution.get("require_distinct_new_classes_across_rounds")
            is not True
        ):
            errors.append("incremental_round_execution_contract_invalid")
        try:
            round_registry = load_incremental_round_registry(
                configured_round_registry
            )
            minimum_rounds = int(
                config["incremental"].get("minimum_distinct_new_class_rounds")
                or 0
            )
            round_rows = list(round_registry["rounds"])
            distinct_new_classes = {
                class_id
                for row in round_rows
                for class_id in row["new_class_ids"]
            }
            if (
                minimum_rounds < 2
                or len(round_rows) < minimum_rounds
                or len(distinct_new_classes) < minimum_rounds
            ):
                errors.append("incremental_round_count_invalid")
            source_workflow = {
                "register": str(
                    config["incremental"].get(
                        "round_candidate_registration_tool"
                    )
                    or ""
                ),
                "summarize": str(
                    config["incremental"].get("round_summary_tool") or ""
                ),
                "promote": str(
                    config["incremental"].get("strict_promotion_tool") or ""
                ),
            }
            expected_workflow = {
                "register": "tools/13_register_incremental_round_candidate.py",
                "summarize": "tools/12_summarize_incremental_rounds.py",
                "promote": "tools/10_promote_scene_aware_4plus2.py",
            }
            if (
                source_workflow != expected_workflow
                or config["incremental"].get("strict_runtime_source")
                != "models/generations.json"
                or round_execution.get("candidate_registration_required")
                is not True
                or round_execution.get(
                    "promotion_requires_complete_round_evidence"
                )
                is not True
                or round_execution.get(
                    "production_switch_only_after_final_round_lock"
                )
                is not True
            ):
                errors.append("incremental_round_source_workflow_invalid")
            elif any(
                not resolve_path(path).is_file()
                for path in source_workflow.values()
            ):
                errors.append("incremental_round_source_tool_missing")
            split_counts: Dict[str, Dict[str, int]] = {}
            for split_role in ("train", "dev", "lock"):
                source_references = {
                    str(row["source_splits"][split_role]) for row in round_rows
                }
                if len(source_references) != 1:
                    errors.append(
                        f"incremental_round_source_split_invalid:{split_role}"
                    )
                    continue
                source_path = resolve_path(source_references.pop())
                source_rows = {
                    value.strip()
                    for value in source_path.read_text(encoding="utf-8").splitlines()
                    if value.strip()
                }
                cumulative_rows: set[str] = set()
                for row in round_rows:
                    round_id = str(row["round_id"])
                    split_path = resolve_path(row["splits"][split_role])
                    values = [
                        value.strip()
                        for value in split_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if value.strip()
                    ]
                    if (
                        not values
                        or len(values) != len(set(values))
                        or cumulative_rows & set(values)
                    ):
                        errors.append(
                            f"incremental_round_split_invalid:{round_id}:{split_role}"
                        )
                    cumulative_rows.update(values)
                    split_counts.setdefault(round_id, {})[split_role] = len(values)
                if cumulative_rows != source_rows:
                    errors.append(
                        f"incremental_round_split_coverage_invalid:{split_role}"
                    )
            split_manifest = json.loads(
                (ROOT / "splits" / "strict_4plus2" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest_round_contract = dict(
                split_manifest.get("increment_rounds") or {}
            )
            manifest_rounds = list(
                manifest_round_contract.get("rounds") or []
            )
            manifest_owners = dict(split_manifest.get("owners") or {})
            if (
                manifest_round_contract.get("registry")
                != configured_round_registry
                or manifest_round_contract.get("materializer")
                != "tools/11_prepare_incremental_round_splits.py"
                or len(manifest_rounds) != len(round_rows)
            ):
                errors.append("incremental_round_split_manifest_invalid")
            else:
                for expected, recorded in zip(round_rows, manifest_rounds):
                    model_id = str(expected["specialist"]["model_id"])
                    owner = dict(manifest_owners.get(model_id) or {})
                    if (
                        recorded.get("round_id") != expected["round_id"]
                        or recorded.get("round_index")
                        != expected["round_index"]
                        or recorded.get("parent_generation_id")
                        != expected["parent_generation_id"]
                        or recorded.get("generation_id")
                        != expected["generation_id"]
                        or recorded.get("new_global_class_ids")
                        != expected["new_class_ids"]
                        or dict(recorded.get("counts") or {})
                        != split_counts.get(str(expected["round_id"]), {})
                        or owner.get("round_id") != expected["round_id"]
                        or owner.get("global_class_ids")
                        != expected["new_class_ids"]
                        or {
                            int(key): int(value)
                            for key, value in dict(
                                owner.get("local_to_global") or {}
                            ).items()
                        }
                        != expected["specialist"]["local_to_global"]
                    ):
                        errors.append(
                            "incremental_round_split_manifest_invalid:"
                            f"{expected['round_id']}"
                        )
            scene_boundary = incremental_policy.get("scene_system_boundary", {})
            if (
                config["incremental"].get("scene_sensor_is_incremental_learner")
                is not False
                or scene_boundary.get("scene_sensor_is_incremental_learner")
                is not False
                or scene_boundary.get("phase") != "system_calibration"
                or scene_boundary.get("detector_weights_updated") is not False
            ):
                errors.append("incremental_scene_system_boundary_invalid")
            round_registry_summary = {
                "path": rel_path(Path(round_registry["path"])),
                "protocol": round_registry["protocol_id"],
                "round_count": len(round_rows),
                "distinct_new_class_count": len(distinct_new_classes),
                "split_counts": split_counts,
                "source_workflow": source_workflow,
                "runtime_source": "models/generations.json",
                "scene_sensor_is_incremental_learner": False,
            }
        except (KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"incremental_round_registry_invalid:{exc}")

    inference_configs = {}
    expected_base_imgsz = int(manifest.get("base_model", {}).get("imgsz") or 0)
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
        if int(details["imgsz"] or 0) != expected_base_imgsz:
            errors.append(f"inference_imgsz_mismatch:{name}")
        if int(details["batch"] or 0) != 32:
            errors.append(f"inference_batch_not_32:{name}")

    demo_path = resolve_demo_evidence(config)
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

    release_blockers = [
        blocker
        for blocker in state.get("current_blockers", [])
        if blocker
        not in {
            "official_test_not_ready",
            "official_format_not_confirmed",
            "official_test_dir_missing",
        }
    ]

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
            "valid": not any(
                error.startswith(
                    (
                        "incremental_detection_",
                        "incremental_round_",
                        "incremental_scene_",
                    )
                )
                for error in errors
            ),
            "task_type": incremental_policy.get("task_type"),
        },
        "incremental_round_registry": round_registry_summary,
        "evidence_mode": state.get("evidence", {}).get("mode"),
        "blockers": release_blockers,
    }
