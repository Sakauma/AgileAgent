from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.hashes import sha256_file


EXPECTED_CONTRACTS = {
    "base": ([1, 608, 736, 3],),
    "specialist": ([1, 608, 736, 3],),
    "context": ([1, 160, 160, 3],),
}
REQUIRED_VALIDATION_REPORTS = ("golden", "accuracy", "performance")
FULL_SCORE_VALIDATION_REPORTS = ("accuracy", "performance")
FULL_SCORE_ACCURACY_GATES = {
    "base_map50": 0.80,
    "new_map50": 0.60,
    "krr": 0.95,
}
FULL_SCORE_BATCH_IMAGE_COUNT = 20
FULL_SCORE_BATCH_ROUNDS = 3
FULL_SCORE_BATCH_FPS_MIN = 30.0
FULL_SCORE_FPS_CALCULATION = "total_frames_divided_by_total_elapsed_seconds"
FULL_SCORE_TIMED_COMPONENTS = (
    "image_decode",
    "scene_model",
    "decision_model",
    "base_detector",
    "incremental_detector",
    "postprocess",
    "formal_result_write",
)
LEGACY_FULL_SCORE_PERFORMANCE_SCHEMAS = {5, 6, 7}


def _load_json(path: Path, errors: List[str], label: str) -> Dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing_{label}:{path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_{label}:{path}:{exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"invalid_{label}:{path}:top_level_not_mapping")
        return {}
    return payload


def _verify_file_entry(
    entry: Any,
    errors: List[str],
    label: str,
) -> Path | None:
    if not isinstance(entry, Mapping) or not entry.get("path"):
        errors.append(f"invalid_file_entry:{label}")
        return None
    path = resolve_path(str(entry["path"]))
    digest = entry.get("sha256")
    if not path.is_file():
        errors.append(f"missing_file:{label}:{path}")
        return path
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append(f"missing_sha256:{label}")
    elif sha256_file(path) != digest:
        errors.append(f"sha256_mismatch:{label}:{path}")
    return path


def _verify_full_score_method_entry(
    entry: Any,
    errors: List[str],
) -> int | None:
    path = _verify_file_entry(entry, errors, "full_score_method")
    if path is None or not path.is_file():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        errors.append(f"invalid_full_score_method:{path}:{exc}")
        return None
    if not isinstance(payload, Mapping) or payload.get("kind") != "ascend310b_full_score_method":
        errors.append("full_score_method_kind_invalid")
        return None
    schema_version = payload.get("schema_version")
    if schema_version not in {1, 2}:
        errors.append("full_score_method_schema_invalid")
        return None
    accuracy = (payload.get("competition") or {}).get("accuracy_gates") or {}
    performance = (payload.get("competition") or {}).get("performance_gate") or {}
    expected_accuracy = {
        "base_map50_min": FULL_SCORE_ACCURACY_GATES["base_map50"],
        "new_map50_min": FULL_SCORE_ACCURACY_GATES["new_map50"],
        "krr_min": FULL_SCORE_ACCURACY_GATES["krr"],
    }
    try:
        accuracy_matches = all(
            float(accuracy.get(name, -1.0)) == minimum
            for name, minimum in expected_accuracy.items()
        )
        performance_matches = (
            int(performance.get("batch_image_count", 0))
            == FULL_SCORE_BATCH_IMAGE_COUNT
            and int(performance.get("batch_rounds", 0))
            == FULL_SCORE_BATCH_ROUNDS
        )
        if schema_version == 1:
            performance_matches = performance_matches and (
                float(performance.get("median_fps_min", -1.0))
                == FULL_SCORE_BATCH_FPS_MIN
            )
        else:
            performance_matches = performance_matches and (
                float(performance.get("aggregate_fps_min", -1.0))
                == FULL_SCORE_BATCH_FPS_MIN
                and performance.get("calculation")
                == FULL_SCORE_FPS_CALCULATION
                and performance.get("includes_result_persistence") is True
                and tuple(performance.get("required_components") or ())
                == FULL_SCORE_TIMED_COMPONENTS
            )
    except (TypeError, ValueError):
        accuracy_matches = False
        performance_matches = False
    if not accuracy_matches or not performance_matches:
        errors.append("full_score_method_gate_contract_invalid")
    return int(schema_version)


def _verify_full_score_accuracy_report(
    report: Mapping[str, Any],
    errors: List[str],
) -> None:
    if report.get("schema_version") != 2:
        errors.append("full_score_accuracy_schema_invalid")
    if report.get("unlabeled_predictions_frozen_before_labels") is not True:
        errors.append("full_score_predictions_not_frozen_before_labels")
    metrics = report.get("metrics") or {}
    try:
        for name, minimum in FULL_SCORE_ACCURACY_GATES.items():
            if float(metrics.get(name, -1.0)) < minimum:
                errors.append(f"full_score_accuracy_gate_failed:{name}")
    except (TypeError, ValueError):
        errors.append("full_score_accuracy_metrics_invalid")
    gates = report.get("competition_gates")
    if not isinstance(gates, Mapping) or set(gates) != set(FULL_SCORE_ACCURACY_GATES):
        errors.append("full_score_accuracy_gates_invalid")
    elif any(value is not True for value in gates.values()):
        errors.append("full_score_accuracy_gate_report_failed")
    if report.get("score_passed", report.get("passed")) is not True:
        errors.append("full_score_accuracy_report_not_passed")


def _verify_full_score_performance_report(
    report: Mapping[str, Any],
    errors: List[str],
    *,
    method_schema_version: int | None = None,
) -> None:
    schema_version = report.get("schema_version")
    allowed_schemas = LEGACY_FULL_SCORE_PERFORMANCE_SCHEMAS | {8}
    if schema_version not in allowed_schemas:
        errors.append("full_score_performance_schema_invalid")
    if (
        method_schema_version == 1
        and schema_version not in LEGACY_FULL_SCORE_PERFORMANCE_SCHEMAS
    ) or (method_schema_version == 2 and schema_version != 8):
        errors.append("full_score_method_performance_schema_mismatch")
    protocol = report.get("protocol") or {}
    competition = report.get("competition") or {}
    rounds = competition.get("batch_rounds")
    try:
        contract_matches = (
            int(protocol.get("batch_probe_size", 0))
            == FULL_SCORE_BATCH_IMAGE_COUNT
            and int(protocol.get("batch_rounds", 0)) == FULL_SCORE_BATCH_ROUNDS
            and float(protocol.get("target_batch_fps", -1.0))
            == FULL_SCORE_BATCH_FPS_MIN
            and int(competition.get("batch_image_count", 0))
            == FULL_SCORE_BATCH_IMAGE_COUNT
            and isinstance(rounds, list)
            and len(rounds) == FULL_SCORE_BATCH_ROUNDS
        )
    except (TypeError, ValueError):
        contract_matches = False
    if not contract_matches:
        errors.append("full_score_performance_contract_invalid")
    if schema_version == 7 and competition.get("batch_timing_source") != (
        "api_timings.batch_engine_ms"
    ):
        errors.append("full_score_performance_timing_source_invalid")
    if schema_version == 8:
        expected_frames = FULL_SCORE_BATCH_IMAGE_COUNT * FULL_SCORE_BATCH_ROUNDS
        try:
            total_frames = int(competition.get("batch_total_frames", 0))
            total_elapsed_ms = float(
                competition.get("batch_total_elapsed_ms", 0.0)
            )
            fps = float(competition.get("batch_fps", -1.0))
            round_frames = sum(
                int(row.get("image_count", 0))
                for row in rounds
                if isinstance(row, Mapping)
            )
            round_elapsed_ms = sum(
                float(row.get("full_pipeline_wall_ms", 0.0))
                for row in rounds
                if isinstance(row, Mapping)
            )
            current_contract_matches = (
                protocol.get("fps_calculation") == FULL_SCORE_FPS_CALCULATION
                and tuple(protocol.get("timed_components") or ())
                == FULL_SCORE_TIMED_COMPONENTS
                and protocol.get("formal_result_format")
                == "class_id x_center y_center width height confidence"
                and competition.get("batch_timing_source")
                == "client_full_pipeline_wall_ms"
                and competition.get("batch_fps_calculation")
                == FULL_SCORE_FPS_CALCULATION
                and competition.get("includes_result_persistence") is True
                and competition.get("formal_results_valid") is True
                and total_frames == expected_frames
                and round_frames == expected_frames
                and total_elapsed_ms > 0.0
                and math.isclose(
                    round_elapsed_ms,
                    total_elapsed_ms,
                    rel_tol=1e-9,
                    abs_tol=1e-6,
                )
                and math.isclose(
                    fps,
                    total_frames * 1000.0 / total_elapsed_ms,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                and all(
                    isinstance(row, Mapping)
                    and int(row.get("image_count", 0))
                    == FULL_SCORE_BATCH_IMAGE_COUNT
                    and float(row.get("full_pipeline_wall_ms", 0.0)) > 0.0
                    and int(row.get("result_file_count", 0))
                    == FULL_SCORE_BATCH_IMAGE_COUNT
                    and row.get("formal_results_valid") is True
                    for row in rounds
                )
            )
        except (TypeError, ValueError, ZeroDivisionError):
            current_contract_matches = False
        if not current_contract_matches:
            errors.append("full_score_performance_full_pipeline_contract_invalid")
    try:
        if float(competition.get("batch_fps", -1.0)) < FULL_SCORE_BATCH_FPS_MIN:
            errors.append("full_score_performance_gate_failed")
    except (TypeError, ValueError):
        errors.append("full_score_performance_metric_invalid")
    if competition.get("batch_fps_passed") is not True:
        errors.append("full_score_performance_report_not_passed")
    gates = report.get("gates")
    if isinstance(gates, Mapping) and any(value is not True for value in gates.values()):
        errors.append("full_score_performance_aux_gate_failed")


def _configured_om_rows(options: Mapping[str, Any]) -> tuple[set[tuple[str, str]], tuple[str, str]]:
    detector_rows = {
        (str(resolve_path(str(entry["path"]))), str(entry.get("sha256") or ""))
        for entry in dict(options.get("models") or {}).values()
        if isinstance(entry, Mapping) and entry.get("path")
    }
    context = options.get("context_model") or {}
    context_row = (
        str(resolve_path(str(context.get("path") or ""))),
        str(context.get("sha256") or ""),
    )
    return detector_rows, context_row


def _configured_detector_contracts(
    options: Mapping[str, Any],
) -> Dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (
            str(resolve_path(str(entry["path"]))),
            str(entry.get("sha256") or ""),
        ): entry
        for entry in dict(options.get("models") or {}).values()
        if isinstance(entry, Mapping) and entry.get("path")
    }


def verify_ascend_artifacts(
    options: Mapping[str, Any],
    *,
    require_validation: bool | None = None,
) -> Dict[str, Any]:
    """Verify an Ascend v2 candidate without loading ACL or an OM."""

    errors: List[str] = []
    model_layout = str(options.get("model_layout") or "")
    if model_layout != "independent_yolo26_e2e_v1":
        errors.append("model_layout_invalid")
    require_validation = (
        options.get("validated") is True
        if require_validation is None
        else bool(require_validation)
    )
    if options.get("encoded_preprocessing") != "dvpp":
        return {
            "status": "not_applicable",
            "errors": [],
            "reason": "encoded_preprocessing_not_dvpp",
        }
    if options.get("execution_mode") != "async_stream":
        errors.append("dvpp_requires_async_stream")

    manifest_path = resolve_path(str(options.get("build_manifest") or ""))
    manifest_digest = str(options.get("build_manifest_sha256") or "")
    manifest = _load_json(manifest_path, errors, "build_manifest")
    if manifest_path.is_file():
        if len(manifest_digest) != 64:
            errors.append("build_manifest_sha256_missing")
        elif sha256_file(manifest_path) != manifest_digest:
            errors.append("build_manifest_sha256_mismatch")
    if manifest:
        if int(manifest.get("schema_version") or 0) != 1:
            errors.append("build_manifest_schema_invalid")
        for key in ("soc_version", "cann_version", "precision"):
            if manifest.get(key) != options.get(key):
                errors.append(f"build_manifest_{key}_mismatch")
        git_sha = str(manifest.get("git_sha") or "")
        if git_sha != "unknown" and (len(git_sha) != 40 or any(c not in "0123456789abcdef" for c in git_sha.lower())):
            errors.append("build_manifest_git_sha_invalid")

    artifacts = manifest.get("artifacts") if isinstance(manifest, Mapping) else None
    if not isinstance(artifacts, Mapping):
        errors.append("build_manifest_artifacts_invalid")
        artifacts = {}
    by_role: Dict[str, list[Mapping[str, Any]]] = {
        role: [] for role in EXPECTED_CONTRACTS
    }
    for model_id, raw in artifacts.items():
        if not isinstance(raw, Mapping):
            errors.append(f"build_artifact_invalid:{model_id}")
            continue
        role = str(raw.get("role") or "")
        if role not in by_role:
            errors.append(f"build_artifact_role_invalid:{model_id}:{role}")
            continue
        by_role[role].append(raw)
        for name in ("source_weight", "onnx", "aipp", "om", "atc_log"):
            _verify_file_entry(raw.get(name), errors, f"{model_id}.{name}")
        command = str(raw.get("atc_command") or "")
        for marker in (
            "--framework=5",
            "--input_format=NCHW",
            "--soc_version=Ascend310B1",
            "--precision_mode_v2=mixed_float16",
            "--insert_op_conf=",
        ):
            if marker not in command:
                errors.append(f"atc_command_marker_missing:{model_id}:{marker}")
        if "conda-env" in command:
            errors.append(f"atc_command_uses_removed_environment:{model_id}")
        contract = raw.get("input_contract")
        expected_shapes = EXPECTED_CONTRACTS[role]
        if not isinstance(contract, Mapping) or (
            contract.get("dtype") != "uint8"
            or contract.get("layout") != "NHWC"
            or contract.get("shape") not in expected_shapes
        ):
            errors.append(f"input_contract_mismatch:{model_id}:{role}")
    expected_role_counts = {"base": 1, "specialist": 1, "context": 1}
    for role, expected_count in expected_role_counts.items():
        rows = by_role[role]
        if len(rows) != expected_count:
            errors.append(f"artifact_role_count_invalid:{role}:{len(rows)}")

    configured_detectors, configured_context = _configured_om_rows(options)
    configured_contracts = _configured_detector_contracts(options)
    detector_roles = ("base", "specialist")
    manifest_detectors = {
        (
            str(resolve_path(str(row.get("om", {}).get("path") or ""))),
            str(row.get("om", {}).get("sha256") or ""),
        )
        for role in detector_roles
        for row in by_role[role]
    }
    manifest_context = {
        (
            str(resolve_path(str(row.get("om", {}).get("path") or ""))),
            str(row.get("om", {}).get("sha256") or ""),
        )
        for row in by_role["context"]
    }
    if configured_detectors != manifest_detectors:
        errors.append("configured_detector_oms_do_not_match_manifest")
    for role in detector_roles:
        for row in by_role[role]:
            key = (
                str(resolve_path(str(row.get("om", {}).get("path") or ""))),
                str(row.get("om", {}).get("sha256") or ""),
            )
            configured = configured_contracts.get(key)
            configured_name = str((configured or {}).get("output_contract") or "")
            manifest_name = str(row.get("output_contract") or "")
            if manifest_name != configured_name:
                errors.append(f"output_contract_mismatch:{role}")
                continue
            contract = row.get("postprocess_contract")
            if configured_name == "yolo26_e2e_v1":
                max_det = int((configured or {}).get("max_det", 0))
                class_count = int((configured or {}).get("class_count", 0))
                expected = {
                    "max_det": max_det,
                    "class_count": class_count,
                    "outputs": {
                        "detections": {
                            "shape": [1, max_det, 6],
                            "dtype": "float32",
                        }
                    },
                }
                if max_det <= 0 or class_count <= 0 or contract != expected:
                    errors.append(f"postprocess_contract_mismatch:{role}")
            else:
                errors.append(f"output_contract_invalid:{role}")
    if manifest_context != {configured_context}:
        errors.append("configured_context_om_does_not_match_manifest")

    validation_summary: Dict[str, Any] = {}
    if require_validation:
        report_path = resolve_path(str(options.get("validation_report") or ""))
        report_digest = str(options.get("validation_report_sha256") or "")
        validation_summary = _load_json(report_path, errors, "validation_report")
        if report_path.is_file():
            if len(report_digest) != 64:
                errors.append("validation_report_sha256_missing")
            elif sha256_file(report_path) != report_digest:
                errors.append("validation_report_sha256_mismatch")
        if validation_summary:
            if validation_summary.get("build_manifest_sha256") != manifest_digest:
                errors.append("validation_build_manifest_sha256_mismatch")
            full_score_release = (
                validation_summary.get("kind")
                == "ascend310b_full_score_release_validation"
            )
            required_reports = (
                FULL_SCORE_VALIDATION_REPORTS
                if full_score_release
                else REQUIRED_VALIDATION_REPORTS
            )
            if full_score_release:
                method_schema_version = _verify_full_score_method_entry(
                    validation_summary.get("method_config"), errors
                )
                validity = validation_summary.get("validity")
                required_validity = {
                    "incremental_data_isolation",
                    "asset_hashes_verified",
                    "predictions_frozen_before_labels",
                }
                required_validity.add("base_model_frozen")
                if (
                    not isinstance(validity, Mapping)
                    or any(validity.get(name) is not True for name in required_validity)
                ):
                    errors.append("full_score_validity_prerequisites_failed")
            for name in required_reports:
                entry = validation_summary.get(name)
                report = _load_json(
                    resolve_path(str(entry.get("path") or ""))
                    if isinstance(entry, Mapping)
                    else resolve_path(""),
                    errors,
                    f"{name}_report",
                )
                if not isinstance(entry, Mapping) or entry.get("passed") is not True:
                    errors.append(f"validation_gate_not_passed:{name}")
                elif report:
                    digest = str(entry.get("sha256") or "")
                    report_path = resolve_path(str(entry["path"]))
                    if len(digest) != 64 or sha256_file(report_path) != digest:
                        errors.append(f"validation_report_link_sha256_mismatch:{name}")
                    elif full_score_release and name == "accuracy":
                        _verify_full_score_accuracy_report(report, errors)
                    elif full_score_release and name == "performance":
                        _verify_full_score_performance_report(
                            report,
                            errors,
                            method_schema_version=method_schema_version,
                        )
            if validation_summary.get("passed") is not True:
                errors.append("validation_summary_not_passed")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "build_manifest": rel_path(manifest_path),
        "build_manifest_sha256": manifest_digest,
        "artifact_count": len(artifacts),
        "validation_required": require_validation,
        "validation_summary": validation_summary,
    }


def require_validated_ascend_artifacts(options: Mapping[str, Any]) -> None:
    if options.get("validated") is not True or options.get("encoded_preprocessing") != "dvpp":
        return
    result = verify_ascend_artifacts(options, require_validation=True)
    if result["status"] != "passed":
        raise RuntimeError(
            "Ascend DVPP发布资产未通过可复现性校验：" + "; ".join(result["errors"])
        )


def require_ascend_runtime_artifacts(options: Mapping[str, Any]) -> None:
    """Gate production artifacts and explicitly authorized validation candidates."""

    if options.get("validated") is True:
        require_validated_ascend_artifacts(options)
        return
    if options.get("validation_candidate") is not True:
        raise RuntimeError("Ascend后端尚未通过完整验收，禁止加载OM。")
    if os.environ.get("AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION") != "1":
        raise RuntimeError(
            "Ascend候选仅允许在AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION=1的受控验证进程中加载。"
        )
    result = verify_ascend_artifacts(options, require_validation=False)
    if result["status"] != "passed":
        raise RuntimeError(
            "Ascend候选构建资产未通过可复现性校验：" + "; ".join(result["errors"])
        )
