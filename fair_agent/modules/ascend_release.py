from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.hashes import sha256_file


EXPECTED_CONTRACTS = {
    "base": [1, 736, 896, 3],
    "specialist": [1, 512, 640, 3],
    "dual_detector": [1, 736, 896, 3],
    "context": [1, 160, 160, 3],
}
REQUIRED_VALIDATION_REPORTS = ("golden", "accuracy", "performance")
DETECTIONS_V1_OUTPUTS = {
    "boxes": {"shape": [300, 4], "dtype": "float32"},
    "scores": {"shape": [300], "dtype": "float32"},
    "class_ids": {"shape": [300], "dtype": "int32"},
    "valid_count": {"shape": [1], "dtype": "int32"},
}


def _decoded_candidates_v1_outputs(
    candidate_capacity: int,
    anchor_count: int,
    class_count: int,
) -> Dict[str, Dict[str, Any]]:
    return {
        "boxes": {
            "shape": [int(candidate_capacity), 4],
            "dtype": "float32",
        },
        "scores": {
            "shape": [int(candidate_capacity)],
            "dtype": "float32",
        },
        "class_ids": {
            "shape": [int(candidate_capacity)],
            "dtype": "int32",
        },
        "anchor_ids": {
            "shape": [int(candidate_capacity)],
            "dtype": "int32",
        },
        "valid_count": {"shape": [1], "dtype": "int32"},
        "overflow": {"shape": [1], "dtype": "int32"},
        "raw_output": {
            "shape": [1, 4 + int(class_count), int(anchor_count)],
            "dtype": "float32",
        },
    }


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


def _structural_logical_heads(value: Any) -> Dict[str, Any]:
    """Return the OM-bound dual-head contract without Host-only thresholds."""

    if not isinstance(value, Mapping):
        return {}
    normalized: Dict[str, Any] = {}
    for name, raw in value.items():
        if not isinstance(raw, Mapping):
            normalized[str(name)] = raw
            continue
        head = dict(raw)
        head.pop("candidate_confidence", None)
        normalized[str(name)] = head
    return normalized


def verify_ascend_artifacts(
    options: Mapping[str, Any],
    *,
    require_validation: bool | None = None,
) -> Dict[str, Any]:
    """Verify a DVPP/AIPP candidate without loading ACL or an OM.

    The build manifest is device-produced and immutable. The separate validation
    summary links golden, 89-image accuracy, and 890-request performance reports.
    """

    errors: List[str] = []
    model_layout = str(
        options.get("model_layout", "independent_models_v1")
    )
    shared_dual_head = model_layout == "shared_backbone_dual_head_v1"
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
        expected_shape = EXPECTED_CONTRACTS[role]
        if not isinstance(contract, Mapping) or (
            contract.get("dtype") != "uint8"
            or contract.get("layout") != "NHWC"
            or contract.get("shape") != expected_shape
        ):
            errors.append(f"input_contract_mismatch:{model_id}:{role}")
    expected_role_counts = (
        {"base": 0, "specialist": 0, "dual_detector": 1, "context": 1}
        if shared_dual_head
        else {"base": 1, "specialist": 1, "dual_detector": 0, "context": 1}
    )
    for role, expected_count in expected_role_counts.items():
        rows = by_role[role]
        if len(rows) != expected_count:
            errors.append(f"artifact_role_count_invalid:{role}:{len(rows)}")

    configured_detectors, configured_context = _configured_om_rows(options)
    configured_contracts = _configured_detector_contracts(options)
    detector_roles = (
        ("dual_detector",) if shared_dual_head else ("base", "specialist")
    )
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
            configured_name = str(
                (configured or {}).get("output_contract", "raw_yolo_v1")
            )
            manifest_name = str(row.get("output_contract", "raw_yolo_v1"))
            if manifest_name != configured_name:
                errors.append(f"output_contract_mismatch:{role}")
                continue
            contract = row.get("postprocess_contract")
            if configured_name == "raw_dual_head_v1":
                configured_heads = dict(
                    (configured or {}).get("logical_heads") or {}
                )
                if _structural_logical_heads(
                    row.get("logical_heads")
                ) != _structural_logical_heads(configured_heads):
                    errors.append(f"logical_heads_contract_mismatch:{role}")
                if contract is not None:
                    errors.append(f"raw_output_has_postprocess_contract:{role}")
            elif configured_name == "detections_v1":
                expected = {
                    "candidate_confidence": float(
                        (configured or {}).get("candidate_confidence", -1.0)
                    ),
                    "iou_threshold": float(
                        (configured or {}).get("iou_threshold", -1.0)
                    ),
                    "max_det": int((configured or {}).get("max_det", 0)),
                    "outputs": DETECTIONS_V1_OUTPUTS,
                }
                if contract != expected:
                    errors.append(f"postprocess_contract_mismatch:{role}")
            elif configured_name == "decoded_candidates_v1":
                expected = {
                    "candidate_confidence": float(
                        (configured or {}).get("candidate_confidence", -1.0)
                    ),
                    "candidate_capacity": int(
                        (configured or {}).get("candidate_capacity", 0)
                    ),
                    "anchor_count": int(
                        (configured or {}).get("anchor_count", 0)
                    ),
                    "class_count": int(
                        (configured or {}).get("class_count", 0)
                    ),
                    "outputs": _decoded_candidates_v1_outputs(
                        int((configured or {}).get("candidate_capacity", 0)),
                        int((configured or {}).get("anchor_count", 0)),
                        int((configured or {}).get("class_count", 0)),
                    ),
                }
                if contract != expected:
                    errors.append(f"postprocess_contract_mismatch:{role}")
            elif contract is not None:
                errors.append(f"raw_output_has_postprocess_contract:{role}")
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
            for name in REQUIRED_VALIDATION_REPORTS:
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
