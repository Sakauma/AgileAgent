from __future__ import annotations

import json
from pathlib import Path

import pytest

from fair_agent.core.hashes import sha256_file
from fair_agent.modules.ascend_release import (
    require_ascend_runtime_artifacts,
    verify_ascend_artifacts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _candidate(tmp_path: Path) -> dict:
    specs = {
        "base_detector": ("base", [1, 736, 896, 3]),
        "incremental_detector": ("specialist", [1, 512, 640, 3]),
        "scene_sensor_net": ("context", [1, 160, 160, 3]),
    }
    artifacts = {}
    for model_id, (role, shape) in specs.items():
        entries = {}
        for name in ("source_weight", "onnx", "aipp", "om", "atc_log"):
            suffix = "cfg" if name == "aipp" else name
            path = tmp_path / f"{model_id}.{suffix}"
            path.write_bytes(f"{model_id}:{name}".encode())
            entries[name] = {"path": str(path), "sha256": sha256_file(path)}
        artifacts[model_id] = {
            "role": role,
            **entries,
            "atc_command": (
                "atc --framework=5 --input_format=NCHW "
                "--soc_version=Ascend310B1 --precision_mode_v2=mixed_float16 "
                f"--insert_op_conf={entries['aipp']['path']}"
            ),
            "input_contract": {"dtype": "uint8", "layout": "NHWC", "shape": shape},
        }
    manifest = tmp_path / "build-manifest.json"
    _write_json(manifest, {
        "schema_version": 1,
        "git_sha": "a" * 40,
        "soc_version": "Ascend310B1",
        "cann_version": "7.0.RC1",
        "precision": "mixed_float16",
        "artifacts": artifacts,
    })
    manifest_sha256 = sha256_file(manifest)

    validation_entries = {}
    for name in ("golden", "accuracy", "performance"):
        report = tmp_path / f"{name}.json"
        _write_json(report, {"passed": True})
        validation_entries[name] = {
            "path": str(report),
            "sha256": sha256_file(report),
            "passed": True,
        }
    validation = tmp_path / "validation-summary.json"
    _write_json(validation, {
        "schema_version": 1,
        "build_manifest_sha256": manifest_sha256,
        **validation_entries,
        "passed": True,
    })
    return {
        "device_id": 0,
        "soc_version": "Ascend310B1",
        "cann_version": "7.0.RC1",
        "precision": "mixed_float16",
        "execution_mode": "async_stream",
        "encoded_preprocessing": "dvpp",
        "memory_mode": "pageable",
        "validated": True,
        "build_manifest": str(manifest),
        "build_manifest_sha256": manifest_sha256,
        "validation_report": str(validation),
        "validation_report_sha256": sha256_file(validation),
        "models": {
            "base.pt": artifacts["base_detector"]["om"],
            "incremental.pt": artifacts["incremental_detector"]["om"],
        },
        "context_model": artifacts["scene_sensor_net"]["om"],
    }


def test_verified_ascend_candidate_binds_manifest_oms_and_reports(tmp_path: Path) -> None:
    result = verify_ascend_artifacts(_candidate(tmp_path))
    assert result["status"] == "passed", result["errors"]
    assert result["artifact_count"] == 3
    assert result["validation_required"] is True


def test_ascend_candidate_rejects_hash_and_input_contract_drift(tmp_path: Path) -> None:
    options = _candidate(tmp_path)
    manifest_path = Path(options["build_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"]["base_detector"]["input_contract"]["shape"] = [1, 736, 895, 3]
    _write_json(manifest_path, payload)

    result = verify_ascend_artifacts(options)

    assert result["status"] == "failed"
    assert "build_manifest_sha256_mismatch" in result["errors"]
    assert "input_contract_mismatch:base_detector:base" in result["errors"]


def test_unvalidated_candidate_can_verify_build_before_gate_reports(tmp_path: Path) -> None:
    options = _candidate(tmp_path)
    options["validated"] = False
    options["validation_report"] = None
    options["validation_report_sha256"] = None

    result = verify_ascend_artifacts(options)

    assert result["status"] == "passed", result["errors"]
    assert result["validation_required"] is False


def test_detections_v1_candidate_binds_fixed_postprocess_contract(tmp_path: Path) -> None:
    options = _candidate(tmp_path)
    manifest_path = Path(options["build_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = payload["artifacts"]["base_detector"]
    base["output_contract"] = "detections_v1"
    base["postprocess_contract"] = {
        "candidate_confidence": 0.01,
        "iou_threshold": 0.7,
        "max_det": 300,
        "outputs": {
            "boxes": {"shape": [300, 4], "dtype": "float32"},
            "scores": {"shape": [300], "dtype": "float32"},
            "class_ids": {"shape": [300], "dtype": "int32"},
            "valid_count": {"shape": [1], "dtype": "int32"},
        },
    }
    _write_json(manifest_path, payload)
    options["build_manifest_sha256"] = sha256_file(manifest_path)
    base_entry = next(
        entry for entry in options["models"].values()
        if entry["path"] == base["om"]["path"]
    )
    base_entry.update({
        "output_contract": "detections_v1",
        "candidate_confidence": 0.01,
        "iou_threshold": 0.7,
        "max_det": 300,
    })
    result = verify_ascend_artifacts(options, require_validation=False)
    assert result["status"] == "passed", result["errors"]

    base_entry["iou_threshold"] = 0.6
    result = verify_ascend_artifacts(options, require_validation=False)
    assert "postprocess_contract_mismatch:base" in result["errors"]


def test_decoded_candidates_v1_manifest_binds_raw_fallback_contract(
    tmp_path: Path,
) -> None:
    options = _candidate(tmp_path)
    manifest_path = Path(options["build_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = payload["artifacts"]["base_detector"]
    base["output_contract"] = "decoded_candidates_v1"
    base["postprocess_contract"] = {
        "candidate_confidence": 0.01,
        "candidate_capacity": 5,
        "anchor_count": 3,
        "class_count": 2,
        "outputs": {
            "boxes": {"shape": [5, 4], "dtype": "float32"},
            "scores": {"shape": [5], "dtype": "float32"},
            "class_ids": {"shape": [5], "dtype": "int32"},
            "anchor_ids": {"shape": [5], "dtype": "int32"},
            "valid_count": {"shape": [1], "dtype": "int32"},
            "overflow": {"shape": [1], "dtype": "int32"},
            "raw_output": {"shape": [1, 6, 3], "dtype": "float32"},
        },
    }
    _write_json(manifest_path, payload)
    options["build_manifest_sha256"] = sha256_file(manifest_path)
    base_entry = next(
        entry
        for entry in options["models"].values()
        if entry["path"] == base["om"]["path"]
    )
    base_entry.update(
        {
            "output_contract": "decoded_candidates_v1",
            "candidate_confidence": 0.01,
            "candidate_capacity": 5,
            "anchor_count": 3,
            "class_count": 2,
        }
    )

    result = verify_ascend_artifacts(options, require_validation=False)
    assert result["status"] == "passed", result["errors"]

    base_entry["candidate_capacity"] = 4
    result = verify_ascend_artifacts(options, require_validation=False)
    assert "postprocess_contract_mismatch:base" in result["errors"]


def test_shared_dual_head_manifest_uses_one_physical_detector(
    tmp_path: Path,
) -> None:
    options = _candidate(tmp_path)
    manifest_path = Path(options["build_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = payload["artifacts"]
    dual = artifacts.pop("base_detector")
    artifacts.pop("incremental_detector")
    logical_heads = {
        "old": {
            "owner": "frozen_base_model",
            "class_map": {"0": 0, "1": 1, "2": 3},
            "class_count": 3,
            "anchor_count": 13524,
            "output_index": 0,
        },
        "new": {
            "owner": "incremental_model",
            "class_map": {"0": 2},
            "class_count": 1,
            "anchor_count": 13524,
            "output_index": 1,
        },
    }
    dual.update(
        {
            "role": "dual_detector",
            "output_contract": "raw_dual_head_v1",
            "logical_heads": logical_heads,
        }
    )
    artifacts["shared_backbone_dual_head"] = dual
    _write_json(manifest_path, payload)

    options.update(
        {
            "model_layout": "shared_backbone_dual_head_v1",
            "build_manifest_sha256": sha256_file(manifest_path),
            "models": {
                "base.pt": {
                    **dual["om"],
                    "output_contract": "raw_dual_head_v1",
                    "logical_heads": logical_heads,
                }
            },
        }
    )
    result = verify_ascend_artifacts(options, require_validation=False)
    assert result["status"] == "passed", result["errors"]
    assert result["artifact_count"] == 2

    options["models"]["base.pt"]["logical_heads"]["old"][
        "candidate_confidence"
    ] = 0.03
    options["models"]["base.pt"]["logical_heads"]["new"][
        "candidate_confidence"
    ] = 0.20
    result = verify_ascend_artifacts(options, require_validation=False)
    assert result["status"] == "passed", result["errors"]

    options["models"]["base.pt"]["logical_heads"]["new"][
        "owner"
    ] = "frozen_base_model"
    result = verify_ascend_artifacts(options, require_validation=False)
    assert "logical_heads_contract_mismatch:dual_detector" in result["errors"]


def test_shared_dual_head_formal_release_uses_competition_gates(
    tmp_path: Path,
) -> None:
    options = _candidate(tmp_path)
    manifest_path = Path(options["build_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = payload["artifacts"]
    dual = artifacts.pop("base_detector")
    artifacts.pop("incremental_detector")
    logical_heads = {
        "old": {
            "owner": "frozen_base_model",
            "class_map": {"0": 0, "1": 1, "2": 3},
            "class_count": 3,
            "anchor_count": 13524,
            "output_index": 0,
            "candidate_confidence": 0.05,
        },
        "new": {
            "owner": "incremental_model",
            "class_map": {"0": 2},
            "class_count": 1,
            "anchor_count": 13524,
            "output_index": 1,
            "candidate_confidence": 0.30,
        },
    }
    dual.update(
        {
            "role": "dual_detector",
            "output_contract": "raw_dual_head_v1",
            "logical_heads": logical_heads,
        }
    )
    artifacts["shared_backbone_dual_head"] = dual
    _write_json(manifest_path, payload)
    manifest_digest = sha256_file(manifest_path)

    accuracy = tmp_path / "full-score-accuracy.json"
    _write_json(
        accuracy,
        {
            "schema_version": 2,
            "unlabeled_predictions_frozen_before_labels": True,
            "metrics": {
                "base_map50": 0.8049,
                "new_map50": 0.6050,
                "krr": 1.0,
            },
            "competition_gates": {
                "base_map50": True,
                "new_map50": True,
                "krr": True,
            },
            "diagnostic_warnings": [
                "business_json_equivalence",
                "lock_precision",
            ],
            "score_passed": True,
            "passed": True,
        },
    )
    performance = tmp_path / "full-score-performance.json"
    _write_json(
        performance,
        {
            "schema_version": 5,
            "protocol": {
                "batch_probe_size": 20,
                "batch_rounds": 3,
                "target_batch_fps": 30.0,
            },
            "competition": {
                "batch_image_count": 20,
                "batch_fps": 30.066,
                "batch_fps_passed": True,
                "batch_rounds": [
                    {"round": 1, "fps": 30.066},
                    {"round": 2, "fps": 30.071},
                    {"round": 3, "fps": 30.039},
                ],
            },
            "gates": {
                "sample_count": True,
                "request_failures": True,
                "batch_fps": True,
            },
        },
    )
    method = Path("configs/ascend310b/full_score_method.yaml").resolve()
    validation = tmp_path / "full-score-validation.json"
    _write_json(
        validation,
        {
            "schema_version": 1,
            "kind": "ascend310b_full_score_release_validation",
            "build_manifest_sha256": manifest_digest,
            "method_config": {
                "path": str(method),
                "sha256": sha256_file(method),
            },
            "accuracy": {
                "path": str(accuracy),
                "sha256": sha256_file(accuracy),
                "passed": True,
            },
            "performance": {
                "path": str(performance),
                "sha256": sha256_file(performance),
                "passed": True,
            },
            "validity": {
                "incremental_data_isolation": True,
                "shared_max_drift_zero": True,
                "asset_hashes_verified": True,
                "predictions_frozen_before_labels": True,
            },
            "passed": True,
        },
    )
    options.update(
        {
            "model_layout": "shared_backbone_dual_head_v1",
            "build_manifest_sha256": manifest_digest,
            "validation_report": str(validation),
            "validation_report_sha256": sha256_file(validation),
            "models": {
                "base.pt": {
                    **dual["om"],
                    "output_contract": "raw_dual_head_v1",
                    "logical_heads": logical_heads,
                }
            },
        }
    )

    result = verify_ascend_artifacts(options, require_validation=True)
    assert result["status"] == "passed", result["errors"]

    score_payload = json.loads(accuracy.read_text(encoding="utf-8"))
    score_payload["metrics"]["new_map50"] = 0.59
    _write_json(accuracy, score_payload)
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    validation_payload["accuracy"]["sha256"] = sha256_file(accuracy)
    _write_json(validation, validation_payload)
    options["validation_report_sha256"] = sha256_file(validation)
    result = verify_ascend_artifacts(options, require_validation=True)
    assert "full_score_accuracy_gate_failed:new_map50" in result["errors"]


def test_candidate_runtime_requires_explicit_process_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    options = _candidate(tmp_path)
    options.update({
        "validated": False,
        "validation_candidate": True,
        "validation_report": None,
        "validation_report_sha256": None,
    })
    monkeypatch.delenv("AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION", raising=False)
    with pytest.raises(RuntimeError, match="受控验证进程"):
        require_ascend_runtime_artifacts(options)

    monkeypatch.setenv("AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION", "1")
    require_ascend_runtime_artifacts(options)
