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
