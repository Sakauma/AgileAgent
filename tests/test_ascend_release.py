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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _artifact(
    tmp_path: Path,
    model_id: str,
    role: str,
    shape: list[int],
    *,
    class_count: int | None = None,
) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name in ("source_weight", "onnx", "aipp", "om", "atc_log"):
        suffix = "cfg" if name == "aipp" else name
        path = tmp_path / f"{model_id}.{suffix}"
        path.write_bytes(f"{model_id}:{name}".encode())
        entries[name] = {"path": str(path), "sha256": sha256_file(path)}
    artifact = {
        "role": role,
        **entries,
        "atc_command": (
            "atc --framework=5 --input_format=NCHW "
            "--soc_version=Ascend310B1 --precision_mode_v2=mixed_float16 "
            f"--insert_op_conf={entries['aipp']['path']}"
        ),
        "input_contract": {
            "dtype": "uint8",
            "layout": "NHWC",
            "shape": shape,
        },
    }
    if class_count is not None:
        artifact.update(
            {
                "output_contract": "yolo26_e2e_v1",
                "postprocess_contract": {
                    "max_det": 300,
                    "class_count": class_count,
                    "outputs": {
                        "detections": {
                            "shape": [1, 300, 6],
                            "dtype": "float32",
                        }
                    },
                },
            }
        )
    return artifact


def _candidate(tmp_path: Path) -> dict:
    artifacts = {
        "base_detector": _artifact(
            tmp_path,
            "base_detector",
            "base",
            [1, 608, 736, 3],
            class_count=4,
        ),
        "incremental_detector": _artifact(
            tmp_path,
            "incremental_detector",
            "specialist",
            [1, 608, 736, 3],
            class_count=2,
        ),
        "scene_sensor_net": _artifact(
            tmp_path,
            "scene_sensor_net",
            "context",
            [1, 160, 160, 3],
        ),
    }
    manifest = tmp_path / "build-manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "git_sha": "a" * 40,
            "soc_version": "Ascend310B1",
            "cann_version": "7.0.RC1",
            "precision": "mixed_float16",
            "model_layout": "independent_yolo26_e2e_v1",
            "artifacts": artifacts,
        },
    )
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
    _write_json(
        validation,
        {
            "schema_version": 1,
            "build_manifest_sha256": manifest_sha256,
            **validation_entries,
            "passed": True,
        },
    )

    def model_entry(model_id: str, class_count: int) -> dict:
        return {
            **artifacts[model_id]["om"],
            "output_contract": "yolo26_e2e_v1",
            "max_det": 300,
            "class_count": class_count,
        }

    return {
        "device_id": 0,
        "soc_version": "Ascend310B1",
        "cann_version": "7.0.RC1",
        "precision": "mixed_float16",
        "execution_mode": "async_stream",
        "encoded_preprocessing": "dvpp",
        "memory_mode": "pageable",
        "model_layout": "independent_yolo26_e2e_v1",
        "validated": True,
        "build_manifest": str(manifest),
        "build_manifest_sha256": manifest_sha256,
        "validation_report": str(validation),
        "validation_report_sha256": sha256_file(validation),
        "models": {
            "base.pt": model_entry("base_detector", 4),
            "incremental.pt": model_entry("incremental_detector", 2),
        },
        "context_model": artifacts["scene_sensor_net"]["om"],
    }


def test_verified_independent_yolo26_candidate_binds_three_oms(
    tmp_path: Path,
) -> None:
    result = verify_ascend_artifacts(_candidate(tmp_path))

    assert result["status"] == "passed", result["errors"]
    assert result["artifact_count"] == 3
    assert result["validation_required"] is True


def test_candidate_rejects_hash_and_input_contract_drift(
    tmp_path: Path,
) -> None:
    options = _candidate(tmp_path)
    manifest_path = Path(options["build_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"]["base_detector"]["input_contract"]["shape"] = [
        1,
        608,
        735,
        3,
    ]
    _write_json(manifest_path, payload)

    result = verify_ascend_artifacts(options)

    assert result["status"] == "failed"
    assert "build_manifest_sha256_mismatch" in result["errors"]
    assert "input_contract_mismatch:base_detector:base" in result["errors"]


def test_unvalidated_candidate_can_verify_before_gate_reports(
    tmp_path: Path,
) -> None:
    options = _candidate(tmp_path)
    options["validated"] = False
    options["validation_report"] = None
    options["validation_report_sha256"] = None

    result = verify_ascend_artifacts(options)

    assert result["status"] == "passed", result["errors"]
    assert result["validation_required"] is False


def test_only_current_yolo26_e2e_layout_is_accepted(tmp_path: Path) -> None:
    options = _candidate(tmp_path)
    options["model_layout"] = "unknown_layout"
    result = verify_ascend_artifacts(options, require_validation=False)
    assert "model_layout_invalid" in result["errors"]

    options = _candidate(tmp_path / "other")
    model = next(iter(options["models"].values()))
    model["output_contract"] = "unknown_output"
    result = verify_ascend_artifacts(options, require_validation=False)
    assert "output_contract_mismatch:base" in result["errors"]


def test_formal_release_uses_competition_gates_and_validity_prerequisites(
    tmp_path: Path,
) -> None:
    options = _candidate(tmp_path)
    manifest_digest = options["build_manifest_sha256"]
    method = Path("configs/ascend310b/full_score_method.yaml").resolve()
    accuracy = tmp_path / "full-score-accuracy.json"
    _write_json(
        accuracy,
        {
            "schema_version": 2,
            "unlabeled_predictions_frozen_before_labels": True,
            "metrics": {
                "base_map50": 0.8256706047,
                "new_map50": 0.6188591828,
                "krr": 1.0,
            },
            "competition_gates": {
                "base_map50": True,
                "new_map50": True,
                "krr": True,
            },
            "score_passed": True,
        },
    )
    performance = tmp_path / "full-score-performance.json"
    _write_json(
        performance,
        {
            "schema_version": 6,
            "protocol": {
                "batch_probe_size": 20,
                "batch_rounds": 3,
                "target_batch_fps": 30.0,
            },
            "competition": {
                "batch_image_count": 20,
                "batch_fps": 39.3,
                "batch_fps_passed": True,
                "batch_rounds": [
                    {"round": 1, "fps": 39.2},
                    {"round": 2, "fps": 39.3},
                    {"round": 3, "fps": 39.4},
                ],
            },
            "gates": {
                "sample_count": True,
                "request_failures": True,
                "batch_fps": True,
            },
        },
    )
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
                "base_model_frozen": True,
                "asset_hashes_verified": True,
                "predictions_frozen_before_labels": True,
            },
            "passed": True,
        },
    )
    options["validation_report"] = str(validation)
    options["validation_report_sha256"] = sha256_file(validation)

    result = verify_ascend_artifacts(options, require_validation=True)
    assert result["status"] == "passed", result["errors"]

    score = json.loads(accuracy.read_text(encoding="utf-8"))
    score["metrics"]["new_map50"] = 0.59
    _write_json(accuracy, score)
    summary = json.loads(validation.read_text(encoding="utf-8"))
    summary["accuracy"]["sha256"] = sha256_file(accuracy)
    _write_json(validation, summary)
    options["validation_report_sha256"] = sha256_file(validation)
    result = verify_ascend_artifacts(options, require_validation=True)
    assert "full_score_accuracy_gate_failed:new_map50" in result["errors"]


def test_candidate_runtime_requires_explicit_process_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    options = _candidate(tmp_path)
    options.update(
        {
            "validated": False,
            "validation_candidate": True,
            "validation_report": None,
            "validation_report_sha256": None,
        }
    )
    monkeypatch.delenv(
        "AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION",
        raising=False,
    )
    with pytest.raises(RuntimeError, match="受控验证进程"):
        require_ascend_runtime_artifacts(options)

    monkeypatch.setenv("AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION", "1")
    require_ascend_runtime_artifacts(options)
