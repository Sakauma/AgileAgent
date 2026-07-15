from __future__ import annotations

import copy
import json
import shutil
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

from fair_agent.backends.inference import TensorRTEngineBackend, TensorRTNativeBackend
from fair_agent.core.config import apply_overrides, load_config, redact_config, validate_config
from fair_agent.core.hashes import sha256_file
from fair_agent.models.context import validate_context_input_shape
from fair_agent.modules.configuration import set_persistent_value
from fair_agent.modules.api_benchmark import _server_session
from fair_agent.modules.generation_management import promote_generation, rollback_generation
from fair_agent.modules.model_generations import load_generation_registry
from fair_agent.modules.tensorrt_export import (
    export_or_verify_engines,
    export_plan,
    write_export_hashes,
)


def clean_config() -> dict:
    config = load_config()
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def test_effective_config_rejects_unknown_runtime_field() -> None:
    config = clean_config()
    config["inference"]["magic_speed"] = True
    with pytest.raises(ValueError, match="未知字段"):
        validate_config(config)


def test_competition_incremental_labels_prefer_class_ids_with_bbox_fallback() -> None:
    config = clean_config()
    assert config["incremental_workbench"]["allowed_label_formats"] == ["class_id_bbox", "bbox_only"]
    config["incremental_workbench"]["allowed_label_formats"] = ["unknown"]
    with pytest.raises(ValueError, match="allowed_label_formats"):
        validate_config(config)


def test_cli_overrides_are_typed_and_process_local() -> None:
    config = clean_config()
    effective = apply_overrides(config, ["inference.confidence_default=0.61", "ui.history_limit=7"])
    validate_config(effective)
    assert effective["inference"]["confidence_default"] == 0.61
    assert effective["ui"]["history_limit"] == 7
    assert config["inference"]["confidence_default"] == 0.5


def test_sensitive_values_are_redacted() -> None:
    assert redact_config({"runtime": {"token": "secret", "server_port": 8501}}) == {
        "runtime": {"token": "***", "server_port": 8501}
    }


def test_persistent_config_rejects_protected_generation_field(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(clean_config(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="受保护参数"):
        set_persistent_value(path, "generation.registry", "models/other.json")


def test_persistent_config_rejects_protected_context_engine(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump(clean_config(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="受保护参数"):
        set_persistent_value(path, "tensorrt_backend.context_engine.path", "other.engine")


def test_tensorrt_backend_has_no_cpu_fallback(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="尚未通过"):
        TensorRTNativeBackend({
            "validated": False,
            "library": str(tmp_path / "missing.so"),
            "base_engine": str(tmp_path / "base.engine"),
            "context_engine": str(tmp_path / "context.engine"),
        })


def _mock_tensorrt_runtime(monkeypatch, version: str = "10.8.0.43", capability=(8, 9)) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__=version))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_capability=lambda _index: capability,
    )))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda *_args, **_kwargs: object()))


def _engine_options(engine: Path, weights: Path) -> dict:
    return {
        "validated": True,
        "expected_version": "10.8.0.43",
        "require_exact_gpu": True,
        "expected_compute_capability": "8.9",
        "engines": {
            weights.as_posix(): {
                "path": str(engine),
                "sha256": sha256_file(engine),
                "imgsz": 640,
                "batch_size": 1,
            }
        },
    }


def test_tensorrt_engine_rejects_version_capability_and_hash(monkeypatch, tmp_path: Path) -> None:
    engine = tmp_path / "model.engine"
    weights = tmp_path / "model.pt"
    engine.write_bytes(b"engine")
    options = _engine_options(engine, weights)

    _mock_tensorrt_runtime(monkeypatch, version="10.9")
    with pytest.raises(RuntimeError, match="版本不匹配"):
        TensorRTEngineBackend(weights, "0", options)

    _mock_tensorrt_runtime(monkeypatch, capability=(8, 6))
    with pytest.raises(RuntimeError, match="计算能力不匹配"):
        TensorRTEngineBackend(weights, "0", options)

    _mock_tensorrt_runtime(monkeypatch)
    options["engines"][weights.as_posix()]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="哈希不匹配"):
        TensorRTEngineBackend(weights, "0", options)


def test_context_engine_contract_rejects_bad_batch_and_shape() -> None:
    assert validate_context_input_shape((8, 3, 160, 160), 160, 20) == 8
    with pytest.raises(RuntimeError, match="batch范围"):
        validate_context_input_shape((21, 3, 160, 160), 160, 20)
    with pytest.raises(RuntimeError, match="输入尺寸"):
        validate_context_input_shape((1, 3, 224, 224), 160, 20)


def test_tensorrt_export_plan_is_fully_config_driven() -> None:
    rows = export_plan(load_config())
    assert [row["kind"] for row in rows] == ["yolo", "yolo", "context"]
    assert rows[-1]["batch_size"] == 20
    assert rows[-1]["imgsz"] == 160


def _unset_engine_hashes(config: dict) -> dict:
    config["tensorrt_backend"]["validated"] = False
    config["tensorrt_backend"]["expected_version"] = "10.8.0.43"
    config["tensorrt_backend"]["expected_compute_capability"] = "8.9"
    config["tensorrt_backend"]["context_engine"]["sha256"] = None
    for entry in config["tensorrt_backend"]["engines"].values():
        entry["sha256"] = None
    return config


def test_default_profile_uses_portable_cuda_weights_without_local_engines() -> None:
    config = clean_config()
    assert config["inference"]["backend"] == "ultralytics_cuda"
    assert config["tensorrt_backend"]["validated"] is False
    assert config["tensorrt_backend"]["context_engine"]["sha256"] is None
    assert all(entry["sha256"] is None for entry in config["tensorrt_backend"]["engines"].values())
    validate_config(config)


def test_unverified_export_profile_accepts_only_explicitly_allowed_null_hashes() -> None:
    config = _unset_engine_hashes(clean_config())
    config["inference"]["backend"] = "tensorrt_engine"
    validate_config(config, allow_unverified_tensorrt_hashes=True)
    with pytest.raises(ValueError, match="SHA256"):
        validate_config(config)
    config["tensorrt_backend"]["validated"] = True
    with pytest.raises(ValueError, match="SHA256"):
        validate_config(config, allow_unverified_tensorrt_hashes=True)


def test_verify_only_rejects_unregistered_engine_hashes() -> None:
    config = _unset_engine_hashes(clean_config())
    with pytest.raises(ValueError, match="真实SHA256"):
        export_or_verify_engines(config, verify_only=True)


def test_export_hashes_are_verified_and_atomically_written(tmp_path: Path) -> None:
    config = _unset_engine_hashes(clean_config())
    config["logging"]["root"] = str(tmp_path / "logs")
    profile = tmp_path / "device.yaml"
    engine_rows = []
    for index, (source, entry) in enumerate(config["tensorrt_backend"]["engines"].items()):
        target = tmp_path / f"detector-{index}.engine"
        target.write_bytes(f"detector-{index}".encode())
        entry["path"] = str(target)
        engine_rows.append({
            "kind": "yolo",
            "source": source,
            "target": str(target),
            "status": "exported",
            "sha256": sha256_file(target),
        })
    context_target = tmp_path / "context.engine"
    context_target.write_bytes(b"context")
    config["tensorrt_backend"]["context_engine"]["path"] = str(context_target)
    engine_rows.append({
        "kind": "context",
        "source": config["tensorrt_backend"]["export"]["context_checkpoint"],
        "target": str(context_target),
        "status": "exported",
        "sha256": sha256_file(context_target),
    })
    profile.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    update = write_export_hashes(profile, {"engines": engine_rows})

    stored = yaml.safe_load(profile.read_text(encoding="utf-8"))
    assert update["updated"] is True
    assert update["engine_hashes_recorded"] == 3
    assert stored["tensorrt_backend"]["validated"] is False
    assert stored["tensorrt_backend"]["context_engine"]["sha256"] == sha256_file(context_target)
    assert all(entry["sha256"] for entry in stored["tensorrt_backend"]["engines"].values())
    validate_config(stored)


def test_api_benchmark_can_require_an_existing_server(monkeypatch) -> None:
    config = load_config()
    config["performance"]["auto_start_server"] = False
    monkeypatch.setattr("fair_agent.modules.api_benchmark._health_available", lambda *_args: False)
    with pytest.raises(RuntimeError, match="检测服务未启动"):
        with _server_session(config, "http://127.0.0.1:65534"):
            pass


def test_generation_registry_keeps_verified_rollback_point() -> None:
    registry = load_generation_registry("models/generations.json")
    assert registry["channels"]["production"] == "incremental_detection_generation"
    assert registry["generations_by_id"]["base_detection_generation"]["status"] == "active"
    assert registry["models_by_id"]["incremental_detector"]["activation_threshold"] == 0.63


def test_promotion_rejects_failed_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"candidate": "incremental_detection_generation", "accepted": False}), encoding="utf-8")
    config = copy.deepcopy(load_config())
    config["logging"]["root"] = str(tmp_path / "logs")
    with pytest.raises(ValueError, match="未通过"):
        promote_generation(config, "incremental_detection_generation", manifest)
    from fair_agent.core.runtime_log import event_log_from_config

    rows = event_log_from_config(config).query(generation_id="incremental_detection_generation")
    assert [row["event"] for row in rows][:2] == [
        "generation.production_switch.failed", "generation.production_switch.started"
    ]
    assert rows[0]["details"]["production_before"] == "incremental_detection_generation"


def test_rollback_updates_only_copied_registry(tmp_path: Path) -> None:
    registry_copy = tmp_path / "generations.json"
    shutil.copy2("models/generations.json", registry_copy)
    config = copy.deepcopy(load_config())
    config["generation"]["registry"] = str(registry_copy)
    config["logging"]["root"] = str(tmp_path / "logs")
    result = rollback_generation(config, "base_detection_generation")
    assert result["production"] == "base_detection_generation"
    assert json.loads(registry_copy.read_text(encoding="utf-8"))["channels"]["production"] == "base_detection_generation"
    assert json.loads(Path("models/generations.json").read_text(encoding="utf-8"))["channels"]["production"] == "incremental_detection_generation"
    from fair_agent.core.runtime_log import event_log_from_config

    rows = event_log_from_config(config).query(generation_id="base_detection_generation")
    completed = next(row for row in rows if row["event"] == "generation.rollback.completed")
    assert completed["details"]["production_before"] == "incremental_detection_generation"
    assert completed["details"]["production_after"] == "base_detection_generation"
    assert completed["details"]["registry_sha256_before"] != completed["details"]["registry_sha256_after"]
