from __future__ import annotations

import copy
import json
import shutil
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml
from PIL import Image

from fair_agent.backends.inference import TensorRTEngineBackend, TensorRTNativeBackend
from fair_agent.core.config import apply_overrides, load_config, redact_config, validate_config
from fair_agent.core.hashes import business_payload_sha256, sha256_file
from fair_agent.models.context import validate_context_input_shape
from fair_agent.modules.configuration import set_persistent_value
from fair_agent.modules.api_benchmark import _server_session
from fair_agent.modules.generation_management import promote_generation, rollback_generation
from fair_agent.modules.model_generations import load_generation_registry
from fair_agent.modules.tensorrt_export import (
    _apply_mixed_precision,
    _optimal_detector_shape,
    export_or_verify_engines,
    export_plan,
    prepare_calibration_manifest,
    write_export_hashes,
)
from fair_agent.modules.api_benchmark import _performance_assessment
from fair_agent.modules.tensorrt_validation import (
    _apply_protocol_thresholds,
    _competition_accuracy_gates,
)


def clean_config() -> dict:
    config = load_config()
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def test_effective_config_rejects_unknown_runtime_field() -> None:
    config = clean_config()
    config["inference"]["magic_speed"] = True
    with pytest.raises(ValueError, match="未知字段"):
        validate_config(config)


def test_ascend_dvpp_scene_resize_stages_require_even_bounded_shapes() -> None:
    config = clean_config()
    config["ascend_backend"]["dvpp_scene_resize_stages"] = [
        [208, 192],
        [288, 230],
    ]
    validate_config(config)

    config["ascend_backend"]["dvpp_scene_resize_stages"] = [[207, 192]]
    with pytest.raises(ValueError, match="偶数宽高"):
        validate_config(config)


def test_ascend_p4_runtime_ablation_fields_are_validated() -> None:
    config = clean_config()
    config["ascend_backend"]["schedule_mode"] = "unified_enqueue"
    config["ascend_backend"]["detailed_event_timing"] = False
    validate_config(config)

    config["ascend_backend"]["schedule_mode"] = "serial"
    with pytest.raises(ValueError, match="schedule_mode"):
        validate_config(config)

    config["ascend_backend"]["schedule_mode"] = "threaded_execute"
    config["ascend_backend"]["detailed_event_timing"] = "false"
    with pytest.raises(ValueError, match="detailed_event_timing"):
        validate_config(config)


def test_ascend_p5_order_and_priority_fields_are_validated() -> None:
    config = clean_config()
    config["ascend_backend"]["submit_order"] = [
        "specialist",
        "base",
        "scene",
    ]
    config["ascend_backend"]["collect_order"] = [
        "base",
        "specialist",
        "scene",
    ]
    config["ascend_backend"]["stream_priorities"] = {
        "scene": "normal",
        "base": "high",
        "specialist": "low",
    }
    validate_config(config)

    config["ascend_backend"]["submit_order"] = ["base", "base", "scene"]
    with pytest.raises(ValueError, match="submit_order"):
        validate_config(config)

    config["ascend_backend"]["submit_order"] = ["scene", "base", "specialist"]
    config["ascend_backend"]["stream_priorities"]["specialist"] = "urgent"
    with pytest.raises(ValueError, match="stream_priorities"):
        validate_config(config)


def test_ascend_yolo26_e2e_contract_is_explicit_and_fully_pinned() -> None:
    config = clean_config()
    entry = next(iter(config["ascend_backend"]["models"].values()))

    assert config["ascend_backend"]["model_layout"] == (
        "independent_yolo26_e2e_v1"
    )
    assert entry["output_contract"] == "yolo26_e2e_v1"
    assert entry["max_det"] == 300
    assert entry["class_count"] in {2, 4}
    validate_config(config)

    entry.pop("class_count")
    with pytest.raises(ValueError, match="缺少固定输出参数"):
        validate_config(config)

    config = clean_config()
    entry = next(iter(config["ascend_backend"]["models"].values()))
    entry["output_contract"] = "shape_guess"
    with pytest.raises(ValueError, match="output_contract"):
        validate_config(config)

    config = clean_config()
    config["ascend_backend"]["model_layout"] = "unknown_layout"
    with pytest.raises(ValueError, match="model_layout"):
        validate_config(config)


def test_global_cross_class_suppression_configuration_is_explicitly_validated() -> None:
    config = clean_config()
    policy = config["routing"]["cross_class_suppression"]

    assert policy == {
        "enabled": True,
        "strategy": "highest_confidence",
        "scope": "all_classes",
        "iou": 0.50,
        "smaller_box_coverage": 0.95,
    }
    validate_config(config)

    config["routing"]["cross_class_suppression"]["scope"] = "known_pairs"
    with pytest.raises(ValueError, match="scope必须为all_classes"):
        validate_config(config)


def test_model_score_calibration_requires_both_frozen_sources() -> None:
    config = clean_config()
    config["routing"]["score_calibration"] = {
        "enabled": True,
        "method": "logit_affine",
        "source_split": "mixed_dev_only",
        "sources": {
            "frozen_base_model": {"temperature": 1.0, "bias": 0.0},
            "incremental_model": {"temperature": 1.25, "bias": -0.25},
        },
    }
    config["routing"]["cross_class_suppression"][
        "incremental_over_base_margin"
    ] = 0.10

    validate_config(config)

    del config["routing"]["score_calibration"]["sources"]["incremental_model"]
    with pytest.raises(ValueError, match="完整登记Base与Specialist"):
        validate_config(config)


def test_ascend_context_mode_requires_the_registered_scene_model() -> None:
    config = clean_config()
    config["ascend_backend"]["context_mode"] = "fixed_neutral_v1"
    with pytest.raises(ValueError, match="context_mode必须为model"):
        validate_config(config)

    config["ascend_backend"]["context_mode"] = "filename_derived"
    with pytest.raises(ValueError, match="context_mode"):
        validate_config(config)


def test_api_business_signature_ignores_only_request_timing_fields() -> None:
    payload = {
        "filename": "sample.png",
        "detections": [{"class_id": 1, "confidence": 0.75}],
        "agent": {"mode": "automatic", "activated_classes": []},
        "inference_ms": 10.0,
        "timings": {"engine_total_ms": 12.0},
        "queue_wait_ms": 0.1,
        "system_total_ms": 13.0,
    }
    changed_timing = copy.deepcopy(payload)
    changed_timing.update(
        {
            "inference_ms": 99.0,
            "timings": {"engine_total_ms": 101.0},
            "queue_wait_ms": 2.0,
            "system_total_ms": 105.0,
        }
    )
    changed_business = copy.deepcopy(payload)
    changed_business["detections"][0]["class_id"] = 2

    assert business_payload_sha256(payload) == business_payload_sha256(
        changed_timing
    )
    assert business_payload_sha256(payload) != business_payload_sha256(
        changed_business
    )


def test_cli_overrides_are_typed_and_process_local() -> None:
    config = clean_config()
    effective = apply_overrides(config, ["inference.confidence_default=0.61", "ui.history_limit=7"])
    validate_config(effective)
    assert effective["inference"]["confidence_default"] == 0.61
    assert effective["ui"]["history_limit"] == 7
    assert config["inference"]["confidence_default"] == 0.01


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
    assert rows[-1]["batch_size"] == 32
    assert [
        (row["min_batch_size"], row["opt_batch_size"], row["batch_size"])
        for row in rows
    ] == [(1, 8, 32), (1, 8, 32), (1, 8, 32)]
    assert rows[-1]["imgsz"] == 224


def test_int8_calibration_manifest_is_deterministic_and_rejects_lock(tmp_path: Path) -> None:
    config = clean_config()
    settings = config["tensorrt_backend"]["int8_calibration"]
    settings.update({
        "cache_root": str(tmp_path / "calibration"),
        "batch_size": 2,
        "max_images": 6,
        "minimum_images_per_class": 2,
        "seed": 20260705,
    })
    images = []
    for index in range(8):
        image = tmp_path / f"sample-{index}.png"
        Image.new("RGB", (32, 24), (index, 0, 0)).save(image)
        image.with_suffix(".txt").write_text(
            f"{index % 2} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
        images.append(image)
    first = prepare_calibration_manifest(
        config, "model.pt", images, [0, 1], "unit-test", preprocessing={"imgsz": 640}
    )
    second = prepare_calibration_manifest(
        config, "model.pt", list(reversed(images)), [0, 1], "unit-test", preprocessing={"imgsz": 640}
    )
    assert first["fingerprint"] == second["fingerprint"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["image_count"] == 6
    changed = prepare_calibration_manifest(
        config, "model.pt", images, [0, 1], "unit-test", preprocessing={"imgsz": 512}
    )
    assert changed["fingerprint"] != first["fingerprint"]
    with pytest.raises(ValueError, match="封存lock重叠"):
        prepare_calibration_manifest(
            config, "model.pt", images, [0, 1], "unit-test", [first["images"][0].stem]
        )


def test_int8_optimal_shape_matches_rectangular_ultralytics_preprocessing(tmp_path: Path) -> None:
    images = []
    for index in range(3):
        image = tmp_path / f"rect-{index}.png"
        Image.new("RGB", (640, 512)).save(image)
        images.append(image)
    assert _optimal_detector_shape(images, 640) == (512, 640)
    assert _optimal_detector_shape(images, 512) == (416, 512)


def test_mixed_precision_forces_configured_layers_to_fp16() -> None:
    class Output:
        def __init__(self, name: str) -> None:
            self.name = name
            self.dtype = "fp32"
            self.is_shape_tensor = False

    class Layer:
        def __init__(self, name: str) -> None:
            self.name = name
            self.type = "convolution"
            self.outputs = [Output(name + ":0")]
            self.num_outputs = 1
            self.precision = None
            self.output_types = {}

        def get_output(self, index: int):
            return self.outputs[index]

        def set_output_type(self, index: int, dtype: str) -> None:
            self.output_types[index] = dtype

    layers = [Layer("/model.5/conv/Conv"), Layer("/model.6/cv1/conv/Conv"), Layer("/model.23/cv3/Conv")]
    network = SimpleNamespace(num_layers=len(layers), get_layer=lambda index: layers[index])
    flags = []
    builder_config = SimpleNamespace(set_flag=flags.append)
    trt = SimpleNamespace(
        float16="fp16",
        float32="fp32",
        BuilderFlag=SimpleNamespace(
            OBEY_PRECISION_CONSTRAINTS="obey",
            PREFER_PRECISION_CONSTRAINTS="prefer",
        ),
        LayerType=SimpleNamespace(
            ACTIVATION="activation",
            CONVOLUTION="convolution",
            DECONVOLUTION="deconvolution",
            ELEMENTWISE="elementwise",
            MATRIX_MULTIPLY="matrix_multiply",
            NORMALIZATION="normalization",
            PARAMETRIC_RELU="parametric_relu",
            POOLING="pooling",
            SOFTMAX="softmax",
        ),
    )
    report = _apply_mixed_precision(network, builder_config, {
        "precision": "int8",
        "mixed_precision": {
            "enabled": True,
            "constraint": "obey",
            "fp16_layer_patterns": ["/model.[6-9]/*", "/model.2[0-3]/*"],
            "minimum_matched_layers": 2,
        },
    }, trt)
    assert report["matched_layer_count"] == 2
    assert layers[0].precision is None
    assert layers[1].precision == layers[2].precision == "fp16"
    assert flags == ["obey"]


def test_mixed_precision_requires_int8_backend() -> None:
    config = clean_config()
    config["tensorrt_backend"]["mixed_precision"]["enabled"] = True
    with pytest.raises(ValueError, match="仅在precision=int8"):
        validate_config(config)


def test_fixed_mixed_precision_profile_keeps_only_modules_zero_and_one_int8() -> None:
    profile = load_config()["tensorrt_backend"]["mixed_precision"]
    assert profile["fp16_layer_patterns"] == [
        "/model.[2-9]/*",
        "/model.1[0-9]/*",
        "/model.2[0-3]/*",
    ]
    assert profile["minimum_matched_layers"] == 269


def test_non_competition_diagnostics_do_not_fail_quantized_candidate() -> None:
    config = load_config()
    gates = _competition_accuracy_gates(
        {
            "base_map50": 0.81,
            "new_map50": 0.78,
            "krr": 1.0,
            "combined_map50": 0.803,
        },
        config["gates"]["official_hard"],
        True,
    )
    assert gates == {
        "quantized_thresholds_calibrated": True,
        "base_map50": True,
        "new_map50": True,
        "krr": True,
    }

    competition, diagnostics = _performance_assessment(
        {
            "batch_fps": 47.9,
            "median_round_mean_server_ms": 46.9,
            "all_p95_server_ms": 60.0,
            "concurrent_success_count": 8,
        },
        config["performance"],
        8,
    )
    assert competition == {"batch_fps": True}
    assert diagnostics == {"mean_api_ms": False, "p95_api_ms": False, "concurrency": True}


def test_validated_int8_profile_uses_device_calibrated_threshold(
    tmp_path: Path,
) -> None:
    from fair_agent.web.app import build_web_settings

    report = tmp_path / "validation.json"
    report.write_text(
        json.dumps({
            "accepted": True,
            "threshold_calibration": {"thresholds": {"4": 0.38}},
        }),
        encoding="utf-8",
    )
    config = clean_config()
    config["inference"]["backend"] = "tensorrt_engine"
    config["tensorrt_backend"]["precision"] = "int8"
    config["tensorrt_backend"]["validated"] = True
    config["tensorrt_backend"]["validation_report"] = str(report)

    settings = build_web_settings(config)

    protocol = settings["protocols"]["incremental_detector"]
    assert protocol["activation_threshold"] is None
    assert protocol["activation_thresholds"] == {4: 0.38, 5: 0.82}


def test_quantized_threshold_override_updates_single_class_protocol() -> None:
    protocols = {
        "expert": {
            "global_class_ids": [2],
            "activation_thresholds": {2: 0.63},
            "activation_threshold": 0.63,
        }
    }
    _apply_protocol_thresholds(protocols, {2: 0.47})
    assert protocols["expert"]["activation_thresholds"] == {2: 0.47}
    assert protocols["expert"]["activation_threshold"] == 0.47


def test_int8_precision_requires_enabled_calibration() -> None:
    config = clean_config()
    config["tensorrt_backend"]["precision"] = "int8"
    config["tensorrt_backend"]["int8_calibration"]["enabled"] = False
    with pytest.raises(ValueError, match="必须启用INT8校准"):
        validate_config(config)


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
    assert registry["channels"]["production"] == (
        "incremental_detection_generation_4plus2"
    )
    assert registry["generations_by_id"][
        "base_detection_generation_4plus2"
    ]["status"] == "active"
    assert registry["models_by_id"]["incremental_detector"][
        "per_class_thresholds"
    ] == {4: 0.57, 5: 0.82}


def test_generation_registry_rejects_missing_context_prior_asset(tmp_path: Path) -> None:
    payload = json.loads(Path("models/generations.json").read_text(encoding="utf-8"))
    expert = next(item for item in payload["models"] if item["id"] == "incremental_detector")
    expert["context_gate"]["prior_source"] = str(tmp_path / "missing-context-prior.json")
    registry_path = tmp_path / "generations.json"
    registry_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="场景软门控资产非法"):
        load_generation_registry(registry_path)


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
    assert rows[0]["details"]["production_before"] == (
        "incremental_detection_generation_4plus2"
    )


def test_rollback_updates_only_copied_registry(tmp_path: Path) -> None:
    registry_copy = tmp_path / "generations.json"
    shutil.copy2("models/generations.json", registry_copy)
    config = copy.deepcopy(load_config())
    config["generation"]["registry"] = str(registry_copy)
    config["logging"]["root"] = str(tmp_path / "logs")
    result = rollback_generation(config, "base_detection_generation_4plus2")
    assert result["production"] == "base_detection_generation_4plus2"
    assert json.loads(registry_copy.read_text(encoding="utf-8"))["channels"]["production"] == "base_detection_generation_4plus2"
    assert json.loads(Path("models/generations.json").read_text(encoding="utf-8"))["channels"]["production"] == "incremental_detection_generation_4plus2"
    from fair_agent.core.runtime_log import event_log_from_config

    rows = event_log_from_config(config).query(
        generation_id="base_detection_generation_4plus2"
    )
    completed = next(row for row in rows if row["event"] == "generation.rollback.completed")
    assert completed["details"]["production_before"] == (
        "incremental_detection_generation_4plus2"
    )
    assert completed["details"]["production_after"] == (
        "base_detection_generation_4plus2"
    )
    assert completed["details"]["registry_sha256_before"] != completed["details"]["registry_sha256_after"]
