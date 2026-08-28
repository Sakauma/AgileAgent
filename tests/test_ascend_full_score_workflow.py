from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from fair_agent.core.config import load_config


ROOT = Path(__file__).resolve().parents[1]
RELEASE = (
    ROOT
    / "models"
    / "ascend310b"
    / "full-score"
    / "20260824-4plus2-yolo26-runtime-calibration-v1"
)


def load_tool(name: str):
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATERIALIZE = load_tool("112_materialize_ascend_yolo26_candidate.py")
SCORE = load_tool("94_score_ascend_agent.py")
PROMOTE = load_tool("111_promote_ascend_full_score_release.py")
FREEZE = load_tool("98_freeze_ascend_predictions.py")


def method_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/ascend310b/full_score_method.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_full_score_method_pins_current_independent_yolo26_contract() -> None:
    method = method_config()

    MATERIALIZE.validate_method(method)
    PROMOTE.validate_method(method)

    assert method["schema_version"] == 2
    assert method["competition"]["performance_gate"] == {
        "batch_image_count": 20,
        "batch_rounds": 3,
        "aggregate_fps_min": 30.0,
        "calculation": "total_frames_divided_by_total_elapsed_seconds",
        "includes_result_persistence": True,
        "required_components": [
            "image_decode",
            "scene_model",
            "decision_model",
            "base_detector",
            "incremental_detector",
            "postprocess",
            "formal_result_write",
        ],
    }
    assert method["training"]["method"] == (
        "phase_separated_independent_yolo26s"
    )
    assert method["training"]["checkpoint_metric"] == "map50"
    assert method["training"]["reference_export_checkpoint"] == "best"
    assert method["training"]["epochs"] == 500
    assert method["training"]["patience"] == 50
    assert method["export"]["model_layout"] == "independent_yolo26_e2e_v1"
    assert method["export"]["output_contract"] == "yolo26_e2e_v1"
    assert method["export"]["input_shape_nchw"] == [1, 3, 608, 736]
    assert method["export"]["input_shape_aipp_nhwc"] == [1, 608, 736, 3]
    assert method["runtime"]["context_mode"] == "model"
    assert method["runtime"]["content_execution_gate"] == {
        "enabled": True,
        "policy": "skip_specialist_on_scene_and_base_evidence_v1",
        "action": "skip_specialist",
        "scene": "air",
        "scene_probability_min": 0.5,
        "base_evidence_class_ids": [1],
        "base_evidence_mode": "any",
        "online_inputs": ["scene_probabilities", "base_detections"],
        "label_aware_online_routing": False,
        "filename_aware_online_routing": False,
        "learning_data_scope": "frozen_system_calibration",
    }
    assert method["benchmark"]["image_contract"] == {
        "root_glob": "*.png",
        "width": 640,
        "height": 512,
        "bit_depth": 8,
        "color_types": [0, 2, 6],
    }
    assert method["reference_result"]["base_map50"] == pytest.approx(
        0.8166630282
    )
    assert method["reference_result"]["new_map50"] == pytest.approx(
        0.6114608956
    )
    assert method["reference_result"]["krr"] == 1.0
    assert min(
        method["reference_result"]["public_8501_batch_median_fps"]
    ) >= 30

    invalid = copy.deepcopy(method)
    invalid["reference_result"]["artifact"] = "/home/user/candidate.om"
    with pytest.raises(ValueError, match="绝对路径"):
        MATERIALIZE.validate_method(invalid)


def test_score_contract_uses_registered_class_ids_and_method_gates() -> None:
    assert SCORE.parse_class_ids("4, 7,9") == [4, 7, 9]
    with pytest.raises(SCORE.argparse.ArgumentTypeError, match="互异"):
        SCORE.parse_class_ids("4,4")

    assert SCORE.load_accuracy_gates(
        ROOT / "configs/ascend310b/full_score_method.yaml"
    ) == {
        "base_map50": 0.80,
        "new_map50": 0.60,
        "krr": 0.95,
    }


def test_calibration_probe_keeps_models_but_neutralizes_policy_layers(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(load_config())

    audit = FREEZE.prepare_calibration_probe(config, tmp_path, 0.00001)
    probe_registry = json.loads(
        Path(config["web"]["generation_registry"]).read_text(encoding="utf-8")
    )

    assert audit["confidence_floor"] == 0.00001
    assert config["routing"]["fusion_iou"] == 1.0
    assert config["routing"]["score_calibration"] == {"enabled": False}
    assert config["routing"]["cross_class_suppression"]["enabled"] is False
    for model in probe_registry["models"]:
        assert set(model["per_class_thresholds"].values()) == {0.00001}
        assert model["context_gate"] == {"enabled": False}
        assert model["context_prior"] == {}
        if model["role"] == "class_incremental_expert":
            assert model["content_execution_gate"] == {"enabled": False}


def test_packaged_score_and_release_match_the_reference_method() -> None:
    method = method_config()
    score = json.loads(
        (RELEASE / "validation/score.json").read_text(encoding="utf-8")
    )
    release = json.loads(
        (RELEASE / "release.json").read_text(encoding="utf-8")
    )

    metrics = PROMOTE.validate_score(score)

    assert metrics == release["metrics"]
    assert metrics["base_map50"] == pytest.approx(
        method["reference_result"]["base_map50"]
    )
    assert metrics["new_map50"] == pytest.approx(
        method["reference_result"]["new_map50"]
    )
    assert metrics["krr"] == method["reference_result"]["krr"]
    assert score["metrics"]["old_prediction_equivalent"] is True


def score_payload(base: float, new: float, krr: float) -> dict:
    gates = {
        "base_map50": base >= 0.80,
        "new_map50": new >= 0.60,
        "krr": krr >= 0.95,
    }
    return {
        "schema_version": 2,
        "unlabeled_predictions_frozen_before_labels": True,
        "metrics": {"base_map50": base, "new_map50": new, "krr": krr},
        "competition_gates": gates,
        "score_passed": all(gates.values()),
    }


def benchmark_payload(round_fps: list[float], schema_version: int = 8) -> dict:
    round_elapsed_ms = [20 * 1000.0 / fps for fps in round_fps]
    total_elapsed_ms = sum(round_elapsed_ms)
    aggregate_fps = 60 * 1000.0 / total_elapsed_ms
    return {
        "schema_version": schema_version,
        "protocol": {
            "batch_probe_size": 20,
            "batch_rounds": 3,
            "target_batch_fps": 30.0,
            "fps_calculation": "total_frames_divided_by_total_elapsed_seconds",
            "formal_result_format": (
                "class_id x_center y_center width height confidence"
            ),
            "timed_components": [
                "image_decode",
                "scene_model",
                "decision_model",
                "base_detector",
                "incremental_detector",
                "postprocess",
                "formal_result_write",
            ],
        },
        "competition": {
            "batch_image_count": 20,
            "batch_timing_source": "client_full_pipeline_wall_ms",
            "batch_fps_calculation": (
                "total_frames_divided_by_total_elapsed_seconds"
            ),
            "batch_total_frames": 60,
            "batch_total_elapsed_ms": total_elapsed_ms,
            "includes_result_persistence": True,
            "formal_results_valid": True,
            "batch_fps": aggregate_fps,
            "batch_fps_passed": aggregate_fps >= 30.0,
            "batch_rounds": [
                {
                    "round": index + 1,
                    "image_count": 20,
                    "full_pipeline_wall_ms": elapsed_ms,
                    "result_file_count": 20,
                    "formal_results_valid": True,
                    "fps": fps,
                }
                for index, (fps, elapsed_ms) in enumerate(
                    zip(round_fps, round_elapsed_ms)
                )
            ],
        },
        "gates": {
            "sample_count": True,
            "request_failures": True,
            "formal_result_write": True,
            "batch_fps": aggregate_fps >= 30.0,
        },
    }


def test_promotion_requires_all_accuracy_and_three_round_fps_gates() -> None:
    metrics = PROMOTE.validate_score(score_payload(0.81, 0.61, 1.0))
    assert metrics["base_map50"] == pytest.approx(0.81)
    expected_fps = 60 * 1000.0 / sum(
        20 * 1000.0 / fps for fps in [39.2, 39.4, 39.3]
    )
    assert PROMOTE.validate_benchmark(
        benchmark_payload([39.2, 39.4, 39.3]),
        "primary",
    ) == pytest.approx(expected_fps)

    with pytest.raises(ValueError, match="new_map50"):
        PROMOTE.validate_score(score_payload(0.81, 0.59, 1.0))
    with pytest.raises(ValueError, match="30 FPS"):
        PROMOTE.validate_benchmark(
            benchmark_payload([29.0, 29.2, 29.1]),
            "primary",
        )
    with pytest.raises(ValueError, match="schema v8"):
        PROMOTE.validate_benchmark(
            benchmark_payload([39.2, 39.4, 39.3], schema_version=7),
            "legacy-engine-only",
        )
    invalid_calculation = benchmark_payload([39.2, 39.4, 39.3])
    invalid_calculation["competition"]["batch_fps"] += 1.0
    with pytest.raises(ValueError, match="总帧数除以总耗时"):
        PROMOTE.validate_benchmark(invalid_calculation, "miscalculated")


def test_score_gate_and_build_scripts_use_current_yolo26_workflow() -> None:
    gate = (ROOT / "scripts/run_ascend310b_score_gate.sh").read_text(
        encoding="utf-8"
    )
    build = (ROOT / "scripts/build_ascend_yolo26_e2e_oms.sh").read_text(
        encoding="utf-8"
    )
    materialize = (
        ROOT / "tools/112_materialize_ascend_yolo26_candidate.py"
    ).read_text(encoding="utf-8")
    freeze = (ROOT / "tools/98_freeze_ascend_predictions.py").read_text(
        encoding="utf-8"
    )

    assert gate.count("AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION=1") == 2
    assert '--confidence "$SCORING_CONFIDENCE"' in gate
    assert '--method-config "$METHOD_CONFIG"' in gate
    assert '--batch-rounds "$BATCH_ROUNDS"' in gate
    assert 'pattern != "*.png"' in gate
    assert "atc --version" in gate
    assert "context_mode=model" in build
    assert "--input_shape=images:1,3,608,736" in build
    assert "skip_specialist_on_scene_and_base_evidence_v1" in materialize
    assert '"label_aware_online_routing": False' in materialize
    assert '"passed": len(records) == args.expected_images' in freeze


def test_primary_route_and_installer_preserve_service_boundaries() -> None:
    route = (ROOT / "scripts/manage_ascend310b_primary_route.sh").read_text(
        encoding="utf-8"
    )
    installer = (
        ROOT / "scripts/install_ascend310b_primary_services.sh"
    ).read_text(encoding="utf-8")

    assert "independent_yolo26_e2e_v1" in route
    assert 'PUBLIC_PORT=8501' in route
    assert 'MAIN_PORT="${2:-18501}"' in route
    assert 'COMMENT="AGILE_AGENT_ASCEND310B_PRIMARY"' in route
    assert "independent_yolo26_e2e_v1" in installer
    assert 'MAIN_PORT="${3:-18501}"' in installer
    assert 'PUBLIC_PORT=8501' in installer
    assert 'require_loopback_listener "$PUBLIC_PORT"' in installer
    assert 'require_loopback_listener "$MAIN_PORT"' in installer
