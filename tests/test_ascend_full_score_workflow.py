from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATERIALIZE = load_tool("109_materialize_ascend_full_score_candidate.py")
SELECT = load_tool("110_select_ascend_full_score_candidate.py")
SCORE = load_tool("94_score_ascend_agent.py")
TRAIN = load_tool("107_train_shared_dual_head.py")
PROMOTE = load_tool("111_promote_ascend_full_score_release.py")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/ascend310b/full_score_method.yaml").read_text(encoding="utf-8")
    )


def test_full_score_method_preserves_reference_and_has_no_machine_paths() -> None:
    method = method_config()

    MATERIALIZE.validate_method(method)

    assert method["training"]["method"] == "residual_adapter"
    assert method["training"]["checkpoint_metric"] == "map50"
    assert method["training"]["export_checkpoints"] == ["best", "last"]
    assert method["training"]["reference_export_checkpoint"] == "last"
    assert method["export"]["model_layout"] == "shared_backbone_dual_head_v1"
    assert method["export"]["output_contract"] == "raw_dual_head_v1"
    assert method["runtime"]["context_mode"] == "fixed_neutral_v1"
    assert (
        method["export"]["aipp_config"] == "configs/ascend310b/aipp/base_detector.cfg"
    )
    assert method["benchmark"]["image_contract"] == {
        "root_glob": "*.png",
        "width": 640,
        "height": 512,
        "bit_depth": 8,
        "color_types": [2, 6],
    }
    assert method["threshold_search"]["current_seed"] == {
        "old": 0.05,
        "new": 0.30,
    }
    assert method["threshold_search"]["stage_1_new"]["values"] == [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
    ]
    assert method["threshold_search"]["stage_2_old"]["values"] == [
        0.03,
        0.05,
        0.10,
        0.20,
        0.30,
    ]
    assert method["reference_result"]["base_map50"] == pytest.approx(0.8049006528)
    assert method["reference_result"]["new_map50"] == pytest.approx(0.6050327631)
    assert method["reference_result"]["krr"] == 1.0
    assert method["reference_result"]["dual_head_onnx_sha256"] == (
        "5d6651a25cdc227a6feaf3135d754f1d132f0740117156f9b6d0651c32104c5e"
    )
    assert method["reference_result"]["export_checkpoint_sha256"] == (
        "6d1e7098015134615b32a7cedeeab9352bb83adf812f227b85044a3e64da9c6a"
    )

    invalid = copy.deepcopy(method)
    invalid["reference_result"]["artifact"] = "/home/user/candidate.om"
    with pytest.raises(ValueError, match="绝对路径"):
        MATERIALIZE.validate_method(invalid)


def test_score_class_contract_can_follow_a_new_dataset_mapping() -> None:
    assert SCORE.parse_class_ids("4, 7,9") == [4, 7, 9]
    with pytest.raises(SCORE.argparse.ArgumentTypeError, match="互异"):
        SCORE.parse_class_ids("4,4")


def test_training_and_score_tools_read_the_single_method_contract(
    tmp_path: Path,
) -> None:
    method = method_config()
    method["competition"]["accuracy_gates"]["base_map50_min"] = 0.83
    path = tmp_path / "method.yaml"
    path.write_text(yaml.safe_dump(method, sort_keys=False), encoding="utf-8")

    _, training = TRAIN._training_contract(path)
    gates = SCORE.load_accuracy_gates(path)

    assert training["epochs"] == 100
    assert training["export_checkpoints"] == ["best", "last"]
    assert gates == {"base_map50": 0.83, "new_map50": 0.60, "krr": 0.95}
    with pytest.raises(ValueError, match="--epochs"):
        TRAIN._locked_option(99, training["epochs"], "--epochs")


def write_build_inputs(tmp_path: Path, method: dict) -> tuple[Path, Path, Path]:
    dual = tmp_path / "shared_backbone_dual_head.om"
    context = tmp_path / "scene_sensor_net.om"
    dual.write_bytes(b"dual-om")
    context.write_bytes(b"context-om")
    training_report = tmp_path / "training-report.json"
    export_manifest = tmp_path / "export-manifest.json"
    training_report.write_text("{}", encoding="utf-8")
    export_manifest.write_text("{}", encoding="utf-8")
    method_path = ROOT / "configs/ascend310b/full_score_method.yaml"
    heads = copy.deepcopy(method["export"]["logical_heads"])
    for head in heads.values():
        head.pop("output_shape")
    manifest = tmp_path / "build-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "soc_version": "Ascend310B1",
                "cann_version": "7.0.RC1",
                "precision": "mixed_float16",
                "model_layout": "shared_backbone_dual_head_v1",
                "method_config": {
                    "path": str(method_path),
                    "sha256": digest(method_path),
                },
                "training_report": {
                    "path": str(training_report),
                    "sha256": digest(training_report),
                },
                "export_manifest": {
                    "path": str(export_manifest),
                    "sha256": digest(export_manifest),
                },
                "artifacts": {
                    "dual": {
                        "role": "dual_detector",
                        "om": {"path": str(dual), "sha256": digest(dual)},
                        "output_contract": "raw_dual_head_v1",
                        "logical_heads": heads,
                    },
                    "context": {
                        "role": "context",
                        "om": {"path": str(context), "sha256": digest(context)},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return dual, context, manifest


def test_materialize_forces_8502_and_injects_verified_dual_head(tmp_path: Path) -> None:
    method = method_config()
    dual, context, manifest = write_build_inputs(tmp_path, method)
    base = yaml.safe_load(
        (ROOT / "configs/agent_pipeline_ascend310b.yaml").read_text(encoding="utf-8")
    )

    result = MATERIALIZE.build_candidate_config(
        base,
        method,
        dual_om=dual,
        context_om=context,
        build_manifest=manifest,
        old_threshold=0.05,
        new_threshold=0.30,
        report_root="reports/ascend310b/test",
        method_config=ROOT / "configs/ascend310b/full_score_method.yaml",
    )

    assert result["runtime"]["server_port"] == 8502
    assert result["ascend_backend"]["validated"] is False
    assert result["ascend_backend"]["validation_candidate"] is True
    assert result["ascend_backend"]["context_mode"] == "fixed_neutral_v1"
    assert len(result["ascend_backend"]["models"]) == 1
    model = next(iter(result["ascend_backend"]["models"].values()))
    assert model["sha256"] == digest(dual)
    assert model["logical_heads"]["old"]["candidate_confidence"] == 0.05
    assert model["logical_heads"]["new"]["candidate_confidence"] == 0.30

    searched = MATERIALIZE.build_candidate_config(
        base,
        method,
        dual_om=dual,
        context_om=context,
        build_manifest=manifest,
        old_threshold=0.03,
        new_threshold=0.20,
        report_root="reports/ascend310b/test-search",
        method_config=ROOT / "configs/ascend310b/full_score_method.yaml",
    )
    searched_model = next(iter(searched["ascend_backend"]["models"].values()))
    assert searched_model["logical_heads"]["old"]["candidate_confidence"] == 0.03
    assert searched_model["logical_heads"]["new"]["candidate_confidence"] == 0.20

    original_manifest = manifest.read_text(encoding="utf-8")
    mismatched = json.loads(original_manifest)
    mismatched["method_config"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(mismatched), encoding="utf-8")
    with pytest.raises(ValueError, match="method_config|方法配置"):
        MATERIALIZE.build_candidate_config(
            base,
            method,
            dual_om=dual,
            context_om=context,
            build_manifest=manifest,
            old_threshold=0.05,
            new_threshold=0.30,
            report_root="reports/ascend310b/test-method-mismatch",
            method_config=ROOT / "configs/ascend310b/full_score_method.yaml",
        )
    manifest.write_text(original_manifest, encoding="utf-8")

    dual.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        MATERIALIZE.build_candidate_config(
            base,
            method,
            dual_om=dual,
            context_om=context,
            build_manifest=manifest,
            old_threshold=0.05,
            new_threshold=0.30,
            report_root="reports/ascend310b/test",
            method_config=ROOT / "configs/ascend310b/full_score_method.yaml",
        )


def score_payload(base: float, new: float, krr: float, warnings=None) -> dict:
    return {
        "schema_version": 2,
        "unlabeled_predictions_frozen_before_labels": True,
        "metrics": {"base_map50": base, "new_map50": new, "krr": krr},
        "diagnostic_warnings": list(warnings or []),
    }


def benchmark_payload(round_fps: list[float]) -> dict:
    median = sorted(round_fps)[len(round_fps) // 2]
    return {
        "schema_version": 5,
        "protocol": {
            "batch_probe_size": 20,
            "batch_rounds": 3,
            "target_batch_fps": 30.0,
        },
        "competition": {
            "batch_image_count": 20,
            "batch_fps": median,
            "batch_fps_passed": median >= 30.0,
            "batch_rounds": [
                {"round": index + 1, "fps": fps} for index, fps in enumerate(round_fps)
            ],
        },
    }


def test_formal_promotion_uses_only_score_gates_but_keeps_validity_prerequisites() -> None:
    score = score_payload(
        0.8049006528,
        0.6050327631,
        1.0,
        ["business_json_equivalence", "lock_precision"],
    )
    score.update(
        {
            "competition_gates": {
                "base_map50": True,
                "new_map50": True,
                "krr": True,
            },
            "score_passed": True,
        }
    )
    metrics = PROMOTE.validate_score(score)
    assert metrics["base_map50"] == pytest.approx(0.8049006528)

    benchmark = benchmark_payload([30.066, 30.071, 30.039])
    benchmark["gates"] = {
        "sample_count": True,
        "request_failures": True,
        "batch_fps": True,
    }
    assert PROMOTE.validate_benchmark(benchmark, "primary") == pytest.approx(30.066)

    validity = PROMOTE.validate_training_report(
        {
            "kind": "shared_backbone_dual_head_training",
            "shared_max_drift": 0.0,
            "dataset_audit": {
                "old_raw_image_count": 0,
                "old_raw_label_count": 0,
                "old_feature_cache_count": 0,
                "original_data_modified": False,
            },
        }
    )
    assert validity == {
        "incremental_data_isolation": True,
        "shared_max_drift_zero": True,
    }

    invalid = copy.deepcopy(score)
    invalid["metrics"]["new_map50"] = 0.59
    with pytest.raises(ValueError, match="new_map50"):
        PROMOTE.validate_score(invalid)


def write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path.name


def candidate_row(
    tmp_path: Path,
    candidate_id: str,
    *,
    score: dict,
    benchmark: dict,
) -> dict:
    return {
        "id": candidate_id,
        "score": write_json(tmp_path / f"{candidate_id}-score.json", score),
        "benchmark": write_json(tmp_path / f"{candidate_id}-benchmark.json", benchmark),
        "prerequisites": {
            "incremental_data_isolation": True,
            "asset_hashes_verified": True,
        },
    }


def test_selector_uses_only_four_score_gates_and_accuracy_headroom(
    tmp_path: Path,
) -> None:
    rows = [
        candidate_row(
            tmp_path,
            "reference",
            score=score_payload(0.8049, 0.6050, 1.0, ["lock_precision"]),
            benchmark=benchmark_payload([30.066, 30.071, 30.039]),
        ),
        candidate_row(
            tmp_path,
            "more-headroom",
            score=score_payload(0.81, 0.61, 1.0, ["false_activation_rate"]),
            benchmark=benchmark_payload([30.01, 30.02, 30.03]),
        ),
    ]
    index = tmp_path / "candidates.json"
    payload = {"schema_version": 1, "candidates": rows}

    result = SELECT.select_candidates(index, payload)

    assert result["status"] == "full_score_winner"
    assert result["selected_candidate"] == "more-headroom"
    selected = next(row for row in result["candidates"] if row["id"] == "more-headroom")
    assert selected["full_score"] is True
    assert selected["diagnostic_warnings"] == ["false_activation_rate"]


def test_selector_marks_best_sub_30_fps_candidate_intermediate(tmp_path: Path) -> None:
    rows = [
        candidate_row(
            tmp_path,
            "slower",
            score=score_payload(0.82, 0.62, 1.0),
            benchmark=benchmark_payload([29.0, 29.1, 29.2]),
        ),
        candidate_row(
            tmp_path,
            "faster",
            score=score_payload(0.81, 0.61, 1.0),
            benchmark=benchmark_payload([29.4, 29.5, 29.6]),
        ),
    ]

    result = SELECT.select_candidates(
        tmp_path / "candidates.json", {"schema_version": 1, "candidates": rows}
    )

    assert result["status"] == "intermediate_only"
    assert result["selected_candidate"] == "faster"
    assert result["full_score_achieved"] is False


def test_selector_rejects_invalid_evidence_even_when_metrics_pass(
    tmp_path: Path,
) -> None:
    row = candidate_row(
        tmp_path,
        "invalid",
        score=score_payload(0.82, 0.62, 1.0),
        benchmark=benchmark_payload([31.0, 31.1, 31.2]),
    )
    row["prerequisites"]["asset_hashes_verified"] = False

    result = SELECT.select_candidates(
        tmp_path / "candidates.json",
        {"schema_version": 1, "candidates": [row]},
    )

    assert result["status"] == "no_eligible_candidate"
    assert result["selected_candidate"] is None
    assert result["candidates"][0]["prerequisite_failures"] == ["asset_hashes_verified"]


def test_selector_reads_score_thresholds_from_method_config(tmp_path: Path) -> None:
    row = candidate_row(
        tmp_path,
        "method-driven",
        score=score_payload(0.81, 0.61, 1.0),
        benchmark=benchmark_payload([30.1, 30.2, 30.3]),
    )
    method = method_config()
    method["competition"]["accuracy_gates"]["base_map50_min"] = 0.82

    result = SELECT.select_candidates(
        tmp_path / "candidates.json",
        {"schema_version": 1, "candidates": [row]},
        method,
    )

    assert result["status"] == "no_eligible_candidate"
    assert result["candidates"][0]["accuracy_failures"] == ["base_map50"]


def test_score_gate_authorizes_candidates_and_uses_method_contract() -> None:
    gate = (ROOT / "scripts/run_ascend310b_score_gate.sh").read_text(encoding="utf-8")
    build = (ROOT / "scripts/build_ascend_dual_head_om.sh").read_text(encoding="utf-8")
    freeze = (ROOT / "tools/98_freeze_ascend_predictions.py").read_text(
        encoding="utf-8"
    )

    assert gate.count("AGILE_AGENT_ASCEND_CANDIDATE_VALIDATION=1") == 2
    assert '--confidence "$SCORING_CONFIDENCE"' in gate
    assert '--method-config "$METHOD_CONFIG"' in gate
    assert '--batch-rounds "$BATCH_ROUNDS"' in gate
    assert 'pattern != "*.png"' in gate
    assert "atc --version" in gate
    assert "candidate_confidence" not in build
    assert "AGILE_AGENT_LOGICAL_HEADS_JSON" in build
    assert "fixed_reference_evidence_compatibility" in build
    assert 'training_report.get("checkpoints")' in build
    assert '"passed": len(records) == args.expected_images' in freeze
