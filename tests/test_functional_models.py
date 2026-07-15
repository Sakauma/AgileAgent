from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml
import pytest

from fair_agent.models.context import require_cuda_device
from fair_agent.modules.functional_models import validate_functional_models


def test_three_distinct_functional_models_are_registered() -> None:
    result = validate_functional_models("configs/functional_models.yaml")
    assert result["valid"] is True, result["errors"]
    assert result["model_count"] == 3
    assert result["distinct_function_count"] == 3
    assert result["all_x86_gpu_ready"] is True
    assert result["all_ascend_310b_ready"] is False
    assert result["strict_class_incremental"]["true_class_incremental_verified"] is True
    assert result["strict_class_incremental"]["production_profile"]["model_id"] == "incremental_detector"
    assert {item["function"] for item in result["models"]} == {
        "context_perception",
        "multimodal_target_detection",
        "incremental_object_detection",
    }
    incremental = next(
        item for item in result["models"] if item["function"] == "incremental_object_detection"
    )
    assert incremental["evidence"]["summary"]["true_class_incremental_verified"] is True
    assert incremental["evidence"]["summary"]["production_class_incremental"]["activation_threshold"] == 0.63
    assert incremental["artifact_count"] == 1
    assert incremental["evidence"]["summary"]["protocol_count"] == 1


def test_functional_registry_rejects_tampered_hash(tmp_path: Path) -> None:
    registry = yaml.safe_load(Path("configs/functional_models.yaml").read_text(encoding="utf-8"))
    registry = deepcopy(registry)
    registry["models"][0]["artifacts"][0]["sha256"] = "0" * 64
    path = tmp_path / "functional_models.yaml"
    path.write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True), encoding="utf-8")
    result = validate_functional_models(path)
    assert result["valid"] is False
    assert any(error.startswith("functional_artifact_invalid") for error in result["errors"])


def test_context_model_lock_metrics_pass_declared_thresholds() -> None:
    metrics = json.loads(Path("models/context/scene_sensor_metrics.json").read_text(encoding="utf-8"))
    assert metrics["acceptance"]["passed"] is True
    assert metrics["lock"]["sensor_accuracy"] >= 0.90
    assert metrics["lock"]["scene_accuracy"] >= 0.70
    assert metrics["lock"]["joint_accuracy"] >= 0.65


def test_context_model_rejects_cpu_execution() -> None:
    with pytest.raises(ValueError, match="不提供 CPU"):
        require_cuda_device("cpu")


def test_registry_describes_the_real_policy_integration() -> None:
    registry = yaml.safe_load(Path("configs/functional_models.yaml").read_text(encoding="utf-8"))
    detector = next(item for item in registry["models"] if item["function"] == "multimodal_target_detection")
    assert detector["inputs"] == ["image_rgb"]
    first_edge = registry["collaboration"][0]
    assert first_edge["payload"] == "agent_policy_context"
    assert first_edge["status"] == "implemented"


def test_incremental_status_is_derived_from_frozen_evidence(tmp_path: Path) -> None:
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    manifest["incremental_models"][0]["acceptance"] = "failed"
    evidence_path = tmp_path / "manifest.json"
    evidence_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    registry = yaml.safe_load(Path("configs/functional_models.yaml").read_text(encoding="utf-8"))
    incremental = next(item for item in registry["models"] if item["function"] == "incremental_object_detection")
    incremental["evidence"]["path"] = str(evidence_path)
    incremental["evidence"]["sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    registry_path = tmp_path / "functional_models.yaml"
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True), encoding="utf-8")

    result = validate_functional_models(registry_path)
    assert result["valid"] is False
    assert "functional_status_mismatch:incremental_model_bank_v1" in result["errors"]
