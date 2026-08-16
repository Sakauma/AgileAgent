from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from fair_agent.core.blackboard import build_blackboard
from fair_agent.core.config import load_config, validate_config
from fair_agent.executors.local import run_command
from fair_agent.policies.decision import build_decision


RELEASE_ASSETS_AVAILABLE = Path(
    "models/production/incremental_detection/three_class_base_detector.pt"
).exists()


@pytest.mark.skipif(not RELEASE_ASSETS_AVAILABLE, reason="release weights are not present")
def test_active_inference_uses_verified_frozen_weight() -> None:
    config = load_config()
    state = build_blackboard(config)
    inference = state["frozen_assets"]["inference_weights"]
    assert inference["path"] == "models/production/incremental_detection/three_class_base_detector.pt"
    assert inference["matches_expected"] is True
    assert inference["same_frozen_path"] is True
    assert state["frozen_assets"]["checksums"]["valid"] is True
    checksum_count = sum(
        1
        for line in Path("models/SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    assert state["frozen_assets"]["checksums"]["checked"] == checksum_count
    assert state["functional_models"]["valid"] is True
    assert state["functional_models"]["distinct_function_count"] == 3
    assert all(state["frozen_assets"]["artifacts"].values())
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    assert state["detector"]["imgsz"] == manifest["base_model"]["imgsz"]
    assert state["detector"]["candidate_status"] == "verified"


@pytest.mark.skipif(not RELEASE_ASSETS_AVAILABLE, reason="release weights are not present")
def test_blackboard_uses_verified_production_generation() -> None:
    state = build_blackboard(load_config())
    generation = state["model_generation"]
    assert generation["valid"] is True
    assert generation["production"] == "incremental_detection_generation"
    assert generation["incremental_verified"] is True
    assert "incremental_compliant_threshold_not_met" not in state["current_blockers"]


def test_decision_uses_current_generation() -> None:
    config = load_config()
    state = build_blackboard(config)
    decision = build_decision(config, state, {"sensor": "sar", "scene": "urban", "class_focus": "soldier"})
    assert "sar_soldier" not in state
    assert state["incremental_learning"]["source"] == "models/generations.json"
    assert state["incremental_learning"]["passed"] is True
    assert {item["action"] for item in decision["candidates"]} == {"formal_submission", "refresh_blackboard"}
    assert decision["recommended_action"]["action"] == "wait_for_external_input"


def test_executor_logs_timeout(tmp_path: Path) -> None:
    log = tmp_path / "actions.jsonl"
    code = run_command([sys.executable, "-c", "import time; time.sleep(2)"], tmp_path, log, "timeout", timeout=1)
    events = [json.loads(line)["event"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert code == 124
    assert events == ["start", "timeout"]


def test_config_validation_fails_closed() -> None:
    broken = {"model": {"weights": "x", "expected_sha256": "bad"}, "decision": {"actions": {}}}
    try:
        validate_config(broken)
    except ValueError as exc:
        assert "expected_sha256" in str(exc)
    else:
        raise AssertionError("invalid config was accepted")


def test_default_x86_inference_uses_gpu_zero() -> None:
    config = load_config()
    inference = yaml.safe_load(Path("configs/local_infer_gpu.yaml").read_text(encoding="utf-8"))
    assert config["runtime"]["default_device"] == "0"
    assert inference["predict"]["device"] == "0"
    assert inference["predict"]["batch"] == 32


def test_runtime_rejects_non_gpu_device() -> None:
    config = load_config()
    config["runtime"]["default_device"] = "cpu"
    with pytest.raises(ValueError, match="GPU 编号"):
        validate_config(config)


def test_start_scripts_only_launch_configured_agent() -> None:
    path = Path("scripts/start_agent.sh")
    content = path.read_text(encoding="utf-8")
    assert "fair_agent.cli" in content
    assert "doctor" in content
    assert "doctor --quiet" in content
    assert "refresh" in content
    assert "decide" in content
    assert "serve" in content
    assert '"--cli"' in content
    assert "fair_agent.cli console" in content
    assert "pip install" not in content
    assert "torch" not in content.lower()
    assert "-m venv" not in content.lower()
