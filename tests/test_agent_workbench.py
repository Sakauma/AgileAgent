from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from fair_agent.core.blackboard import build_blackboard
from fair_agent.core.config import load_config, validate_config
from fair_agent.executors.local import run_command
from fair_agent.modules.status import output_freshness, parse_incremental, parse_specialist
from fair_agent.policies.decision import build_decision


RELEASE_ASSETS_AVAILABLE = Path("models/base/yolo11s_ir_sar_imgsz640.pt").exists()
PRIVATE_REPORTS_AVAILABLE = Path("reports/agent_sar_soldier_casebank/sar_soldier_replay_metrics.csv").exists()


@pytest.mark.skipif(not RELEASE_ASSETS_AVAILABLE, reason="release weights are not present")
def test_active_inference_uses_verified_frozen_weight() -> None:
    config = load_config()
    state = build_blackboard(config)
    inference = state["frozen_assets"]["inference_weights"]
    assert inference["path"] == "models/base/yolo11s_ir_sar_imgsz640.pt"
    assert inference["matches_expected"] is True
    assert inference["same_frozen_path"] is True
    assert state["frozen_assets"]["checksums"]["valid"] is True
    assert state["frozen_assets"]["checksums"]["checked"] == 5
    assert all(state["frozen_assets"]["artifacts"].values())
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    assert state["detector"]["imgsz"] == manifest["base_model"]["imgsz"]
    assert state["detector"]["candidate_status"] == "stability_rechecked"


@pytest.mark.skipif(not PRIVATE_REPORTS_AVAILABLE, reason="private competition reports are not distributed")
def test_completed_diagnosis_does_not_loop() -> None:
    config = load_config()
    state = build_blackboard(config)
    decision = build_decision(config, state, {"sensor": "sar", "scene": "urban", "class_focus": "soldier"})
    candidate = next(item for item in decision["candidates"] if item["action"] == "diagnose_sar_soldier")
    assert candidate["status"] == "completed"
    assert candidate["can_execute"] is False
    assert decision["recommended_action"]["action"] in {"review_incremental_learning", "wait_for_external_input"}


@pytest.mark.skipif(not PRIVATE_REPORTS_AVAILABLE, reason="private competition reports are not distributed")
def test_stale_low_risk_diagnosis_becomes_executable() -> None:
    config = load_config()
    state = deepcopy(build_blackboard(config))
    state["sar_soldier"]["case_bank_freshness"] = {"freshness": "stale", "reason": "inputs_newer_than_outputs"}
    decision = build_decision(config, state, {"sensor": "sar", "scene": "urban", "class_focus": "soldier"})
    action = decision["recommended_action"]
    assert action["action"] == "diagnose_sar_soldier"
    assert action["can_execute"] is True
    assert action["risk_level"] == "low"


@pytest.mark.skipif(not PRIVATE_REPORTS_AVAILABLE, reason="private competition reports are not distributed")
def test_metrics_are_parsed_instead_of_inferred_from_report_existence() -> None:
    config = load_config()
    specialist = parse_specialist(config)
    incremental = parse_incremental(config)
    assert specialist["status"] == "rejected"
    assert specialist["deltas"]["lock_all_map50"] < 0
    assert incremental["complete"] is True
    assert incremental["compliance_required"] is True
    if incremental["compliance_verified"]:
        by_name = {row["protocol"]: row for row in incremental["protocols"]}
        assert by_name["p01_new_small_aircraft"]["passed"] is False
        assert all(by_name[name]["passed"] for name in ["p02_new_warship", "p03_new_tank", "p04_new_soldier"])
        assert incremental["passed"] is False
    else:
        assert incremental["passed"] is False


def test_output_freshness_detects_stale_output(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    output.write_text("old", encoding="utf-8")
    source.write_text("new", encoding="utf-8")
    os.utime(output, (1, 1))
    os.utime(source, (2, 2))
    assert output_freshness([str(source)], [str(output)])["freshness"] == "stale"


def test_missing_incremental_inputs_are_not_executable() -> None:
    config = load_config()
    state = deepcopy(build_blackboard(config))
    state["sar_soldier"]["case_bank_freshness"] = {
        "freshness": "missing", "reason": "missing_inputs", "missing": ["private.csv"]
    }
    state["incremental_learning"]["complete"] = False
    state["incremental_learning"]["freshness"] = {
        "freshness": "missing", "reason": "missing_inputs", "missing": ["private.csv"]
    }
    decision = build_decision(config, state, {"sensor": "sar", "scene": "urban", "class_focus": "soldier"})
    review = next(item for item in decision["candidates"] if item["action"] == "review_incremental_learning")
    assert review["status"] == "blocked"
    assert review["can_execute"] is False
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
    assert not Path("configs/local_infer_cpu.yaml").exists()
