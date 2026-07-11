from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from fair_agent import cli
from fair_agent.core.audit import make_run_dir
from fair_agent.core.blackboard import build_blackboard
from fair_agent.core.config import load_config
from fair_agent.modules.incremental_review import write_incremental_review
from fair_agent.modules.operator_view import build_operator_snapshot, render_snapshot
from fair_agent.modules.release_verification import verify_release
from fair_agent.policies.decision import build_decision
from fair_agent.ui.console import ConsoleFrontend, render_page


def test_serve_is_bound_to_loopback(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(cli, "check_module", lambda _name: True)

    def fake_call(command, cwd):
        captured["command"] = command
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    code = cli.cmd_serve(argparse.Namespace(config="configs/agent_pipeline.yaml"))
    assert code == 0
    assert "--server.address=127.0.0.1" in captured["command"]
    assert "--server.port=8501" in captured["command"]


def test_serve_stops_cleanly_on_keyboard_interrupt(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "check_module", lambda _name: True)

    def interrupted_call(_command, cwd):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.subprocess, "call", interrupted_call)
    code = cli.cmd_serve(argparse.Namespace(config="configs/agent_pipeline.yaml"))
    output = capsys.readouterr().out
    assert code == 0
    assert "工作台已停止" in output


def test_cli_has_no_retired_image_size_commands() -> None:
    parser = cli.build_parser()
    subcommands = next(action for action in parser._actions if action.dest == "command").choices
    assert "model-recheck" not in subcommands
    assert "freeze-candidate" not in subcommands
    assert "status" in subcommands
    assert "console" in subcommands


def test_run_directories_are_unique(tmp_path: Path) -> None:
    first = make_run_dir("dryrun", str(tmp_path))
    second = make_run_dir("dryrun", str(tmp_path))
    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_decision_commands_use_the_active_python() -> None:
    config = load_config()
    state = build_blackboard(config)
    decision = build_decision(config, state, {"sensor": "sar", "scene": "urban", "class_focus": "soldier"})
    commands = [item for item in decision["candidates"] if item.get("argv")]
    assert commands
    for item in commands:
        assert item["argv"][0] == sys.executable
        assert all(value != "None" for value in item["argv"])
        assert "None" not in item["command"]


def test_shell_entrypoints_have_valid_syntax() -> None:
    for script in ["scripts/bootstrap_x86.sh", "scripts/start_agent.sh"]:
        result = subprocess.run(["bash", "-n", script], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_bootstrap_selects_only_supported_python() -> None:
    content = Path("scripts/bootstrap_x86.sh").read_text(encoding="utf-8")
    assert "python3.12 python3.11 python3.10" in content
    assert 'uv venv --python "${UV_PYTHON:-3.12}" --seed' in content
    assert "python3.8" not in content
    assert "nvidia-smi" in content
    assert "2.5.1+cu124" in content
    assert "0.20.1+cu124" in content


def test_doctor_fails_when_workbench_dependency_is_missing(monkeypatch, capsys) -> None:
    external = {
        "returncode": 0,
        "modules": {
            "yaml": True, "PIL": True, "pandas": True, "streamlit": False,
            "ultralytics": True, "cv2": True, "torch": True,
        },
        "accelerator": {"cuda_available": True, "cuda_device_count": 1, "cuda_devices": ["test-gpu"]},
    }
    monkeypatch.setattr(cli, "check_external_python", lambda _path: external)
    monkeypatch.setattr(cli, "configured_python", lambda _config: Path(sys.executable))
    code = cli.cmd_doctor(argparse.Namespace(config="configs/agent_pipeline.yaml"))
    capsys.readouterr()
    assert code == 1


def test_pipeline_execute_advances_until_no_action(monkeypatch, tmp_path: Path) -> None:
    config = {
        "automation": {"run_root": str(tmp_path / "runs"), "max_steps_per_run": 8},
        "blackboard": {},
        "decision": {"outputs": {}},
    }
    state = {"stage": 0, "generated_at": "now", "current_blockers": [], "frozen_assets": {}}
    executed = []

    def decision_for(_config, current, _context):
        stage = current["stage"]
        if stage < 2:
            name = f"action_{stage + 1}"
            action = {"action": name, "status": "ready", "freshness": "current", "reason": name, "risk_level": "low", "can_execute": True}
        else:
            action = {"action": "wait_for_external_input", "status": "blocked", "freshness": "current", "reason": "done", "risk_level": "low", "can_execute": False}
        return {"recommended_action": action, "candidates": [], "current_blockers": []}

    def execute(_config, action, _log):
        executed.append(action["action"])
        state["stage"] += 1
        return {"action": action["action"], "returncode": 0, "status": "completed"}

    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(cli, "build_blackboard", lambda _config: state)
    monkeypatch.setattr(cli, "build_decision", decision_for)
    monkeypatch.setattr(cli, "execute_low_risk_action", execute)
    monkeypatch.setattr(cli, "write_blackboard", lambda *_args: {})
    monkeypatch.setattr(cli, "write_decision", lambda *_args: {})
    args = argparse.Namespace(config="unused", mode="execute", sensor="sar", scene="all", class_focus="soldier")
    assert cli.cmd_pipeline(args) == 0
    assert executed == ["action_1", "action_2"]
    run_dir = next((tmp_path / "runs").iterdir())
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan["termination"] == "no_executable_action"
    assert [step["name"] for step in plan["steps"]] == executed


def test_blackboard_uses_demo_evidence_without_private_reports(tmp_path: Path) -> None:
    config = deepcopy(load_config())
    missing_root = tmp_path / "missing"
    for key, value in list(config["inputs"].items()):
        if isinstance(value, list):
            config["inputs"][key] = [str(missing_root / f"{index}.missing") for index, _ in enumerate(value)]
        else:
            config["inputs"][key] = str(missing_root / f"{key}.missing")
    config["decision"]["actions"]["diagnose_sar_soldier"]["inputs"] = [str(missing_root / "diagnose.csv")]
    config["decision"]["actions"]["diagnose_sar_soldier"]["outputs"] = [str(missing_root / "case.csv")]
    config["decision"]["actions"]["review_incremental_learning"]["inputs"] = [str(missing_root / "summary.csv")]
    config["decision"]["actions"]["review_incremental_learning"]["outputs"] = [str(missing_root / "summary.md")]
    state = build_blackboard(config)
    assert state["evidence"]["mode"] == "demo"
    assert state["dataset"]["image_count"] == 750
    assert state["sar_soldier"]["case_bank"]["case_count"] == 53
    assert len(state["incremental_learning"]["protocols"]) == 4
    assert state["submission"]["dryrun_valid"] is True
    assert state["submission"]["smoke_valid"] is True


def test_incremental_review_writes_declared_output(tmp_path: Path) -> None:
    output = tmp_path / "summary.md"
    config = {"decision": {"actions": {"review_incremental_learning": {"outputs": [str(output)]}}}}
    summary = {
        "complete": True,
        "passed": False,
        "compliance_verified": True,
        "acceptance": {"min_new_class_map50": 0.6, "min_krr": 0.95},
        "protocols": [{"protocol": "p01", "new_class": "small_aircraft", "new_map50": 0.55, "krr": 1.0, "old_raw_image_count": 0, "compliant": True, "passed": False}],
        "warnings": [],
    }
    assert write_incremental_review(config, summary) == output
    text = output.read_text(encoding="utf-8")
    assert "合规增量学习复核报告" in text
    assert "p01" in text


def test_static_release_verification_passes() -> None:
    result = verify_release()
    assert result["status"] == "passed", result["errors"]
    assert isinstance(result["required_assets"], dict)
    assert len(result["required_assets"]) == 9
    assert result["functional_models"]["valid"] is True
    assert result["functional_models"]["distinct_function_count"] == 3


def test_low_risk_action_output_cannot_escape_allowlist(tmp_path: Path) -> None:
    config = {"automation": {"allowed_output_roots": [str(tmp_path / "allowed")]}}
    safe = {"outputs": [str(tmp_path / "allowed" / "result.json")]}
    unsafe = {"outputs": [str(tmp_path / "outside" / "result.json")]}
    assert cli.validate_action_outputs(config, safe) is None
    assert cli.validate_action_outputs(config, unsafe).startswith("output_outside_allowlist")


def test_operator_snapshot_is_shared_by_text_and_json() -> None:
    config = load_config()
    state = build_blackboard(config)
    decision = build_decision(config, state, {"sensor": "sar", "scene": "urban", "class_focus": "soldier"})
    snapshot = build_operator_snapshot(state, decision)
    text = render_snapshot(snapshot, "text")
    payload = json.loads(render_snapshot(snapshot, "json"))
    assert "AgileAgent 终端工作台" in text
    assert snapshot["detector"]["name"] in text
    assert payload["detector"] == snapshot["detector"]
    assert payload["deployment"]["x86_nvidia_gpu"] == "ready"
    assert payload["deployment"]["ascend_310b"] == "waiting_for_hardware"


def test_status_command_supports_machine_readable_output(monkeypatch, capsys) -> None:
    config = load_config()
    state = build_blackboard(config)
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(cli, "load_or_build_state", lambda _config, refresh=False: state)
    monkeypatch.setattr(cli, "write_decision", lambda *_args: {})
    code = cli.cmd_status(argparse.Namespace(config="unused", format="json", refresh=True))
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["schema_version"] == 1
    assert payload["deployment"]["x86_nvidia_gpu"] == "ready"


def test_cli_frontend_navigates_all_operator_pages(monkeypatch) -> None:
    config = load_config()
    state = build_blackboard(config)
    decision = build_decision(config, state, {"sensor": "sar", "scene": "urban", "class_focus": "soldier"})
    answers = iter(["2", "3", "4", "5", "q"])
    output = []
    frontend = ConsoleFrontend(input_fn=lambda _prompt: next(answers), output_fn=output.append, clear_screen=False)
    monkeypatch.setattr(frontend, "_state", lambda: (state, decision))
    assert frontend.run() == 0
    rendered = "\n".join(output)
    assert "AgileAgent 终端工作台" in rendered
    assert "功能模型" in rendered
    assert "数据与诊断" in rendered
    assert "增量目标检测" in rendered
    assert "部署与提交" in rendered
    assert "终端工作台已退出" in rendered


def test_cli_frontend_pages_share_operator_state() -> None:
    config = load_config()
    state = build_blackboard(config)
    decision = build_decision(config, state, {"sensor": "sar", "scene": "all", "class_focus": "soldier"})
    assert "0.91202" in render_page("overview", state, decision)
    assert "waiting_for_hardware" in render_page("deployment", state, decision)
