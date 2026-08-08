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
from fair_agent.modules.operator_view import build_operator_snapshot, render_snapshot
from fair_agent.modules.release_verification import _validate_model_manifest, verify_release
from fair_agent.policies.decision import build_decision
from fair_agent.ui.console import ConsoleFrontend, render_page


def test_serve_is_bound_to_loopback(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(cli, "check_module", lambda _name: True)

    def fake_call(command, cwd, env=None):
        captured["command"] = command
        captured["cwd"] = cwd
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    code = cli.cmd_serve(argparse.Namespace(config="configs/agent_pipeline.yaml"))
    assert code == 0
    assert captured["command"][:4] == [sys.executable, "-m", "uvicorn", "fair_agent.web.app:app"]
    assert captured["command"][captured["command"].index("--host") + 1] == "127.0.0.1"
    assert captured["command"][captured["command"].index("--port") + 1] == "8501"
    assert "--no-access-log" in captured["command"]


def test_serve_stops_cleanly_on_keyboard_interrupt(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "check_module", lambda _name: True)

    def interrupted_call(_command, cwd, env=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.subprocess, "call", interrupted_call)
    code = cli.cmd_serve(argparse.Namespace(config="configs/agent_pipeline.yaml"))
    output = capsys.readouterr().out
    assert code == 0
    assert "工作台已停止" in output


def test_web_ui_has_no_collapsible_sidebar_dependency() -> None:
    content = Path("fair_agent/web/static/index.html").read_text(encoding="utf-8")
    assert "mobileMenu" in content
    assert "main-nav" in content
    assert "sidebar" not in content.lower()


def test_online_detection_config_only_selects_decoder() -> None:
    config = load_config()
    assert config["decoding"] == {"backend": "opencv", "workers": 4}
    assert "limits" not in config
    runtime = Path("fair_agent/modules/web_inference.py").read_text(encoding="utf-8")
    assert "max_image_pixels" not in runtime
    assert "content_task_id" not in runtime


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


def test_decision_actions_are_fully_declared() -> None:
    config = load_config()
    state = build_blackboard(config)
    decision = build_decision(config, state, {"sensor": "sar", "scene": "urban", "class_focus": "soldier"})
    assert decision["candidates"]
    assert all(item.get("argv") or item.get("handler") for item in decision["candidates"])
    commands = [item for item in decision["candidates"] if item.get("argv")]
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
    assert 'scripts/smoke_models.py --load-only' in content


def test_readme_keeps_tensorrt_out_of_ascend_deployment_path() -> None:
    content = Path("README.md").read_text(encoding="utf-8")
    assert "TensorRT 不属于 310B 部署链路" in content
    assert "ATC 编译生成设备专用 OM" in content
    assert "AscendCL 负责模型加载" in content
    for retired_command in (
        "export_tensorrt_engines.sh",
        "tensorrt validate",
        "tensorrt calibrate",
        ".[workbench,inference,tensorrt,export]",
    ):
        assert retired_command not in content


def test_bootstrap_reuses_compatible_cuda_environment() -> None:
    content = Path("scripts/bootstrap_x86.sh").read_text(encoding="utf-8")
    assert "AGILE_AGENT_PYTHON" in content
    assert "torch.version.cuda" in content
    assert 'tuple(map(int, match.groups())) < (2, 0)' in content
    assert "torchvision.ops.nms" in content
    assert "dependencies_compatible" in content
    assert "project_entrypoint_compatible" in content
    assert 'pip install -e . --no-deps' in content
    assert "> .agent-python" in content
    assert "python -m pip install --upgrade pip" not in content
    assert "--force-reinstall" not in content
    assert "-c constraints-agent.txt" not in content


def test_start_script_reuses_bootstrap_selected_python() -> None:
    content = Path("scripts/start_agent.sh").read_text(encoding="utf-8")
    assert '[[ -f "${ROOT_DIR}/.agent-python" ]]' in content
    assert 'IFS= read -r AGENT_PYTHON < "${ROOT_DIR}/.agent-python"' in content


def test_doctor_fails_when_workbench_dependency_is_missing(monkeypatch, capsys) -> None:
    external = {
        "returncode": 0,
        "modules": {
            "yaml": True, "PIL": True, "pandas": True, "starlette": True,
            "uvicorn": True, "multipart": False,
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


def test_blackboard_uses_demo_dataset_without_legacy_reports(tmp_path: Path) -> None:
    config = deepcopy(load_config())
    missing_root = tmp_path / "missing"
    for key, value in list(config["inputs"].items()):
        if isinstance(value, list):
            config["inputs"][key] = [str(missing_root / f"{index}.missing") for index, _ in enumerate(value)]
        else:
            config["inputs"][key] = str(missing_root / f"{key}.missing")
    state = build_blackboard(config)
    assert state["evidence"]["mode"] == "mixed"
    assert state["dataset"]["image_count"] == 750
    assert len(state["incremental_learning"]["protocols"]) == 1
    assert state["incremental_learning"]["protocols"][0]["protocol"] == "incremental_detector"
    assert state["incremental_learning"]["source"] == "models/generations.json"
    assert "dryrun_valid" not in state["submission"]
    assert "smoke_valid" not in state["submission"]


def test_static_release_verification_passes() -> None:
    result = verify_release()
    assert result["status"] == "passed", result["errors"]
    assert isinstance(result["required_assets"], dict)
    assert {
        "models/production/incremental_detection/calibration.json",
        "models/production/incremental_detection/context_prior.json",
        "models/production/incremental_detection/metrics.json",
        "models/production/incremental_detection/profile.json",
    } <= set(result["required_assets"])
    assert all(item["exists"] for item in result["required_assets"].values())
    assert result["functional_models"]["valid"] is True
    assert result["functional_models"]["distinct_function_count"] == 3
    assert result["model_generations"]["production"] == "incremental_detection_generation"
    assert result["model_generations"]["production_classes"] == [0, 1, 2, 3]
    assert result["model_generations"]["candidate_status"] == "active"


def test_manifest_blocks_uncalibrated_or_overlapping_true_new_class() -> None:
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    assert _validate_model_manifest(manifest) == []
    candidate = manifest["incremental_models"][0]
    candidate.pop("calibration_source", None)
    assert "manifest_new_class_calibration_missing:incremental_detector" in _validate_model_manifest(manifest)
    candidate["calibration_source"] = "incremental_val/calibration.json"
    candidate["base_class_ids"].append(2)
    assert "manifest_new_class_overlaps_base:incremental_detector" in _validate_model_manifest(manifest)
    candidate["base_class_ids"].remove(2)
    candidate["deployment_accepted"] = False
    assert "manifest_deployment_gates_missing:incremental_detector" in _validate_model_manifest(manifest)


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
    assert "灵动Agent终端工作台" in text
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
    assert "灵动Agent终端工作台" in rendered
    assert "功能模型" in rendered
    assert "数据概况" in rendered
    assert "增量目标检测" in rendered
    assert "部署与提交" in rendered
    assert "终端工作台已退出" in rendered


def test_cli_frontend_pages_share_operator_state() -> None:
    config = load_config()
    state = build_blackboard(config)
    decision = build_decision(config, state, {"sensor": "sar", "scene": "all", "class_focus": "soldier"})
    overview = render_page("overview", state, decision)
    assert "0.8141423771111025" in overview
    assert "0.91202" not in overview
    assert "waiting_for_hardware" in render_page("deployment", state, decision)
