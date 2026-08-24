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
from fair_agent.ui.console import MENU, ConsoleFrontend, render_menu, render_page


def test_serve_is_bound_to_loopback(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(cli, "check_module", lambda _name: True)

    def fake_call(command, cwd, env=None):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    code = cli.cmd_serve(argparse.Namespace(config="configs/agent_pipeline.yaml"))
    assert code == 0
    assert captured["command"][:4] == [sys.executable, "-m", "uvicorn", "fair_agent.web.app:app"]
    assert captured["command"][captured["command"].index("--host") + 1] == "127.0.0.1"
    assert captured["command"][captured["command"].index("--port") + 1] == "8501"
    assert "--no-access-log" in captured["command"]
    assert Path(captured["env"]["AGILE_AGENT_CONFIG"]).resolve() == Path(
        "configs/agent_pipeline.yaml"
    ).resolve()


def test_serve_preserves_automatic_architecture_selection(monkeypatch) -> None:
    captured = {}
    monkeypatch.delenv("AGILE_AGENT_CONFIG", raising=False)
    monkeypatch.setattr(cli, "check_module", lambda _name: True)

    def fake_call(command, cwd, env=None):
        captured["env"] = env
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    code = cli.cmd_serve(argparse.Namespace(config="auto"))
    assert code == 0
    assert "AGILE_AGENT_CONFIG" not in captured["env"]


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
    assert config["decoding"] == {
        "backend": "opencv",
        "workers": 4,
        "opencv_threads": 0,
    }
    assert "limits" not in config
    runtime = Path("fair_agent/modules/web_inference.py").read_text(encoding="utf-8")
    assert "max_image_pixels" not in runtime
    assert "content_task_id" not in runtime


def test_cli_has_no_retired_image_size_commands() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["status"]).config == "auto"
    subcommands = next(action for action in parser._actions if action.dest == "command").choices
    assert "model-recheck" not in subcommands
    assert "freeze-candidate" not in subcommands
    assert "status" in subcommands
    assert "console" in subcommands


def test_detect_cli_exposes_result_controls_without_model_tuning() -> None:
    parser = cli.build_parser()
    subcommands = next(
        action for action in parser._actions if action.dest == "command"
    ).choices
    detect_parser = subcommands["detect"]
    options = {
        option
        for action in detect_parser._actions
        for option in action.option_strings
    }
    assert "--source" in options
    assert "--output" in options
    assert "--recursive" in options
    assert "--format" in options
    assert "--confidence" not in options
    assert "--profile" not in options

    args = parser.parse_args(["detect", "--source", "sample.png"])
    assert args.source == "sample.png"
    assert args.output is None
    assert args.recursive is False
    assert args.format == "text"
    assert not hasattr(args, "confidence")
    assert not hasattr(args, "profile")


def test_console_prioritizes_detection_without_manual_model_controls() -> None:
    source = Path("fair_agent/ui/console.py").read_text(encoding="utf-8")
    assert "[1] 单图识别" in MENU
    assert "[2] 批量识别" in MENU
    assert "传感器 [sar/ir" not in source
    assert "关注类别 [soldier" not in source
    assert "--confidence" not in source
    assert "--profile" not in source


def test_console_menu_groups_actions_and_only_shows_home_off_home_page() -> None:
    home = render_menu("home")
    status = render_menu("status")
    assert "识别" in home
    assert "查看" in home
    assert "系统" in home
    assert "[0] 返回首页" not in home
    assert "[0] 返回首页" in status


def test_interactive_console_refreshes_state_before_rendering(monkeypatch) -> None:
    refresh_values = []
    config = {
        "_config_path": "configs/agent_pipeline.yaml",
        "decision": {"default_context": {}},
    }

    class InteractiveInput:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setattr(cli.sys, "stdin", InteractiveInput())
    monkeypatch.setattr(cli, "load_args_config", lambda _args: config)
    monkeypatch.setattr(
        cli,
        "load_or_build_state",
        lambda _config, refresh=False: refresh_values.append(refresh) or {},
    )
    monkeypatch.setattr(cli, "build_decision", lambda *_args: {})
    monkeypatch.setattr(cli, "write_decision", lambda *_args: {})
    monkeypatch.setattr(cli, "run_console_frontend", lambda _config_path: 0)
    args = argparse.Namespace(once=False, refresh=False)
    assert cli.cmd_console(args) == 0
    assert refresh_values == [True]


def test_config_get_prints_scalars_without_yaml_document_markers(
    monkeypatch, capsys
) -> None:
    config = {
        "runtime": {"server_host": "127.0.0.1", "server_port": 8501}
    }
    monkeypatch.setattr(cli, "load_config", lambda _path, _overrides: config)

    for key, expected in (
        ("runtime.server_host", "127.0.0.1"),
        ("runtime.server_port", "8501"),
    ):
        args = argparse.Namespace(
            config_action="get",
            config="auto",
            config_overrides=[],
            key=key,
        )
        assert cli.cmd_config(args) == 0
        assert capsys.readouterr().out == f"{expected}\n"


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


def test_bootstrap_reuses_compatible_cuda_environment() -> None:
    content = Path("scripts/bootstrap_x86.sh").read_text(encoding="utf-8")
    assert "AGILE_AGENT_PYTHON" in content
    assert "python_environment_supported" in content
    assert 'AGENT_PYTHON="${VIRTUAL_ENV}/bin/python"' in content
    assert 'AGENT_PYTHON="${CONDA_PREFIX}/bin/python"' in content
    assert 'IFS= read -r REGISTERED_PYTHON < .agent-python' in content
    project_venv = content.index('elif [[ -x .venv/bin/python ]]')
    assert content.index('elif [[ -n "${VIRTUAL_ENV:-}" ]]') < project_venv
    assert content.index('elif [[ -n "${CONDA_PREFIX:-}" ]]') < project_venv
    assert "torch.version.cuda" in content
    assert 'tuple(map(int, match.groups())) < (2, 0)' in content
    assert "torchvision.ops.nms" in content
    assert "dependencies_compatible" in content
    assert "project_entrypoint_compatible" in content
    assert 'pip install -e . --no-deps' in content
    registration = 'printf \'%s\\n\' "${AGENT_PYTHON}" > .agent-python'
    assert registration in content
    assert content.index(registration) < content.index("torch_stack_compatible()")
    assert "python -m pip install --upgrade pip" not in content
    assert "--force-reinstall" not in content
    assert "-c constraints-agent.txt" not in content


def test_start_script_reuses_bootstrap_selected_python() -> None:
    content = Path("scripts/start_agent.sh").read_text(encoding="utf-8")
    assert '[[ "${AGENT_PLATFORM}" == x86 && -f "${ROOT_DIR}/.agent-python" ]]' in content
    assert 'IFS= read -r REGISTERED_PYTHON < "${ROOT_DIR}/.agent-python"' in content
    assert 'AGENT_PYTHON="${VIRTUAL_ENV}/bin/python"' in content
    assert 'AGENT_PYTHON="${CONDA_PREFIX}/bin/python"' in content
    assert 'MACHINE_ARCH_RAW="$(uname -m)"' in content
    assert 'MACHINE_ARCH="${MACHINE_ARCH_RAW,,}"' in content
    assert 'export AGILE_AGENT_PYTHON="${AGENT_PYTHON}"' in content
    assert "AGILE_AGENT_ASCEND_ENV" in content
    assert "agent_pipeline_ascend310b.yaml" not in content


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


def test_ascend_doctor_does_not_require_unused_pandas() -> None:
    x86_modules = cli.workbench_modules_for_backend("ultralytics_cuda")
    ascend_modules = cli.workbench_modules_for_backend("ascend_acl")
    assert "pandas" in x86_modules
    assert "pandas" not in ascend_modules
    assert ascend_modules == ["starlette", "uvicorn", "multipart"]


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


def test_blackboard_uses_demo_dataset_state(tmp_path: Path) -> None:
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
        "models/production/incremental_detection/base_context_prior.json",
        "models/production/incremental_detection/incremental_context_prior.json",
        "models/production/incremental_detection/metrics.json",
        "models/production/incremental_detection/profile.json",
    } <= set(result["required_assets"])
    assert all(item["exists"] for item in result["required_assets"].values())
    assert result["functional_models"]["valid"] is True
    assert result["functional_models"]["distinct_function_count"] == 3
    assert result["model_generations"]["production"] == (
        "incremental_detection_generation_4plus2"
    )
    assert result["model_generations"]["production_classes"] == [0, 1, 2, 3, 4, 5]
    assert result["model_generations"]["candidate_status"] == "active"
    assert result["blockers"] == []


def test_manifest_blocks_uncalibrated_or_overlapping_true_new_class() -> None:
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    assert _validate_model_manifest(manifest) == []
    candidate = manifest["incremental_models"][0]
    candidate.pop("calibration_source", None)
    calibration_sources = candidate.pop("calibration_sources")
    assert "manifest_new_class_calibration_missing:incremental_detector" in _validate_model_manifest(manifest)
    candidate["calibration_sources"] = calibration_sources
    candidate["base_class_ids"].append(4)
    assert "manifest_new_class_overlaps_base:incremental_detector" in _validate_model_manifest(manifest)
    candidate["base_class_ids"].remove(4)
    candidate["deployment_accepted"] = False
    assert _validate_model_manifest(manifest) == []
    candidate["competition_accepted"] = False
    assert "manifest_competition_gates_missing:incremental_detector" in _validate_model_manifest(manifest)


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
    assert payload["deployment"]["ascend_310b"] == "ready"
    assert any(item["external"] for item in payload["blockers"])
    assert "外部门禁" not in text
    assert "本地赛题评测输入未配置" not in text
    assert "官方提交格式尚未确认" not in text


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
    answers = iter(["4", "5", "h", "q"])
    output = []
    frontend = ConsoleFrontend(input_fn=lambda _prompt: next(answers), output_fn=output.append, clear_screen=False)
    monkeypatch.setattr(frontend, "_state", lambda: (state, decision))
    assert frontend.run() == 0
    rendered = "\n".join(output)
    assert "灵动 Agent · 视觉识别终端" in rendered
    assert "SSH 视觉识别终端" not in rendered
    assert "CLI 自动使用正式模型" not in rendered
    assert "运行状态" in rendered
    assert "当前正式模型" in rendered
    assert "CLI 使用帮助" in rendered
    assert "视觉识别终端已退出" in rendered


def test_cli_frontend_pages_share_operator_state() -> None:
    config = load_config()
    state = build_blackboard(config)
    decision = build_decision(config, state, {"sensor": "sar", "scene": "all", "class_focus": "soldier"})
    overview = render_page("overview", state, decision)
    assert "灵动 Agent · 视觉识别终端" in overview
    assert "incremental_detection_generation_4plus2" in overview
    assert "人员 / 小型飞行器 / 舰船 / 坦克 / 巡逻艇 / 装甲车辆" in overview
    assert "当前正式模型" in render_page("models", state, decision)
    status = render_page("status", state, decision)
    assert "外部门禁" not in status
    assert "本地赛题评测输入未配置" not in status
    assert "官方提交格式尚未确认" not in status
    assert "当前没有影响本机识别的问题" in status


def test_cli_frontend_runs_single_detection_and_returns_home(monkeypatch) -> None:
    config = load_config()
    state = build_blackboard(config)
    decision = build_decision(
        config,
        state,
        {"sensor": "sar", "scene": "all", "class_focus": "soldier"},
    )
    answers = iter(["1", '"/tmp/input image.png"', "/tmp/result", "", "q"])
    output = []
    calls = []
    frontend = ConsoleFrontend(
        input_fn=lambda _prompt: next(answers),
        output_fn=output.append,
        clear_screen=False,
    )
    monkeypatch.setattr(frontend, "_state", lambda: (state, decision))

    def fake_call(arguments, *, timeout=600):
        calls.append((arguments, timeout))
        return subprocess.CompletedProcess(arguments, 0, "识别完成并已保存", "")

    monkeypatch.setattr(frontend, "_call_cli", fake_call)
    assert frontend.run() == 0
    assert calls == [
        (
            [
                "detect",
                "--source",
                "/tmp/input image.png",
                "--output",
                "/tmp/result",
            ],
            3600,
        )
    ]
    assert "识别完成并已保存" in "\n".join(output)
