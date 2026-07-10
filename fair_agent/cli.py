from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fair_agent.core.audit import make_run_dir, write_pipeline_artifacts
from fair_agent.core.blackboard import build_blackboard, write_blackboard
from fair_agent.core.config import ROOT, configured_python, load_config, rel_path, resolve_path
from fair_agent.executors.local import append_log, run_command
from fair_agent.modules.status import parse_incremental
from fair_agent.policies.decision import build_decision, write_decision


REQUIRED_MODULES = ["yaml", "pexpect", "PIL"]
WORKBENCH_MODULES = ["pandas", "streamlit"]
INFERENCE_MODULES = ["ultralytics", "cv2", "torch"]
ALL_MODULES = REQUIRED_MODULES + WORKBENCH_MODULES + INFERENCE_MODULES


def check_module(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def check_external_python(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.exists(), "modules": {}}
    if not path.exists():
        return result
    code = (
        "import importlib.util as u, json, sys; "
        f"mods={ALL_MODULES!r}; "
        "print(json.dumps({'executable': sys.executable, 'version': sys.version.split()[0], "
        "'modules': {m: bool(u.find_spec(m)) for m in mods}}, ensure_ascii=False))"
    )
    proc = subprocess.run([str(path), "-c", code], text=True, capture_output=True, timeout=60)
    result["returncode"] = proc.returncode
    if proc.returncode == 0 and proc.stdout.strip():
        result.update(json.loads(proc.stdout.strip()))
    else:
        result["stderr"] = proc.stderr[-1000:]
    return result


def cmd_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    py = configured_python(config)
    external = check_external_python(py)
    state = build_blackboard(config)
    artifacts = state.get("frozen_assets", {}).get("artifacts", {})
    result = {
        "runtime": external,
        "dependency_groups": {
            "required": REQUIRED_MODULES,
            "workbench": WORKBENCH_MODULES,
            "inference": INFERENCE_MODULES,
        },
        "weights": state.get("frozen_assets", {}).get("weights"),
        "inference_weights": state.get("frozen_assets", {}).get("inference_weights"),
        "frozen_checksums": state.get("frozen_assets", {}).get("checksums"),
        "core_artifacts": artifacts,
        "current_blockers": state.get("current_blockers"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    modules = external.get("modules", {})
    missing_required = [name for name in REQUIRED_MODULES if not modules.get(name)]
    missing_workbench = [name for name in WORKBENCH_MODULES if not modules.get(name)]
    missing_inference = [name for name in INFERENCE_MODULES if not modules.get(name)]
    if missing_required:
        print("Missing required modules:", ", ".join(missing_required))
    if missing_workbench:
        print("Missing workbench modules:", ", ".join(missing_workbench))
        print(f"Install workbench deps with: {py.parent / 'pip'} install -r requirements-agent.txt")
    if missing_inference:
        print("Missing optional inference modules:", ", ".join(missing_inference))
        print("Install inference deps only when local prediction is needed; avoid unplanned CUDA/Torch upgrades.")
    inference_ok = bool(result["inference_weights"].get("matches_expected")) and bool(result["inference_weights"].get("same_frozen_path"))
    core_artifacts_ok = all(bool(value) for value in artifacts.values())
    checksums_ok = bool(result["frozen_checksums"].get("valid"))
    return 1 if missing_required or not inference_ok or not core_artifacts_ok or not checksums_ok or external.get("returncode") != 0 else 0


def cmd_refresh(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    state = build_blackboard(config)
    paths = write_blackboard(config, state)
    print(rel_path(paths["state"]))
    print(rel_path(paths["report"]))
    return 0


def decision_context(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    defaults = config.get("decision", {}).get("default_context", {})
    return {
        "sensor": args.sensor or defaults.get("sensor", "sar"),
        "scene": args.scene or defaults.get("scene", "all"),
        "class_focus": args.class_focus or defaults.get("class_focus", "soldier"),
    }


def cmd_decide(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    state_path = resolve_path(config.get("blackboard", {}).get("output_dir", "reports/agent_blackboard")) / config.get("blackboard", {}).get("state_json", "blackboard_state.json")
    if state_path.exists() and args.cached and not args.refresh:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = build_blackboard(config)
        write_blackboard(config, state)
    decision = build_decision(config, state, decision_context(config, args))
    paths = write_decision(config, decision)
    print(rel_path(paths["json"]))
    print(rel_path(paths["report"]))
    print(decision["recommended_action"]["action"])
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    state = build_blackboard(config)
    context = {
        "sensor": args.sensor,
        "scene": args.scene,
        "class_focus": args.class_focus,
    }
    decision = build_decision(config, state, context)
    automation = config.get("automation", {})
    run_dir = make_run_dir("dryrun" if args.mode == "dryrun" else "execute", automation.get("run_root", "reports/agent_runs"))
    recommended = decision["recommended_action"]
    plan = {
        "mode": args.mode,
        "audit_before_execute": bool(config.get("automation", {}).get("audit_before_execute", True)),
        "context": context,
        "steps": [{
            "name": recommended["action"],
            "execute": args.mode == "execute" and bool(recommended.get("can_execute")),
            "reason": recommended["reason"],
        }],
    }
    paths = write_pipeline_artifacts(run_dir, plan, state, decision)
    action_results: List[Dict[str, Any]] = []
    if args.mode == "execute" and recommended.get("can_execute"):
        result = execute_low_risk_action(config, recommended, paths["log"])
        action_results.append(result)
        state = build_blackboard(config)
        decision = build_decision(config, state, context)
        write_pipeline_artifacts(run_dir, plan, state, decision, action_results)
    write_blackboard(config, state)
    write_decision(config, decision)
    print(rel_path(run_dir))
    print(rel_path(paths["plan"]))
    print(rel_path(paths["manifest"]))
    print(rel_path(paths["report"]))
    return 1 if any(item.get("returncode") != 0 for item in action_results) else 0


def execute_low_risk_action(config: Dict[str, Any], action: Dict[str, Any], log_path: Path) -> Dict[str, Any]:
    name = str(action["action"])
    if not action.get("can_execute") or action.get("risk_level") != "low":
        return {"action": name, "returncode": 2, "status": "refused"}
    handler = action.get("handler")
    if handler == "refresh_blackboard":
        append_log(log_path, {"event": "start", "name": name, "time": datetime.now().isoformat(timespec="seconds")})
        write_blackboard(config, build_blackboard(config))
        append_log(log_path, {"event": "finish", "name": name, "returncode": 0, "time": datetime.now().isoformat(timespec="seconds")})
        return {"action": name, "returncode": 0, "status": "completed"}
    if handler == "review_incremental_learning":
        append_log(log_path, {"event": "start", "name": name, "time": datetime.now().isoformat(timespec="seconds")})
        summary = parse_incremental(config)
        code = 0 if summary.get("complete") else 1
        append_log(log_path, {"event": "finish", "name": name, "returncode": code, "summary": summary, "time": datetime.now().isoformat(timespec="seconds")})
        return {"action": name, "returncode": code, "status": "completed" if code == 0 else "failed"}
    argv = [str(value) for value in action.get("argv", [])]
    if not argv:
        return {"action": name, "returncode": 2, "status": "missing_command"}
    timeout = action.get("timeout_seconds") or config.get("automation", {}).get("default_timeout_seconds", 300)
    code = run_command(argv, ROOT, log_path, name, int(timeout))
    return {"action": name, "returncode": code, "status": "completed" if code == 0 else "failed"}


def cmd_serve(args: argparse.Namespace) -> int:
    app_path = ROOT / "fair_agent" / "ui" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        return subprocess.call(command, cwd=str(ROOT))
    except FileNotFoundError:
        print("Streamlit is not installed. Run: pip install -r requirements-agent.txt")
        return 1


def cmd_model_recheck(args: argparse.Namespace) -> int:
    try:
        from fair_agent.modules.model_recheck import run_model_recheck
        reuse = resolve_path(args.reuse_predictions) if args.reuse_predictions else None
        output = run_model_recheck(resolve_path(args.recheck_config), reuse)
    except ImportError as exc:
        print(f"Model recheck requires the remote inference environment: {exc}")
        return 1
    print(rel_path(output))
    return 0


def cmd_freeze_candidate(args: argparse.Namespace) -> int:
    import yaml
    from fair_agent.modules.model_recheck import freeze_candidate

    recheck_config = resolve_path(args.recheck_config)
    config = yaml.safe_load(recheck_config.read_text(encoding="utf-8"))
    report_dir = resolve_path(args.report)
    metrics = json.loads((report_dir / "stability_metrics.json").read_text(encoding="utf-8"))
    selected = int(metrics.get("selected_imgsz") or 0)
    if metrics.get("status") != "passed" or selected != int(args.imgsz):
        print("Refusing freeze: report is not passed or selected_imgsz does not match --imgsz")
        return 1
    freeze_candidate(config, selected, report_dir)
    print(f"frozen_imgsz={selected}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IR/SAR fast-learning agent workbench")
    parser.add_argument("--config", default="configs/agent_pipeline.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor").set_defaults(func=cmd_doctor)
    sub.add_parser("refresh").set_defaults(func=cmd_refresh)

    decide = sub.add_parser("decide")
    decide.add_argument("--sensor", choices=["ir", "sar"])
    decide.add_argument("--scene", choices=["all", "air", "forest", "sea", "urban"])
    decide.add_argument("--class-focus", choices=["soldier", "small_aircraft", "warship", "tank"])
    decide.add_argument("--refresh", action="store_true")
    decide.add_argument("--cached", action="store_true", help="Reuse persisted blackboard state instead of rebuilding it.")
    decide.set_defaults(func=cmd_decide)

    pipeline = sub.add_parser("pipeline")
    pipeline.add_argument("--mode", choices=["dryrun", "execute"], default="dryrun")
    pipeline.add_argument("--sensor", choices=["ir", "sar"], default="sar")
    pipeline.add_argument("--scene", choices=["all", "air", "forest", "sea", "urban"], default="all")
    pipeline.add_argument("--class-focus", choices=["soldier", "small_aircraft", "warship", "tank"], default="soldier")
    pipeline.set_defaults(func=cmd_pipeline)

    recheck = sub.add_parser("model-recheck")
    recheck.add_argument("--config", dest="recheck_config", default="configs/inference_size_stability.yaml")
    recheck.add_argument("--reuse-predictions", help="Reuse predictions_640.json and predictions_768.json from a prior failed validation run.")
    recheck.set_defaults(func=cmd_model_recheck)

    freeze = sub.add_parser("freeze-candidate")
    freeze.add_argument("--config", dest="recheck_config", default="configs/inference_size_stability.yaml")
    freeze.add_argument("--report", required=True)
    freeze.add_argument("--imgsz", type=int, choices=[640, 768], required=True)
    freeze.set_defaults(func=cmd_freeze_candidate)

    sub.add_parser("serve").set_defaults(func=cmd_serve)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
