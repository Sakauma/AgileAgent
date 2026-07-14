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
from fair_agent.modules.incremental_review import write_incremental_review
from fair_agent.modules.functional_models import validate_functional_models
from fair_agent.modules.operator_view import build_operator_snapshot, render_snapshot
from fair_agent.modules.status import parse_incremental
from fair_agent.policies.decision import build_decision, write_decision
from fair_agent.ui.console import run_console_frontend


REQUIRED_MODULES = ["yaml", "PIL"]
WORKBENCH_MODULES = ["pandas", "starlette", "uvicorn", "multipart"]
INFERENCE_MODULES = ["ultralytics", "cv2", "torch", "torchvision"]
ALL_MODULES = REQUIRED_MODULES + WORKBENCH_MODULES + INFERENCE_MODULES


def check_module(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def check_external_python(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"path": str(path), "exists": path.exists(), "modules": {}}
    if not path.exists():
        return result
    code = "\n".join([
        "import importlib.util as u, json, sys",
        f"mods = {ALL_MODULES!r}",
        "module_status = {m: bool(u.find_spec(m)) for m in mods}",
        "accelerator = {'cuda_available': False, 'cuda_device_count': 0, 'cuda_devices': []}",
        "if module_status.get('torch'):",
        "    import torch",
        "    accelerator = {",
        "        'torch_version': torch.__version__,",
        "        'torch_cuda_version': torch.version.cuda,",
        "        'cuda_available': bool(torch.cuda.is_available()),",
        "        'cuda_device_count': int(torch.cuda.device_count()),",
        "        'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],",
        "    }",
        "print(json.dumps({'executable': sys.executable, 'version': sys.version.split()[0], 'modules': module_status, 'accelerator': accelerator}, ensure_ascii=False))",
    ])
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
    runtime_cfg = config.get("runtime", {})
    default_device = str(runtime_cfg.get("default_device", "0"))
    accelerator = external.get("accelerator", {})
    gpu_ready = (
        bool(accelerator.get("cuda_available"))
        and int(default_device) < int(accelerator.get("cuda_device_count") or 0)
    )
    state = build_blackboard(config)
    artifacts = state.get("frozen_assets", {}).get("artifacts", {})
    result = {
        "runtime": external,
        "device": {
            "default": default_device,
            "gpu_required": True,
            "ready": gpu_ready,
        },
        "server": {
            "host": runtime_cfg.get("server_host"),
            "port": runtime_cfg.get("server_port"),
        },
        "dependency_groups": {
            "required": REQUIRED_MODULES,
            "workbench": WORKBENCH_MODULES,
            "inference": INFERENCE_MODULES,
        },
        "weights": state.get("frozen_assets", {}).get("weights"),
        "inference_weights": state.get("frozen_assets", {}).get("inference_weights"),
        "frozen_checksums": state.get("frozen_assets", {}).get("checksums"),
        "core_artifacts": artifacts,
        "functional_models": state.get("functional_models"),
        "current_blockers": state.get("current_blockers"),
    }
    quiet = bool(getattr(args, "quiet", False))
    if not quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    modules = external.get("modules", {})
    missing_required = [name for name in REQUIRED_MODULES if not modules.get(name)]
    missing_workbench = [name for name in WORKBENCH_MODULES if not modules.get(name)]
    missing_inference = [name for name in INFERENCE_MODULES if not modules.get(name)]
    if missing_required:
        print("缺少必需模块：", ", ".join(missing_required))
    if missing_workbench:
        print("缺少工作台模块：", ", ".join(missing_workbench))
        print(f"安装命令：{py} -m pip install -e \".[workbench]\"")
    if missing_inference:
        print("缺少推理模块：", ", ".join(missing_inference))
    if not gpu_ready:
        print("默认设备为 NVIDIA GPU，但当前环境无法使用 CUDA。请安装 CUDA 版 PyTorch 并检查显卡驱动。")
    inference_ok = bool(result["inference_weights"].get("matches_expected")) and bool(result["inference_weights"].get("same_frozen_path"))
    core_artifacts_ok = all(bool(value) for value in artifacts.values())
    checksums_ok = bool(result["frozen_checksums"].get("valid"))
    functional_ok = bool(result["functional_models"].get("valid")) and bool(result["functional_models"].get("all_x86_gpu_ready"))
    return 1 if missing_required or missing_workbench or missing_inference or not gpu_ready or not inference_ok or not core_artifacts_ok or not checksums_ok or not functional_ok or external.get("returncode") != 0 else 0


def cmd_refresh(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    state = build_blackboard(config)
    paths = write_blackboard(config, state)
    print(rel_path(paths["state"]))
    print(rel_path(paths["report"]))
    return 0


def load_or_build_state(config: Dict[str, Any], refresh: bool = False) -> Dict[str, Any]:
    output_dir = config.get("blackboard", {}).get("output_dir", "reports/agent_blackboard")
    state_name = config.get("blackboard", {}).get("state_json", "blackboard_state.json")
    state_path = resolve_path(output_dir) / state_name
    if state_path.exists() and not refresh:
        return json.loads(state_path.read_text(encoding="utf-8"))
    state = build_blackboard(config)
    write_blackboard(config, state)
    return state


def cmd_status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    state = load_or_build_state(config, refresh=args.refresh)
    decision_path = resolve_path(
        config.get("decision", {}).get("outputs", {}).get(
            "decision_json", "reports/agent_blackboard/agent_decision.json"
        )
    )
    if args.refresh or not decision_path.exists():
        context = dict(config.get("decision", {}).get("default_context", {}))
        decision = build_decision(config, state, context)
        write_decision(config, decision)
    else:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    snapshot = build_operator_snapshot(state, decision)
    print(render_snapshot(snapshot, args.format))
    return 0


def cmd_console(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    state = load_or_build_state(config, refresh=True)
    context = decision_context(config, args)
    decision = build_decision(config, state, context)
    write_decision(config, decision)
    if args.once or not sys.stdin.isatty():
        print(render_snapshot(build_operator_snapshot(state, decision), "text"))
        return 0
    return run_console_frontend(args.config)


def infer_image_context(config: Dict[str, Any], source: str | Path) -> Dict[str, Any]:
    from PIL import Image

    from fair_agent.models.context import load_context_model, predict_context

    functional_cfg = config.get("functional_models", {})
    registry = validate_functional_models(functional_cfg.get("registry", "configs/functional_models.yaml"))
    if not registry.get("valid"):
        raise RuntimeError(f"功能模型注册表无效：{registry.get('errors')}")
    context_entry = next(item for item in registry["models"] if item["function"] == "context_perception")
    weights = resolve_path(context_entry["artifacts"][0]["path"])
    device_index = str(config.get("runtime", {}).get("default_device", "0"))
    device = f"cuda:{device_index}"
    model, checkpoint = load_context_model(weights, device)
    source_path = resolve_path(source)
    with Image.open(source_path) as image:
        prediction = predict_context(model, checkpoint, image, device)
    return {"source": rel_path(source_path), "model_id": checkpoint["model_id"], **prediction}


def decision_context(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    defaults = config.get("decision", {}).get("default_context", {})
    inferred = infer_image_context(config, args.source) if getattr(args, "source", None) else {}
    return {
        "sensor": args.sensor or inferred.get("sensor") or defaults.get("sensor", "sar"),
        "scene": args.scene or inferred.get("scene") or defaults.get("scene", "all"),
        "class_focus": args.class_focus or defaults.get("class_focus", "soldier"),
        "context_model": inferred or None,
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
    context = decision_context(config, args)
    decision = build_decision(config, state, context)
    automation = config.get("automation", {})
    run_dir = make_run_dir("dryrun" if args.mode == "dryrun" else "execute", automation.get("run_root", "reports/agent_runs"))
    plan = {
        "mode": args.mode,
        "audit_before_execute": bool(config.get("automation", {}).get("audit_before_execute", True)),
        "context": context,
        "steps": [],
        "termination": None,
    }
    paths = write_pipeline_artifacts(run_dir, plan, state, decision)
    action_results: List[Dict[str, Any]] = []
    if args.mode == "dryrun":
        recommended = decision["recommended_action"]
        plan["steps"].append({
            "name": recommended["action"],
            "execute": False,
            "reason": recommended["reason"],
        })
        plan["termination"] = "dryrun"
    else:
        max_steps = int(automation.get("max_steps_per_run", 8))
        executed_names = set()
        while len(action_results) < max_steps:
            recommended = decision["recommended_action"]
            name = str(recommended["action"])
            if not recommended.get("can_execute"):
                plan["termination"] = "no_executable_action"
                break
            if name in executed_names:
                plan["termination"] = f"repeated_action:{name}"
                break
            plan["steps"].append({
                "name": name,
                "execute": True,
                "reason": recommended["reason"],
            })
            write_pipeline_artifacts(run_dir, plan, state, decision, action_results)
            result = execute_low_risk_action(config, recommended, paths["log"])
            action_results.append(result)
            executed_names.add(name)
            if result.get("returncode") != 0:
                plan["termination"] = f"action_failed:{name}"
                break
            state = build_blackboard(config)
            decision = build_decision(config, state, context)
        else:
            plan["termination"] = "max_steps_reached"
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
    path_error = validate_action_outputs(config, action)
    if path_error:
        append_log(log_path, {"event": "refused", "name": name, "reason": path_error, "time": datetime.now().isoformat(timespec="seconds")})
        return {"action": name, "returncode": 2, "status": "refused", "reason": path_error}
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
        output = None
        if code == 0:
            output = write_incremental_review(config, summary)
        append_log(log_path, {"event": "finish", "name": name, "returncode": code, "summary": summary, "time": datetime.now().isoformat(timespec="seconds")})
        return {"action": name, "returncode": code, "status": "completed" if code == 0 else "failed", "output": rel_path(output) if output else None}
    argv = [str(value) for value in action.get("argv", [])]
    if not argv:
        return {"action": name, "returncode": 2, "status": "missing_command"}
    timeout = action.get("timeout_seconds") or config.get("automation", {}).get("default_timeout_seconds", 300)
    code = run_command(argv, ROOT, log_path, name, int(timeout))
    return {"action": name, "returncode": code, "status": "completed" if code == 0 else "failed"}


def validate_action_outputs(config: Dict[str, Any], action: Dict[str, Any]) -> Optional[str]:
    roots = [resolve_path(path).resolve() for path in config.get("automation", {}).get("allowed_output_roots", [])]
    if not roots:
        return "allowed_output_roots_missing"
    for value in action.get("outputs", []):
        target = resolve_path(value).resolve()
        if not any(target == root or root in target.parents for root in roots):
            return f"output_outside_allowlist:{rel_path(target)}"
    return None


def cmd_serve(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    runtime = config.get("runtime", {})
    host = str(runtime.get("server_host", "127.0.0.1"))
    port = int(runtime.get("server_port", 8501))
    missing = [name for name in ("starlette", "uvicorn", "multipart") if not check_module(name)]
    if missing:
        print(f"缺少 Web 服务模块：{', '.join(missing)}。请先运行 scripts/bootstrap_x86.sh 完成环境配置。")
        return 1
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "fair_agent.web.app:app",
        "--host",
        host,
        "--port",
        str(port),
        "--no-access-log",
    ]
    try:
        return subprocess.call(command, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\n工作台已停止。")
        return 0
    except FileNotFoundError:
        print("未找到 Uvicorn。请先运行 scripts/bootstrap_x86.sh 完成环境配置。")
        return 1


def cmd_context_predict(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    try:
        result = infer_image_context(config, args.source)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"上下文认知失败：{exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    from fair_agent.modules.web_inference import WebInferenceEngine, validate_image_bytes
    from fair_agent.modules.strict_incremental import load_experiment_profile
    from fair_agent.web.app import build_web_settings

    source = resolve_path(args.source)
    try:
        if not 0.01 <= float(args.confidence) <= 1.0:
            raise ValueError("置信度必须位于0.01到1.00之间。")
        data = source.read_bytes()
        image, task_id = validate_image_bytes(data, source.name)
        settings = build_web_settings()
        if args.profile:
            profile = load_experiment_profile(args.profile)
            class_names = {int(key): str(value) for key, value in profile["class_names"].items()}
            base_mapping = {int(key): int(value) for key, value in profile["base_local_to_global"].items()}
            settings.update({
                "detector_path": resolve_path(profile["base_weight"]),
                "class_names": class_names,
                "base_class_ids": list(base_mapping.values()),
                "base_local_to_global": base_mapping,
                "protocols": {
                    profile["profile_id"]: {
                        "id": profile["profile_id"],
                        "class_name": profile["new_class"],
                        "new_class": profile["new_class"],
                        "global_class_id": int(profile["new_global_id"]),
                        "incremental_mode": "class_incremental",
                        "weights": resolve_path(profile["specialist_weight"]),
                        "new_map50": float(profile["new_map50"]),
                        "krr": float(profile["krr"]),
                        "available": True,
                        "activation_threshold": float(profile["activation_threshold"]),
                        "calibration_source": profile["calibration_source"],
                        "routing_prior": 0.5,
                        "context_prior": {},
                    }
                },
            })
        engine = WebInferenceEngine(
            settings["detector_path"],
            settings["context_path"],
            device_index=settings["device_index"],
            predict_options=settings["predict"],
            incremental_protocols=settings["protocols"],
            class_names=settings["class_names"],
            base_class_ids=settings["base_class_ids"],
            base_local_to_global=settings.get("base_local_to_global"),
            routing_options=settings["routing"],
        )
        result = engine.predict(image, source.name, float(args.confidence), task_id, "auto")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"自动检测失败：{exc}")
        return 1
    public = {key: value for key, value in result.items() if key not in {"annotated_png", "task_id"}}
    print(json.dumps(public, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IR/SAR 快速学习智能体工作台")
    parser.add_argument("--config", default="configs/agent_pipeline.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--quiet", action="store_true", help="成功时不输出完整诊断信息。")
    doctor.set_defaults(func=cmd_doctor)
    sub.add_parser("refresh").set_defaults(func=cmd_refresh)

    status = sub.add_parser("status", help="输出面向运维或外部程序的统一状态摘要。")
    status.add_argument("--format", choices=["text", "json"], default="text")
    status.add_argument("--refresh", action="store_true", help="读取证据并重建黑板。")
    status.set_defaults(func=cmd_status)

    console = sub.add_parser("console", help="在无浏览器环境运行终端工作台。")
    console.add_argument("--sensor", choices=["ir", "sar"])
    console.add_argument("--scene", choices=["all", "air", "forest", "sea", "urban"])
    console.add_argument("--class-focus", choices=["soldier", "small_aircraft", "warship", "tank"])
    console.add_argument("--source", help="使用 Scene-SensorNet 从图像推断传感器和场景。")
    console.add_argument("--once", action="store_true", help="只打印一次终端总览，不进入交互界面。")
    console.set_defaults(func=cmd_console)

    decide = sub.add_parser("decide")
    decide.add_argument("--sensor", choices=["ir", "sar"])
    decide.add_argument("--scene", choices=["all", "air", "forest", "sea", "urban"])
    decide.add_argument("--class-focus", choices=["soldier", "small_aircraft", "warship", "tank"])
    decide.add_argument("--source", help="使用 Scene-SensorNet 从图像自动推断传感器和场景。")
    decide.add_argument("--refresh", action="store_true")
    decide.add_argument("--cached", action="store_true", help="复用已保存的黑板状态，不重新采集证据。")
    decide.set_defaults(func=cmd_decide)

    pipeline = sub.add_parser("pipeline")
    pipeline.add_argument("--mode", choices=["dryrun", "execute"], default="dryrun")
    pipeline.add_argument("--sensor", choices=["ir", "sar"])
    pipeline.add_argument("--scene", choices=["all", "air", "forest", "sea", "urban"])
    pipeline.add_argument("--class-focus", choices=["soldier", "small_aircraft", "warship", "tank"], default="soldier")
    pipeline.add_argument("--source", help="使用 Scene-SensorNet 从图像自动推断传感器和场景。")
    pipeline.set_defaults(func=cmd_pipeline)

    context_predict = sub.add_parser("context-predict")
    context_predict.add_argument("--source", required=True)
    context_predict.set_defaults(func=cmd_context_predict)

    detect = sub.add_parser("detect", help="执行完整自动路由检测并输出详细决策轨迹。")
    detect.add_argument("--source", required=True)
    detect.add_argument("--confidence", type=float, default=0.50)
    detect.add_argument("--profile", choices=["strict-p01", "strict-p02"], help="使用已通过验收的严格 3+1 实验档。")
    detect.set_defaults(func=cmd_detect)

    sub.add_parser("serve").set_defaults(func=cmd_serve)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
