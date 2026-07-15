from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from fair_agent.core.audit import make_run_dir, write_pipeline_artifacts
from fair_agent.core.blackboard import build_blackboard, write_blackboard
from fair_agent.core.config import ROOT, configured_python, get_key, load_config, rel_path, resolve_path
from fair_agent.core.hashes import sha256_file
from fair_agent.core.runtime_log import StructuredEventLog, event_log_from_config
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
TENSORRT_MODULES = ["tensorrt"]
ALL_MODULES = REQUIRED_MODULES + WORKBENCH_MODULES + INFERENCE_MODULES + TENSORRT_MODULES


def load_args_config(args: argparse.Namespace) -> Dict[str, Any]:
    overrides = list(getattr(args, "config_overrides", []) or [])
    return load_config(args.config, overrides) if overrides else load_config(args.config)


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
        "        'cuda_capabilities': ['.'.join(map(str, torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())],",
        "    }",
        "tensorrt_version = None",
        "if module_status.get('tensorrt'):",
        "    import tensorrt",
        "    tensorrt_version = tensorrt.__version__",
        "print(json.dumps({'executable': sys.executable, 'version': sys.version.split()[0], 'modules': module_status, 'accelerator': accelerator, 'tensorrt_version': tensorrt_version}, ensure_ascii=False))",
    ])
    proc = subprocess.run([str(path), "-c", code], text=True, capture_output=True, timeout=60)
    result["returncode"] = proc.returncode
    if proc.returncode == 0 and proc.stdout.strip():
        result.update(json.loads(proc.stdout.strip()))
    else:
        result["stderr"] = proc.stderr[-1000:]
    return result


def cmd_doctor(args: argparse.Namespace) -> int:
    config = load_args_config(args)
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
    backend_name = str(config["inference"]["backend"])
    native_assets = {
        key: resolve_path(config["native_backend"][key]).is_file()
        for key in ("library", "base_engine", "context_engine")
    }
    tensorrt_cfg = config["tensorrt_backend"]
    tensorrt_assets = {}
    for source, entry in tensorrt_cfg["engines"].items():
        path = resolve_path(entry["path"])
        actual = sha256_file(path) if path.is_file() else None
        tensorrt_assets[source] = {
            "path": rel_path(path),
            "exists": path.is_file(),
            "sha256_valid": actual == str(entry["sha256"]),
        }
    context_entry = tensorrt_cfg["context_engine"]
    context_path = resolve_path(context_entry["path"])
    context_actual = sha256_file(context_path) if context_path.is_file() else None
    tensorrt_assets["scene_sensor_net"] = {
        "path": rel_path(context_path),
        "exists": context_path.is_file(),
        "sha256_valid": context_actual == str(context_entry["sha256"]),
    }
    capabilities = list(accelerator.get("cuda_capabilities") or [])
    capability = capabilities[int(default_device)] if int(default_device) < len(capabilities) else None
    tensorrt_ready = (
        bool(external.get("modules", {}).get("tensorrt"))
        and str(external.get("tensorrt_version")) == str(tensorrt_cfg["expected_version"])
        and (
            not tensorrt_cfg["require_exact_gpu"]
            or capability == str(tensorrt_cfg["expected_compute_capability"])
        )
        and tensorrt_cfg["validated"] is True
        and all(item["exists"] and item["sha256_valid"] for item in tensorrt_assets.values())
    )
    backend_ready = (
        backend_name == "ultralytics_cuda"
        or (backend_name == "tensorrt_engine" and tensorrt_ready)
        or (
            backend_name == "tensorrt_native"
            and all(native_assets.values())
            and config["native_backend"]["validated"] is True
        )
    )
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
        "inference_backend": {
            "name": backend_name,
            "ready": backend_ready,
            "native_assets": native_assets if backend_name == "tensorrt_native" else None,
            "tensorrt_assets": tensorrt_assets if backend_name == "tensorrt_engine" else None,
            "cpu_fallback": False,
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
    required_inference_modules = INFERENCE_MODULES + (
        TENSORRT_MODULES if backend_name in {"tensorrt_engine", "tensorrt_native"} else []
    )
    missing_inference = [name for name in required_inference_modules if not modules.get(name)]
    if missing_required:
        print("缺少必需模块：", ", ".join(missing_required))
    if missing_workbench:
        print("缺少工作台模块：", ", ".join(missing_workbench))
        print(f"安装命令：{py} -m pip install -e \".[workbench]\"")
    if missing_inference:
        print("缺少推理模块：", ", ".join(missing_inference))
    if not gpu_ready:
        print("默认设备为 NVIDIA GPU，但当前环境无法使用 CUDA。请安装 CUDA 版 PyTorch 并检查显卡驱动。")
    if not backend_ready:
        print("推理后端依赖、资产、GPU或验收状态不完整；已拒绝启动，且不会回退到CPU。")
    inference_ok = bool(result["inference_weights"].get("matches_expected")) and bool(result["inference_weights"].get("same_frozen_path"))
    core_artifacts_ok = all(bool(value) for value in artifacts.values())
    checksums_ok = bool(result["frozen_checksums"].get("valid"))
    functional_ok = bool(result["functional_models"].get("valid")) and bool(result["functional_models"].get("all_x86_gpu_ready"))
    return 1 if missing_required or missing_workbench or missing_inference or not gpu_ready or not backend_ready or not inference_ok or not core_artifacts_ok or not checksums_ok or not functional_ok or external.get("returncode") != 0 else 0


def cmd_refresh(args: argparse.Namespace) -> int:
    config = load_args_config(args)
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
    config = load_args_config(args)
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
    config = load_args_config(args)
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
    config = load_args_config(args)
    state_path = resolve_path(config.get("blackboard", {}).get("output_dir", "reports/agent_blackboard")) / config.get("blackboard", {}).get("state_json", "blackboard_state.json")
    if state_path.exists() and args.cached and not args.refresh:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = build_blackboard(config)
        write_blackboard(config, state)
    context = decision_context(config, args)
    decision = build_decision(config, state, context)
    paths = write_decision(config, decision)
    event_log_from_config(config).append(
        "agent.decision.completed",
        component="policy",
        details={
            "context": context,
            "recommended_action": decision.get("recommended_action"),
            "candidate_count": len(decision.get("candidates", [])),
            "decision_json": rel_path(paths["json"]),
            "decision_json_sha256": sha256_file(paths["json"]),
        },
    )
    print(rel_path(paths["json"]))
    print(rel_path(paths["report"]))
    print(decision["recommended_action"]["action"])
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    config = load_args_config(args)
    state = build_blackboard(config)
    context = decision_context(config, args)
    decision = build_decision(config, state, context)
    automation = config.get("automation", {})
    run_dir = make_run_dir("dryrun" if args.mode == "dryrun" else "execute", automation.get("run_root", "reports/agent_runs"))
    pipeline_log = event_log_from_config(config)
    pipeline_trace_id = f"pipeline_{run_dir.name}"
    pipeline_log.append(
        "agent.pipeline.started", component="pipeline", trace_id=pipeline_trace_id,
        run_id=run_dir.name, details={"mode": args.mode, "context": context},
    )
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
    pipeline_log.append(
        "agent.pipeline.completed",
        level="error" if any(item.get("returncode") != 0 for item in action_results) else "info",
        component="pipeline",
        trace_id=pipeline_trace_id,
        run_id=run_dir.name,
        details={
            "mode": args.mode,
            "termination": plan.get("termination"),
            "steps": plan.get("steps", []),
            "action_results": action_results,
            "recommended_action": decision.get("recommended_action"),
            "manifest": rel_path(paths["manifest"]),
            "manifest_sha256": sha256_file(paths["manifest"]),
        },
    )
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
    timeout = action.get("timeout_seconds") or config["automation"]["default_timeout_seconds"]
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
    config = load_args_config(args)
    runtime = config["runtime"]
    host = str(runtime["server_host"])
    port = int(runtime["server_port"])
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
        environment = dict(os.environ)
        environment["AGILE_AGENT_CONFIG"] = str(resolve_path(args.config))
        environment["AGILE_AGENT_OVERRIDES"] = json.dumps(list(getattr(args, "config_overrides", []) or []))
        return subprocess.call(command, cwd=str(ROOT), env=environment)
    except KeyboardInterrupt:
        print("\n工作台已停止。")
        return 0
    except FileNotFoundError:
        print("未找到 Uvicorn。请先运行 scripts/bootstrap_x86.sh 完成环境配置。")
        return 1


def cmd_context_predict(args: argparse.Namespace) -> int:
    config = load_args_config(args)
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
        config = load_args_config(args)
        inference = config["inference"]
        confidence = float(
            inference["confidence_default"] if args.confidence is None else args.confidence
        )
        if not float(inference["confidence_min"]) <= confidence <= float(inference["confidence_max"]):
            raise ValueError(
                f"置信度必须位于{float(inference['confidence_min']):.2f}到"
                f"{float(inference['confidence_max']):.2f}之间。"
            )
        data = source.read_bytes()
        image, task_id = validate_image_bytes(data, source.name, config["limits"])
        settings = build_web_settings(config)
        if args.profile:
            profile = load_experiment_profile(args.profile)
            class_names = {int(key): str(value) for key, value in profile["class_names"].items()}
            base_mapping = {int(key): int(value) for key, value in profile["base_local_to_global"].items()}
            settings.update({
                "detector_path": resolve_path(profile["base_weight"]),
                "class_names": class_names,
                "base_class_ids": list(base_mapping.values()),
                "base_local_to_global": base_mapping,
                "generation_id": f"experiment-{profile['profile_id']}",
                "base_model_id": f"{profile['profile_id']}_base",
                "class_owners": {
                    **{global_id: f"{profile['profile_id']}_base" for global_id in base_mapping.values()},
                    int(profile["new_global_id"]): profile["profile_id"],
                },
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
                        "routing_prior": float(config["routing"]["default_routing_prior"]),
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
            generation_id=settings["generation_id"],
            base_model_id=settings["base_model_id"],
            class_owners=settings["class_owners"],
            backend_name=settings["backend"],
            native_options=settings["native_backend"],
        )
        result = engine.predict(image, source.name, confidence, task_id, "auto")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"自动检测失败：{exc}")
        return 1
    public = {key: value for key, value in result.items() if key not in {"annotated_png", "task_id"}}
    print(json.dumps(public, ensure_ascii=False, indent=2))
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    from fair_agent.modules.incremental_experiment import (
        reproduce_experiment,
        run_experiment,
        validate_experiment,
    )

    try:
        if args.experiment_action == "validate":
            result = validate_experiment(args.experiment_config)
        elif args.experiment_action == "run":
            result = run_experiment(args.experiment_config, run_id=args.run_id)
        else:
            result = reproduce_experiment(args.manifest, run_id=args.run_id)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"增量实验失败：{exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("valid", result.get("returncode", 0) == 0) else 1


def cmd_config(args: argparse.Namespace) -> int:
    from fair_agent.modules.configuration import (
        config_diff,
        migrate_config,
        render_effective_config,
        set_persistent_value,
        unset_persistent_value,
    )

    overrides = list(getattr(args, "config_overrides", []) or [])
    try:
        if args.config_action == "validate":
            config = load_config(args.config, overrides)
            print(json.dumps({
                "valid": True,
                "config": config["_config_path"],
                "sha256": config["_config_sha256"],
                "overrides": overrides,
                "restart_required": bool(overrides),
            }, ensure_ascii=False, indent=2))
        elif args.config_action == "show":
            print(render_effective_config(args.config, overrides, args.format))
        elif args.config_action == "get":
            value = get_key(load_config(args.config, overrides), args.key)
            print(yaml.safe_dump(value, allow_unicode=True, sort_keys=False).rstrip())
        elif args.config_action == "set":
            path = set_persistent_value(args.config, args.key, args.value)
            print(json.dumps({"updated": rel_path(path), "key": args.key, "restart_required": True}, ensure_ascii=False))
        elif args.config_action == "unset":
            path = unset_persistent_value(args.config, args.key)
            print(json.dumps({"updated": rel_path(path), "key": args.key, "restart_required": True}, ensure_ascii=False))
        elif args.config_action == "diff":
            print(json.dumps(config_diff(args.config, overrides), ensure_ascii=False, indent=2))
        else:
            path = migrate_config(args.input, args.output)
            print(json.dumps({"migrated": rel_path(path), "restart_required": True}, ensure_ascii=False))
    except (FileExistsError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"配置操作失败：{exc}")
        return 1
    return 0


def cmd_generation(args: argparse.Namespace) -> int:
    from fair_agent.modules.generation_management import (
        promote_generation,
        recheck_generation,
        rollback_generation,
    )

    config = load_args_config(args)
    try:
        if args.generation_action == "recheck":
            result = recheck_generation(config, args.candidate)
        elif args.generation_action == "promote":
            result = promote_generation(config, args.candidate, args.manifest)
        else:
            result = rollback_generation(config, args.to)
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"代际操作失败：{exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_benchmark_api(args: argparse.Namespace) -> int:
    from httpx import HTTPError

    from fair_agent.modules.api_benchmark import benchmark_api

    try:
        result = benchmark_api(load_args_config(args))
    except (HTTPError, OSError, RuntimeError, ValueError) as exc:
        print(f"API性能验收失败：{exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["accepted"] else 2


def _incremental_services(config: Dict[str, Any]):
    from fair_agent.modules.incremental_workbench import IncrementalBatchStore, TrainingJobManager
    from fair_agent.modules.model_generations import generation_web_settings, load_generation_registry

    log_config = config["logging"]
    event_log = StructuredEventLog(
        log_config["root"], int(log_config["max_file_bytes"]), int(log_config["retained_files"])
    )
    generation = generation_web_settings(
        load_generation_registry(config["web"]["generation_registry"]),
        str(config["web"]["generation_channel"]),
    )
    active_classes = {
        class_id: generation["class_names"][class_id] for class_id in generation["active_class_ids"]
    }
    store = IncrementalBatchStore(config["incremental_workbench"], event_log, active_classes)
    return store, TrainingJobManager(store, config["incremental_workbench"], event_log), event_log


def cmd_incremental_data(args: argparse.Namespace) -> int:
    config = load_args_config(args)
    store, manager, _event_log = _incremental_services(config)
    try:
        if args.incremental_action == "list":
            payload = store.list()
        elif args.incremental_action == "upload":
            archive = resolve_path(args.archive)
            payload = store.create(archive.name, archive.read_bytes(), args.name, args.class_names)
        elif args.incremental_action == "show":
            payload = store.get(args.batch_id)
        elif args.incremental_action == "inject":
            payload = store.inject(args.batch_id)
        elif args.incremental_action == "rename":
            values = [value.strip() for value in args.class_names.split(",") if value.strip()]
            current = store.get(args.batch_id, include_files=False)
            source_ids = [
                int(item["source_class_id"])
                for item in current.get("audit", {}).get("class_bindings", [])
            ]
            if len(values) != len(source_ids):
                raise ValueError("类别名称数量必须与当前批次类别数量一致。")
            payload = store.rename_classes(args.batch_id, dict(zip(source_ids, values)))
        elif args.incremental_action == "train":
            payload = manager.start(args.batch_id)
        elif args.incremental_action == "jobs":
            payload = manager.list(args.batch_id)
        elif args.incremental_action == "logs":
            print(manager.read_log(args.batch_id, args.job_id, args.tail))
            return 0
        else:
            raise ValueError(f"未知增量数据动作：{args.incremental_action}")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if isinstance(payload, dict) and payload.get("status") == "REJECTED" else 0
    except (OSError, KeyError, ValueError) as exc:
        print(f"增量数据操作失败：{exc}", file=sys.stderr)
        return 2


def cmd_logs(args: argparse.Namespace) -> int:
    config = load_args_config(args)
    _store, _manager, event_log = _incremental_services(config)
    rows = event_log.query(
        limit=args.limit, level=args.level, component=args.component,
        trace_id=args.trace_id, batch_id=args.batch_id, job_id=args.job_id,
        experiment_id=args.experiment_id, run_id=args.run_id,
        protocol_id=args.protocol_id, generation_id=args.generation_id,
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IR/SAR 快速学习智能体工作台")
    parser.add_argument("--config", default="configs/agent_pipeline.yaml")
    parser.add_argument("--set", dest="config_overrides", action="append", default=[], metavar="KEY=VALUE", help="仅覆盖当前命令的配置，可重复使用。")
    sub = parser.add_subparsers(dest="command", required=True)

    config_cmd = sub.add_parser("config", help="校验、查看或修改Agent配置。")
    config_sub = config_cmd.add_subparsers(dest="config_action", required=True)
    for name in ("validate", "show", "get", "set", "unset", "diff"):
        action = config_sub.add_parser(name)
        action.add_argument("--config", default=argparse.SUPPRESS)
        action.set_defaults(func=cmd_config)
        if name == "show":
            action.add_argument("--effective", action="store_true", help="显示环境变量与CLI覆盖后的有效配置。")
            action.add_argument("--format", choices=["yaml", "json"], default="yaml")
        elif name in {"get", "unset"}:
            action.add_argument("key")
        elif name == "set":
            action.add_argument("key")
            action.add_argument("value")
    migrate = config_sub.add_parser("migrate")
    migrate.add_argument("--input", required=True)
    migrate.add_argument("--output", required=True)
    migrate.set_defaults(func=cmd_config)

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
    detect.add_argument("--confidence", type=float)
    detect.add_argument("--profile", choices=["incremental-detection"], help="使用已注册并通过验收的增量检测档。")
    detect.set_defaults(func=cmd_detect)

    experiment = sub.add_parser("experiment", help="验证、执行或复现可审计的增量学习实验。")
    experiment_sub = experiment.add_subparsers(dest="experiment_action", required=True)
    experiment_validate = experiment_sub.add_parser("validate", help="只读校验配置、数据划分和访问边界。")
    experiment_validate.add_argument("--config", dest="experiment_config", required=True)
    experiment_validate.set_defaults(func=cmd_experiment)
    experiment_run = experiment_sub.add_parser("run", help="创建不可变run并执行训练适配器。")
    experiment_run.add_argument("--config", dest="experiment_config", required=True)
    experiment_run.add_argument("--run-id")
    experiment_run.set_defaults(func=cmd_experiment)
    experiment_reproduce = experiment_sub.add_parser("reproduce", help="按父manifest和相同数据指纹复现实验。")
    experiment_reproduce.add_argument("--manifest", required=True)
    experiment_reproduce.add_argument("--run-id")
    experiment_reproduce.set_defaults(func=cmd_experiment)

    generation = sub.add_parser("generation", help="复核、上线或回滚模型代际。")
    generation_sub = generation.add_subparsers(dest="generation_action", required=True)
    generation_recheck = generation_sub.add_parser("recheck", help="在lock-val上执行一次不可调参的部署复核。")
    generation_recheck.add_argument("--candidate", required=True)
    generation_recheck.set_defaults(func=cmd_generation)
    generation_promote = generation_sub.add_parser("promote", help="依据通过门禁的manifest原子切换production。")
    generation_promote.add_argument("--candidate", required=True)
    generation_promote.add_argument("--manifest", required=True)
    generation_promote.set_defaults(func=cmd_generation)
    generation_rollback = generation_sub.add_parser("rollback", help="回滚到已验证的历史代际。")
    generation_rollback.add_argument("--to", required=True)
    generation_rollback.set_defaults(func=cmd_generation)

    incremental_data = sub.add_parser("incremental-data", help="管理上传、审计、注入和训练增量数据批次。")
    incremental_sub = incremental_data.add_subparsers(dest="incremental_action", required=True)
    incremental_sub.add_parser("list", help="列出本机增量批次。").set_defaults(func=cmd_incremental_data)
    upload = incremental_sub.add_parser("upload", help="保存并审计本地ZIP数据包。")
    upload.add_argument("--archive", required=True)
    upload.add_argument("--name")
    upload.add_argument("--class-names", help="逗号分隔的本地类别名称；包内有data.yaml时可省略。")
    upload.set_defaults(func=cmd_incremental_data)
    for action_name in ("show", "inject", "train"):
        action = incremental_sub.add_parser(action_name)
        action.add_argument("--batch-id", required=True)
        action.set_defaults(func=cmd_incremental_data)
    rename = incremental_sub.add_parser("rename", help="按源类别ID顺序更新类别显示名称。")
    rename.add_argument("--batch-id", required=True)
    rename.add_argument("--class-names", required=True, help="逗号分隔，顺序与批次类别列表一致。")
    rename.set_defaults(func=cmd_incremental_data)
    jobs = incremental_sub.add_parser("jobs")
    jobs.add_argument("--batch-id")
    jobs.set_defaults(func=cmd_incremental_data)
    job_logs = incremental_sub.add_parser("logs")
    job_logs.add_argument("--batch-id", required=True)
    job_logs.add_argument("--job-id", required=True)
    job_logs.add_argument("--tail", type=int, default=300)
    job_logs.set_defaults(func=cmd_incremental_data)

    logs = sub.add_parser("logs", help="查询Agent结构化运行日志。")
    logs.add_argument("--limit", type=int, default=200)
    logs.add_argument("--level")
    logs.add_argument("--component")
    logs.add_argument("--trace-id")
    logs.add_argument("--batch-id")
    logs.add_argument("--job-id")
    logs.add_argument("--experiment-id")
    logs.add_argument("--run-id")
    logs.add_argument("--protocol-id")
    logs.add_argument("--generation-id")
    logs.set_defaults(func=cmd_logs)

    sub.add_parser("benchmark-api", help="按YAML配置执行检测API端到端性能验收。").set_defaults(func=cmd_benchmark_api)

    sub.add_parser("serve").set_defaults(func=cmd_serve)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        returncode = int(args.func(args))
    except Exception as exc:
        returncode = 1
        try:
            config = load_args_config(args)
            log_config = config["logging"]
            StructuredEventLog(log_config["root"], int(log_config["max_file_bytes"]), int(log_config["retained_files"])).append(
                "cli.command.failed", level="error", component="cli", message=str(exc),
                duration_ms=(time.perf_counter() - started) * 1000,
                details={"command": args.command},
            )
        except Exception:
            pass
        raise
    try:
        config = load_args_config(args)
        log_config = config["logging"]
        StructuredEventLog(log_config["root"], int(log_config["max_file_bytes"]), int(log_config["retained_files"])).append(
            "cli.command.completed", level="info" if returncode == 0 else "warning", component="cli",
            duration_ms=(time.perf_counter() - started) * 1000,
            details={"command": args.command, "returncode": returncode},
        )
    except Exception:
        pass
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
