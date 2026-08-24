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
from fair_agent.core.config import (
    AUTO_CONFIG,
    ROOT,
    configured_python,
    get_key,
    load_config,
    rel_path,
    resolve_path,
    runtime_platform_info,
)
from fair_agent.core.hashes import sha256_file
from fair_agent.core.runtime_log import StructuredEventLog, event_log_from_config
from fair_agent.executors.local import append_log, run_command
from fair_agent.modules.functional_models import validate_functional_models
from fair_agent.modules.operator_view import build_operator_snapshot, render_snapshot
from fair_agent.policies.decision import build_decision, write_decision
from fair_agent.ui.console import run_console_frontend


REQUIRED_MODULES = ["yaml", "PIL"]
WORKBENCH_MODULES = ["pandas", "starlette", "uvicorn", "multipart"]
INFERENCE_MODULES = ["ultralytics", "cv2", "torch", "torchvision"]
ASCEND_INFERENCE_MODULES = ["numpy", "cv2", "acl"]
TENSORRT_MODULES = ["tensorrt"]
ALL_MODULES = list(dict.fromkeys(
    REQUIRED_MODULES
    + WORKBENCH_MODULES
    + INFERENCE_MODULES
    + ASCEND_INFERENCE_MODULES
    + TENSORRT_MODULES
))


def workbench_modules_for_backend(backend_name: str) -> list[str]:
    if backend_name == "ascend_acl":
        return [name for name in WORKBENCH_MODULES if name != "pandas"]
    return list(WORKBENCH_MODULES)


def load_args_config(args: argparse.Namespace) -> Dict[str, Any]:
    overrides = list(getattr(args, "config_overrides", []) or [])
    config_path = getattr(args, "config", AUTO_CONFIG)
    return load_config(config_path, overrides) if overrides else load_config(config_path)


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
    runtime_platform = runtime_platform_info(config)
    py = configured_python(config)
    external = check_external_python(py)
    modules = external.get("modules", {})
    runtime_cfg = config.get("runtime", {})
    default_device = str(runtime_cfg.get("default_device", "0"))
    accelerator = external.get("accelerator", {})
    cuda_ready = (
        bool(accelerator.get("cuda_available"))
        and int(default_device) < int(accelerator.get("cuda_device_count") or 0)
    )
    state = build_blackboard(config)
    backend_name = str(config["inference"]["backend"])
    required_workbench_modules = workbench_modules_for_backend(backend_name)
    required_inference_modules = (
        ASCEND_INFERENCE_MODULES
        if backend_name == "ascend_acl"
        else INFERENCE_MODULES
        + (
            TENSORRT_MODULES
            if backend_name in {"tensorrt_engine", "tensorrt_native"}
            else []
        )
    )
    ascend_verification: Dict[str, Any] | None = None
    if backend_name == "ascend_acl":
        try:
            from fair_agent.modules.ascend_release import verify_ascend_artifacts

            ascend_verification = verify_ascend_artifacts(
                config["ascend_backend"], require_validation=True
            )
        except (OSError, RuntimeError, ValueError) as exc:
            ascend_verification = {"status": "failed", "errors": [str(exc)]}
    device_ready = (
        bool(modules.get("acl"))
        and bool(ascend_verification)
        and ascend_verification.get("status") == "passed"
        if backend_name == "ascend_acl"
        else cuda_ready
    )
    native_cfg = config["native_backend"]
    native_assets = {
        "library": resolve_path(native_cfg["library"]).is_file(),
        "context_engine": resolve_path(native_cfg["context_engine"]).is_file(),
    }
    for source, entry in native_cfg["engines"].items():
        path = resolve_path(entry["path"])
        native_assets[source] = bool(
            path.is_file() and entry.get("sha256") and sha256_file(path) == str(entry["sha256"])
        )
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
        or (
            backend_name == "ascend_acl"
            and config["ascend_backend"]["validated"] is True
            and device_ready
        )
    )
    artifacts = state.get("frozen_assets", {}).get("artifacts", {})
    result = {
        "runtime": external,
        "runtime_platform": runtime_platform,
        "device": {
            "default": default_device,
            "accelerator_required": True,
            "gpu_required": backend_name != "ascend_acl",
            "npu_required": backend_name == "ascend_acl",
            "ready": device_ready,
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
            "ascend_assets": ascend_verification if backend_name == "ascend_acl" else None,
            "cpu_fallback": False,
        },
        "dependency_groups": {
            "required": REQUIRED_MODULES,
            "workbench": required_workbench_modules,
            "inference": required_inference_modules,
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
    missing_required = [name for name in REQUIRED_MODULES if not modules.get(name)]
    missing_workbench = [
        name for name in required_workbench_modules if not modules.get(name)
    ]
    missing_inference = [name for name in required_inference_modules if not modules.get(name)]
    if missing_required:
        print("缺少必需模块：", ", ".join(missing_required))
    if missing_workbench:
        print("缺少工作台模块：", ", ".join(missing_workbench))
        print(f"安装命令：{py} -m pip install -e \".[workbench]\"")
    if missing_inference:
        print("缺少推理模块：", ", ".join(missing_inference))
    if not device_ready:
        if backend_name == "ascend_acl":
            print("已选择 ARM/Ascend 模型，但 PyACL、CANN 环境或正式 OM 资产不可用。")
        else:
            print("已选择 x86/CUDA 模型，但当前环境无法使用 CUDA。请检查 CUDA 版 PyTorch 与显卡驱动。")
    if not backend_ready:
        print("推理后端依赖、资产、GPU或验收状态不完整；已拒绝启动，且不会回退到CPU。")
    if backend_name == "ascend_acl":
        inference_ok = bool(ascend_verification) and ascend_verification.get("status") == "passed"
        core_artifacts_ok = inference_ok
        checksums_ok = inference_ok
        functional_ok = bool(result["functional_models"].get("valid")) and bool(
            result["functional_models"].get("all_ascend_310b_ready")
        )
    else:
        inference_ok = bool(result["inference_weights"].get("matches_expected")) and bool(result["inference_weights"].get("same_frozen_path"))
        core_artifacts_ok = all(bool(value) for value in artifacts.values())
        checksums_ok = bool(result["frozen_checksums"].get("valid"))
        functional_ok = bool(result["functional_models"].get("valid")) and bool(result["functional_models"].get("all_x86_gpu_ready"))
    return 1 if missing_required or missing_workbench or missing_inference or not device_ready or not backend_ready or not inference_ok or not core_artifacts_ok or not checksums_ok or not functional_ok or external.get("returncode") != 0 else 0


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
    interactive = not args.once and sys.stdin.isatty()
    state = load_or_build_state(
        config,
        refresh=interactive or bool(getattr(args, "refresh", False)),
    )
    context = decision_context(config, args)
    decision = build_decision(config, state, context)
    write_decision(config, decision)
    if not interactive:
        print(render_snapshot(build_operator_snapshot(state, decision), "text"))
        return 0
    return run_console_frontend(config["_config_path"])


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
        "sensor": getattr(args, "sensor", None) or inferred.get("sensor") or defaults.get("sensor", "sar"),
        "scene": getattr(args, "scene", None) or inferred.get("scene") or defaults.get("scene", "all"),
        "class_focus": getattr(args, "class_focus", None) or defaults.get("class_focus", "soldier"),
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
        requested_config = str(getattr(args, "config", AUTO_CONFIG) or AUTO_CONFIG)
        if requested_config.strip().lower() != AUTO_CONFIG:
            environment["AGILE_AGENT_CONFIG"] = str(
                resolve_path(config["_config_path"])
            )
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


def build_detection_engine(config: Dict[str, Any]) -> Any:
    from fair_agent.modules.web_inference import WebInferenceEngine
    from fair_agent.web.app import build_web_settings

    settings = build_web_settings(config)
    return WebInferenceEngine(
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
        unified_class_gates=settings.get("unified_class_gates"),
    )


def cmd_detect(args: argparse.Namespace) -> int:
    from fair_agent.modules.cli_detection import (
        DEFAULT_ANNOTATION_FORMAT,
        DEFAULT_ANNOTATION_WORKERS,
        DEFAULT_BATCH_SIZE,
        DetectionArtifactWriter,
        LocalDetectionApiClient,
        create_result_dir,
        discover_detection_inputs,
        probe_local_detection_api,
        render_detection_summary,
        write_detection_reports,
    )
    from fair_agent.modules.web_inference import decode_batch_images, decode_image_bytes

    engine = None
    artifact_writer = None
    operation_started = time.perf_counter()
    try:
        config = load_args_config(args)
        inference = config["inference"]
        confidence = float(inference["confidence_default"])
        if not float(inference["confidence_min"]) <= confidence <= float(inference["confidence_max"]):
            raise ValueError(
                "冻结的默认置信度超出有效范围："
                f"{float(inference['confidence_min']):.2f}到"
                f"{float(inference['confidence_max']):.2f}。"
            )
        source, inputs = discover_detection_inputs(
            args.source,
            recursive=bool(args.recursive),
        )
        total_inputs = len(inputs)
        progress_interval = max(1, (total_inputs + 9) // 10)

        def report_batch_progress(completed: int) -> None:
            if total_inputs <= 1:
                return
            if completed != total_inputs and completed % progress_interval:
                return
            percent = completed * 100.0 / total_inputs
            print(
                f"批量识别进度：{completed}/{total_inputs} · {percent:.0f}%",
                file=sys.stderr,
            )

        input_read_started = time.perf_counter()
        source_bytes = [item.path.read_bytes() for item in inputs]
        input_read_ms = (time.perf_counter() - input_read_started) * 1000
        run_dir = create_result_dir(args.output, source)
        artifact_writer = DetectionArtifactWriter(
            run_dir,
            annotation_format=DEFAULT_ANNOTATION_FORMAT,
            workers=min(
                DEFAULT_ANNOTATION_WORKERS,
                max(1, int(os.cpu_count() or 1)),
            ),
        )
        api_base_url = probe_local_detection_api(config)
        results: list[Dict[str, Any]] = []
        request_count = 0
        service_processing_ms = 0.0
        client_request_wall_ms = 0.0
        performance_options = config.get("performance", {})
        configured_batch_size = max(
            1,
            int(performance_options.get("batch_probe_size") or DEFAULT_BATCH_SIZE),
        )
        batch_size = min(total_inputs, configured_batch_size)
        detection_started = time.perf_counter()

        def store_results(start: int, chunk_results: List[Dict[str, Any]]) -> None:
            if len(chunk_results) + start > total_inputs:
                raise RuntimeError("识别结果数量超过输入数量")
            for offset, result in enumerate(chunk_results):
                index = start + offset
                results.append(result)
                artifact_writer.submit(
                    index + 1,
                    inputs[index],
                    source_bytes[index],
                    result,
                )

        if api_base_url is not None:
            transport = "local_api"
            request_timeout = max(
                1.0,
                float(config.get("performance", {}).get("request_timeout_seconds") or 180),
            )
            with LocalDetectionApiClient(api_base_url, request_timeout) as client:
                if total_inputs == 1:
                    result, wall_ms = client.detect(source_bytes[0], inputs[0].name)
                    request_count = 1
                    client_request_wall_ms = wall_ms
                    service_processing_ms = float(
                        result.get("timings", {}).get("engine_total_ms")
                        or result.get("system_total_ms")
                        or result.get("inference_ms")
                        or 0.0
                    )
                    store_results(0, [result])
                else:
                    for start in range(0, total_inputs, batch_size):
                        stop = min(total_inputs, start + batch_size)
                        rows = [
                            (inputs[index].name, source_bytes[index])
                            for index in range(start, stop)
                        ]
                        batch_payload, wall_ms = client.detect_batch(rows)
                        chunk_results = list(batch_payload["results"])
                        store_results(start, chunk_results)
                        request_count += 1
                        client_request_wall_ms += wall_ms
                        service_processing_ms += float(
                            batch_payload.get("timings", {}).get(
                                "batch_engine_ms"
                            )
                            or batch_payload.get("system_total_ms")
                            or batch_payload.get("inference_ms")
                            or 0.0
                        )
                        for completed in range(start + 1, stop + 1):
                            report_batch_progress(completed)
        else:
            transport = "direct_engine"
            print("正在准备识别引擎，请稍候……", file=sys.stderr)
            engine = build_detection_engine(config)
            accepts_encoded = getattr(engine, "accepts_encoded", None)
            predict_encoded_batch = getattr(engine, "predict_encoded_batch", None)
            predict_batch = getattr(engine, "predict_batch", None)
            for start in range(0, total_inputs, batch_size):
                stop = min(total_inputs, start + batch_size)
                rows = [
                    (inputs[index].name, source_bytes[index])
                    for index in range(start, stop)
                ]
                if (
                    callable(accepts_encoded)
                    and callable(predict_encoded_batch)
                    and all(accepts_encoded(data) for _name, data in rows)
                ):
                    chunk_results = list(
                        predict_encoded_batch(
                            [(data, name) for name, data in rows],
                            confidence,
                            "auto",
                            True,
                        )
                    )
                elif callable(predict_batch) and len(rows) > 1:
                    decoded = decode_batch_images(
                        rows,
                        str(config["decoding"]["backend"]),
                        int(config["decoding"].get("workers") or 1),
                    )
                    chunk_results = list(
                        predict_batch(
                            [(image, name) for name, _data, image in decoded],
                            confidence,
                            "auto",
                        )
                    )
                else:
                    chunk_results = []
                    for name, data in rows:
                        if callable(accepts_encoded) and accepts_encoded(data):
                            result = engine.predict_encoded(
                                data,
                                name,
                                confidence,
                                "auto",
                                True,
                            )
                        else:
                            image = decode_image_bytes(
                                data,
                                name,
                                str(config["decoding"]["backend"]),
                            )
                            result = engine.predict(
                                image,
                                name,
                                confidence,
                                "auto",
                            )
                        chunk_results.append(result)
                if len(chunk_results) != len(rows):
                    raise RuntimeError("本进程批量识别结果数量与输入不一致")
                store_results(start, chunk_results)
                request_count += 1
                for completed in range(start + 1, stop + 1):
                    report_batch_progress(completed)
            service_processing_ms = sum(
                float(
                    result.get("timings", {}).get("engine_total_ms")
                    or result.get("system_total_ms")
                    or result.get("inference_ms")
                    or 0.0
                )
                for result in results
            )
        detection_finished = time.perf_counter()
        detection_wall_ms = (detection_finished - detection_started) * 1000
        model_inference_ms = sum(
            float(result.get("inference_ms") or 0.0) for result in results
        )
        target_fps = float(performance_options.get("target_api_fps") or 30.0)

        def throughput(duration_ms: float) -> float | None:
            return (
                total_inputs * 1000.0 / duration_ms
                if duration_ms > 0.0
                else None
            )

        end_to_end_inference_fps = throughput(service_processing_ms)
        performance = {
            "measurement_scope": "competition_end_to_end_inference",
            "timing_source": (
                "api_engine_total_ms"
                if transport == "local_api"
                else "engine_total_ms"
            ),
            "strategy": (
                "single_api"
                if total_inputs == 1 and transport == "local_api"
                else "chunked_batch_api"
                if transport == "local_api"
                else "direct_engine_batch"
            ),
            "target_fps": round(target_fps, 3),
            "batch_size": batch_size,
            "request_count": request_count,
            "input_read_ms": round(input_read_ms, 3),
            "model_inference_ms": round(model_inference_ms, 3),
            "end_to_end_inference_ms": round(service_processing_ms, 3),
            "end_to_end_inference_fps": (
                round(float(end_to_end_inference_fps), 3)
                if end_to_end_inference_fps is not None
                else None
            ),
            "service_processing_ms": round(service_processing_ms, 3),
            "service_processing_fps": (
                round(float(end_to_end_inference_fps), 3)
                if end_to_end_inference_fps is not None
                else None
            ),
            "client_request_wall_ms": round(client_request_wall_ms, 3),
            "detection_wall_ms": round(detection_wall_ms, 3),
            "detection_wall_fps": (
                round(float(throughput(detection_wall_ms)), 3)
                if throughput(detection_wall_ms) is not None
                else None
            ),
        }
        payload = artifact_writer.finalize(
            source,
            transport=transport,
            performance=performance,
        )
        artifact_writer = None
        completed_at = time.perf_counter()
        cli_total_wall_ms = (completed_at - operation_started) * 1000
        result_finalize_ms = (completed_at - detection_finished) * 1000
        cli_total_fps = throughput(cli_total_wall_ms)
        payload["performance"].update(
            {
                "result_finalize_ms": round(result_finalize_ms, 3),
                "cli_total_wall_ms": round(cli_total_wall_ms, 3),
                "cli_total_fps": (
                    round(float(cli_total_fps), 3)
                    if cli_total_fps is not None
                    else None
                ),
                "target_met": bool(
                    end_to_end_inference_fps is not None
                    and end_to_end_inference_fps >= target_fps
                ),
            }
        )
        write_detection_reports(payload)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"自动检测失败：{exc}")
        return 1
    finally:
        if artifact_writer is not None:
            artifact_writer.abort()
        if engine is not None:
            close = getattr(engine, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    print(f"警告：推理引擎释放资源失败：{exc}", file=sys.stderr)
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_detection_summary(payload))
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    from fair_agent.modules.configuration import (
        config_diff,
        render_effective_config,
        set_persistent_value,
        unset_persistent_value,
    )

    overrides = list(getattr(args, "config_overrides", []) or [])
    try:
        if args.config_action == "validate":
            config = load_config(args.config, overrides)
            runtime_platform = runtime_platform_info(config)
            print(json.dumps({
                "valid": True,
                "config": config["_config_path"],
                "sha256": config["_config_sha256"],
                "overrides": overrides,
                "restart_required": bool(overrides),
                "runtime": runtime_platform,
            }, ensure_ascii=False, indent=2))
        elif args.config_action == "show":
            print(render_effective_config(args.config, overrides, args.format))
        elif args.config_action == "get":
            value = get_key(load_config(args.config, overrides), args.key)
            rendered = yaml.safe_dump(
                value, allow_unicode=True, sort_keys=False
            ).rstrip()
            if rendered.endswith("\n..."):
                rendered = rendered[:-4].rstrip()
            print(rendered)
        elif args.config_action == "set":
            path = set_persistent_value(args.config, args.key, args.value)
            print(json.dumps({"updated": rel_path(path), "key": args.key, "restart_required": True}, ensure_ascii=False))
        elif args.config_action == "unset":
            path = unset_persistent_value(args.config, args.key)
            print(json.dumps({"updated": rel_path(path), "key": args.key, "restart_required": True}, ensure_ascii=False))
        elif args.config_action == "diff":
            print(json.dumps(config_diff(args.config, overrides), ensure_ascii=False, indent=2))
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


def cmd_tensorrt(args: argparse.Namespace) -> int:
    from fair_agent.modules.tensorrt_export import export_or_verify_engines, write_export_hashes
    from fair_agent.modules.tensorrt_validation import validate_tensorrt

    try:
        config = load_args_config(args)
        if args.tensorrt_action == "calibrate":
            if config["tensorrt_backend"]["precision"] != "int8":
                raise ValueError("INT8自动校准要求设备配置使用precision: int8。")
            export_result = export_or_verify_engines(config, verify_only=False)
            config_update = write_export_hashes(config["_config_path"], export_result)
            config = load_config(config["_config_path"])
            validation = validate_tensorrt(config, activate=bool(args.activate))
            result = {
                "calibration": export_result,
                "config_update": config_update,
                "validation": validation,
                "accepted": bool(validation["accepted"]),
            }
        else:
            result = validate_tensorrt(config, activate=bool(args.activate))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"TensorRT验收失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["accepted"] else 2


def _incremental_services(config: Dict[str, Any]):
    from fair_agent.modules.generation_management import active_generation_registry
    from fair_agent.modules.incremental_workbench import IncrementalBatchStore, TrainingJobManager
    from fair_agent.modules.model_generations import generation_web_settings, load_generation_registry

    log_config = config["logging"]
    event_log = StructuredEventLog(
        log_config["root"], int(log_config["max_file_bytes"]), int(log_config["retained_files"])
    )
    generation = generation_web_settings(
        load_generation_registry(active_generation_registry(config)),
        str(config["web"]["generation_channel"]),
    )
    active_classes = {
        class_id: generation["class_names"][class_id] for class_id in generation["active_class_ids"]
    }
    store = IncrementalBatchStore(
        config["incremental_workbench"], event_log, active_classes, generation["class_names"]
    )
    return store, TrainingJobManager(
        store, config["incremental_workbench"], event_log, config
    ), event_log


def cmd_incremental(args: argparse.Namespace) -> int:
    config = load_args_config(args)
    store, manager, _event_log = _incremental_services(config)
    try:
        if args.incremental_action in {"audit", "run"}:
            source = Path(args.batch)
            if source.is_file():
                archive = resolve_path(source)
                manifest = store.create(archive.name, archive.read_bytes(), getattr(args, "name", None), getattr(args, "class_names", None))
            else:
                manifest = store.get(args.batch)
            if args.incremental_action == "audit":
                payload = manifest
            else:
                if manifest["status"] == "AUDITED":
                    manifest = store.inject(manifest["batch_id"])
                if manifest["status"] not in {"INJECTED", "FAILED"}:
                    raise ValueError(f"批次状态不允许启动完整生命周期：{manifest['status']}")
                payload = manager.start(manifest["batch_id"], wait=True)
        else:
            matches = [row for row in manager.list() if str(row["job_id"]) == str(args.run_id)]
            if not matches:
                raise KeyError(args.run_id)
            payload = matches[0]
            payload["batch"] = store.get(str(payload["batch_id"]), include_files=False)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if isinstance(payload, dict) and payload.get("status") in {"REJECTED", "FAILED", "ROLLED_BACK", "ROLLBACK_FAILED"} else 0
    except (OSError, KeyError, ValueError) as exc:
        print(f"增量生命周期操作失败：{exc}", file=sys.stderr)
        return 2


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
    parser = argparse.ArgumentParser(
        description="灵动 Agent：IR/SAR 视觉识别主界面",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "常用示例：\n"
            "  agile-agent\n"
            "  agile-agent detect --source image.png\n"
            "  agile-agent detect --source images/ --recursive --output runs/case01\n"
            "  agile-agent status --format json"
        ),
    )
    parser.add_argument(
        "--config",
        default=AUTO_CONFIG,
        help="运行配置路径；默认 auto，x86 选择 CUDA/PT，ARM 选择 Ascend/OM。",
    )
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
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--quiet", action="store_true", help="成功时不输出完整诊断信息。")
    doctor.set_defaults(func=cmd_doctor)
    sub.add_parser("refresh").set_defaults(func=cmd_refresh)

    status = sub.add_parser("status", help="输出面向运维或外部程序的统一状态摘要。")
    status.add_argument("--format", choices=["text", "json"], default="text")
    status.add_argument("--refresh", action="store_true", help="读取证据并重建黑板。")
    status.set_defaults(func=cmd_status)

    console = sub.add_parser("console", help="进入视觉识别交互主界面。")
    console.add_argument("--once", action="store_true", help="只打印一次终端总览，不进入交互界面。")
    console.add_argument("--refresh", action="store_true", help="打印总览前重新采集运行状态。")
    console.set_defaults(func=cmd_console)

    decide = sub.add_parser("decide")
    decide.add_argument("--sensor", choices=["ir", "sar"])
    decide.add_argument("--scene", choices=["all", "air", "forest", "sea", "urban"])
    decide.add_argument(
        "--class-focus",
        choices=[
            "soldier",
            "small_aircraft",
            "warship",
            "tank",
            "patrol_boat",
            "armored_vehicle",
        ],
    )
    decide.add_argument("--source", help="使用 Scene-SensorNet 从图像自动推断传感器和场景。")
    decide.add_argument("--refresh", action="store_true")
    decide.add_argument("--cached", action="store_true", help="复用已保存的黑板状态，不重新采集证据。")
    decide.set_defaults(func=cmd_decide)

    pipeline = sub.add_parser("pipeline")
    pipeline.add_argument("--mode", choices=["dryrun", "execute"], default="dryrun")
    pipeline.add_argument("--sensor", choices=["ir", "sar"])
    pipeline.add_argument("--scene", choices=["all", "air", "forest", "sea", "urban"])
    pipeline.add_argument(
        "--class-focus",
        choices=[
            "soldier",
            "small_aircraft",
            "warship",
            "tank",
            "patrol_boat",
            "armored_vehicle",
        ],
        default="soldier",
    )
    pipeline.add_argument("--source", help="使用 Scene-SensorNet 从图像自动推断传感器和场景。")
    pipeline.set_defaults(func=cmd_pipeline)

    context_predict = sub.add_parser("context-predict")
    context_predict.add_argument("--source", required=True)
    context_predict.set_defaults(func=cmd_context_predict)

    detect = sub.add_parser("detect", help="识别单图或目录，并自动保存完整本地结果。")
    detect.add_argument("--source", required=True, help="图像文件或图像目录。")
    detect.add_argument("--recursive", action="store_true", help="目录输入时递归扫描子目录。")
    detect.add_argument("--output", help="本次结果目录；必须不存在或为空。")
    detect.add_argument("--format", choices=["text", "json"], default="text", help="终端输出格式。")
    detect.set_defaults(func=cmd_detect)

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

    incremental = sub.add_parser("incremental", help="运行可审计的增量学习完整生命周期。")
    incremental_lifecycle_sub = incremental.add_subparsers(dest="incremental_action", required=True)
    incremental_audit = incremental_lifecycle_sub.add_parser("audit", help="上传或读取批次并执行数据血缘审计。")
    incremental_audit.add_argument("--batch", required=True, help="增量ZIP路径或已有batch_id。")
    incremental_audit.add_argument("--name")
    incremental_audit.add_argument("--class-names")
    incremental_audit.set_defaults(func=cmd_incremental)
    incremental_run = incremental_lifecycle_sub.add_parser("run", help="从审计继续执行到自动门禁与受控上线。")
    incremental_run.add_argument("--batch", required=True, help="增量ZIP路径或已有batch_id。")
    incremental_run.add_argument("--name")
    incremental_run.add_argument("--class-names")
    incremental_run.set_defaults(func=cmd_incremental)
    incremental_status = incremental_lifecycle_sub.add_parser("status", help="按训练run_id查询完整生命周期状态。")
    incremental_status.add_argument("--run-id", required=True)
    incremental_status.set_defaults(func=cmd_incremental)

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

    tensorrt = sub.add_parser("tensorrt", help="执行TensorRT精度、性能和启用门禁。")
    tensorrt_sub = tensorrt.add_subparsers(dest="tensorrt_action", required=True)
    tensorrt_validate = tensorrt_sub.add_parser("validate", help="对CUDA与TensorRT执行同集精度对齐和API性能验收。")
    tensorrt_validate.add_argument("--activate", action="store_true", help="全部门禁通过后原子写回并启用当前设备配置。")
    tensorrt_validate.set_defaults(func=cmd_tensorrt)
    tensorrt_calibrate = tensorrt_sub.add_parser(
        "calibrate", help="自动选择代表性数据、执行INT8 PTQ并完成精度和性能门禁。"
    )
    tensorrt_calibrate.add_argument(
        "--activate", action="store_true", help="全部门禁通过后原子启用INT8设备配置。"
    )
    tensorrt_calibrate.set_defaults(func=cmd_tensorrt)

    sub.add_parser("serve").set_defaults(func=cmd_serve)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if not effective_argv:
        effective_argv = ["console"]
    args = parser.parse_args(effective_argv)
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
