from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from fair_agent.core.blackboard import build_blackboard
from fair_agent.core.config import ROOT, load_config, rel_path, resolve_path
from fair_agent.core.hashes import hash_if_exists, verify_sha256s


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是映射：{path}")
    return data


def verify_release(config_path: str | Path = "configs/agent_pipeline.yaml") -> Dict[str, Any]:
    config = load_config(config_path)
    errors: List[str] = []
    assets = config["assets"]
    checksums = verify_sha256s(resolve_path(assets["checksums"]))
    if not checksums["valid"]:
        errors.extend(f"model_checksum:{item}" for item in checksums["errors"])

    required = {}
    for name in assets["required"]:
        status = hash_if_exists(resolve_path(name))
        required[name] = status
        if not status["exists"]:
            errors.append(f"missing_required_asset:{name}")

    model = config["model"]
    model_status = hash_if_exists(resolve_path(model["weights"]))
    if model_status.get("sha256") != model["expected_sha256"]:
        errors.append("base_model_sha256_mismatch")
    manifest_path = resolve_path(assets["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if manifest.get("base_model", {}).get("sha256") != model["expected_sha256"]:
        errors.append("manifest_base_hash_mismatch")
    incremental_models = manifest.get("incremental_models", [])
    if len(incremental_models) != 4:
        errors.append("manifest_incremental_model_count_invalid")

    inference_configs = {}
    for name in ["configs/local_infer_gpu.yaml", config["detector"]["config"]]:
        path = resolve_path(name)
        data = _load_yaml(path)
        predict = data.get("predict", {})
        details = {
            "model": data.get("model"),
            "device": str(predict.get("device")),
            "imgsz": predict.get("imgsz"),
            "batch": predict.get("batch"),
        }
        inference_configs[rel_path(path)] = details
        if data.get("model") != model["weights"]:
            errors.append(f"inference_model_mismatch:{name}")
        if data.get("expected_sha256") != model["expected_sha256"]:
            errors.append(f"inference_hash_mismatch:{name}")
        if not details["device"].isdigit():
            errors.append(f"inference_device_not_gpu:{name}")
        if int(details["imgsz"] or 0) != 640:
            errors.append(f"inference_imgsz_not_640:{name}")
        if int(details["batch"] or 0) != 32:
            errors.append(f"inference_batch_not_32:{name}")

    demo_path = resolve_path(config["blackboard"]["demo_evidence"])
    demo_text = demo_path.read_text(encoding="utf-8") if demo_path.exists() else ""
    demo = json.loads(demo_text) if demo_text else {}
    if not demo:
        errors.append("demo_evidence_missing")
    for forbidden in ["datasets_r1_base_train", ".png", ".jpg", "visualizations/"]:
        if forbidden in demo_text:
            errors.append(f"demo_contains_private_reference:{forbidden}")
    manifest_metrics = {item.get("protocol"): float(item.get("new_map50") or 0) for item in incremental_models}
    demo_metrics = {item.get("protocol"): float(item.get("new_map50") or 0) for item in demo.get("incremental_learning", {}).get("protocols", [])}
    if set(manifest_metrics) != set(demo_metrics) or any(
        abs(manifest_metrics[name] - demo_metrics[name]) > 1e-4 for name in manifest_metrics
    ):
        errors.append("demo_incremental_metrics_do_not_match_manifest")

    start_script = ROOT / "scripts" / "start_agent.sh"
    start_text = start_script.read_text(encoding="utf-8") if start_script.exists() else ""
    for forbidden in ["pip install", "-m venv", "torch==", "bootstrap_x86.sh\nexec"]:
        if forbidden in start_text:
            errors.append(f"start_script_mutates_environment:{forbidden}")
    bootstrap_script = ROOT / "scripts" / "bootstrap_x86.sh"
    bootstrap_text = bootstrap_script.read_text(encoding="utf-8") if bootstrap_script.exists() else ""
    for required_marker in ["python3.12 python3.11 python3.10", "uv venv --python 3.12 --seed", "nvidia-smi", "uname -m"]:
        if required_marker not in bootstrap_text:
            errors.append(f"bootstrap_gate_missing:{required_marker}")

    state = build_blackboard(config)
    if int(state.get("dataset", {}).get("image_count") or 0) != 750:
        errors.append("blackboard_dataset_evidence_missing")
    if len(state.get("incremental_learning", {}).get("protocols", [])) != 4:
        errors.append("blackboard_incremental_evidence_missing")
    if config["runtime"]["server_host"] not in {"127.0.0.1", "localhost", "::1"}:
        errors.append("server_not_loopback")

    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "config": rel_path(resolve_path(config_path)),
        "checksums": checksums,
        "required_assets": required,
        "inference_configs": inference_configs,
        "evidence_mode": state.get("evidence", {}).get("mode"),
        "blockers": state.get("current_blockers", []),
    }
