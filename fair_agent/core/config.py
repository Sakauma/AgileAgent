from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "agent_pipeline.yaml"


def resolve_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_config(path: Union[str, Path] = DEFAULT_CONFIG) -> Dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Agent config must be a mapping: {config_path}")
    data["_config_path"] = rel_path(config_path)
    validate_config(data)
    return data


def validate_config(config: Dict[str, Any]) -> None:
    errors: List[str] = []
    runtime = config.get("runtime", {})
    default_device = str(runtime.get("default_device", ""))
    if not default_device.isdigit() or int(default_device) < 0:
        errors.append("runtime.default_device 必须是非负 GPU 编号")
    server_host = str(runtime.get("server_host", ""))
    if server_host not in {"127.0.0.1", "localhost", "::1"}:
        errors.append("runtime.server_host 必须是本机回环地址")
    try:
        server_port = int(runtime.get("server_port"))
    except (TypeError, ValueError):
        server_port = 0
    if not 1 <= server_port <= 65535:
        errors.append("runtime.server_port 必须在 1-65535 之间")
    model = config.get("model", {})
    if not model.get("weights"):
        errors.append("model.weights is required")
    expected = str(model.get("expected_sha256") or "")
    if len(expected) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in expected):
        errors.append("model.expected_sha256 must be a 64-character hexadecimal digest")
    assets = config.get("assets", {})
    if not assets.get("manifest"):
        errors.append("assets.manifest is required")
    if not assets.get("checksums"):
        errors.append("assets.checksums is required")
    if not isinstance(assets.get("required"), list) or not assets.get("required"):
        errors.append("assets.required must be a non-empty list")
    functional = config.get("functional_models", {})
    if not functional.get("registry"):
        errors.append("functional_models.registry is required")
    try:
        required_functional_count = int(functional.get("required_count"))
    except (TypeError, ValueError):
        required_functional_count = 0
    if required_functional_count < 3:
        errors.append("functional_models.required_count must be at least 3")
    automation = config.get("automation", {})
    if not isinstance(automation.get("allowed_output_roots"), list) or not automation.get("allowed_output_roots"):
        errors.append("automation.allowed_output_roots must be a non-empty list")
    try:
        max_steps = int(automation.get("max_steps_per_run"))
    except (TypeError, ValueError):
        max_steps = 0
    if not 1 <= max_steps <= 100:
        errors.append("automation.max_steps_per_run must be in 1-100")
    actions = config.get("decision", {}).get("actions", {})
    if not isinstance(actions, dict) or not actions:
        errors.append("decision.actions must be a non-empty mapping")
    else:
        for name, action in actions.items():
            if not isinstance(action, dict):
                errors.append(f"decision.actions.{name} must be a mapping")
                continue
            if action.get("risk_level") not in {"low", "medium", "high"}:
                errors.append(f"decision.actions.{name}.risk_level is invalid")
            if not action.get("handler") and not action.get("argv"):
                errors.append(f"decision.actions.{name} needs handler or argv")
    if errors:
        raise ValueError("Invalid agent config: " + "; ".join(errors))


def configured_python(config: Dict[str, Any]) -> Path:
    runtime = config.get("runtime", {})
    configured = runtime.get("local_python")
    return Path(configured).expanduser() if configured else Path(sys.executable)
