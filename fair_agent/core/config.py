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
    default_device = str(config.get("runtime", {}).get("default_device", ""))
    if not default_device.isdigit() or int(default_device) < 0:
        errors.append("runtime.default_device 必须是非负 GPU 编号")
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
