from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from fair_agent.core.config import (
    apply_overrides,
    get_key,
    is_protected_key,
    load_config,
    parse_yaml_value,
    redact_config,
    runtime_config_path,
    set_key,
    unset_key,
    write_config,
)


def raw_config(path: str | Path) -> Dict[str, Any]:
    resolved = runtime_config_path(path)
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是映射：{resolved}")
    return value


def set_persistent_value(path: str | Path, key: str, value: str) -> Path:
    if is_protected_key(key):
        raise ValueError(f"受保护参数必须使用generation专用命令修改：{key}")
    resolved = runtime_config_path(path)
    data = raw_config(resolved)
    set_key(data, key, parse_yaml_value(value), create=False)
    return write_config(resolved, data, f"set:{key}")


def unset_persistent_value(path: str | Path, key: str) -> Path:
    if is_protected_key(key):
        raise ValueError(f"受保护参数必须使用generation专用命令修改：{key}")
    resolved = runtime_config_path(path)
    data = raw_config(resolved)
    unset_key(data, key)
    return write_config(resolved, data, f"unset:{key}")


def flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).startswith("_"):
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.update(flatten(item, child))
    else:
        rows[prefix] = value
    return rows


def config_diff(path: str | Path, overrides: list[str]) -> Dict[str, Any]:
    before = flatten(raw_config(path))
    effective = flatten(load_config(path, overrides))
    keys = sorted(set(before) | set(effective))
    return {
        key: {"yaml": before.get(key), "effective": effective.get(key)}
        for key in keys
        if before.get(key) != effective.get(key)
    }


def render_effective_config(path: str | Path, overrides: list[str], output_format: str) -> str:
    config = redact_config(load_config(path, overrides))
    visible = {key: value for key, value in config.items() if not str(key).startswith("_")}
    if output_format == "json":
        return json.dumps(visible, ensure_ascii=False, indent=2)
    return yaml.safe_dump(visible, allow_unicode=True, sort_keys=False).rstrip()
