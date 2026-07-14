from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from fair_agent.core.config import (
    DEFAULT_CONFIG,
    apply_overrides,
    get_key,
    is_protected_key,
    load_config,
    parse_yaml_value,
    redact_config,
    resolve_path,
    set_key,
    unset_key,
    validate_config,
    write_config,
)


def raw_config(path: str | Path) -> Dict[str, Any]:
    resolved = resolve_path(path)
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是映射：{resolved}")
    return value


def set_persistent_value(path: str | Path, key: str, value: str) -> Path:
    if is_protected_key(key):
        raise ValueError(f"受保护参数必须使用generation专用命令修改：{key}")
    data = raw_config(path)
    set_key(data, key, parse_yaml_value(value), create=False)
    return write_config(path, data, f"set:{key}")


def unset_persistent_value(path: str | Path, key: str) -> Path:
    if is_protected_key(key):
        raise ValueError(f"受保护参数必须使用generation专用命令修改：{key}")
    data = raw_config(path)
    unset_key(data, key)
    return write_config(path, data, f"unset:{key}")


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


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def migrate_config(input_path: str | Path, output_path: str | Path) -> Path:
    source = raw_config(input_path)
    template = raw_config(DEFAULT_CONFIG)
    source.pop("schema_version", None)
    migrated = _deep_merge(template, source)
    migrated["schema_version"] = template["schema_version"]
    validate_config(migrated)
    destination = resolve_path(output_path)
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已有迁移目标：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(migrated, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return destination


def render_effective_config(path: str | Path, overrides: list[str], output_format: str) -> str:
    config = redact_config(load_config(path, overrides))
    visible = {key: value for key, value in config.items() if not str(key).startswith("_")}
    if output_format == "json":
        return json.dumps(visible, ensure_ascii=False, indent=2)
    return yaml.safe_dump(visible, allow_unicode=True, sort_keys=False).rstrip()
