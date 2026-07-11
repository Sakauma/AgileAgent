from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.hashes import hash_if_exists


def fingerprints(paths: Iterable[str]) -> Dict[str, Dict[str, object]]:
    return {path: hash_if_exists(resolve_path(path)) for path in paths}


def output_freshness(inputs: Iterable[str], outputs: Iterable[str]) -> Dict[str, Any]:
    input_paths = [resolve_path(path) for path in inputs]
    output_paths = [resolve_path(path) for path in outputs]
    missing_inputs = [rel_path(path) for path in input_paths if not path.exists()]
    missing_outputs = [rel_path(path) for path in output_paths if not path.exists()]
    if missing_inputs:
        return {"freshness": "missing", "reason": "missing_inputs", "missing": missing_inputs}
    if missing_outputs:
        return {"freshness": "missing", "reason": "missing_outputs", "missing": missing_outputs}
    newest_input = max((path.stat().st_mtime for path in input_paths), default=0.0)
    oldest_output = min((path.stat().st_mtime for path in output_paths), default=0.0)
    if oldest_output < newest_input:
        return {"freshness": "stale", "reason": "inputs_newer_than_outputs", "missing": []}
    return {"freshness": "current", "reason": "outputs_current", "missing": []}


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    target = resolve_path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_incremental(config: Dict[str, Any]) -> Dict[str, Any]:
    inputs = config.get("inputs", {})
    compliant_path = inputs.get("incremental_compliant_metrics")
    compliant_rows = read_csv_rows(compliant_path) if compliant_path else []
    compliance_required = bool(config.get("incremental", {}).get("require_compliant_no_old_data", False))
    paths = inputs.get("incremental_metrics", [])
    rows: List[Dict[str, str]] = []
    if compliant_rows:
        rows = compliant_rows
    elif not compliance_required:
        for path in paths:
            rows.extend(read_csv_rows(path))
    compliance_verified = bool(compliant_rows)
    min_new = float(config.get("thresholds", {}).get("min_new_class_map50", 0.60))
    min_krr = float(config.get("thresholds", {}).get("min_krr", 0.85))
    incremental_cfg = config.get("incremental", {})
    required_task_type = incremental_cfg.get("task_type", "incremental_object_detection")
    primary_mode = incremental_cfg.get("primary_mode", "class_incremental")
    supported_modes = set(incremental_cfg.get("supported_modes", ["class_incremental", "target_incremental"]))
    required_scope = incremental_cfg.get("learning_data_scope", "incremental_dataset_only")
    protocols = []
    warnings = []
    for row in rows:
        new_map = float(row.get("new_map50_after") or 0.0)
        krr = float(row.get("krr") or 0.0)
        old_raw = int(float(row.get("old_raw_image_count") or 0)) if compliance_verified else None
        scope_verified = str(row.get("learning_scope_verified", "")).lower() in {"true", "1", "yes"}
        task_type = row.get("task_type")
        incremental_mode = row.get("incremental_mode")
        learning_scope = row.get("learning_data_scope")
        row_compliant = bool(
            compliance_verified
            and old_raw == 0
            and scope_verified
            and task_type == required_task_type
            and incremental_mode in supported_modes
            and learning_scope == required_scope
        )
        passed = new_map >= min_new and krr >= min_krr and (row_compliant or not compliance_required)
        if passed and (new_map - min_new < 0.03 or krr - min_krr < 0.03):
            warnings.append(f"{row.get('protocol')} is close to an acceptance threshold")
        protocols.append({
            "protocol": row.get("protocol"),
            "new_class": row.get("new_classes"),
            "task_type": task_type,
            "incremental_mode": incremental_mode,
            "learning_data_scope": learning_scope,
            "learning_scope_verified": scope_verified,
            "new_map50": new_map,
            "krr": krr,
            "old_raw_image_count": old_raw,
            "compliant": row_compliant,
            "passed": passed,
        })
    expected = set(config.get("incremental", {}).get("expected_protocols", []))
    seen = {str(item.get("protocol")) for item in protocols}
    complete = bool(protocols) and (not expected or expected.issubset(seen))
    return {
        "protocols": protocols,
        "complete": complete,
        "passed": complete and all(item["passed"] for item in protocols),
        "source": compliant_path if compliance_verified else ("missing_compliant_metrics" if compliance_required else "legacy_metrics"),
        "compliance_required": compliance_required,
        "compliance_verified": compliance_verified,
        "warnings": warnings,
        "acceptance": {"min_new_class_map50": min_new, "min_krr": min_krr},
        "task_type": required_task_type,
        "primary_mode": primary_mode,
        "supported_modes": sorted(supported_modes),
        "learning_data_scope": required_scope,
    }


def parse_specialist(config: Dict[str, Any]) -> Dict[str, Any]:
    path = config.get("inputs", {}).get(
        "specialist_metrics", "reports/agent_sar_soldier_casebank/sar_soldier_replay_metrics.csv"
    )
    rows = read_csv_rows(path)
    keyed = {(row.get("phase"), row.get("eval_set")): row for row in rows}

    def value(phase: str, eval_set: str, field: str) -> float | None:
        row = keyed.get((phase, eval_set))
        return float(row[field]) if row and row.get(field) not in {None, ""} else None

    required = [
        ("before", "lock_all", "map50"), ("after", "lock_all", "map50"),
        ("before", "lock_ir", "soldier_map50"), ("after", "lock_ir", "soldier_map50"),
        ("before", "lock_sar", "soldier_map50"), ("after", "lock_sar", "soldier_map50"),
    ]
    if any(value(*item) is None for item in required):
        return {"status": "not_run", "metrics_path": path, "deltas": {}, "accepted": False}
    deltas = {
        "lock_all_map50": value("after", "lock_all", "map50") - value("before", "lock_all", "map50"),
        "lock_ir_soldier": value("after", "lock_ir", "soldier_map50") - value("before", "lock_ir", "soldier_map50"),
        "lock_sar_soldier": value("after", "lock_sar", "soldier_map50") - value("before", "lock_sar", "soldier_map50"),
    }
    policy = config.get("specialist_acceptance", {})
    accepted = (
        deltas["lock_all_map50"] >= float(policy.get("min_lock_all_delta", 0.0))
        and deltas["lock_ir_soldier"] >= float(policy.get("min_lock_ir_soldier_delta", -0.01))
        and deltas["lock_sar_soldier"] >= float(policy.get("min_lock_sar_soldier_delta", 0.01))
    )
    return {"status": "accepted" if accepted else "rejected", "metrics_path": path, "deltas": deltas, "accepted": accepted}
