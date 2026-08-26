#!/usr/bin/env python3
"""Materialize a gate-passed edge Adapter in an isolated demo channel."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import yaml

from fair_agent.core.config import load_config

from .protocol import load_protocol


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid JSON report: {path}")
    return payload


def build_demo_manifest(
    *,
    run_id: str,
    protocol_id: str,
    export_report: Mapping[str, Any],
    evaluation_report: Mapping[str, Any],
    benchmark_report: Mapping[str, Any],
    om_path: Path,
) -> dict[str, Any]:
    """Validate all pre-deployment gates and build the runtime manifest."""

    if evaluation_report.get("competition_passed") is not True:
        raise ValueError("edge Adapter failed the mixed-lock accuracy gate")
    numerical = benchmark_report.get("numerical_equivalence") or {}
    projected = benchmark_report.get("projected_integrated_pipeline") or {}
    if numerical.get("passed") is not True:
        raise ValueError("edge Adapter OM failed numerical equivalence")
    if projected.get("fps_gate_30_passed") is not True:
        raise ValueError("edge Adapter candidate failed the 30 FPS gate")
    if export_report.get("protocol_id") != protocol_id:
        raise ValueError("edge Adapter export protocol does not match the registry")
    if export_report.get("feature_contract") != "candidate_confidence_context_v1":
        raise ValueError("edge Adapter export uses an incompatible feature contract")
    class_order = [int(value) for value in export_report.get("class_order") or ()]
    raw_weights = export_report.get("effective_weights") or {}
    if set(class_order) != {int(value) for value in raw_weights}:
        raise ValueError("edge Adapter export weights are incomplete")
    metrics = evaluation_report.get("edge_adapter") or {}
    return {
        "schema_version": 1,
        "kind": "ascend_edge_incremental_demo_adapter",
        "channel": "isolated_demo",
        "run_id": run_id,
        "protocol_id": protocol_id,
        "feature_contract": "candidate_confidence_context_v1",
        "class_order": class_order,
        "effective_weights": raw_weights,
        "adapter_om": str(om_path),
        "accuracy": {
            key: metrics.get(key)
            for key in ("base_map50", "new_map50", "krr", "full_map50")
        },
        "projected_fps": projected.get("projected_fps"),
        "accepted": True,
        "production_modified": False,
        "created_at_epoch": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将通过门禁的板端 Adapter 部署到隔离演示通道。"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--om", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    protocol = load_protocol(args.registry, repo_root)
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    export_report = _load_json(args.export_report)
    evaluation_report = _load_json(args.evaluation_report)
    benchmark_report = _load_json(args.benchmark_report)
    source_om = args.om.expanduser().resolve()
    if not source_om.is_file():
        raise FileNotFoundError(source_om)
    final_om = destination / "edge_adapter_bank.om"
    manifest = build_demo_manifest(
        run_id=str(args.run_id),
        protocol_id=protocol.protocol_id,
        export_report=export_report,
        evaluation_report=evaluation_report,
        benchmark_report=benchmark_report,
        om_path=final_om,
    )

    temporary.mkdir(parents=True)
    try:
        shutil.copy2(source_om, temporary / final_om.name)
        final_manifest = destination / "adapter_manifest.json"
        (temporary / final_manifest.name).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        base_config = yaml.safe_load(
            args.base_config.expanduser().resolve().read_text(encoding="utf-8")
        )
        if not isinstance(base_config, dict):
            raise ValueError("invalid Ascend base config")
        routing = base_config.setdefault("routing", {})
        routing["edge_incremental_adapter"] = {
            "enabled": True,
            "manifest": str(final_manifest),
            "require_accepted": True,
            "required_protocol_id": protocol.protocol_id,
        }
        final_config = destination / "agent_pipeline_ascend310b_demo.yaml"
        (temporary / final_config.name).write_text(
            yaml.safe_dump(base_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
        load_config(final_config)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    report = {
        "schema_version": 1,
        "status": "deployed_to_isolated_demo_channel",
        "run_id": args.run_id,
        "protocol_id": protocol.protocol_id,
        "manifest": str(destination / "adapter_manifest.json"),
        "config": str(destination / "agent_pipeline_ascend310b_demo.yaml"),
        "om": str(destination / "edge_adapter_bank.om"),
        "production_modified": False,
        "runtime_acceptance_pending": True,
    }
    report_path = destination / "deployment_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
