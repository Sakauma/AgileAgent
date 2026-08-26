#!/usr/bin/env python3
"""One-command, network-free 4->4+2 incremental-learning demonstration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from .demo_contract import materialize_demo_contract
from .workflow import build_parser as build_workflow_parser
from .workflow import build_stages, plan_payload, run as run_workflow


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _offline_environment(repo_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{existing}" if existing else str(repo_root)
    )
    environment.update(
        {
            "PIP_NO_INDEX": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "YOLO_CONFIG_DIR": str(repo_root / "runs/yolo_config_offline"),
        }
    )
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    ):
        environment.pop(key, None)
    return environment


def _command(python: Path, module: str, *arguments: object) -> list[str]:
    return [str(python), "-m", module, *(str(value) for value in arguments)]


def _revoke_manifest(path: Path, reason: str) -> None:
    if not path.is_file():
        return
    payload = _read_json(path)
    payload["accepted"] = False
    payload["runtime_acceptance"] = {"passed": False, "reason": reason}
    _write_json(path, payload)


def _workflow_args(
    args: argparse.Namespace,
    registry: Path,
    workflow_output: Path,
) -> argparse.Namespace:
    values = [
        "run",
        "--repo-root",
        str(args.repo_root),
        "--registry",
        str(registry),
        "--method-config",
        str(args.method_config),
        "--context-prior",
        str(args.context_prior),
        "--ascend-config",
        str(args.ascend_config),
        "--output-root",
        str(workflow_output),
        "--training-python",
        str(args.training_python),
        "--production-python",
        str(args.production_python),
        "--baseline-fps",
        str(args.baseline_fps),
        "--device-id",
        str(args.device_id),
        "--epochs",
        str(args.epochs),
        "--seeds",
        args.seeds,
        "--learning-rates",
        args.learning_rates,
        "--training-rows",
        str(args.training_rows),
        "--batch-size",
        str(args.batch_size),
        "--candidate-slots",
        str(args.candidate_slots),
    ]
    if args.encoded:
        values.append("--encoded")
    if args.include_all_diagnostics:
        values.append("--include-all-diagnostics")
    if args.opp_source is not None:
        values.extend(("--opp-source", str(args.opp_source)))
    return build_workflow_parser().parse_args(values)


def run_demo(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.expanduser().resolve()
    args.repo_root = repo_root
    session = args.output_root.expanduser().resolve()
    if session.exists():
        raise FileExistsError(session)
    run_id = session.name
    started = time.time()
    state_path = session / "demo_state.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "preparing",
        "offline": True,
        "production_modified": False,
        "started_at_epoch": started,
    }
    contract_root = repo_root / ".edge_incremental_demo_contracts" / run_id
    workflow_output = session / "workflow"
    deployment_root = session / "deployment"
    registry = contract_root / "incremental_round_registry_4plus2.yaml"
    manifest = deployment_root / "adapter_manifest.json"
    environment = _offline_environment(repo_root)
    os.environ.clear()
    os.environ.update(environment)
    try:
        audit = materialize_demo_contract(
            repo_root=repo_root,
            reference_registry=args.reference_registry,
            incremental_data=args.incremental_data,
            base_data=args.base_data,
            contract_root=contract_root,
        )
        state.update({"status": "contract_ready", "input_audit": audit})
        _write_json(state_path, state)
        workflow_args = _workflow_args(args, registry, workflow_output)
        stages = build_stages(workflow_args)
        plan = {
            "schema_version": 1,
            "kind": "offline_ascend310b_4_to_4plus2_demo",
            "run_id": run_id,
            "offline": True,
            "input_audit": audit,
            "edge_workflow": plan_payload(workflow_args, stages),
            "post_training_stages": [
                "deploy_isolated_demo_candidate",
                "benchmark_complete_runtime_pipeline",
                "accept_or_revoke_demo_candidate",
            ],
            "production_modified": False,
        }
        _write_json(session / "demo_plan.json", plan)
        if args.plan_only:
            state.update({"status": "planned", "finished_at_epoch": time.time()})
            _write_json(state_path, state)
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0

        state["status"] = "training_and_edge_acceptance"
        _write_json(state_path, state)
        run_workflow(workflow_args, stages)
        state["status"] = "deploying_isolated_demo_candidate"
        _write_json(state_path, state)
        subprocess.run(
            _command(
                args.production_python,
                "extras.ascend_edge_incremental.promote_demo",
                "--repo-root",
                repo_root,
                "--registry",
                registry,
                "--run-id",
                run_id,
                "--base-config",
                args.ascend_config,
                "--export-report",
                workflow_output / "evidence/onnx_export.json",
                "--evaluation-report",
                workflow_output / "evaluation/lock/evaluation_report.json",
                "--benchmark-report",
                workflow_output / "evaluation/adapter_om_benchmark.json",
                "--om",
                workflow_output / "export/edge_adapter_bank.om",
                "--output-dir",
                deployment_root,
            ),
            cwd=repo_root,
            env=environment,
            check=True,
        )
        state["status"] = "runtime_acceptance"
        _write_json(state_path, state)
        runtime_report_path = session / "evaluation/runtime_benchmark.json"
        subprocess.run(
            _command(
                args.production_python,
                "extras.ascend_edge_incremental.benchmark_demo_runtime",
                "--repo-root",
                repo_root,
                "--registry",
                registry,
                "--config",
                deployment_root / "agent_pipeline_ascend310b_demo.yaml",
                "--output",
                runtime_report_path,
                "--probe-size",
                args.runtime_probe_size,
                "--warmup-rounds",
                args.runtime_warmup_rounds,
                "--rounds",
                args.runtime_rounds,
                "--target-fps",
                args.target_fps,
            ),
            cwd=repo_root,
            env=environment,
            check=True,
        )
        runtime_report = _read_json(runtime_report_path)
        manifest_payload = _read_json(manifest)
        manifest_payload["runtime_acceptance"] = {
            "passed": True,
            "report": str(runtime_report_path),
            "median_fps": runtime_report["median_fps"],
        }
        _write_json(manifest, manifest_payload)
        evaluation = _read_json(
            workflow_output / "evaluation/lock/evaluation_report.json"
        )
        workflow_state = _read_json(workflow_output / "workflow_state.json")
        training_seconds = sum(
            float(row["wall_seconds"])
            for row in workflow_state.get("completed_stages") or ()
            if row.get("name") == "train_registered_rounds"
        )
        finished = time.time()
        report = {
            "schema_version": 1,
            "kind": "offline_ascend310b_4_to_4plus2_demo_result",
            "run_id": run_id,
            "status": "accepted",
            "offline": True,
            "training_device": "npu:0",
            "incremental_learning_scope": "incremental_train_dev_only",
            "base_weights_frozen": True,
            "old_expert_weights_frozen": True,
            "production_modified": False,
            "demo_config": str(
                deployment_root / "agent_pipeline_ascend310b_demo.yaml"
            ),
            "adapter_manifest": str(manifest),
            "metrics": evaluation["edge_adapter"],
            "runtime_median_fps": runtime_report["median_fps"],
            "training_wall_seconds": training_seconds,
            "total_wall_seconds": finished - started,
            "passed": True,
        }
        _write_json(session / "demo_report.json", report)
        state.update(
            {
                "status": "accepted",
                "finished_at_epoch": finished,
                "report": str(session / "demo_report.json"),
            }
        )
        _write_json(state_path, state)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        _revoke_manifest(manifest, f"{type(exc).__name__}: {exc}")
        state.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at_epoch": time.time(),
                "production_modified": False,
            }
        )
        _write_json(state_path, state)
        raise


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    run_id = time.strftime("edge-demo-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    parser = argparse.ArgumentParser(
        description="Ascend310B 断网环境一键演示当前 4->4+2 增量学习。"
    )
    parser.add_argument("--incremental-data", type=Path, required=True)
    parser.add_argument("--base-data", type=Path)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "runs/ascend_edge_incremental_demo" / run_id,
    )
    parser.add_argument(
        "--reference-registry",
        type=Path,
        default=repo_root / "configs/incremental_round_registry_4plus2.yaml",
    )
    parser.add_argument(
        "--method-config",
        type=Path,
        default=repo_root / "configs/ascend310b/full_score_method.yaml",
    )
    parser.add_argument(
        "--context-prior",
        type=Path,
        default=(
            repo_root
            / "models/production/incremental_detection/incremental_context_prior.json"
        ),
    )
    parser.add_argument(
        "--ascend-config",
        type=Path,
        default=repo_root / "configs/agent_pipeline_ascend310b.yaml",
    )
    parser.add_argument(
        "--training-python",
        type=Path,
        default=Path.home() / "agileagent/envs/agileagent_train/bin/python",
    )
    parser.add_argument(
        "--production-python",
        type=Path,
        default=Path("/usr/local/miniconda3/envs/agileagent/bin/python"),
    )
    parser.add_argument("--opp-source", type=Path)
    parser.add_argument("--baseline-fps", type=float, default=38.2175)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", default="20260825,11,26,4090,310")
    parser.add_argument("--learning-rates", default="0.01,0.05,0.1")
    parser.add_argument("--training-rows", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--candidate-slots", type=int, default=512)
    parser.add_argument("--runtime-probe-size", type=int, default=20)
    parser.add_argument("--runtime-warmup-rounds", type=int, default=1)
    parser.add_argument("--runtime-rounds", type=int, default=3)
    parser.add_argument("--decoded", dest="encoded", action="store_false")
    parser.add_argument("--include-all-diagnostics", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.set_defaults(encoded=True)
    return parser


def main() -> int:
    return run_demo(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
