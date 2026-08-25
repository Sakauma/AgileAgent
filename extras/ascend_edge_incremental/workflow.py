#!/usr/bin/env python3
"""Plan or execute the isolated Ascend310B edge-training workflow."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .protocol import ensure_isolated_output, load_protocol


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    use_training_opp: bool = False

    def public(self) -> dict[str, object]:
        return {
            "name": self.name,
            "command": list(self.command),
            "shell": shlex.join(self.command),
            "uses_training_opp_overlay": self.use_training_opp,
        }


def module_command(python: Path, module: str, *arguments: object) -> tuple[str, ...]:
    return (str(python), "-m", module, *(str(value) for value in arguments))


def build_stages(args: argparse.Namespace) -> list[Stage]:
    repo_root = args.repo_root.expanduser().resolve()
    protocol = load_protocol(args.registry, repo_root)
    output = args.output_root.expanduser().resolve()
    training_python = args.training_python.expanduser().resolve()
    production_python = args.production_python.expanduser().resolve()
    registry = protocol.registry_path
    method = args.method_config.expanduser().resolve()
    context_prior = args.context_prior.expanduser().resolve()
    config = args.ascend_config.expanduser().resolve()
    module = "extras.ascend_edge_incremental"
    stages: list[Stage] = []
    if args.opp_source is not None:
        stages.append(
            Stage(
                "prepare_opp_overlay",
                module_command(
                    production_python,
                    f"{module}.prepare_opp_overlay",
                    "--source",
                    args.opp_source.expanduser().resolve(),
                    "--output",
                    output / "opp_overlay",
                    "--manifest",
                    output / "evidence/opp_overlay.json",
                ),
            )
        )
    stages.append(
        Stage(
            "npu_backward_probe",
            module_command(
                training_python,
                f"{module}.smoke_backward",
                "--output",
                output / "evidence/npu_backward.json",
                "--device-id",
                args.device_id,
            ),
            use_training_opp=args.opp_source is not None,
        )
    )
    probe_paths: dict[str, tuple[Path, Path, Path, int]] = {}
    scopes = ["training", "selection", "lock"]
    if args.include_all_diagnostics:
        scopes.append("all")
    for scope in scopes:
        view = output / f"probes/{scope}_images"
        manifest = output / f"evidence/{scope}_inputs.json"
        predictions = output / f"probes/{scope}_probe.jsonl"
        summary = output / f"evidence/{scope}_probe.json"
        count = len(protocol.image_paths(scope))
        probe_paths[scope] = (view, predictions, summary, count)
        stages.append(
            Stage(
                f"prepare_{scope}_inputs",
                module_command(
                    production_python,
                    f"{module}.prepare_inputs",
                    "--repo-root",
                    repo_root,
                    "--registry",
                    registry,
                    "--scope",
                    scope,
                    "--output",
                    view,
                    "--manifest",
                    manifest,
                ),
            )
        )
        freeze_arguments: list[object] = [
            "--config",
            config,
            "--image-root",
            view,
            "--output",
            predictions,
            "--summary",
            summary,
            "--expected-images",
            count,
        ]
        if args.encoded:
            freeze_arguments.append("--encoded")
        stages.append(
            Stage(
                f"freeze_{scope}_probe",
                module_command(
                    production_python,
                    f"{module}.freeze_probe",
                    *freeze_arguments,
                ),
            )
        )
        if scope == "training":
            stages.append(
                Stage(
                    "train_registered_rounds",
                    module_command(
                        training_python,
                        f"{module}.train",
                        "--repo-root",
                        repo_root,
                        "--registry",
                        registry,
                        "--method-config",
                        method,
                        "--context-prior",
                        context_prior,
                        "--probe",
                        predictions,
                        "--output-dir",
                        output / "training",
                        "--epochs",
                        args.epochs,
                        "--seeds",
                        args.seeds,
                        "--learning-rates",
                        args.learning_rates,
                        "--training-rows",
                        args.training_rows,
                        "--batch-size",
                        args.batch_size,
                        "--device-id",
                        args.device_id,
                    ),
                    use_training_opp=args.opp_source is not None,
                )
            )
        elif scope == "selection":
            stages.append(
                Stage(
                    "select_adapter_scales",
                    module_command(
                        training_python,
                        f"{module}.calibrate",
                        "--repo-root",
                        repo_root,
                        "--registry",
                        registry,
                        "--method-config",
                        method,
                        "--context-prior",
                        context_prior,
                        "--probe",
                        predictions,
                        "--checkpoint",
                        output / "training/combined_adapter_bank.pt",
                        "--output",
                        output / "calibration/adapter_scales.json",
                    ),
                )
            )
        elif scope in {"lock", "all"}:
            evaluation_arguments: list[object] = [
                "--repo-root",
                repo_root,
                "--registry",
                registry,
                "--method-config",
                method,
                "--context-prior",
                context_prior,
                "--scope",
                scope,
                "--probe",
                predictions,
                "--checkpoint",
                output / "training/combined_adapter_bank.pt",
                "--adapter-scales",
                output / "calibration/adapter_scales.json",
                "--output-dir",
                output / f"evaluation/{scope}",
            ]
            if scope == "lock":
                evaluation_arguments.append("--write-predictions")
                if args.expected_baseline_score is not None:
                    evaluation_arguments.extend(
                        (
                            "--expected-baseline-score",
                            args.expected_baseline_score.expanduser().resolve(),
                        )
                    )
            stages.append(
                Stage(
                    f"evaluate_{scope}",
                    module_command(
                        training_python,
                        f"{module}.evaluate",
                        *evaluation_arguments,
                    ),
                )
            )

    onnx_path = output / "export/edge_adapter_bank.onnx"
    om_prefix = output / "export/edge_adapter_bank"
    om_path = output / "export/edge_adapter_bank.om"
    stages.extend(
        (
            Stage(
                "export_onnx",
                module_command(
                    training_python,
                    f"{module}.export_onnx",
                    "--repo-root",
                    repo_root,
                    "--registry",
                    registry,
                    "--checkpoint",
                    output / "training/combined_adapter_bank.pt",
                    "--scales",
                    output / "calibration/adapter_scales.json",
                    "--candidate-slots",
                    args.candidate_slots,
                    "--output",
                    onnx_path,
                    "--report",
                    output / "evidence/onnx_export.json",
                ),
            ),
            Stage(
                "compile_om",
                (
                    "atc",
                    f"--model={onnx_path}",
                    "--framework=5",
                    f"--output={om_prefix}",
                    "--input_format=ND",
                    f"--input_shape=adapter_features:{args.candidate_slots},8",
                    "--soc_version=Ascend310B1",
                    "--precision_mode=allow_fp32_to_fp16",
                ),
            ),
            Stage(
                "benchmark_om",
                module_command(
                    production_python,
                    f"{module}.benchmark_om",
                    "--repo-root",
                    repo_root,
                    "--registry",
                    registry,
                    "--checkpoint",
                    output / "training/combined_adapter_bank.pt",
                    "--scales",
                    output / "calibration/adapter_scales.json",
                    "--candidate-slots",
                    args.candidate_slots,
                    "--om",
                    om_path,
                    "--baseline-fps",
                    args.baseline_fps,
                    "--output",
                    output / "evaluation/adapter_om_benchmark.json",
                ),
            ),
        )
    )
    return stages


def plan_payload(args: argparse.Namespace, stages: Sequence[Stage]) -> dict[str, object]:
    protocol = load_protocol(args.registry, args.repo_root)
    return {
        "schema_version": 1,
        "kind": "ascend310b_edge_incremental_workflow",
        "protocol_id": protocol.protocol_id,
        "registry": str(protocol.registry_path),
        "new_class_ids": list(protocol.new_class_ids),
        "output_root": str(args.output_root.expanduser().resolve()),
        "production_assets_modified": False,
        "cpu_fallback_allowed": False,
        "include_all_diagnostics": args.include_all_diagnostics,
        "stage_count": len(stages),
        "stages": [stage.public() for stage in stages],
    }


def run(args: argparse.Namespace, stages: Sequence[Stage]) -> int:
    repo_root = args.repo_root.expanduser().resolve()
    output = ensure_isolated_output(args.output_root, repo_root)
    for executable in (args.training_python, args.production_python):
        resolved = executable.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
    if shutil.which("atc") is None:
        raise RuntimeError("atc is unavailable; source the CANN environment first")
    output.mkdir(parents=True)
    plan = plan_payload(args, stages)
    (output / "workflow_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    state: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "started_at_epoch": time.time(),
        "completed_stages": [],
    }
    state_path = output / "workflow_state.json"
    base_environment = os.environ.copy()
    previous_pythonpath = base_environment.get("PYTHONPATH")
    base_environment["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{previous_pythonpath}"
        if previous_pythonpath
        else str(repo_root)
    )
    original_opp = base_environment.get("ASCEND_OPP_PATH")
    try:
        for index, stage in enumerate(stages, 1):
            print(f"\n[{index}/{len(stages)}] {stage.name}", flush=True)
            print(f"$ {shlex.join(stage.command)}", flush=True)
            environment = base_environment.copy()
            if stage.use_training_opp:
                environment["ASCEND_OPP_PATH"] = str(output / "opp_overlay")
            elif original_opp is not None:
                environment["ASCEND_OPP_PATH"] = original_opp
            started = time.perf_counter()
            subprocess.run(
                stage.command,
                cwd=repo_root,
                env=environment,
                check=True,
            )
            elapsed = time.perf_counter() - started
            completed = state["completed_stages"]
            assert isinstance(completed, list)
            completed.append({"name": stage.name, "wall_seconds": elapsed})
            state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["finished_at_epoch"] = time.time()
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    state["status"] = "completed"
    state["finished_at_epoch"] = time.time()
    state["total_wall_seconds"] = float(state["finished_at_epoch"]) - float(
        state["started_at_epoch"]
    )
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="计划或执行隔离的 Ascend310B 轻量增量训练流水线。"
    )
    parser.add_argument("mode", choices=("plan", "run"))
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--registry",
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
    parser.add_argument("--output-root", type=Path, required=True)
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
    parser.add_argument("--baseline-fps", type=float, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", default="20260825,11,26,4090,310")
    parser.add_argument("--learning-rates", default="0.01,0.05,0.1")
    parser.add_argument("--training-rows", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--candidate-slots", type=int, default=512)
    parser.add_argument("--encoded", action="store_true")
    parser.add_argument("--include-all-diagnostics", action="store_true")
    parser.add_argument("--expected-baseline-score", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    args.repo_root = repo_root
    ensure_isolated_output(args.output_root, repo_root)
    stages = build_stages(args)
    if args.mode == "plan":
        print(json.dumps(plan_payload(args, stages), ensure_ascii=False, indent=2))
        return 0
    return run(args, stages)


if __name__ == "__main__":
    raise SystemExit(main())
