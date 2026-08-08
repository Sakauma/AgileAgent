#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import resolve_path
from fair_agent.modules.ascend_preflight import (
    benchmark_local_optimization_candidates,
    benchmark_onnx_pipeline,
    compare_fixed_agent_metrics,
    compare_raw_outputs,
    convert_fixed_onnx_assets_to_mixed_fp16,
    export_fixed_onnx_assets,
    production_onnx_plan,
    write_golden_bundle,
)
from fair_agent.modules.strict_incremental import read_split


def _unique_samples(paths: Sequence[Path], count: int) -> list[Path]:
    selected = []
    seen = set()
    for path in paths:
        if path.stem in seen:
            continue
        selected.append(path)
        seen.add(path.stem)
        if len(selected) >= int(count):
            break
    return selected


def _dev_samples(count: int) -> list[Path]:
    base = read_split(resolve_path("splits/strict_3plus1/base_dev.txt"))
    incremental = read_split(resolve_path("splits/strict_3plus1/increment_dev.txt"))
    base_count = max(1, (int(count) + 1) // 2)
    incremental_count = max(1, int(count) - base_count)
    base_selected = _unique_samples(base, base_count)
    incremental_selected = _unique_samples(incremental, incremental_count)
    selected = []
    for index in range(max(len(base_selected), len(incremental_selected))):
        if index < len(base_selected):
            selected.append(base_selected[index])
        if index < len(incremental_selected):
            selected.append(incremental_selected[index])
    return _unique_samples([*selected, *base, *incremental], count)


def _write_report(root: str | Path, name: str, payload: Any) -> Path:
    path = resolve_path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在无Ascend板卡时导出并验证固定shape ONNX与Agent预处理。"
    )
    parser.add_argument(
        "action",
        choices=[
            "export",
            "convert-fp16",
            "raw-align",
            "metric-align",
            "golden",
            "benchmark",
            "optimize",
            "all",
        ],
    )
    parser.add_argument("--output-root", default="runs/ascend310b")
    parser.add_argument(
        "--source-root",
        default="runs/ascend310b",
        help="convert-fp16所读取的固定FP32 ONNX根目录",
    )
    parser.add_argument("--shape-mode", choices=["rect", "square"], default="rect")
    parser.add_argument("--device", default="0")
    parser.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--context-fp32",
        action="store_true",
        help="convert-fp16时只转换两个检测器，Scene-SensorNet保持FP32",
    )
    args = parser.parse_args()

    output_root = resolve_path(args.output_root)
    alignment_root = output_root / "alignment" / args.shape_mode
    if args.overwrite and args.action == "all" and alignment_root.exists():
        shutil.rmtree(alignment_root)
    elif args.overwrite and args.action == "metric-align":
        for path in (
            alignment_root / "pytorch",
            alignment_root / "onnx",
            alignment_root / "summary.json",
        ):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    result: dict[str, Any] = {"action": args.action, "shape_mode": args.shape_mode}
    if args.action in {"export", "all"}:
        result["export"] = export_fixed_onnx_assets(
            output_root,
            shape_mode=args.shape_mode,
            device=args.device,
            opset=args.opset,
            simplify=True,
            overwrite=args.overwrite,
        )

    if args.action == "convert-fp16":
        result["mixed_fp16"] = convert_fixed_onnx_assets_to_mixed_fp16(
            args.source_root,
            output_root,
            shape_mode=args.shape_mode,
            keep_context_fp32=args.context_fp32,
            overwrite=args.overwrite,
        )

    samples = (
        _dev_samples(args.samples)
        if args.action in {"raw-align", "golden", "benchmark", "all"}
        else []
    )
    if args.action in {"raw-align", "all"}:
        assets = production_onnx_plan(output_root, shape_mode=args.shape_mode)
        raw = compare_raw_outputs(
            assets, samples, device=args.device, provider=args.provider
        )
        raw_path = _write_report(
            output_root, f"alignment/{args.shape_mode}/raw_outputs.json", raw
        )
        result["raw_alignment"] = {**raw, "report": str(raw_path)}

    if args.action in {"metric-align", "all"}:
        result["metric_alignment"] = compare_fixed_agent_metrics(
            output_root,
            shape_mode=args.shape_mode,
            device=args.device,
            provider=args.provider,
        )

    if args.action in {"golden", "all"}:
        result["golden"] = write_golden_bundle(
            output_root,
            shape_mode=args.shape_mode,
            image_paths=samples[: min(3, len(samples))],
            device=args.device,
            provider=args.provider,
            overwrite=args.overwrite,
        )

    if args.action in {"benchmark", "all"}:
        result["benchmark"] = benchmark_onnx_pipeline(
            output_root,
            shape_mode=args.shape_mode,
            image_paths=samples,
            device=args.device,
            provider=args.provider,
            warmup=args.warmup,
            rounds=args.rounds,
        )

    if args.action in {"optimize", "all"}:
        mixed_test = read_split(resolve_path("splits/strict_3plus1/mixed_test.txt"))
        result["local_optimization"] = benchmark_local_optimization_candidates(
            output_root,
            shape_mode=args.shape_mode,
            image_paths=samples,
            correctness_paths=mixed_test,
            device=args.device,
            warmup=args.warmup,
            rounds=args.rounds,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    passed = []
    if "raw_alignment" in result:
        passed.append(bool(result["raw_alignment"]["passed"]))
    if "metric_alignment" in result:
        passed.append(bool(result["metric_alignment"]["passed"]))
    if "local_optimization" in result:
        passed.append(bool(result["local_optimization"]["passed"]))
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
