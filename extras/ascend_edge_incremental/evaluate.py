#!/usr/bin/env python3
"""Compare frozen production candidates with an edge-trained adapter bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import (
    accuracy_gates,
    adapt_probe,
    image_sizes,
    load_adapter_bank,
    load_calibration_module,
    load_context_prior,
    load_method,
    load_scales,
    require_exact_probe,
    score_view,
)
from .protocol import load_protocol


def public_record(
    image_id: str,
    context: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "filename": f"{image_id}.png",
        "context": context,
        "detections": list(detections),
    }


def write_predictions(path: Path, probe: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    grouped = {image_id: [] for image_id in probe.image_ids}
    for row in rows:
        grouped[str(row["image_id"])].append(dict(row))
    path.write_text(
        "".join(
            json.dumps(
                public_record(image_id, probe.contexts[image_id], grouped[image_id]),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for image_id in sorted(probe.image_ids)
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在冻结 lock 或全部图像诊断范围比较板端 Adapter。"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--context-prior", type=Path, required=True)
    parser.add_argument("--scope", choices=("lock", "all"), required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-scales", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-baseline-score", type=Path)
    parser.add_argument("--write-predictions", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output_dir}")
    repo_root = args.repo_root.expanduser().resolve()
    protocol = load_protocol(args.registry, repo_root)
    from fair_agent.modules.strict_incremental import (  # noqa: PLC0415
        evaluate_ap50,
        precision_recall,
        retention_metrics,
        subset_rows,
        yolo_ground_truth,
    )

    calibration = load_calibration_module(repo_root)
    probe = calibration.load_probe(args.probe.expanduser().resolve())
    paths = protocol.image_paths(args.scope)
    require_exact_probe(probe, paths, args.scope)
    _, states = load_adapter_bank(args.checkpoint, protocol)
    scales, scale_source = load_scales(
        args.adapter_scales, protocol.new_class_ids, protocol.protocol_id
    )
    method = load_method(args.method_config.expanduser().resolve())
    context_prior = load_context_prior(args.context_prior.expanduser().resolve())
    ground_truth = yolo_ground_truth(paths)
    base_ids = protocol.base_image_ids(args.scope)
    sizes = image_sizes(paths)
    adapted_probe = adapt_probe(calibration, probe, states, sizes, scales)
    baseline, baseline_rows = score_view(
        calibration,
        protocol,
        method,
        probe,
        ground_truth,
        base_ids,
        context_prior,
        evaluate_ap50,
        precision_recall,
        retention_metrics,
        subset_rows,
    )
    adapted, adapted_rows = score_view(
        calibration,
        protocol,
        method,
        adapted_probe,
        ground_truth,
        base_ids,
        context_prior,
        evaluate_ap50,
        precision_recall,
        retention_metrics,
        subset_rows,
    )
    differences = {
        key: float(adapted[key]) - float(baseline[key])
        for key in ("base_map50", "new_map50", "krr", "full_map50")
    }
    expected_reproduction = None
    if args.expected_baseline_score is not None:
        expected = json.loads(
            args.expected_baseline_score.expanduser().read_text(encoding="utf-8")
        )
        expected_metrics = expected.get("metrics") or expected
        reproduction_differences = {
            key: float(baseline[key]) - float(expected_metrics[key])
            for key in ("base_map50", "new_map50", "krr", "full_map50")
        }
        expected_reproduction = {
            "source": str(args.expected_baseline_score.expanduser().resolve()),
            "differences": reproduction_differences,
            "passed": all(abs(value) <= 1e-6 for value in reproduction_differences.values()),
        }
    gates = accuracy_gates(method)
    gate_results = {
        key: float(adapted[key]) >= minimum for key, minimum in gates.items()
    }
    report = {
        "schema_version": 1,
        "phase": "joint_evaluation",
        "platform": "Ascend310B1",
        "protocol_id": protocol.protocol_id,
        "scope": args.scope,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "adapter_scales": {str(key): value for key, value in scales.items()},
        "adapter_scale_source": scale_source,
        "lock_opened_after_training_frozen": args.scope == "lock",
        "selection_used_lock": False,
        "baseline_reproduction": expected_reproduction,
        "baseline": baseline,
        "edge_adapter": adapted,
        "delta": differences,
        "competition_thresholds": gates,
        "competition_gates": gate_results,
        "competition_passed": all(gate_results.values()),
        "diagnostic_only": args.scope == "all",
        "production_replaced": False,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.write_predictions:
        write_predictions(output_dir / "baseline_predictions.jsonl", probe, baseline_rows)
        write_predictions(
            output_dir / "edge_adapter_predictions.jsonl",
            adapted_probe,
            adapted_rows,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.scope == "lock" and not report["competition_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
