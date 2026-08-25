#!/usr/bin/env python3
"""Export a calibrated registry-ordered Adapter bank to static ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from .core import FEATURE_DIM, load_adapter_bank, load_scales
from .protocol import load_protocol


class ExportBank(torch.nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weights", weights.reshape(-1, FEATURE_DIM))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features.unsqueeze(1) * self.weights.unsqueeze(0)).sum(dim=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate-slots", type=int, default=512)
    parser.add_argument("--scales", type=Path)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("ONNX output or report already exists")
    if args.candidate_slots <= 0:
        raise ValueError("candidate slots must be positive")
    repo_root = args.repo_root.expanduser().resolve()
    protocol = load_protocol(args.registry, repo_root)
    _, states = load_adapter_bank(args.checkpoint, protocol)
    scales, scale_source = load_scales(
        args.scales, protocol.new_class_ids, protocol.protocol_id
    )
    weights = torch.stack(
        [states[class_id]["weights"] * scales[class_id] for class_id in protocol.new_class_ids]
    )
    model = ExportBank(weights).eval()
    example = torch.zeros(args.candidate_slots, FEATURE_DIM, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        output,
        input_names=["adapter_features"],
        output_names=["residual_logits"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    loaded = onnx.load(str(output))
    onnx.checker.check_model(loaded)
    operator_types = sorted({node.op_type for node in loaded.graph.node})
    contains_matmul = any(value in {"MatMul", "Gemm"} for value in operator_types)
    if contains_matmul:
        raise RuntimeError(f"exported Adapter unexpectedly contains MatMul: {operator_types}")
    with torch.no_grad():
        observed = model(example)
    payload = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "class_order": list(protocol.new_class_ids),
        "scales": {str(key): value for key, value in scales.items()},
        "scale_source": scale_source,
        "onnx": str(output),
        "candidate_slots": args.candidate_slots,
        "feature_dim": FEATURE_DIM,
        "output_dim": len(protocol.new_class_ids),
        "opset": 11,
        "onnx_checker_passed": True,
        "operator_types": operator_types,
        "zero_input_output_shape": list(observed.shape),
        "parameter_count": FEATURE_DIM * len(protocol.new_class_ids),
        "contains_matmul": contains_matmul,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
