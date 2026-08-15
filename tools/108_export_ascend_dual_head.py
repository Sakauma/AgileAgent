#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import resolve_path  # noqa: E402
from fair_agent.modules.ascend_detection_export import (  # noqa: E402
    model_output_shapes,
    write_export_manifest,
)
from fair_agent.modules.shared_dual_head import (  # noqa: E402
    build_shared_dual_head_export_module,
    export_shared_dual_head_onnx,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="导出P10共享Base backbone/neck、old/new双逻辑Detect head的固定ONNX。"
    )
    parser.add_argument("--base-weight", required=True)
    parser.add_argument("--new-head-weight", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import torch

    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"输出目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    module, provenance = build_shared_dual_head_export_module(
        resolve_path(args.base_weight),
        resolve_path(args.new_head_weight),
        str(args.device),
    )
    sample = torch.zeros((1, 3, 736, 896), device=str(args.device))
    with torch.inference_mode():
        outputs = module(sample)
    expected = [[1, 7, 13524], [1, 5, 13524]]
    actual = model_output_shapes(outputs)
    if actual != expected:
        raise RuntimeError(f"P10双head输出shape错误：{actual} != {expected}")
    exported = export_shared_dual_head_onnx(
        module,
        sample,
        output_dir / "shared_backbone_dual_head.onnx",
        opset=int(args.opset),
    )
    payload = {
        "schema_version": 1,
        "kind": "shared_backbone_dual_head_v1",
        "device": str(args.device),
        "opset": int(args.opset),
        "input_shape": [1, 3, 736, 896],
        "output_shapes": actual,
        "logical_heads": {
            "old": {
                "owner": "frozen_base_model",
                "class_map": {"0": 0, "1": 1, "2": 3},
                "class_count": 3,
                "anchor_count": 13524,
                "output_index": 0,
            },
            "new": {
                "owner": "incremental_model",
                "class_map": {"0": 2},
                "class_count": 1,
                "anchor_count": 13524,
                "output_index": 1,
            },
        },
        "provenance": provenance,
        "onnx": exported,
    }
    payload["manifest"] = write_export_manifest(
        output_dir / "export-manifest.json", payload
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
