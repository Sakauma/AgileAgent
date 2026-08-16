#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml


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
    parser.add_argument(
        "--method-config",
        default=str(ROOT / "configs/ascend310b/full_score_method.yaml"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opset", type=int)
    args = parser.parse_args()

    import torch

    method = yaml.safe_load(
        resolve_path(args.method_config).read_text(encoding="utf-8")
    )
    if not isinstance(method, dict):
        raise ValueError("满分方法配置必须是YAML mapping")
    export = method.get("export") or {}
    input_shape = list(export.get("input_shape_nchw") or [])
    expected = [
        list((export.get("logical_heads") or {})[name]["output_shape"])
        for name in ("old", "new")
    ]
    method_opset = int(export["opset"])
    if args.opset is not None and int(args.opset) != method_opset:
        raise ValueError("--opset必须与满分方法配置一致")

    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"输出目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    module, provenance = build_shared_dual_head_export_module(
        resolve_path(args.base_weight),
        resolve_path(args.new_head_weight),
        str(args.device),
    )
    sample = torch.zeros(tuple(input_shape), device=str(args.device))
    with torch.inference_mode():
        outputs = module(sample)
    actual = model_output_shapes(outputs)
    if actual != expected:
        raise RuntimeError(f"P10双head输出shape错误：{actual} != {expected}")
    exported = export_shared_dual_head_onnx(
        module,
        sample,
        output_dir / "shared_backbone_dual_head.onnx",
        opset=method_opset,
    )
    logical_heads = copy.deepcopy(export.get("logical_heads"))
    if not isinstance(logical_heads, dict) or set(logical_heads) != {"old", "new"}:
        raise ValueError("满分方法配置必须定义old/new logical_heads")
    for head in logical_heads.values():
        head.pop("output_shape", None)
    if (
        logical_heads["old"].get("owner") != "frozen_base_model"
        or logical_heads["new"].get("owner") != "incremental_model"
        or {
            logical_heads["old"].get("output_index"),
            logical_heads["new"].get("output_index"),
        }
        != {0, 1}
    ):
        raise ValueError("满分方法logical head owner/output_index非法")
    payload = {
        "schema_version": 1,
        "kind": export["model_layout"],
        "device": str(args.device),
        "opset": method_opset,
        "input_shape": input_shape,
        "output_shapes": actual,
        "logical_heads": logical_heads,
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
