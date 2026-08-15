#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import resolve_path
from fair_agent.modules.ascend_detection_export import (
    build_detections_v1_module,
    export_detections_v1_onnx,
    model_output_shapes,
    write_export_manifest,
)


def _probe(args: argparse.Namespace) -> dict:
    import torch

    class RawProbe(torch.nn.Module):
        def forward(self, predictions):
            return predictions

    module = build_detections_v1_module(
        RawProbe(),
        class_count=int(args.classes),
        candidate_confidence=float(args.candidate_confidence),
        iou_threshold=float(args.iou_threshold),
        max_det=int(args.max_det),
        nms_backend=str(args.nms_backend),
    )
    sample = torch.zeros(
        (1, 4 + int(args.classes), int(args.anchors)), dtype=torch.float32
    )
    sample[:, 2:4] = 2.0
    with torch.inference_mode():
        outputs = module(sample)
    exported = export_detections_v1_onnx(
        module,
        sample,
        resolve_path(args.output),
        input_name="raw_predictions",
        opset=int(args.opset),
    )
    return {
        "schema_version": 1,
        "kind": "cann_nms_probe",
        "input_shape": list(sample.shape),
        "output_shapes": model_output_shapes(outputs),
        "candidate_confidence": float(args.candidate_confidence),
        "iou_threshold": float(args.iou_threshold),
        "max_det": int(args.max_det),
        "nms_backend": str(args.nms_backend),
        "onnx": exported,
    }


def _detector_specs() -> tuple[dict, ...]:
    return (
        {
            "model_id": "base_detector",
            "source": "models/production/incremental_detection/three_class_base_detector.pt",
            "input_shape": (1, 3, 736, 896),
            "class_count": 3,
        },
        {
            "model_id": "incremental_detector",
            "source": "models/production/incremental_detection/incremental_detector.pt",
            "input_shape": (1, 3, 512, 640),
            "class_count": 1,
        },
    )


def _export(args: argparse.Namespace) -> dict:
    import torch
    from ultralytics import YOLO

    device = torch.device(str(args.device))
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"输出目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in _detector_specs():
        source = resolve_path(spec["source"])
        detector = copy.deepcopy(YOLO(str(source)).model).to(device).eval()
        if hasattr(detector, "fuse"):
            detector = detector.fuse().eval()
        for layer in detector.modules():
            if hasattr(layer, "export"):
                layer.export = True
            if hasattr(layer, "format"):
                layer.format = "onnx"
            if hasattr(layer, "dynamic"):
                layer.dynamic = False
        module = build_detections_v1_module(
            detector,
            class_count=int(spec["class_count"]),
            candidate_confidence=float(args.candidate_confidence),
            iou_threshold=float(args.iou_threshold),
            max_det=int(args.max_det),
            nms_backend=str(args.nms_backend),
        ).to(device)
        sample = torch.zeros(spec["input_shape"], device=device)
        with torch.inference_mode():
            outputs = module(sample)
        exported = export_detections_v1_onnx(
            module,
            sample,
            output_dir / f"{spec['model_id']}.onnx",
            input_name="images",
            opset=int(args.opset),
        )
        rows.append(
            {
                **spec,
                "source": str(source),
                "output_shapes": model_output_shapes(outputs),
                "onnx": exported,
            }
        )
        del module, detector, sample, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "kind": "ascend_detections_v1",
        "candidate_confidence": float(args.candidate_confidence),
        "iou_threshold": float(args.iou_threshold),
        "max_det": int(args.max_det),
        "nms_backend": str(args.nms_backend),
        "opset": int(args.opset),
        "device": str(device),
        "assets": rows,
    }
    payload["manifest"] = write_export_manifest(
        output_dir / "export-manifest.json", payload
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "无新增依赖导出Ascend P6固定检测输出ONNX。导出物必须先通过板端"
            "严格语义探针；当前CANN 7.0.RC1的已测NMS后端均未通过。"
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--output", required=True)
    probe.add_argument("--anchors", type=int, default=64)
    probe.add_argument("--classes", type=int, default=2)
    export = subparsers.add_parser("export")
    export.add_argument("--output-dir", required=True)
    export.add_argument("--device", default="cuda:0")
    for child in (probe, export):
        child.add_argument(
            "--nms-backend",
            choices=["nms_with_mask", "standard_onnx"],
            required=True,
            help="显式选择实验后端；该选择不代表已通过板端语义门禁。",
        )
        child.add_argument("--candidate-confidence", type=float, default=0.01)
        child.add_argument("--iou-threshold", type=float, default=0.7)
        child.add_argument("--max-det", type=int, default=300)
        child.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    result = _probe(args) if args.action == "probe" else _export(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
