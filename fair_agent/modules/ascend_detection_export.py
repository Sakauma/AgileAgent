from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Sequence


DETECTIONS_V1_OUTPUT_NAMES = (
    "boxes",
    "scores",
    "class_ids",
    "valid_count",
)

DECODED_CANDIDATES_V1_OUTPUT_NAMES = (
    "boxes",
    "scores",
    "class_ids",
    "anchor_ids",
    "valid_count",
    "overflow",
    "raw_output",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def _without_optional_onnx_postprocessing() -> Iterator[None]:
    """Allow the legacy Torch exporter to emit protobuf without `onnx`.

    Torch has already serialized the complete standard-operator graph when it
    calls this hook. The hook only adds ONNXScript function definitions for
    registered custom ops; P6 deliberately exports no such functions.
    """

    from torch.onnx._internal import onnx_proto_utils

    original = onnx_proto_utils._add_onnxscript_fn
    onnx_proto_utils._add_onnxscript_fn = (
        lambda model_bytes, _custom_opsets: model_bytes
    )
    try:
        yield
    finally:
        onnx_proto_utils._add_onnxscript_fn = original


def build_detections_v1_module(
    detector: Any,
    *,
    class_count: int,
    candidate_confidence: float,
    iou_threshold: float,
    max_det: int,
    nms_backend: str = "nms_with_mask",
) -> Any:
    """Wrap one raw YOLO head in a fixed, class-aware detections contract."""

    import torch
    from torch.onnx import symbolic_helper
    from torchvision.ops import nms

    if int(class_count) <= 0:
        raise ValueError("class_count必须为正整数")
    if not 0.0 <= float(candidate_confidence) < 1.0:
        raise ValueError("candidate_confidence必须位于[0, 1)")
    if not 0.0 < float(iou_threshold) < 1.0:
        raise ValueError("iou_threshold必须位于(0, 1)")
    if int(max_det) <= 0:
        raise ValueError("max_det必须为正整数")
    if nms_backend not in {
        "batch_multiclass_nms",
        "nms_with_mask",
        "standard_onnx",
    }:
        raise ValueError(f"nms_backend非法：{nms_backend}")

    class StableArgsort(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values, count: int, descending: bool):
            del ctx, count
            return torch.argsort(values, descending=bool(descending), stable=True)

        @staticmethod
        def symbolic(graph, values, count, descending):
            count_value = symbolic_helper._parse_arg(count, "i")
            descending_value = symbolic_helper._parse_arg(descending, "b")
            count_tensor = graph.op(
                "Constant",
                value_t=torch.tensor([count_value], dtype=torch.int64),
            )
            _sorted_values, indices = graph.op(
                "TopK",
                values,
                count_tensor,
                axis_i=0,
                largest_i=int(descending_value),
                sorted_i=1,
                outputs=2,
            )
            return indices

    def stable_argsort(values, *, descending: bool = False):
        return StableArgsort.apply(values, int(values.shape[0]), bool(descending))

    class NMSWithMask(torch.autograd.Function):
        @staticmethod
        def forward(ctx, proposals, threshold: float):
            del ctx
            keep = nms(proposals[:, :4], proposals[:, 4], float(threshold))
            selected_boxes = proposals[:, :5].clone()
            selected_indices = torch.arange(
                proposals.shape[0],
                device=proposals.device,
                dtype=torch.int32,
            )
            selected_mask = torch.zeros(
                proposals.shape[0],
                device=proposals.device,
                dtype=torch.uint8,
            )
            selected_mask[keep] = 1
            return selected_boxes, selected_indices, selected_mask

        @staticmethod
        def symbolic(graph, proposals, threshold):
            threshold_value = symbolic_helper._parse_arg(threshold, "f")
            return graph.op(
                "NPUNmsWithMask",
                proposals,
                iou_threshold_f=float(threshold_value),
                outputs=3,
            )

    class BatchMultiClassNMS(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            batch_boxes,
            batch_scores,
            score_threshold: float,
            iou_threshold: float,
            max_size_per_class: int,
            max_total_size: int,
        ):
            del ctx
            boxes_value = batch_boxes[0, :, 0, :].to(torch.float32)
            scores_value = batch_scores[0].to(torch.float32)
            selected_boxes = []
            selected_scores = []
            selected_classes = []
            for class_id in range(scores_value.shape[1]):
                class_scores = scores_value[:, class_id]
                valid = class_scores > float(score_threshold)
                valid_ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
                if valid_ids.numel() == 0:
                    continue
                keep = nms(
                    boxes_value[valid_ids],
                    class_scores[valid_ids],
                    float(iou_threshold),
                )[: int(max_size_per_class)]
                selected = valid_ids[keep]
                selected_boxes.append(boxes_value[selected])
                selected_scores.append(class_scores[selected])
                selected_classes.append(
                    torch.full(
                        (selected.numel(),),
                        class_id,
                        device=boxes_value.device,
                        dtype=torch.float32,
                    )
                )
            output_boxes = torch.zeros(
                (1, int(max_total_size), 4),
                device=boxes_value.device,
                dtype=torch.float16,
            )
            output_scores = torch.zeros(
                (1, int(max_total_size)),
                device=boxes_value.device,
                dtype=torch.float16,
            )
            output_classes = torch.zeros(
                (1, int(max_total_size)),
                device=boxes_value.device,
                dtype=torch.float16,
            )
            output_count = torch.zeros(
                (1,), device=boxes_value.device, dtype=torch.int32
            )
            if selected_scores:
                candidate_boxes = torch.cat(selected_boxes, dim=0)
                candidate_scores = torch.cat(selected_scores, dim=0)
                candidate_classes = torch.cat(selected_classes, dim=0)
                order = torch.argsort(
                    candidate_scores, descending=True, stable=True
                )[: int(max_total_size)]
                count = int(order.numel())
                output_boxes[0, :count] = candidate_boxes[order].to(torch.float16)
                output_scores[0, :count] = candidate_scores[order].to(torch.float16)
                output_classes[0, :count] = candidate_classes[order].to(
                    torch.float16
                )
                output_count[0] = count
            return output_boxes, output_scores, output_classes, output_count

        @staticmethod
        def symbolic(
            graph,
            batch_boxes,
            batch_scores,
            score_threshold,
            iou_threshold,
            max_size_per_class,
            max_total_size,
        ):
            return graph.op(
                "BatchMultiClassNMS",
                batch_boxes,
                batch_scores,
                change_coordinate_frame_i=0,
                iou_threshold_f=float(
                    symbolic_helper._parse_arg(iou_threshold, "f")
                ),
                max_size_per_class_i=int(
                    symbolic_helper._parse_arg(max_size_per_class, "i")
                ),
                max_total_size_i=int(
                    symbolic_helper._parse_arg(max_total_size, "i")
                ),
                score_threshold_f=float(
                    symbolic_helper._parse_arg(score_threshold, "f")
                ),
                transpose_box_i=0,
                outputs=4,
            )

    class DetectionsV1(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.detector = detector
            self.class_count = int(class_count)
            self.candidate_confidence = float(candidate_confidence)
            self.iou_threshold = float(iou_threshold)
            self.max_det = int(max_det)
            self.nms_backend = str(nms_backend)

        def forward(self, images):
            raw = self.detector(images)
            if isinstance(raw, (tuple, list)):
                raw = raw[0]
            prediction = raw[0]
            xywh = prediction[:4].transpose(0, 1)
            boxes = torch.empty_like(xywh)
            boxes[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2.0
            boxes[:, 2:] = xywh[:, :2] + xywh[:, 2:] / 2.0
            class_scores = prediction[4 : 4 + self.class_count]
            anchor_count = int(prediction.shape[1])
            if self.nms_backend == "batch_multiclass_nms":
                batch_boxes = boxes.unsqueeze(0).unsqueeze(2).to(torch.float16)
                batch_scores = class_scores.transpose(0, 1).unsqueeze(0).to(
                    torch.float16
                )
                result_boxes, result_scores, result_classes, result_count = (
                    BatchMultiClassNMS.apply(
                        batch_boxes,
                        batch_scores,
                        self.candidate_confidence,
                        self.iou_threshold,
                        self.max_det,
                        self.max_det,
                    )
                )
                return (
                    result_boxes[0].to(torch.float32),
                    result_scores[0].to(torch.float32),
                    result_classes[0].to(torch.int32),
                    result_count.to(torch.int32),
                )
            positions = torch.arange(
                anchor_count,
                device=prediction.device,
                dtype=torch.long,
            )
            filler_positions = torch.arange(
                self.max_det,
                device=prediction.device,
                dtype=torch.long,
            )
            per_class_boxes = []
            per_class_scores = []
            per_class_ids = []
            per_class_anchors = []
            for class_id in range(self.class_count):
                scores = class_scores[class_id]
                order = stable_argsort(scores, descending=True)
                ordered_scores = scores[order]
                ordered_boxes = boxes[order]
                valid = ordered_scores > self.candidate_confidence

                if self.nms_backend == "nms_with_mask":
                    proposals = torch.cat(
                        (
                            ordered_boxes,
                            ordered_scores[:, None],
                        ),
                        dim=1,
                    )
                    _selected_boxes, _selected_indices, selected_mask = (
                        NMSWithMask.apply(proposals, self.iou_threshold)
                    )
                    selected_scores = torch.where(
                        selected_mask.to(torch.bool) & valid,
                        ordered_scores,
                        torch.full_like(ordered_scores, -1.0),
                    )
                    keep = stable_argsort(selected_scores, descending=True)[
                        : self.max_det
                    ]
                    per_class_boxes.append(ordered_boxes[keep])
                    per_class_scores.append(ordered_scores[keep])
                    per_class_anchors.append(order[keep])
                else:
                    # Invalid candidates and the explicit fillers are
                    # zero-area, non-suppressing boxes. Fillers guarantee a
                    # fixed slice from standard ONNX NMS.
                    invalid_x = positions.to(boxes.dtype) * 2.0 + 10000.0
                    invalid_boxes = torch.stack(
                        (invalid_x, invalid_x, invalid_x, invalid_x), dim=1
                    )
                    safe_boxes = torch.where(
                        valid[:, None], ordered_boxes, invalid_boxes
                    )
                    filler_x = filler_positions.to(boxes.dtype) * 2.0 + 100000.0
                    filler_boxes = torch.stack(
                        (filler_x, filler_x, filler_x, filler_x), dim=1
                    )
                    nms_boxes = torch.cat((safe_boxes, filler_boxes), dim=0)
                    rank_scores = -torch.arange(
                        anchor_count + self.max_det,
                        device=prediction.device,
                        dtype=boxes.dtype,
                    )
                    keep = nms(nms_boxes, rank_scores, self.iou_threshold)[
                        : self.max_det
                    ]
                    padded_scores = torch.cat(
                        (
                            ordered_scores,
                            torch.full(
                                (self.max_det,),
                                -1.0,
                                device=prediction.device,
                                dtype=ordered_scores.dtype,
                            ),
                        )
                    )
                    padded_boxes = torch.cat(
                        (
                            ordered_boxes,
                            torch.zeros(
                                (self.max_det, 4),
                                device=prediction.device,
                                dtype=boxes.dtype,
                            ),
                        )
                    )
                    padded_anchors = torch.cat((order, filler_positions + anchor_count))
                    per_class_boxes.append(padded_boxes[keep])
                    per_class_scores.append(padded_scores[keep])
                    per_class_anchors.append(padded_anchors[keep])
                per_class_ids.append(
                    torch.full(
                        (self.max_det,),
                        class_id,
                        device=prediction.device,
                        dtype=torch.int32,
                    )
                )

            candidate_boxes = torch.cat(per_class_boxes, dim=0)
            candidate_scores = torch.cat(per_class_scores, dim=0)
            candidate_classes = torch.cat(per_class_ids, dim=0)
            candidate_anchors = torch.cat(per_class_anchors, dim=0)

            # Restore np.where's anchor-major/class-minor order before the
            # stable confidence sort. This makes equal scores deterministic.
            tie_keys = candidate_anchors * self.class_count + candidate_classes.to(
                torch.long
            )
            tie_order = stable_argsort(tie_keys)
            score_order = stable_argsort(candidate_scores[tie_order], descending=True)
            selected = tie_order[score_order[: self.max_det]]
            final_boxes = candidate_boxes[selected]
            final_scores = candidate_scores[selected]
            final_classes = candidate_classes[selected]
            final_valid = final_scores > self.candidate_confidence
            valid_count = final_valid.to(torch.float32).sum().to(torch.int32).reshape(1)
            return (
                torch.where(
                    final_valid[:, None],
                    final_boxes,
                    torch.zeros_like(final_boxes),
                ).to(torch.float32),
                torch.where(
                    final_valid,
                    final_scores,
                    torch.zeros_like(final_scores),
                ).to(torch.float32),
                torch.where(
                    final_valid,
                    final_classes,
                    torch.zeros_like(final_classes),
                ).to(torch.int32),
                valid_count.to(torch.int32),
            )

    return DetectionsV1()


def build_decoded_candidates_v1_module(
    detector: Any,
    *,
    class_count: int,
    candidate_confidence: float,
    candidate_capacity: int,
) -> Any:
    """Decode and gather fixed candidates while leaving exact NMS on Host.

    The fixed TopK key ranks every strict-threshold candidate ahead of every
    invalid entry, then ranks by the flattened anchor-major/class-minor index.
    Unlike NonZero, this keeps all output shapes static for CANN 7.0.RC1.
    """

    import torch

    if int(class_count) <= 0:
        raise ValueError("class_count必须为正整数")
    if float(candidate_confidence) != 0.01:
        raise ValueError("decoded_candidates_v1候选阈值必须固定为0.01")
    if int(candidate_capacity) <= 0:
        raise ValueError("candidate_capacity必须为正整数")

    class DecodedCandidatesV1(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.detector = detector
            self.class_count = int(class_count)
            self.candidate_confidence = float(candidate_confidence)
            self.candidate_capacity = int(candidate_capacity)

        def forward(self, images):
            raw = self.detector(images)
            if isinstance(raw, (tuple, list)):
                raw = raw[0]
            prediction = raw[0]
            anchor_count = int(prediction.shape[1])
            total_candidates = anchor_count * self.class_count
            if self.candidate_capacity > total_candidates:
                raise RuntimeError(
                    "candidate_capacity不能超过anchor_count*class_count"
                )

            xywh = prediction[:4].transpose(0, 1)
            decoded_boxes = torch.cat(
                (
                    xywh[:, :2] - xywh[:, 2:] / 2.0,
                    xywh[:, :2] + xywh[:, 2:] / 2.0,
                ),
                dim=1,
            )
            flat_scores = (
                prediction[4 : 4 + self.class_count]
                .transpose(0, 1)
                .reshape(total_candidates)
            )
            flat_ids = torch.arange(
                total_candidates,
                device=prediction.device,
                dtype=torch.long,
            )
            valid = flat_scores > self.candidate_confidence
            # All integers involved are exactly representable in float32 for
            # the fixed detector sizes used here. Keys are unique, so TopK
            # cannot introduce an equal-key ordering ambiguity.
            rank_keys = (
                valid.to(torch.float32) * float(total_candidates + 1)
                - flat_ids.to(torch.float32)
            )
            _rank_values, selected = torch.topk(
                rank_keys,
                k=self.candidate_capacity,
                largest=True,
                sorted=True,
            )
            selected_scores = flat_scores[selected]
            selected_valid = selected_scores > self.candidate_confidence
            anchor_ids = torch.div(
                selected,
                self.class_count,
                rounding_mode="floor",
            )
            class_ids = torch.remainder(selected, self.class_count)
            boxes = decoded_boxes[anchor_ids]
            # CANN 7.0.RC1 miscompiles ReduceSum over this int32 cast as a
            # boolean-style 0/1 result. Its float32 ReduceSum path preserves
            # the fixed candidate count, so convert only after sum/clamp.
            valid_total = valid.to(torch.float32).sum()
            valid_count = torch.clamp(
                valid_total,
                max=float(self.candidate_capacity),
            ).to(torch.int32).reshape(1)
            overflow = (
                valid_total > float(self.candidate_capacity)
            ).to(torch.int32).reshape(1)
            return (
                torch.where(
                    selected_valid[:, None], boxes, torch.zeros_like(boxes)
                ).to(torch.float32),
                torch.where(
                    selected_valid,
                    selected_scores,
                    torch.zeros_like(selected_scores),
                ).to(torch.float32),
                torch.where(
                    selected_valid,
                    class_ids,
                    torch.zeros_like(class_ids),
                ).to(torch.int32),
                torch.where(
                    selected_valid,
                    anchor_ids,
                    torch.zeros_like(anchor_ids),
                ).to(torch.int32),
                valid_count.to(torch.int32),
                overflow.to(torch.int32),
                raw.to(torch.float32),
            )

    return DecodedCandidatesV1()


def export_detections_v1_onnx(
    module: Any,
    sample: Any,
    target: str | Path,
    *,
    input_name: str,
    opset: int = 17,
) -> dict[str, Any]:
    import torch

    path = Path(target).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    module.eval()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".onnx", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with _without_optional_onnx_postprocessing(), torch.inference_mode():
            torch.onnx.export(
                module,
                sample,
                temporary,
                input_names=[str(input_name)],
                output_names=list(DETECTIONS_V1_OUTPUT_NAMES),
                opset_version=int(opset),
                dynamic_axes=None,
                do_constant_folding=True,
                dynamo=False,
            )
        if temporary.stat().st_size <= 0:
            raise RuntimeError("Torch导出的detections_v1 ONNX为空")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "opset": int(opset),
        "input_name": str(input_name),
        "output_names": list(DETECTIONS_V1_OUTPUT_NAMES),
    }


def export_decoded_candidates_v1_onnx(
    module: Any,
    sample: Any,
    target: str | Path,
    *,
    input_name: str,
    opset: int = 17,
) -> dict[str, Any]:
    import torch

    path = Path(target).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    module.eval()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".onnx", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with _without_optional_onnx_postprocessing(), torch.inference_mode():
            torch.onnx.export(
                module,
                sample,
                temporary,
                input_names=[str(input_name)],
                output_names=list(DECODED_CANDIDATES_V1_OUTPUT_NAMES),
                opset_version=int(opset),
                dynamic_axes=None,
                do_constant_folding=True,
                dynamo=False,
            )
        if temporary.stat().st_size <= 0:
            raise RuntimeError("Torch导出的decoded_candidates_v1 ONNX为空")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "opset": int(opset),
        "input_name": str(input_name),
        "output_names": list(DECODED_CANDIDATES_V1_OUTPUT_NAMES),
    }


def write_export_manifest(
    target: str | Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    path = Path(target).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"path": str(path), "sha256": _sha256(path)}


def model_output_shapes(outputs: Sequence[Any]) -> list[list[int]]:
    return [list(map(int, output.shape)) for output in outputs]
