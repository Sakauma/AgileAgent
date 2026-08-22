from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fair_agent.backends.ascend_acl import (
    AscendAclBackend,
    AscendDualDetectorExecutionHandle,
    AscendEncodedPreprocessor,
    context_tensor,
    decoded_candidates_v1_records,
    decoded_output_copy_plan,
    detections_v1_records,
    detector_tensor,
    validate_dvpp_scene_resize_stages,
    yolo26_e2e_v1_records,
    yolo_detections,
)


def fixed_png_header(
    width: int = 640,
    height: int = 512,
    bit_depth: int = 8,
    color_type: int = 2,
) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes((bit_depth, color_type))
    )


@pytest.mark.parametrize("color_type", [2, 6])
def test_encoded_preprocessor_accepts_only_fixed_rgb_png(color_type: int) -> None:
    assert AscendEncodedPreprocessor.accepts(
        fixed_png_header(color_type=color_type)
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"not-png",
        fixed_png_header(width=641),
        fixed_png_header(height=511),
        fixed_png_header(bit_depth=16),
        fixed_png_header(color_type=0),
    ],
)
def test_encoded_preprocessor_rejects_other_image_contracts(payload: bytes) -> None:
    assert not AscendEncodedPreprocessor.accepts(payload)


def test_detector_tensor_uses_static_310b_shape() -> None:
    image = Image.new("RGB", (640, 512), "black")
    tensor, info = detector_tensor(image, 736, 896)
    assert tensor.shape == (1, 3, 736, 896)
    assert tensor.dtype == np.float32
    assert info["pad_left"] == 0
    assert info["pad_top"] == 9


def test_detector_tensor_reuses_exact_rgb_array() -> None:
    image = Image.new("RGB", (640, 512), (15, 127, 240))
    rgb = np.ascontiguousarray(np.asarray(image))
    expected, expected_info = detector_tensor(image, 736, 896)
    actual, actual_info = detector_tensor(image, 736, 896, rgb)
    assert np.array_equal(actual, expected)
    assert actual_info == expected_info


def test_detector_tensor_emits_contiguous_uint8_nhwc_for_aipp() -> None:
    image = Image.new("RGB", (640, 512), (15, 127, 240))
    tensor, info = detector_tensor(
        image,
        736,
        896,
        input_mode="nhwc_uint8_aipp",
    )
    assert tensor.shape == (1, 736, 896, 3)
    assert tensor.dtype == np.uint8
    assert tensor.flags.c_contiguous
    assert tensor[0, 0, 0].tolist() == [114, 114, 114]
    assert info["pad_top"] == 9


def test_context_tensor_emits_contiguous_uint8_nhwc_for_aipp() -> None:
    image = Image.new("RGB", (200, 160), (15, 127, 240))
    tensor = context_tensor(image, 160, input_mode="nhwc_uint8_aipp")
    assert tensor.shape == (1, 160, 160, 3)
    assert tensor.dtype == np.uint8
    assert tensor.flags.c_contiguous
    assert tensor[0, 0, 0].tolist() == [15, 127, 240]


def test_dvpp_scene_resize_stages_require_bounded_even_dimensions() -> None:
    assert validate_dvpp_scene_resize_stages([[208, 192], [288, 230]]) == (
        (208, 192),
        (288, 230),
    )
    with pytest.raises(ValueError, match="偶数"):
        validate_dvpp_scene_resize_stages([[207, 192]])
    with pytest.raises(ValueError, match="最多4级"):
        validate_dvpp_scene_resize_stages([[32, 32]] * 5)


def test_yolo_nms_applies_global_max_det_order() -> None:
    raw = np.zeros((1, 6, 3), dtype=np.float32)
    raw[0, :4] = np.asarray(
        [
            [10.0, 30.0, 50.0],
            [10.0, 30.0, 50.0],
            [4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    raw[0, 4:] = np.asarray(
        [
            [0.80, 0.70, 0.01],
            [0.01, 0.01, 0.95],
        ],
        dtype=np.float32,
    )
    rows = yolo_detections(
        raw,
        {
            "original_height": 64,
            "original_width": 64,
            "scale": 1.0,
            "pad_left": 0,
            "pad_top": 0,
        },
        confidence=0.5,
        iou=0.7,
        max_det=2,
    )
    assert [row["class_id"] for row in rows] == [1, 0]
    assert [row["confidence"] for row in rows] == pytest.approx([0.95, 0.80])


def test_yolo_nms_keeps_strict_threshold_stable_ties_and_class_boundaries() -> None:
    raw = np.zeros((1, 6, 3), dtype=np.float32)
    raw[0, :4] = np.asarray([
        [1.0, 0.5, 1.0], [1.0, 1.0, 1.0],
        [2.0, 1.0, 2.0], [2.0, 2.0, 2.0],
    ], dtype=np.float32)
    raw[0, 4:] = np.asarray([
        [0.80, 0.80, 0.50], [0.80, 0.01, 0.01],
    ], dtype=np.float32)
    rows = yolo_detections(
        raw,
        {"original_height": 8, "original_width": 8, "scale": 1.0, "pad_left": 0, "pad_top": 0},
        confidence=0.5, iou=0.5, max_det=3,
    )
    assert [row["class_id"] for row in rows] == [0, 1, 0]
    assert np.allclose(
        [row["xyxy"] for row in rows],
        [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 2.0]],
    )
    assert all(row["confidence"] > 0.5 for row in rows)


def _detections_v1_outputs() -> list[np.ndarray]:
    return [
        np.asarray([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0], [0.0] * 4], dtype=np.float32),
        np.asarray([0.90, 0.50, 0.0], dtype=np.float32),
        np.asarray([2, 1, 0], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
    ]


def test_detections_v1_filters_with_strict_runtime_threshold_and_restores_boxes() -> None:
    rows = detections_v1_records(
        _detections_v1_outputs(),
        {"original_height": 10, "original_width": 10, "scale": 2.0, "pad_left": 1, "pad_top": 2},
        confidence=0.5, iou=0.7, max_det=2,
        candidate_confidence=0.01, contract_iou=0.7, contract_max_det=3,
    )
    assert rows == [
        {"class_id": 2, "confidence": pytest.approx(0.9), "xyxy": [0.0, 0.0, 2.0, 2.0]}
    ]


@pytest.mark.parametrize(("mutate", "kwargs", "message"), [
    (lambda values: values[:3], {}, "输出数量"),
    (lambda values: [values[0][:2], *values[1:]], {}, "shape"),
    (lambda values: [values[0].astype(np.float16), *values[1:]], {}, "dtype"),
    (lambda values: values, {"confidence": 0.001}, "低于设备候选阈值"),
    (lambda values: values, {"iou": 0.6}, "IoU"),
    (lambda values: values, {"max_det": 4}, "max_det"),
    (lambda values: [*values[:3], np.asarray([4], dtype=np.int32)], {}, "valid_count"),
])
def test_detections_v1_rejects_contract_drift(mutate, kwargs: dict, message: str) -> None:
    options = {"confidence": 0.5, "iou": 0.7, "max_det": 3}
    options.update(kwargs)
    with pytest.raises(RuntimeError, match=message):
        detections_v1_records(
            mutate(_detections_v1_outputs()),
            {"original_height": 10, "original_width": 10, "scale": 1.0, "pad_left": 0, "pad_top": 0},
            **options,
            candidate_confidence=0.01, contract_iou=0.7, contract_max_det=3,
        )


def test_yolo26_e2e_v1_filters_without_host_nms_and_restores_boxes() -> None:
    output = np.asarray(
        [[
            [1.0, 2.0, 5.0, 6.0, 0.90, 2.0],
            [1.5, 2.5, 5.5, 6.5, 0.80, 2.0],
            [3.0, 4.0, 7.0, 8.0, 0.50, 1.0],
        ]],
        dtype=np.float32,
    )
    rows = yolo26_e2e_v1_records(
        [output],
        {
            "original_height": 10,
            "original_width": 10,
            "scale": 2.0,
            "pad_left": 1,
            "pad_top": 2,
        },
        confidence=0.5,
        max_det=3,
        contract_max_det=3,
        class_count=4,
    )
    assert [row["class_id"] for row in rows] == [2, 2]
    assert [row["confidence"] for row in rows] == pytest.approx([0.9, 0.8])
    assert rows[0]["xyxy"] == [0.0, 0.0, 2.0, 2.0]
    # Both overlapping rows survive: the exported YOLO26 graph owns selection.
    assert len(rows) == 2


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: [value[0].reshape(1, 6)], "shape"),
        (lambda value: [value[0].astype(np.float16)], "dtype"),
        (
            lambda value: [
                np.asarray(
                    [[[1.0, 2.0, 5.0, 6.0, 0.9, 4.0]]], dtype=np.float32
                )
            ],
            "越界",
        ),
    ],
)
def test_yolo26_e2e_v1_rejects_contract_drift(mutate, message: str) -> None:
    output = [
        np.asarray(
            [[[1.0, 2.0, 5.0, 6.0, 0.9, 2.0]]], dtype=np.float32
        )
    ]
    with pytest.raises(RuntimeError, match=message):
        yolo26_e2e_v1_records(
            mutate(output),
            {
                "original_height": 10,
                "original_width": 10,
                "scale": 1.0,
                "pad_left": 0,
                "pad_top": 0,
            },
            confidence=0.5,
            max_det=1,
            contract_max_det=1,
            class_count=4,
        )


def _decoded_candidates_v1_outputs(*, overflow: int = 0) -> list[np.ndarray]:
    boxes = np.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    raw = np.zeros((1, 6, 3), dtype=np.float32)
    raw[0, :4] = np.asarray(
        [
            [1.0, 0.5, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 2.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    raw[0, 4:] = np.asarray(
        [[0.80, 0.80, 0.50], [0.80, 0.01, 0.01]],
        dtype=np.float32,
    )
    return [
        boxes,
        np.asarray([0.80, 0.80, 0.80, 0.50, 0.0], dtype=np.float32),
        np.asarray([0, 1, 0, 0, 0], dtype=np.int32),
        np.asarray([0, 0, 1, 2, 0], dtype=np.int32),
        np.asarray([4], dtype=np.int32),
        np.asarray([overflow], dtype=np.int32),
        raw,
    ]


def test_decoded_candidates_v1_matches_raw_strict_sort_and_nms() -> None:
    outputs = _decoded_candidates_v1_outputs()
    info = {
        "original_height": 8,
        "original_width": 8,
        "scale": 1.0,
        "pad_left": 0,
        "pad_top": 0,
    }

    decoded = decoded_candidates_v1_records(
        outputs,
        info,
        confidence=0.5,
        iou=0.5,
        max_det=3,
        candidate_confidence=0.01,
        candidate_capacity=5,
        anchor_count=3,
        class_count=2,
    )
    raw = yolo_detections(
        outputs[6], info, confidence=0.5, iou=0.5, max_det=3
    )

    assert decoded == raw
    assert [row["class_id"] for row in decoded] == [0, 1, 0]


def test_decoded_candidates_v1_overflow_and_low_threshold_fail_closed() -> None:
    outputs = _decoded_candidates_v1_outputs(overflow=1)
    kwargs = {
        "candidate_confidence": 0.01,
        "candidate_capacity": 5,
        "anchor_count": 3,
        "class_count": 2,
    }
    info = {
        "original_height": 8,
        "original_width": 8,
        "scale": 1.0,
        "pad_left": 0,
        "pad_top": 0,
    }

    with pytest.raises(RuntimeError, match="溢出"):
        decoded_candidates_v1_records(
            outputs, info, confidence=0.5, iou=0.5, max_det=3, **kwargs
        )
    outputs[5][0] = 0
    with pytest.raises(RuntimeError, match="低阈值"):
        decoded_candidates_v1_records(
            outputs, info, confidence=0.001, iou=0.5, max_det=3, **kwargs
        )


def test_decoded_output_copy_plan_selects_only_raw_for_low_threshold() -> None:
    normal = decoded_output_copy_plan(0.5, candidate_confidence=0.01)
    fallback = decoded_output_copy_plan(0.001, candidate_confidence=0.01)

    assert normal["force_raw"] is False
    assert normal["metadata_indices"] == (4, 5)
    assert fallback["force_raw"] is True
    assert fallback["raw_index"] == 6


def _single_anchor_raw(class_scores: list[float]) -> np.ndarray:
    raw = np.zeros((1, 4 + len(class_scores), 1), dtype=np.float32)
    raw[0, :4, 0] = [10.0, 10.0, 4.0, 4.0]
    raw[0, 4:, 0] = class_scores
    return raw


def test_shared_dual_head_parses_two_logical_outputs_without_double_timing() -> None:
    backend = AscendAclBackend.__new__(AscendAclBackend)
    backend.is_shared_dual_head = True
    backend.logical_heads = {
        "old": {"output_index": 0},
        "new": {"output_index": 1, "candidate_confidence": 0.9},
    }
    backend._last_timings = {}

    old, new = backend._dual_results_from_outputs(
        [
            _single_anchor_raw([0.1, 0.9, 0.2]),
            _single_anchor_raw([0.8]),
        ],
        {
            "original_height": 32,
            "original_width": 32,
            "scale": 1.0,
            "pad_left": 0,
            "pad_top": 0,
        },
        {"conf": 0.5, "iou": 0.7, "max_det": 300},
        1.25,
        {
            "inference_ms": 4.5,
            "submit_ms": 0.2,
            "wait_ms": 0.3,
            "input_copy_ms": 0.4,
            "output_copy_ms": 0.5,
        },
    )

    assert old.records[0]["class_id"] == 1
    assert not new.records
    assert old.speed["preprocess"] == 1.25
    assert old.speed["inference"] == 4.5
    assert old.speed["ascend_output_copy"] == 0.5
    assert new.speed["preprocess"] == 0.0
    assert new.speed["inference"] == 0.0
    assert "ascend_output_copy" not in new.speed


def test_shared_dual_execution_handle_resolves_physical_model_once() -> None:
    calls = {"model": 0, "parse": 0}

    class ModelHandle:
        def result(self):
            calls["model"] += 1
            return ["old", "new"], {"inference_ms": 1.0}

    class Backend:
        def _dual_results_from_outputs(self, outputs, *_args):
            calls["parse"] += 1
            return tuple(outputs)

    handle = AscendDualDetectorExecutionHandle(
        Backend(), ModelHandle(), {}, {}, 0.0
    )

    first = handle.result()
    second = handle.result()

    assert first == ("old", "new")
    assert second is first
    assert calls == {"model": 1, "parse": 1}
