from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fair_agent.backends.ascend_acl import (
    AscendEncodedPreprocessor,
    context_tensor,
    detector_tensor,
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
