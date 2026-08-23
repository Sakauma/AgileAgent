from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fair_agent.backends.ascend_acl import (
    AscendEncodedPreprocessor,
    context_tensor,
    detector_tensor,
    validate_dvpp_scene_resize_stages,
    yolo26_e2e_v1_records,
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


@pytest.mark.parametrize("color_type", [0, 2, 6])
def test_encoded_preprocessor_accepts_fixed_competition_png(
    color_type: int,
) -> None:
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
        fixed_png_header(color_type=4),
    ],
)
def test_encoded_preprocessor_rejects_other_image_contracts(
    payload: bytes,
) -> None:
    assert not AscendEncodedPreprocessor.accepts(payload)


def test_detector_tensor_uses_current_310b_shape() -> None:
    image = Image.new("RGB", (640, 512), "black")
    tensor, info = detector_tensor(image, 608, 736)

    assert tensor.shape == (1, 3, 608, 736)
    assert tensor.dtype == np.float32
    assert info["pad_left"] == 0
    assert info["pad_top"] == 9


def test_detector_tensor_reuses_exact_rgb_array() -> None:
    image = Image.new("RGB", (640, 512), (15, 127, 240))
    rgb = np.ascontiguousarray(np.asarray(image))
    expected, expected_info = detector_tensor(image, 608, 736)
    actual, actual_info = detector_tensor(image, 608, 736, rgb)

    assert np.array_equal(actual, expected)
    assert actual_info == expected_info


def test_detector_tensor_emits_contiguous_uint8_nhwc_for_aipp() -> None:
    image = Image.new("RGB", (640, 512), (15, 127, 240))
    tensor, info = detector_tensor(
        image,
        608,
        736,
        input_mode="nhwc_uint8_aipp",
    )

    assert tensor.shape == (1, 608, 736, 3)
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


def test_yolo26_e2e_filters_without_host_nms_and_restores_boxes() -> None:
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
    assert len(rows) == 2


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: [value[0].reshape(1, 6)], "shape"),
        (lambda value: [value[0].astype(np.float16)], "dtype"),
        (
            lambda value: [
                np.asarray(
                    [[[1.0, 2.0, 5.0, 6.0, 0.9, 4.0]]],
                    dtype=np.float32,
                )
            ],
            "越界",
        ),
    ],
)
def test_yolo26_e2e_rejects_contract_drift(mutate, message: str) -> None:
    output = [
        np.asarray(
            [[[1.0, 2.0, 5.0, 6.0, 0.9, 2.0]]],
            dtype=np.float32,
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
