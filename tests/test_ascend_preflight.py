from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fair_agent.modules.ascend_preflight import (
    _latency_summary,
    context_tensor,
    detector_tensor,
    fixed_rect_shape,
    production_onnx_plan,
    restore_xyxy,
)


def test_fixed_rect_shapes_match_current_ultralytics_protocol() -> None:
    assert fixed_rect_shape(640, 512, 896) == (736, 896)
    assert fixed_rect_shape(640, 512, 640) == (512, 640)


def test_production_plan_is_static_batch_one(tmp_path: Path) -> None:
    rows = production_onnx_plan(tmp_path, shape_mode="rect")
    assert [row.model_id for row in rows] == [
        "base_detector",
        "incremental_detector",
        "scene_sensor_net",
    ]
    assert [row.input_shape for row in rows] == [
        (1, 3, 736, 896),
        (1, 3, 512, 640),
        (1, 3, 160, 160),
    ]


def test_detector_letterbox_golden_padding_and_box_roundtrip() -> None:
    pytest.importorskip("cv2")
    import numpy as np

    image = np.zeros((512, 640, 3), dtype=np.uint8)
    square, square_info = detector_tensor(image, 896, 896)
    assert square.shape == (3, 896, 896)
    assert (
        square_info.pad_left,
        square_info.pad_top,
        square_info.pad_right,
        square_info.pad_bottom,
    ) == (0, 89, 0, 90)

    rectangular, rectangular_info = detector_tensor(image, 736, 896)
    assert rectangular.shape == (3, 736, 896)
    assert (
        rectangular_info.pad_left,
        rectangular_info.pad_top,
        rectangular_info.pad_right,
        rectangular_info.pad_bottom,
    ) == (0, 9, 0, 10)

    original = [32.0, 48.0, 320.0, 400.0]
    projected = [
        original[0] * rectangular_info.scale + rectangular_info.pad_left,
        original[1] * rectangular_info.scale + rectangular_info.pad_top,
        original[2] * rectangular_info.scale + rectangular_info.pad_left,
        original[3] * rectangular_info.scale + rectangular_info.pad_top,
    ]
    assert restore_xyxy(projected, rectangular_info) == pytest.approx(original)


def test_context_preprocessing_matches_pytorch_evaluation_transform() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("torchvision")
    import numpy as np

    from fair_agent.models.context import evaluation_transform

    array = np.arange(512 * 640 * 3, dtype=np.uint8).reshape(512, 640, 3)
    image = Image.fromarray(array, mode="RGB")
    expected = evaluation_transform(160)(image).numpy()
    actual = context_tensor(image, 160)
    assert actual.shape == (3, 160, 160)
    assert np.max(np.abs(actual - expected)) <= 1e-6


def test_latency_summary_reports_fps_and_percentiles() -> None:
    result = _latency_summary([20.0, 25.0, 30.0, 35.0])
    assert result["mean_ms"] == pytest.approx(27.5)
    assert result["fps_from_mean"] == pytest.approx(1000.0 / 27.5)
    assert result["p95_ms"] >= result["p50_ms"]
