from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fair_agent.modules.ascend_preflight import (
    LetterboxInfo,
    _latency_summary,
    _postprocess_yolo,
    _topologically_sort_onnx_graph,
    context_tensor,
    decode_png_rgb,
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


def test_opencv_png_decode_matches_pillow_exactly(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    import numpy as np

    array = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    path = tmp_path / "sample.png"
    Image.fromarray(array, mode="RGB").save(path)
    pillow = decode_png_rgb(path, backend="pillow")
    opencv = decode_png_rgb(path, backend="opencv")
    assert pillow.flags.c_contiguous
    assert opencv.flags.c_contiguous
    assert np.array_equal(pillow, opencv)


def test_yolo_candidate_prefilter_is_exact_and_does_not_mutate_raw() -> None:
    pytest.importorskip("ultralytics")
    import numpy as np

    raw = np.zeros((1, 6, 7), dtype=np.float32)
    raw[0, :4, :] = np.asarray(
        [
            [10.0, 10.5, 40.0, 70.0, 30.0, 50.0, 5.0],
            [10.0, 10.5, 40.0, 70.0, 20.0, 50.0, 5.0],
            [4.0, 4.0, 8.0, 6.0, 5.0, 5.0, 2.0],
            [4.0, 4.0, 8.0, 6.0, 5.0, 5.0, 2.0],
        ]
    )
    raw[0, 4:, :] = np.asarray(
        [
            [0.90, 0.80, 0.05, 0.001, 0.70, 0.60, 0.001],
            [0.10, 0.20, 0.85, 0.50, 0.20, 0.30, 0.001],
        ]
    )
    original = raw.copy()
    info = LetterboxInfo(80, 80, 80, 80, 1.0, 0, 0, 0, 0)
    baseline = _postprocess_yolo(
        raw,
        info,
        confidence=0.01,
        iou=0.5,
        max_det=20,
        candidate_prefilter=False,
    )
    optimized = _postprocess_yolo(
        raw,
        info,
        confidence=0.01,
        iou=0.5,
        max_det=20,
        candidate_prefilter=True,
    )
    assert optimized == baseline
    assert np.array_equal(raw, original)


def test_latency_summary_reports_fps_and_percentiles() -> None:
    result = _latency_summary([20.0, 25.0, 30.0, 35.0])
    assert result["mean_ms"] == pytest.approx(27.5)
    assert result["fps_from_mean"] == pytest.approx(1000.0 / 27.5)
    assert result["p95_ms"] >= result["p50_ms"]


def test_onnx_topological_sort_repairs_out_of_order_cast_style_graph() -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("Add", ["relu_output", "bias"], ["result"], name="add"),
            helper.make_node("Relu", ["images"], ["relu_output"], name="relu"),
        ],
        "out_of_order",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [1])],
        [helper.make_tensor("bias", TensorProto.FLOAT, [1], [1.0])],
    )
    _topologically_sort_onnx_graph(graph)
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    assert [node.name for node in graph.node] == ["relu", "add"]
