from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fair_agent.backends.ascend_acl import detector_tensor, yolo_detections


def test_detector_tensor_uses_static_310b_shape() -> None:
    image = Image.new("RGB", (640, 512), "black")
    tensor, info = detector_tensor(image, 736, 896)
    assert tensor.shape == (1, 3, 736, 896)
    assert tensor.dtype == np.float32
    assert info["pad_left"] == 0
    assert info["pad_top"] == 9


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
