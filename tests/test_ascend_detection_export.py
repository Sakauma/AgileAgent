from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from fair_agent.modules.ascend_detection_export import (
    build_detections_v1_module,
    export_detections_v1_onnx,
)


class RawIdentity(torch.nn.Module):
    def forward(self, predictions):
        return predictions


def _raw_predictions():
    raw = torch.zeros((1, 6, 3), dtype=torch.float32)
    raw[0, :4] = torch.tensor(
        [
            [1.0, 0.5, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 2.0],
            [2.0, 2.0, 2.0],
        ]
    )
    raw[0, 4:] = torch.tensor(
        [
            [0.80, 0.80, 0.50],
            [0.80, 0.01, 0.01],
        ]
    )
    return raw


def test_detections_v1_export_module_preserves_strict_nms_semantics() -> None:
    module = build_detections_v1_module(
        RawIdentity(),
        class_count=2,
        candidate_confidence=0.5,
        iou_threshold=0.5,
        max_det=3,
    )

    boxes, scores, class_ids, valid_count = module(_raw_predictions())

    assert valid_count.tolist() == [3]
    assert class_ids.tolist() == [0, 1, 0]
    assert scores.tolist() == pytest.approx([0.8, 0.8, 0.8])
    assert np.allclose(
        boxes.numpy(),
        [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 2.0]],
    )


def test_detections_v1_probe_exports_without_onnx_package(tmp_path) -> None:
    module = build_detections_v1_module(
        RawIdentity(),
        class_count=2,
        candidate_confidence=0.5,
        iou_threshold=0.5,
        max_det=3,
    )
    target = tmp_path / "probe.onnx"

    result = export_detections_v1_onnx(
        module,
        _raw_predictions(),
        target,
        input_name="raw_predictions",
    )

    assert target.is_file()
    assert result["bytes"] == target.stat().st_size
    assert len(result["sha256"]) == 64
    assert result["output_names"] == [
        "boxes",
        "scores",
        "class_ids",
        "valid_count",
    ]
