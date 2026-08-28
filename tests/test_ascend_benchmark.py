from __future__ import annotations

import runpy
import struct
from pathlib import Path

import pytest

from fair_agent.modules.formal_results import (
    formal_prediction_lines,
    validate_formal_prediction_files,
    write_formal_prediction_files,
)


@pytest.mark.parametrize(
    ("color_type", "expected"),
    [(0, "grayscale"), (2, "rgb"), (6, "rgba")],
)
def test_benchmark_png_contract_accepts_supported_dvpp_formats(
    tmp_path: Path,
    color_type: int,
    expected: str,
) -> None:
    script = runpy.run_path("tools/97_benchmark_ascend_api.py")
    path = tmp_path / f"color-{color_type}.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">IIBBBBB", 640, 512, 8, color_type, 0, 0, 0)
        + b"\x00\x00\x00\x00"
    )

    assert script["validate_png"](path)["color_type"] == expected


def test_score_batch_multipart_uses_files_field_and_strict_fps_gate(
    tmp_path: Path,
) -> None:
    script = runpy.run_path("tools/97_benchmark_ascend_api.py")
    first = tmp_path / "one.png"
    second = tmp_path / "two.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    body = script["batch_multipart_body"](
        [first, second], 0.5, "ScoreBoundary"
    )
    assert body.count(b'name="files"') == 2
    assert b'name="file";' not in body
    assert b'name="confidence"' in body
    assert body.endswith(b"--ScoreBoundary--\r\n")
    assert 20 * 1000.0 / 666.6666666667 < 30.0
    assert 20 * 1000.0 / 666.6666666666 >= 30.0


def test_formal_results_use_six_columns_fixed_decimals_and_empty_files(
    tmp_path: Path,
) -> None:
    detected = {
        "image_width": 640,
        "image_height": 512,
        "detections": [
            {
                "class_id": 5,
                "confidence": 0.875,
                "xyxy": [-10.0, 128.0, 650.0, 384.0],
            }
        ],
    }
    empty = {
        "image_width": 640,
        "image_height": 512,
        "detections": [],
    }

    assert formal_prediction_lines(detected) == [
        "5 0.500000 0.500000 1.000000 0.500000 0.875000"
    ]
    paths = write_formal_prediction_files(
        tmp_path / "formal-results",
        [detected, empty],
        ["first.png", "second.png"],
    )

    assert paths[0].read_text(encoding="utf-8") == (
        "5 0.500000 0.500000 1.000000 0.500000 0.875000\n"
    )
    assert paths[1].read_text(encoding="utf-8") == ""
    assert validate_formal_prediction_files(paths, expected_count=2) is True


def test_aggregate_fps_uses_all_frames_and_all_round_elapsed_time() -> None:
    script = runpy.run_path("tools/97_benchmark_ascend_api.py")
    total_frames, total_elapsed_ms, fps = script["aggregate_fps"](
        [
            {"image_count": 20, "full_pipeline_wall_ms": 500.0},
            {"image_count": 20, "full_pipeline_wall_ms": 625.0},
            {"image_count": 20, "full_pipeline_wall_ms": 375.0},
        ]
    )

    assert total_frames == 60
    assert total_elapsed_ms == 1500.0
    assert fps == 40.0
