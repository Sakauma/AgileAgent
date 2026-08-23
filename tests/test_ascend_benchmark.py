from __future__ import annotations

import runpy
import struct
from pathlib import Path

import pytest


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
