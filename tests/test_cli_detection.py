from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from PIL import Image

from fair_agent import cli
from fair_agent.modules import cli_detection
from fair_agent.modules.cli_detection import (
    DetectionInput,
    create_result_dir,
    discover_detection_inputs,
    render_detection_summary,
    save_detection_results,
)


def image_bytes(color: str = "black") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32), color).save(buffer, format="PNG")
    return buffer.getvalue()


def fake_result(filename: str) -> dict:
    return {
        "filename": filename,
        "image_width": 64,
        "image_height": 32,
        "context": {"sensor": "ir", "scene": "forest"},
        "detections": [
            {
                "class_id": 3,
                "class_name": "tank",
                "confidence": 0.875,
                "xyxy": [8.0, 4.0, 40.0, 20.0],
                "source": "frozen_base_model",
            }
        ],
        "class_counts": {"tank": 1},
        "detection_count": 1,
        "inference_ms": 12.5,
        "agent": {
            "decision": {
                "class_owners": {"3": "four_class_base_detector"},
            }
        },
    }


def test_discover_detection_inputs_supports_single_and_recursive_directory(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    nested = tmp_path / "nested" / "second.jpg"
    nested.parent.mkdir()
    first.write_bytes(image_bytes())
    nested.write_bytes(image_bytes("white"))
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    source, single = discover_detection_inputs(first)
    assert source == first.resolve()
    assert [item.name for item in single] == ["first.png"]

    _source, shallow = discover_detection_inputs(tmp_path)
    assert [item.name for item in shallow] == ["first.png"]

    _source, recursive = discover_detection_inputs(tmp_path, recursive=True)
    assert [item.name for item in recursive] == ["first.png", "nested/second.jpg"]


def test_detection_results_are_saved_as_human_and_machine_readable_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.png"
    data = image_bytes()
    source.write_bytes(data)
    run_dir = create_result_dir(tmp_path / "result", source)
    payload = save_detection_results(
        run_dir,
        source,
        [DetectionInput(source, source.name)],
        [data],
        [fake_result(source.name)],
        transport="local_api",
    )

    assert payload["image_count"] == 1
    assert payload["detection_count"] == 1
    assert payload["class_counts"] == {"tank": 1}
    assert (run_dir / "annotated/001_input.png").is_file()
    assert (run_dir / "predictions/001_input.txt").read_text(encoding="utf-8").startswith("3 ")
    saved = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert saved["transport"] == "local_api"
    assert saved["results"][0]["artifacts"]["annotated_image"] == "annotated/001_input.png"
    with (run_dir / "detections.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["class_name"] == "tank"
    assert rows[0]["owner_model"] == "four_class_base_detector"
    summary = (run_dir / "summary.txt").read_text(encoding="utf-8")
    assert "灵动 Agent · 识别完成" in summary
    assert "坦克 (tank) × 1" in summary
    assert str(run_dir) in summary


def test_detection_summary_limits_terminal_rows() -> None:
    result = fake_result("sample.png")
    result["detections"] = result["detections"] * 3
    result["detection_count"] = 3
    payload = {
        "source": "/tmp/sample.png",
        "transport": "direct_engine",
        "image_count": 1,
        "detection_count": 3,
        "class_counts": {"tank": 3},
        "saved_to": "/tmp/result",
        "results": [result],
    }
    rendered = render_detection_summary(payload, max_detection_rows=1)
    assert "本进程直接推理" in rendered
    assert "另有 2 条记录" in rendered


def test_detect_command_reuses_local_service_and_saves_results(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "sample.png"
    source.write_bytes(image_bytes())
    output = tmp_path / "output"
    config = {
        "inference": {
            "backend": "ascend_acl",
            "confidence_default": 0.05,
            "confidence_min": 0.05,
            "confidence_max": 0.95,
        },
        "decoding": {"backend": "pillow"},
        "performance": {"request_timeout_seconds": 180},
    }
    monkeypatch.setattr(cli, "load_args_config", lambda _args: config)
    monkeypatch.setattr(
        cli_detection,
        "probe_local_detection_api",
        lambda _config: "http://127.0.0.1:18501",
    )
    monkeypatch.setattr(
        cli_detection,
        "detect_via_local_api",
        lambda _base_url, _data, filename, timeout: fake_result(filename),
    )
    args = argparse.Namespace(
        source=str(source),
        recursive=False,
        output=str(output),
        format="json",
    )
    assert cli.cmd_detect(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["transport"] == "local_api"
    assert payload["detection_count"] == 1
    assert (output / "results.json").is_file()
    assert (output / "annotated/001_sample.png").is_file()
