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
    normalize_user_path_text,
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


class FakeLocalDetectionApiClient:
    def __init__(self, _base_url: str, _timeout: float) -> None:
        pass

    def __enter__(self) -> "FakeLocalDetectionApiClient":
        return self

    def __exit__(self, *_exc) -> None:
        pass

    def detect(self, _data: bytes, filename: str) -> tuple[dict, float]:
        result = fake_result(filename)
        result["system_total_ms"] = 15.0
        result["timings"] = {"engine_total_ms": 13.0}
        return result, 18.0

    def detect_batch(
        self,
        rows: list[tuple[str, bytes]],
    ) -> tuple[dict, float]:
        results = [fake_result(filename) for filename, _data in rows]
        return {
            "image_count": len(rows),
            "detection_count": len(rows),
            "inference_ms": 12.5 * len(rows),
            "system_total_ms": 15.0 * len(rows),
            "timings": {"batch_engine_ms": 13.0 * len(rows)},
            "results": results,
        }, 18.0 * len(rows)


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


def test_windows_paths_are_normalized_for_wsl() -> None:
    assert normalize_user_path_text(
        r"\\wsl.localhost\ubuntu2004\home\sakauma\dataset\image.png",
        host_os="posix",
    ) == "/home/sakauma/dataset/image.png"
    assert normalize_user_path_text(
        r"\\wsl$\Ubuntu-20.04\mnt\d\dataset\image.png",
        host_os="posix",
    ) == "/mnt/d/dataset/image.png"
    assert normalize_user_path_text(
        r"D:\dataset\image.png",
        host_os="posix",
    ) == "/mnt/d/dataset/image.png"
    assert normalize_user_path_text(
        r"images\nested\image.png",
        host_os="posix",
    ) == "images/nested/image.png"


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
    assert payload["statistics"]["images_with_detections"] == 1
    assert payload["statistics"]["images_without_detections"] == 0
    assert payload["statistics"]["detections_per_image"] == 1.0
    assert payload["statistics"]["average_processing_ms"] == 12.5
    assert payload["statistics"]["estimated_throughput_fps"] == 80.0
    assert payload["statistics"]["average_confidence"] == 0.875
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
        "performance": {
            "end_to_end_inference_ms": 50.0,
            "end_to_end_inference_fps": 20.0,
        },
        "results": [result],
    }
    rendered = render_detection_summary(payload, max_detection_rows=1)
    assert "统计摘要" in rendered
    assert "端到端推理时间" in rendered
    assert "50.0 ms" in rendered
    assert "端到端推理 FPS" in rendered
    assert "20.00 FPS" in rendered
    assert "处理耗时" not in rendered
    assert "平均耗时" not in rendered
    assert "耗时(ms)" not in rendered
    assert "检测明细" in rendered
    assert "另有 2 条记录" in rendered


def test_batch_detection_summary_only_prints_aggregate_statistics() -> None:
    first = fake_result("first.png")
    second = fake_result("second.png")
    second["context"] = {"sensor": "sar", "scene": "sea"}
    empty = fake_result("empty.png")
    empty["detections"] = []
    empty["class_counts"] = {}
    empty["detection_count"] = 0
    payload = {
        "source": "/tmp/images",
        "transport": "local_api",
        "image_count": 3,
        "detection_count": 2,
        "class_counts": {"tank": 2},
        "saved_to": "/tmp/result",
        "results": [first, second, empty],
    }

    rendered = render_detection_summary(payload)

    assert "批量识别完成" in rendered
    assert "统计摘要" in rendered
    assert "类别分布" in rendered
    assert "上下文分布" in rendered
    assert "有目标" in rendered
    assert "未检出" in rendered
    assert "检测明细" not in rendered
    assert "图像概览" not in rendered
    assert "first.png" not in rendered
    assert "second.png" not in rendered
    assert "empty.png" not in rendered


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
        "LocalDetectionApiClient",
        FakeLocalDetectionApiClient,
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
    assert payload["performance"]["strategy"] == "single_api"
    assert payload["performance"]["end_to_end_inference_ms"] == 13.0
    assert payload["performance"]["end_to_end_inference_fps"] == 76.923
    assert payload["performance"]["cli_total_wall_ms"] > 0
    assert (output / "results.json").is_file()
    assert (output / "annotated/001_sample.jpg").is_file()


def test_batch_detect_command_prints_progress_without_filenames_and_summary_only(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "images"
    source.mkdir()
    for name in ("first.png", "second.png"):
        (source / name).write_bytes(image_bytes())
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
        "LocalDetectionApiClient",
        FakeLocalDetectionApiClient,
    )
    args = argparse.Namespace(
        source=str(source),
        recursive=False,
        output=str(output),
        format="text",
    )

    assert cli.cmd_detect(args) == 0
    captured = capsys.readouterr()
    assert "批量识别完成" in captured.out
    assert "统计摘要" in captured.out
    assert "端到端推理时间" in captured.out
    assert "端到端推理 FPS" in captured.out
    assert "识别墙钟" not in captured.out
    assert "结果收尾" not in captured.out
    assert "服务处理" not in captured.out
    assert "检测明细" not in captured.out
    assert "批量识别进度" in captured.err
    assert "first.png" not in captured.out + captured.err
    assert "second.png" not in captured.out + captured.err
    saved = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert saved["performance"]["strategy"] == "chunked_batch_api"
    assert saved["performance"]["measurement_scope"] == (
        "competition_end_to_end_inference"
    )
    assert saved["performance"]["timing_source"] == "api_engine_total_ms"
    assert saved["performance"]["end_to_end_inference_ms"] == 26.0
    assert saved["performance"]["end_to_end_inference_fps"] == 76.923
    assert saved["performance"]["cli_total_wall_ms"] > 0
    assert saved["performance"]["batch_size"] == 2
    assert saved["performance"]["request_count"] == 1
    assert [item["filename"] for item in saved["results"]] == [
        "first.png",
        "second.png",
    ]
