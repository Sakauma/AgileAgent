from __future__ import annotations

from PIL import Image
from streamlit.testing.v1 import AppTest

import fair_agent.modules.web_inference as web_inference
from fair_agent.modules.web_inference import content_task_id, image_png_bytes


class FakeEngine:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def predict(self, image, filename, confidence=0.15, task_id=None):
        annotated = image_png_bytes(image)
        return {
            "filename": filename,
            "task_id": task_id,
            "context": {
                "sensor": "sar",
                "sensor_confidence": 0.97,
                "scene": "sea",
                "scene_confidence": 0.88,
            },
            "detections": [
                {"class_id": 2, "class_name": "warship", "confidence": 0.91, "xyxy": [1, 2, 10, 12]}
            ],
            "class_counts": {"warship": 1},
            "detection_count": 1,
            "elapsed_ms": 18.2,
            "queue_wait_ms": 0.4,
            "annotated_png": annotated,
        }


def test_public_web_single_image_flow(monkeypatch) -> None:
    monkeypatch.setattr(web_inference, "WebInferenceEngine", FakeEngine)
    image_bytes = image_png_bytes(Image.new("RGB", (32, 24), "white"))
    app = AppTest.from_file("fair_agent/ui/app.py", default_timeout=20).run()
    assert not list(app.exception)
    assert app.radio[0].options == ["智能检测", "批量检测", "任务记录"]
    app.file_uploader[0].upload("sample.png", image_bytes, "image/png").run()
    assert "开始检测" in [button.label for button in app.button]
    start = next(button for button in app.button if button.label == "开始检测")
    start.click().run()
    assert not list(app.exception)
    assert app.session_state["single_result"]["task_id"] == content_task_id(image_bytes)
    assert app.session_state["single_result"]["detection_count"] == 1
    assert len(app.get("download_button")) == 2
    assert [metric.value for metric in app.metric] == ["SAR", "海域", "1", "18.2 ms"]


def test_public_web_batch_flow_and_session_cleanup(monkeypatch) -> None:
    monkeypatch.setattr(web_inference, "WebInferenceEngine", FakeEngine)
    first = image_png_bytes(Image.new("RGB", (24, 24), "white"))
    second = image_png_bytes(Image.new("RGB", (24, 24), "black"))
    app = AppTest.from_file("fair_agent/ui/app.py", default_timeout=20).run()
    app.radio[0].set_value("批量检测").run()
    app.file_uploader[0].set_value(
        [
            ("first.png", first, "image/png"),
            ("second.png", second, "image/png"),
        ]
    ).run()
    start = next(button for button in app.button if button.label == "开始批量检测")
    start.click().run()
    assert not list(app.exception)
    assert len(app.session_state["batch_results"]) == 2
    assert len(app.session_state["task_history"]) == 2
    assert len(app.get("download_button")) == 1
    clear = next(button for button in app.button if button.label == "清除批量结果")
    clear.click().run()
    assert not list(app.exception)
    assert "batch_results" not in app.session_state
    assert "task_history" not in app.session_state
