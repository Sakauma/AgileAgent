from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from PIL import Image
from starlette.testclient import TestClient

from fair_agent.modules.web_inference import content_task_id, image_png_bytes
from fair_agent.web.app import create_app


class FakeEngine:
    def __init__(self) -> None:
        self.calls = []

    def queue_status(self):
        return {"waiting": 0, "active": False, "completed": len(self.calls)}

    def predict(self, image, filename, confidence=0.15, task_id=None):
        self.calls.append((filename, confidence, task_id))
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
            "annotated_png": image_png_bytes(image),
        }


def client_with_engine() -> tuple[TestClient, FakeEngine]:
    engine = FakeEngine()
    return TestClient(create_app(engine_provider=lambda: engine)), engine


def png(color: str = "white") -> bytes:
    return image_png_bytes(Image.new("RGB", (32, 24), color))


def test_health_and_static_product_contract() -> None:
    client, _engine = client_with_engine()
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["device"] == "cuda:0"
    assert health.json()["limits"]["max_batch_files"] == 20

    page = client.get("/")
    assert page.status_code == 200
    assert "智能目标检测" in page.text
    assert "批量检测" in page.text
    assert "会话记录" in page.text
    for private_term in ["SHA256SUMS", "lock-all", "增量协议", "部署门禁"]:
        assert private_term not in page.text
    for internal_hint in ["不会写入训练数据", "当前会话处理", "任务ID", "GPU推理队列", "边界框 XYXY"]:
        assert internal_hint not in page.text
    assert page.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in page.headers["content-security-policy"]


def test_single_detection_api_returns_public_json() -> None:
    client, engine = client_with_engine()
    image = png()
    response = client.post(
        "/api/detect",
        files={"file": ("sample.png", image, "image/png")},
        data={"confidence": "0.21"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == content_task_id(image)
    assert payload["detection_count"] == 1
    assert payload["context"]["sensor"] == "sar"
    assert "annotated_base64" in payload
    assert "annotated_png" not in payload
    assert engine.calls == [("sample.png", 0.21, content_task_id(image))]


def test_single_detection_rejects_invalid_upload_and_confidence() -> None:
    client, engine = client_with_engine()
    invalid = client.post(
        "/api/detect",
        files={"file": ("sample.txt", b"not-an-image", "text/plain")},
        data={"confidence": "0.15"},
    )
    assert invalid.status_code == 400
    assert "无法读取图像" in invalid.json()["error"]
    confidence = client.post(
        "/api/detect",
        files={"file": ("sample.png", png(), "image/png")},
        data={"confidence": "0.99"},
    )
    assert confidence.status_code == 400
    assert "0.01到0.80" in confidence.json()["error"]
    assert not engine.calls


def test_single_detection_rejects_oversized_multipart() -> None:
    client, engine = client_with_engine()
    response = client.post(
        "/api/detect",
        files={"file": ("oversized.png", b"x" * (20 * 1024 * 1024 + 1), "image/png")},
        data={"confidence": "0.15"},
    )
    assert response.status_code == 400
    assert "超过 20MB" in response.json()["error"]
    assert not engine.calls


def test_batch_api_returns_archive_and_summary_headers() -> None:
    client, engine = client_with_engine()
    response = client.post(
        "/api/batch",
        files=[
            ("files", ("first.png", png("white"), "image/png")),
            ("files", ("second.png", png("black"), "image/png")),
        ],
        data={"confidence": "0.18"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["x-image-count"] == "2"
    assert response.headers["x-detection-count"] == "2"
    assert float(response.headers["x-elapsed-ms"]) == 36.4
    assert len(engine.calls) == 2
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert len([name for name in archive.namelist() if name.startswith("annotated/")]) == 2
        summary = json.loads(archive.read("results.json"))
    assert summary["image_count"] == 2


def test_batch_api_rejects_duplicate_content() -> None:
    client, engine = client_with_engine()
    image = png()
    response = client.post(
        "/api/batch",
        files=[
            ("files", ("first.png", image, "image/png")),
            ("files", ("copy.png", image, "image/png")),
        ],
    )
    assert response.status_code == 400
    assert "重复图像" in response.json()["error"]
    assert not engine.calls


def test_custom_frontend_has_complete_interaction_contract() -> None:
    html = Path("fair_agent/web/static/index.html").read_text(encoding="utf-8")
    script = Path("fair_agent/web/static/assets/app.js").read_text(encoding="utf-8")
    styles = Path("fair_agent/web/static/assets/app.css").read_text(encoding="utf-8")
    assert '<script src="/assets/app.js" defer></script>' in html
    for endpoint in ["/api/health", "/api/detect", "/api/batch"]:
        assert endpoint in script
    for capability in ["crypto.subtle.digest", "sessionStorage", "dataTransfer.files", "finally"]:
        assert capability in script
    assert "annotated_base64" in script
    assert "@media (max-width: 620px)" in styles
    assert ".main-nav.is-open" in styles
    assert "streamlit" not in (html + script + styles).lower()
    for internal_hint in ["不会写入训练数据", "当前会话处理", "任务ID", "GPU推理队列", "边界框 XYXY"]:
        assert internal_hint not in (html + script)
