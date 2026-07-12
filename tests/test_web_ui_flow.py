from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from PIL import Image
from starlette.testclient import TestClient

from fair_agent.modules.web_inference import content_task_id, image_png_bytes
from fair_agent.web.app import BatchResultStore, build_web_settings, create_app


class FakeEngine:
    def __init__(self) -> None:
        self.calls = []

    def queue_status(self):
        return {"waiting": 0, "active": False, "completed": len(self.calls)}

    def predict(self, image, filename, confidence=0.50, task_id=None, incremental_protocol=None):
        self.calls.append((filename, confidence, task_id, incremental_protocol))
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
            "confidence_threshold": float(confidence),
            "inference_ms": 18.2,
            "queue_wait_ms": 0.4,
            "agent": {
                "mode": "automatic_orchestration",
                "models_used": ["scene_sensor_net_v1", "unified_yolo11s_v1", "incremental_model_bank_v1"],
                "protocol": None,
                "protocols": [],
                "decision": {
                    "mode": "automatic",
                    "evaluated_specialists": 3,
                    "base_detection_count": 1,
                    "final_detection_count": 1,
                    "activated_classes": [],
                    "reason": "test",
                },
            },
            "annotated_png": image_png_bytes(image),
        }


def client_with_engine() -> tuple[TestClient, FakeEngine]:
    engine = FakeEngine()
    return TestClient(create_app(engine_provider=lambda: engine)), engine


def png(color: str = "white") -> bytes:
    return image_png_bytes(Image.new("RGB", (32, 24), color))


def test_batch_result_store_evicts_old_batches() -> None:
    store = BatchResultStore(max_items=1)
    first = store.put(b"first", [])
    second = store.put(b"second", [])
    assert store.get(first) is None
    assert store.get(second)["archive"] == b"second"


def test_health_and_static_product_contract() -> None:
    client, _engine = client_with_engine()
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["device"] == "cuda:0"
    assert health.json()["limits"]["max_batch_files"] == 20
    capabilities = client.get("/api/capabilities")
    assert capabilities.status_code == 200
    assert len(capabilities.json()["models"]) == 3
    assert any(item["available"] for item in capabilities.json()["protocols"])

    page = client.get("/")
    assert page.status_code == 200
    assert "智能目标检测" in page.text
    assert "灵动Agent" in page.text
    assert '<span class="brand-glyph">灵</span>' in page.text
    assert "AgileAgent 首页" not in page.text
    assert "批量检测" in page.text
    assert "会话记录" in page.text
    for private_term in ["SHA256SUMS", "lock-all", "增量协议", "部署门禁"]:
        assert private_term not in page.text
    for internal_hint in ["不会写入训练数据", "当前会话处理", "任务ID", "GPU推理队列", "边界框 XYXY"]:
        assert internal_hint not in page.text
    assert page.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in page.headers["content-security-policy"]


def test_web_settings_follow_active_config_and_manifest() -> None:
    settings = build_web_settings()
    assert settings["detector_path"].name == "yolo11s_ir_sar_imgsz640.pt"
    assert settings["device_index"] == "0"
    assert settings["predict"] == {"imgsz": 640, "iou": 0.7, "max_det": 300}
    assert settings["protocols"]["p01_new_small_aircraft"]["available"] is False
    assert settings["protocols"]["p02_new_warship"]["available"] is True
    assert settings["protocols"]["p02_new_warship"]["consensus_iou"] == 0.30
    assert "allowed_scenes" not in settings["protocols"]["p02_new_warship"]


def test_health_reports_model_initialization_failure() -> None:
    def broken_provider():
        raise RuntimeError("test failure")

    client = TestClient(create_app(engine_provider=broken_provider))
    response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["status"] == "error"
    assert "模型服务初始化失败" in response.json()["error"]


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
    assert "task_id" not in payload
    assert payload["detection_count"] == 1
    assert payload["context"]["sensor"] == "sar"
    assert "annotated_base64" in payload
    assert "annotated_png" not in payload
    assert payload["inference_ms"] == 18.2
    assert payload["confidence_threshold"] == 0.21
    assert payload["system_total_ms"] >= 0
    assert engine.calls == [("sample.png", 0.21, content_task_id(image), "auto")]


def test_single_detection_ignores_manual_protocol_and_uses_agent_auto_mode() -> None:
    client, engine = client_with_engine()
    image = png()
    response = client.post(
        "/api/detect",
        files={"file": ("sample.png", image, "image/png")},
        data={"confidence": "0.20", "incremental_protocol": "p02_new_warship"},
    )
    assert response.status_code == 200
    assert engine.calls == [("sample.png", 0.20, content_task_id(image), "auto")]


def test_single_detection_defaults_to_half_confidence() -> None:
    client, engine = client_with_engine()
    response = client.post(
        "/api/detect",
        files={"file": ("sample.png", png(), "image/png")},
    )
    assert response.status_code == 200
    assert engine.calls[0][1] == 0.50
    assert engine.calls[0][3] == "auto"


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
        data={"confidence": "1.01"},
    )
    assert confidence.status_code == 400
    assert "0.01到1.00" in confidence.json()["error"]
    assert not engine.calls


def test_single_detection_accepts_full_confidence_upper_bound() -> None:
    client, engine = client_with_engine()
    response = client.post(
        "/api/detect",
        files={"file": ("sample.png", png(), "image/png")},
        data={"confidence": "1.00"},
    )
    assert response.status_code == 200
    assert engine.calls[0][1] == 1.00


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
    payload = response.json()
    assert payload["image_count"] == 2
    assert payload["detection_count"] == 2
    assert payload["inference_ms"] == 36.4
    assert payload["system_total_ms"] >= 0
    assert len(payload["results"]) == 2
    assert payload["results"][0]["preview_url"].endswith("/preview/0")
    assert "annotated_png" not in payload["results"][0]
    assert len(engine.calls) == 2
    preview = client.get(payload["results"][0]["preview_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    download = client.get(payload["download_url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
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
    for capability in ["sessionStorage", "dataTransfer.files", "finally"]:
        assert capability in script
    for internal_term in ["crypto.subtle.digest", "已校验", "singleHash"]:
        assert internal_term not in script
    assert "annotated_base64" in script
    assert "incremental_protocol" not in script
    assert "incrementalProtocol" not in html
    assert "增量演示" not in html
    assert "Agent 自动决策" in script
    assert "纯推理时间" in script
    assert "系统总用时" in script
    assert "本次阈值" in script
    assert "batchPreviewList" in html
    assert "batchPreviewSummary" in html
    assert "batchDetectionRows" in html
    assert "preview_url" in script
    assert "download_url" in script
    assert "renderDetectionRows" in script
    assert html.count('max="1.00"') == 2
    assert "openHistoryItem" in script
    assert "resultCache" in script
    assert "collaborationFlow" in html
    assert 'class="agent-decision-panel is-hidden"' in html
    assert "本次识别过程" in html
    assert "基础候选" in script
    assert "通过" in script and "拒绝" in script
    assert "@media (max-width: 620px)" in styles
    assert ".main-nav.is-open" in styles
    assert ".history-row.is-available" in styles
    assert "streamlit" not in (html + script + styles).lower()
    for internal_hint in ["不会写入训练数据", "当前会话处理", "任务ID", "GPU推理队列", "边界框 XYXY"]:
        assert internal_hint not in (html + script)
