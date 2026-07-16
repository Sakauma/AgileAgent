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
            "image_width": image.width,
            "image_height": image.height,
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
                "models_used": ["scene_sensor_net_v1", "three_class_base_detector"],
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
        }

    def predict_batch(self, items, confidence=0.50, incremental_protocol=None):
        return [
            self.predict(image, filename, confidence, task_id, incremental_protocol)
            for image, filename, task_id in items
        ]


class IncrementalSourceRouter:
    def __init__(self, incremental_ids=()) -> None:
        self.incremental_ids = set(incremental_ids)

    def resolve(self, task_id):
        incremental = task_id in self.incremental_ids
        return {
            "source_scope": "incremental" if incremental else "base",
            "inference_scope": "incremental" if incremental else "base",
            "incremental_protocol": "auto" if incremental else None,
            "known": True,
            "rejected": False,
            "reason": "test_incremental" if incremental else "test_base",
            "generation_id": "generation-1" if incremental else None,
            "batch_id": "batch-1" if incremental else None,
        }


def client_with_engine(source_router=None) -> tuple[TestClient, FakeEngine]:
    engine = FakeEngine()
    return TestClient(create_app(
        engine_provider=lambda: engine,
        inference_source_router=source_router,
    )), engine


def png(color: str = "white") -> bytes:
    return image_png_bytes(Image.new("RGB", (32, 24), color))


def test_batch_result_store_evicts_old_batches() -> None:
    store = BatchResultStore(max_items=1, ttl_seconds=1800, max_bytes=1024)
    first = store.put([])
    second = store.put([])
    assert store.get(first) is None
    assert store.get(second)["results"] == []


def test_batch_result_store_evicts_by_total_bytes() -> None:
    store = BatchResultStore(max_items=4, ttl_seconds=1800, max_bytes=70)
    first = store.put([{"source_bytes": b"a" * 40}])
    second = store.put([{"source_bytes": b"b" * 40}])
    assert store.get(first) is None
    assert store.get(second) is not None
    assert store.total_bytes <= 70


def test_health_and_static_product_contract() -> None:
    client, _engine = client_with_engine()
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["device"] == "cuda:0"
    assert health.json()["limits"]["max_batch_files"] == 20
    public_config = client.get("/api/config/public")
    assert public_config.status_code == 200
    assert public_config.json()["confidence"] == {"min": 0.01, "max": 1.0, "default": 0.5}
    assert "registry" not in json.dumps(public_config.json())
    capabilities = client.get("/api/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["generation_id"] == "incremental_detection_generation"
    assert capabilities.json()["generation_name"] == "增量检测生产代际"
    assert capabilities.json()["active_classes"] == ["soldier", "small_aircraft", "warship", "tank"]
    assert len(capabilities.json()["models"]) == 3
    assert capabilities.json()["protocols"][0]["available"] is True

    page = client.get("/")
    assert page.status_code == 200
    assert "智能目标检测" in page.text
    assert "灵动Agent" in page.text
    assert '<span class="brand-glyph">灵</span>' in page.text
    assert "AgileAgent 首页" not in page.text
    assert "批量检测" in page.text
    assert "会话记录" in page.text
    assert 'id="incrementalClassList"' in page.text
    assert 'id="saveIncrementalClasses"' in page.text
    script = client.get("/assets/app.js")
    assert script.status_code == 200
    assert "/classes`" in script.text
    assert "source_class_id" in script.text
    for private_term in ["SHA256SUMS", "lock-all", "增量协议", "部署门禁"]:
        assert private_term not in page.text
    for internal_hint in ["不会写入训练数据", "当前会话处理", "任务ID", "GPU推理队列", "边界框 XYXY"]:
        assert internal_hint not in page.text
    assert page.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in page.headers["content-security-policy"]


def test_web_settings_follow_active_config_and_manifest() -> None:
    settings = build_web_settings()
    assert settings["detector_path"].name == "three_class_base_detector.pt"
    assert settings["device_index"] == "0"
    assert settings["predict"] == {
            "imgsz": 640, "specialist_imgsz": 512, "iou": 0.7, "max_det": 300, "batch_size": 20,
            "warmup_iterations": 1, "warmup_batch_size": 1,
        "warmup_width": 640, "warmup_height": 512,
        "confidence_default": 0.5, "preload_specialists": True,
        "quantize": None, "cudnn_benchmark": True, "compile": False,
    }
    assert settings["generation_id"] == "incremental_detection_generation"
    assert settings["generation_name"] == "增量检测生产代际"
    assert settings["base_model_id"] == "three_class_base_detector"
    assert settings["base_model_name"] == "三类基础检测器"
    assert settings["active_class_ids"] == [0, 1, 2, 3]
    assert settings["base_local_to_global"] == {0: 0, 1: 1, 2: 3}
    assert list(settings["protocols"]) == ["incremental_detector"]
    assert settings["protocols"]["incremental_detector"]["activation_threshold"] == 0.63
    assert settings["class_names"] == {0: "soldier", 1: "small_aircraft", 2: "warship", 3: "tank"}


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
    assert "annotated_base64" not in payload
    assert payload["image_width"] == 32
    assert payload["image_height"] == 24
    assert "annotated_png" not in payload
    assert payload["inference_ms"] == 18.2
    assert payload["confidence_threshold"] == 0.21
    assert payload["system_total_ms"] >= 0
    assert engine.calls == [("sample.png", 0.21, content_task_id(image), None)]
    assert payload["agent"]["decision"]["source_scope"] == "unknown"
    assert payload["agent"]["decision"]["inference_scope"] == "base"


def test_single_detection_ignores_manual_protocol_and_uses_agent_auto_mode() -> None:
    client, engine = client_with_engine()
    image = png()
    response = client.post(
        "/api/detect",
        files={"file": ("sample.png", image, "image/png")},
        data={"confidence": "0.20", "incremental_protocol": "p02_new_warship"},
    )
    assert response.status_code == 200
    assert engine.calls == [("sample.png", 0.20, content_task_id(image), None)]


def test_single_detection_defaults_to_half_confidence() -> None:
    client, engine = client_with_engine()
    response = client.post(
        "/api/detect",
        files={"file": ("sample.png", png(), "image/png")},
    )
    assert response.status_code == 200
    assert engine.calls[0][1] == 0.50
    assert engine.calls[0][3] is None


def test_registered_incremental_input_activates_agent_route() -> None:
    image = png("navy")
    task_id = content_task_id(image)
    client, engine = client_with_engine(IncrementalSourceRouter({task_id}))

    response = client.post(
        "/api/detect",
        files={"file": ("incremental.png", image, "image/png")},
    )

    assert response.status_code == 200
    assert engine.calls[0][3] == "auto"
    decision = response.json()["agent"]["decision"]
    assert decision["source_scope"] == "incremental"
    assert decision["source_generation_id"] == "generation-1"


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
    assert "20MB" in response.json()["error"]
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
    assert set(payload["timings"]) == {
        "upload_parse_ms", "decode_ms", "queue_wait_ms", "batch_engine_ms", "cache_store_ms"
    }
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


def test_batch_api_groups_base_and_incremental_sources() -> None:
    base_image = png("white")
    incremental_image = png("navy")
    router = IncrementalSourceRouter({content_task_id(incremental_image)})
    client, engine = client_with_engine(router)

    response = client.post(
        "/api/batch",
        files=[
            ("files", ("base.png", base_image, "image/png")),
            ("files", ("incremental.png", incremental_image, "image/png")),
        ],
    )

    assert response.status_code == 200
    assert sorted(call[3] or "base" for call in engine.calls) == ["auto", "base"]
    scopes = [row["agent"]["decision"]["source_scope"] for row in response.json()["results"]]
    assert scopes == ["base", "incremental"]


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
    assert "annotated_base64" not in script
    assert "/api/config/public" in script
    assert "drawDetectionCanvas" in script
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
    assert html.count('max="1.00"') == 0
    assert "openHistoryItem" in script
    assert "resultCache" in script
    assert "collaborationFlow" in html
    assert 'class="agent-decision-panel is-hidden"' in html
    assert "本次识别过程" in html
    assert "基础候选" in script
    assert "已激活" in script and "未激活" in script
    assert "@media (max-width: 620px)" in styles
    assert ".main-nav.is-open" in styles
    assert ".history-row.is-available" in styles
    assert "streamlit" not in (html + script + styles).lower()
    for internal_hint in ["不会写入训练数据", "当前会话处理", "任务ID", "GPU推理队列", "边界框 XYXY"]:
        assert internal_hint not in (html + script)
