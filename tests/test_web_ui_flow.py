from __future__ import annotations

import copy
import io
import json
import zipfile
from pathlib import Path

from PIL import Image
from starlette.testclient import TestClient

from fair_agent.core.config import load_config
from fair_agent.modules.web_inference import image_png_bytes
from fair_agent.web.app import BatchResultStore, build_web_settings, create_app


class FakeEngine:
    def __init__(self) -> None:
        self.calls = []

    def queue_status(self):
        return {"waiting": 0, "active": False, "completed": len(self.calls)}

    def predict(self, image, filename, confidence=0.50, incremental_protocol=None):
        self.calls.append((filename, confidence, incremental_protocol))
        return {
            "filename": filename,
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
                "models_used": [
                    "scene_sensor_net_v1",
                    "four_class_base_detector",
                ],
                "protocol": None,
                "protocols": [],
                "decision": {
                    "mode": "automatic",
                    "input_mode": "unlabeled_image",
                    "inference_scope": "production",
                    "routing_basis": "image_content_and_active_generation",
                    "evaluated_specialists": 1,
                    "base_detection_count": 1,
                    "final_detection_count": 1,
                    "activated_classes": [],
                    "reason": "test",
                },
            },
        }

    def predict_batch(self, items, confidence=0.50, incremental_protocol=None):
        return [
            self.predict(image, filename, confidence, incremental_protocol)
            for image, filename in items
        ]


class EncodedFakeEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.encoded_calls = []

    def accepts_encoded(self, data: bytes) -> bool:
        return data.startswith(b"encoded-test")

    def predict_encoded(
        self, data, filename, confidence=0.50, incremental_protocol=None
    ):
        self.encoded_calls.append(
            (data, filename, confidence, incremental_protocol)
        )
        return self.predict(
            Image.new("RGB", (640, 512)),
            filename,
            confidence,
            incremental_protocol,
        )

    def predict_encoded_batch(
        self, items, confidence=0.50, incremental_protocol=None
    ):
        return [
            self.predict_encoded(
                data,
                filename,
                confidence,
                incremental_protocol,
            )
            for data, filename in items
        ]


def client_with_engine() -> tuple[TestClient, FakeEngine]:
    engine = FakeEngine()
    return TestClient(create_app(
        engine_provider=lambda: engine,
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
    assert "limits" not in health.json()
    public_config = client.get("/api/config/public")
    assert public_config.status_code == 200
    assert public_config.json()["confidence"] == {"min": 0.01, "max": 1.0, "default": 0.01}
    assert public_config.json()["runtime"]["architecture"] == "x86"
    assert public_config.json()["runtime"]["model_format"] == "pt"
    assert "limits" not in public_config.json()
    assert "registry" not in json.dumps(public_config.json())
    capabilities = client.get("/api/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["generation_id"] == (
        "incremental_detection_generation_4plus2"
    )
    assert capabilities.json()["generation_name"] == "4+2 增量检测生产代际"
    assert capabilities.json()["active_classes"] == [
        "soldier",
        "small_aircraft",
        "warship",
        "tank",
        "patrol_boat",
        "armored_vehicle",
    ]
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
    assert settings["detector_path"].name == "four_class_base_detector.pt"
    assert settings["device_index"] == "0"
    assert settings["runtime_platform"]["architecture"] == "x86"
    assert settings["runtime_platform"]["model_format"] == "pt"
    assert settings["model_artifacts"]["base"].suffix == ".pt"
    assert settings["model_artifacts"]["context"].suffix == ".pt"
    assert settings["model_artifacts"]["specialists"]["incremental_detector"].suffix == ".pt"
    assert settings["predict"] == {
            "imgsz": 1280, "specialist_imgsz": 1280, "iou": 0.7, "max_det": 300, "batch_size": 32,
            "warmup_iterations": 1, "warmup_batch_size": 1,
        "warmup_width": 1280, "warmup_height": 1280,
        "confidence_default": 0.01, "preload_specialists": True,
        "quantize": None, "cudnn_benchmark": True, "compile": False,
    }
    assert settings["generation_id"] == "incremental_detection_generation_4plus2"
    assert settings["generation_name"] == "4+2 增量检测生产代际"
    assert settings["base_model_id"] == "four_class_base_detector"
    assert settings["base_model_name"] == "四类冻结基础检测器"
    assert settings["active_class_ids"] == [0, 1, 2, 3, 4, 5]
    assert settings["base_local_to_global"] == {0: 0, 1: 1, 2: 2, 3: 3}
    assert list(settings["protocols"]) == ["incremental_detector"]
    assert settings["protocols"]["incremental_detector"][
        "activation_thresholds"
    ] == {4: 0.57, 5: 0.82}
    assert settings["class_names"] == {
        0: "soldier",
        1: "small_aircraft",
        2: "warship",
        3: "tank",
        4: "patrol_boat",
        5: "armored_vehicle",
    }


def test_health_reports_model_initialization_failure() -> None:
    def broken_provider():
        raise RuntimeError("test failure")

    client = TestClient(create_app(engine_provider=broken_provider))
    response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["status"] == "error"
    assert "模型服务初始化失败" in response.json()["error"]


def test_health_reports_ascend_device_for_ascend_backend() -> None:
    config = copy.deepcopy(load_config())
    config["inference"]["backend"] = "ascend_acl"
    config["ascend_backend"]["validated"] = True
    for entry in config["ascend_backend"]["models"].values():
        entry["sha256"] = "0" * 64
    config["ascend_backend"]["context_model"]["sha256"] = "0" * 64
    settings = build_web_settings(config)
    assert settings["runtime_platform"]["model_format"] == "om"
    assert settings["model_artifacts"]["base"].suffix == ".om"
    assert settings["model_artifacts"]["context"].suffix == ".om"
    assert settings["model_artifacts"]["specialists"]["incremental_detector"].suffix == ".om"
    client = TestClient(create_app(engine_provider=lambda: FakeEngine(), config=config))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["device"] == "ascend:0"
    assert response.json()["model_format"] == "om"


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
    assert engine.calls == [("sample.png", 0.21, "auto")]
    assert payload["agent"]["decision"]["input_mode"] == "unlabeled_image"
    assert payload["agent"]["decision"]["inference_scope"] == "production"


def test_single_detection_uses_encoded_backend_without_cpu_decode(monkeypatch) -> None:
    engine = EncodedFakeEngine()
    client = TestClient(create_app(engine_provider=lambda: engine))

    def fail_decode(*_args, **_kwargs):
        raise AssertionError("CPU decoder must not run for accepted encoded input")

    monkeypatch.setattr("fair_agent.web.app.decode_image_bytes", fail_decode)
    payload = b"encoded-test-png"
    response = client.post(
        "/api/detect",
        files={"file": ("sample.bin", payload, "application/octet-stream")},
        data={"confidence": "0.31"},
    )
    assert response.status_code == 200
    assert engine.encoded_calls == [(payload, "sample.bin", 0.31, "auto")]
    assert response.json()["image_width"] == 640
    assert response.json()["timings"]["decode_ms"] == 0.0


def test_single_detection_keeps_contract_sized_upload_in_memory(monkeypatch) -> None:
    engine = EncodedFakeEngine()
    client = TestClient(create_app(engine_provider=lambda: engine))

    def fail_spooled_file_io(*_args, **_kwargs):
        raise AssertionError("contract-sized detection upload used disk-backed I/O")

    monkeypatch.setattr("starlette.datastructures.run_in_threadpool", fail_spooled_file_io)
    payload = b"encoded-test" + b"x" * (1024 * 1024 + 4096)
    response = client.post(
        "/api/detect",
        files={"file": ("large.png", payload, "image/png")},
        data={"confidence": "0.31"},
    )

    assert response.status_code == 200
    assert engine.encoded_calls == [(payload, "large.png", 0.31, "auto")]


def test_single_detection_falls_back_when_encoded_backend_rejects_input() -> None:
    engine = EncodedFakeEngine()
    client = TestClient(create_app(engine_provider=lambda: engine))
    response = client.post(
        "/api/detect",
        files={"file": ("sample.png", png(), "image/png")},
    )
    assert response.status_code == 200
    assert engine.encoded_calls == []
    assert engine.calls == [("sample.png", 0.01, "auto")]


def test_single_detection_does_not_filter_filename_extension_or_mime() -> None:
    client, engine = client_with_engine()
    response = client.post(
        "/api/detect",
        files={"file": ("official-input.bin", png(), "application/octet-stream")},
    )
    assert response.status_code == 200
    assert engine.calls == [("official-input.bin", 0.01, "auto")]


def test_single_detection_ignores_manual_protocol_and_uses_agent_auto_mode() -> None:
    client, engine = client_with_engine()
    image = png()
    response = client.post(
        "/api/detect",
        files={"file": ("sample.png", image, "image/png")},
        data={"confidence": "0.20", "incremental_protocol": "round_01_patrol_boat"},
    )
    assert response.status_code == 200
    assert engine.calls == [("sample.png", 0.20, "auto")]


def test_single_detection_uses_frozen_default_confidence() -> None:
    client, engine = client_with_engine()
    response = client.post(
        "/api/detect",
        files={"file": ("sample.png", png(), "image/png")},
    )
    assert response.status_code == 200
    assert engine.calls[0][1] == 0.01
    assert engine.calls[0][2] == "auto"


def test_every_unlabeled_input_uses_active_production_generation() -> None:
    image = png("navy")
    client, engine = client_with_engine()

    response = client.post(
        "/api/detect",
        files={"file": ("incremental.png", image, "image/png")},
    )

    assert response.status_code == 200
    assert engine.calls[0][2] == "auto"
    decision = response.json()["agent"]["decision"]
    assert decision["inference_scope"] == "production"
    assert "source_scope" not in decision


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


def test_batch_fast_multipart_preserves_file_order_and_confidence() -> None:
    from fair_agent.web.app import parse_small_batch_multipart

    boundary = "AgileAgentBatchFastPath"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="files"; filename="one.bin"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + b"first" + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="files"; filename="two.bin"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + b"second" + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="confidence"\r\n\r\n'
        "0.31"
        f"\r\n--{boundary}--\r\n"
    ).encode()

    rows, confidence = parse_small_batch_multipart(
        body, f"multipart/form-data; boundary={boundary}"
    )

    assert rows == [("one.bin", b"first"), ("two.bin", b"second")]
    assert all(isinstance(data, memoryview) for _filename, data in rows)
    assert confidence == "0.31"


def test_batch_api_uses_encoded_backend_without_cpu_decode(monkeypatch) -> None:
    engine = EncodedFakeEngine()
    client = TestClient(create_app(engine_provider=lambda: engine))

    def fail_decode(*_args, **_kwargs):
        raise AssertionError("CPU batch decoder must not run for encoded input")

    monkeypatch.setattr("fair_agent.web.app.decode_batch_images", fail_decode)
    first = b"encoded-test-first"
    second = b"encoded-test-second"
    response = client.post(
        "/api/batch",
        files=[
            ("files", ("first.bin", first, "application/octet-stream")),
            ("files", ("second.bin", second, "application/octet-stream")),
        ],
        data={"confidence": "0.29"},
    )

    assert response.status_code == 200
    assert engine.encoded_calls == [
        (first, "first.bin", 0.29, "auto"),
        (second, "second.bin", 0.29, "auto"),
    ]
    payload = response.json()
    assert payload["timings"]["decode_ms"] == 0.0
    assert [row["filename"] for row in payload["results"]] == [
        "first.bin",
        "second.bin",
    ]


def test_batch_api_falls_back_entire_batch_when_one_encoded_input_is_rejected() -> None:
    engine = EncodedFakeEngine()
    first = png("white")
    second = png("black")
    engine.accepts_encoded = lambda data: data == first
    client = TestClient(create_app(engine_provider=lambda: engine))
    response = client.post(
        "/api/batch",
        files=[
            ("files", ("first.png", first, "image/png")),
            ("files", ("second.png", second, "image/png")),
        ],
    )

    assert response.status_code == 200
    assert engine.encoded_calls == []
    assert [call[0] for call in engine.calls] == ["first.png", "second.png"]


def test_batch_api_rejects_empty_batch() -> None:
    client, engine = client_with_engine()
    response = client.post("/api/batch", files=[])

    assert response.status_code == 400
    assert "至少一张图像" in response.json()["error"]
    assert engine.calls == []


def test_batch_api_uses_production_generation_for_every_image() -> None:
    client, engine = client_with_engine()
    base_image = png("white")
    incremental_image = png("navy")

    response = client.post(
        "/api/batch",
        files=[
            ("files", ("base.png", base_image, "image/png")),
            ("files", ("incremental.png", incremental_image, "image/png")),
        ],
    )

    assert response.status_code == 200
    assert [call[2] for call in engine.calls] == ["auto", "auto"]
    assert all(
        row["agent"]["decision"]["inference_scope"] == "production"
        for row in response.json()["results"]
    )


def test_batch_api_does_not_classify_inputs_by_content_identity() -> None:
    client, engine = client_with_engine()
    image = png()
    response = client.post(
        "/api/batch",
        files=[
            ("files", ("first.png", image, "image/png")),
            ("files", ("copy.png", image, "image/png")),
        ],
    )
    assert response.status_code == 200
    assert len(engine.calls) == 2


def test_batch_api_has_no_twenty_image_validation_gate() -> None:
    client, engine = client_with_engine()
    image = png()
    response = client.post(
        "/api/batch",
        files=[
            ("files", (f"image-{index}.data", image, "application/octet-stream"))
            for index in range(21)
        ],
    )
    assert response.status_code == 200
    assert len(engine.calls) == 21


def test_custom_frontend_has_complete_interaction_contract() -> None:
    html = Path("fair_agent/web/static/index.html").read_text(encoding="utf-8")
    script = Path("fair_agent/web/static/assets/app.js").read_text(encoding="utf-8")
    styles = Path("fair_agent/web/static/assets/app.css").read_text(encoding="utf-8")
    assert '<script src="/assets/app.js" defer></script>' in html
    for endpoint in ["/api/health", "/api/detect", "/api/batch"]:
        assert endpoint in script
    for capability in ["sessionStorage", "dataTransfer.files", "finally"]:
        assert capability in script
    for internal_term in [
        "crypto.subtle.digest", "已校验", "singleHash", "validateFile",
        "VALID_TYPES", "VALID_EXTENSIONS", "max_file_bytes", "max_batch_files",
    ]:
        assert internal_term not in script
    assert 'id="singleFile" type="file" accept=' not in html
    assert 'id="batchFiles" type="file" accept=' not in html
    assert "annotated_base64" not in script
    assert "/api/config/public" in script
    assert "drawDetectionCanvas" in script
    assert "incremental_protocol" not in script
    assert "incrementalProtocol" not in html
    assert "增量演示" not in html
    assert "advanced-settings" not in html
    assert 'id="confidence"' not in html
    assert 'id="batchConfidence"' not in html
    assert 'form.append("confidence"' not in script
    assert '$("#confidence")' not in script
    assert '$("#batchConfidence")' not in script
    assert "advanced-settings" not in styles
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
