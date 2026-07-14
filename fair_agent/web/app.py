from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict

import yaml

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from fair_agent.modules.web_inference import (
    MAX_BATCH_BYTES,
    MAX_BATCH_FILES,
    MAX_FILE_BYTES,
    MAX_IMAGE_PIXELS,
    WebInferenceEngine,
    build_batch_zip,
    validate_batch_uploads,
    validate_image_bytes,
)
from fair_agent.core.config import load_config, resolve_path


STATIC_ROOT = Path(__file__).resolve().parent / "static"
EngineProvider = Callable[[], WebInferenceEngine]

_engine: WebInferenceEngine | None = None
_engine_lock = threading.Lock()


class BatchResultStore:
    def __init__(self, max_items: int = 4, ttl_seconds: int = 1800, max_bytes: int = 512 * 1024 * 1024) -> None:
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self._items: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _result_bytes(results: list[Dict[str, Any]]) -> int:
        image_bytes = sum(len(item.get("annotated_png", b"")) for item in results)
        metadata = [
            {key: value for key, value in item.items() if key not in {"annotated_png", "annotated_image"}}
            for item in results
        ]
        return image_bytes + len(json.dumps(metadata, ensure_ascii=False, default=str).encode("utf-8"))

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return sum(int(item["size_bytes"]) for item in self._items.values())

    def put(self, results: list[Dict[str, Any]]) -> str:
        batch_id = uuid.uuid4().hex
        now = time.monotonic()
        size_bytes = self._result_bytes(results)
        if size_bytes > self.max_bytes:
            raise ValueError("批量结果超过会话缓存容量，请减少单批图像数量。")
        with self._lock:
            self._remove_expired(now)
            self._items[batch_id] = {"created_at": now, "size_bytes": size_bytes, "results": results}
            while len(self._items) > self.max_items or sum(
                int(item["size_bytes"]) for item in self._items.values()
            ) > self.max_bytes:
                self._items.popitem(last=False)
        return batch_id

    def get(self, batch_id: str) -> Dict[str, Any] | None:
        now = time.monotonic()
        with self._lock:
            self._remove_expired(now)
            item = self._items.get(batch_id)
            if item is not None:
                self._items.move_to_end(batch_id)
            return item

    def _remove_expired(self, now: float) -> None:
        expired = [
            batch_id
            for batch_id, item in self._items.items()
            if now - float(item["created_at"]) > self.ttl_seconds
        ]
        for batch_id in expired:
            self._items.pop(batch_id, None)


def build_web_settings() -> Dict[str, Any]:
    config = load_config()
    web = config.get("web", {})
    registry_path = resolve_path(web.get("functional_registry", "configs/functional_models.yaml"))
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    context_entry = next(
        item for item in registry["models"] if item.get("function") == "context_perception"
    )
    context_path = resolve_path(context_entry["artifacts"][0]["path"])
    manifest_path = resolve_path(web.get("model_manifest", "models/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_passed = bool(web.get("incremental_demo", {}).get("require_acceptance_passed", True))
    consensus_iou = float(web.get("incremental_demo", {}).get("auto_consensus_iou", 0.30))
    base = manifest.get("base_model", {})
    raw_class_map = base.get("class_map") or {index: name for index, name in enumerate(base.get("classes", []))}
    base_class_names = {int(class_id): str(name) for class_id, name in raw_class_map.items()}
    class_names = dict(base_class_names)
    protocols: Dict[str, Dict[str, Any]] = {}
    for item in manifest.get("incremental_models", []):
        protocol_id = str(item["protocol"])
        class_name = str(item["class_name"])
        global_class_id = int(item["global_class_id"])
        class_names[global_class_id] = class_name
        mode = str(item["incremental_mode"])
        accepted = item.get("acceptance") == "passed"
        available = bool(item.get("available", accepted)) and (accepted or not require_passed)
        if mode == "class_incremental":
            available = available and global_class_id not in base_class_names
            available = available and item.get("activation_threshold") is not None and bool(item.get("calibration_source"))
        protocols[protocol_id] = {
            "id": protocol_id,
            "class_name": class_name,
            "new_class": class_name,
            "global_class_id": global_class_id,
            "incremental_mode": mode,
            "weights": resolve_path(item["path"]),
            "new_map50": float(item["new_map50"]),
            "krr": float(item["krr"]),
            "available": available,
            "activation_threshold": item.get("activation_threshold"),
            "calibration_source": item.get("calibration_source"),
            "routing_prior": float(item.get("routing_prior", 0.5)),
            "context_prior": dict(item.get("context_prior") or {}),
            "evidence_level": item.get("evidence_level"),
            "consensus_iou": consensus_iou,
        }
    incremental_options = dict(web.get("incremental_demo", {}))
    return {
        "detector_path": resolve_path(web.get("detector_weights", config["model"]["weights"])),
        "context_path": context_path,
        "device_index": str(config.get("runtime", {}).get("default_device", "0")),
        "predict": dict(web.get("predict", {})),
        "incremental_enabled": bool(web.get("incremental_demo", {}).get("enabled", True)),
        "class_names": class_names,
        "base_class_ids": list(base_class_names),
        "routing": incremental_options,
        "cache": dict(web.get("cache", {})),
        "protocols": protocols,
    }


WEB_SETTINGS = build_web_settings()


def default_engine_provider() -> WebInferenceEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = WebInferenceEngine(
                    WEB_SETTINGS["detector_path"],
                    WEB_SETTINGS["context_path"],
                    device_index=WEB_SETTINGS["device_index"],
                    predict_options=WEB_SETTINGS["predict"],
                    incremental_protocols=WEB_SETTINGS["protocols"],
                    class_names=WEB_SETTINGS["class_names"],
                    base_class_ids=WEB_SETTINGS["base_class_ids"],
                    routing_options=WEB_SETTINGS["routing"],
                )
    return _engine


def public_result(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = {key: value for key, value in result.items() if key not in {"annotated_png", "task_id"}}
    payload["annotated_base64"] = base64.b64encode(result["annotated_png"]).decode("ascii")
    return payload


def parse_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("置信度参数无效。") from exc
    if not 0.01 <= confidence <= 1.00:
        raise ValueError("置信度必须位于0.01到1.00之间。")
    return confidence


def multipart_error(exc: HTTPException) -> str:
    detail = str(exc.detail or "")
    if "maximum size" in detail.lower():
        return f"单张图像不能超过{MAX_FILE_BYTES // (1024 * 1024)}MB。"
    if "too many files" in detail.lower():
        return f"单批最多上传{MAX_BATCH_FILES}张图像。"
    return "上传请求不符合文件数量或大小限制。"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; font-src 'self'; object-src 'none'"
        )
        return response


async def health(request: Request) -> JSONResponse:
    provider: EngineProvider = request.app.state.engine_provider
    try:
        engine = await run_in_threadpool(provider)
        queue = engine.queue_status()
        return JSONResponse(
            {
                "status": "ready",
                "device": f'cuda:{WEB_SETTINGS["device_index"]}',
                "queue": queue,
                "limits": {
                    "max_file_mb": MAX_FILE_BYTES // (1024 * 1024),
                    "max_batch_files": MAX_BATCH_FILES,
                    "max_batch_mb": MAX_BATCH_BYTES // (1024 * 1024),
                    "max_image_pixels": MAX_IMAGE_PIXELS,
                },
                "classes": list(WEB_SETTINGS["class_names"].values()),
            }
        )
    except (RuntimeError, OSError, ValueError) as exc:
        return JSONResponse({"status": "error", "error": f"模型服务初始化失败：{exc}"}, status_code=503)


async def capabilities(_request: Request) -> JSONResponse:
    protocols = [
        {
            "id": item["id"],
            "class_name": item["class_name"],
            "incremental_mode": item["incremental_mode"],
            "new_map50": item["new_map50"],
            "krr": item["krr"],
            "available": item["available"],
            "evidence_level": item["evidence_level"],
        }
        for item in WEB_SETTINGS["protocols"].values()
    ]
    return JSONResponse(
        {
            "models": [
                {"id": "scene_sensor_net_v1", "name": "场景认知"},
                {"id": "unified_yolo11s_v1", "name": "统一目标检测"},
                {"id": "incremental_model_bank_v1", "name": "专项增强与增量模型库"},
            ],
            "incremental_enabled": WEB_SETTINGS["incremental_enabled"],
            "protocols": protocols,
        }
    )


async def detect(request: Request) -> JSONResponse:
    request_started = time.perf_counter()
    try:
        async with request.form(max_files=1, max_fields=4, max_part_size=MAX_FILE_BYTES) as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise ValueError("请选择一张图像。")
            data = await upload.read()
            image, task_id = validate_image_bytes(data, upload.filename or "image")
            confidence = parse_confidence(form.get("confidence", 0.50))
        provider: EngineProvider = request.app.state.engine_provider
        engine = await run_in_threadpool(provider)
        result = await run_in_threadpool(
            engine.predict,
            image,
            upload.filename or "image",
            confidence,
            task_id,
            "auto",
        )
        payload = public_result(result)
        payload["system_total_ms"] = round((time.perf_counter() - request_started) * 1000, 1)
        return JSONResponse(payload)
    except HTTPException as exc:
        return JSONResponse({"error": multipart_error(exc)}, status_code=400)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except (RuntimeError, OSError) as exc:
        return JSONResponse({"error": f"推理服务暂时不可用：{exc}"}, status_code=503)


async def batch_detect(request: Request) -> Response:
    request_started = time.perf_counter()
    try:
        async with request.form(
            max_files=MAX_BATCH_FILES,
            max_fields=4,
            max_part_size=MAX_FILE_BYTES,
        ) as form:
            uploads = [item for item in form.getlist("files") if isinstance(item, UploadFile)]
            rows = [(item.filename or "image", await item.read()) for item in uploads]
            validated = validate_batch_uploads(rows)
            confidence = parse_confidence(form.get("confidence", 0.50))
        provider: EngineProvider = request.app.state.engine_provider
        engine = await run_in_threadpool(provider)
        results = []
        for filename, _data, image, task_id in validated:
            result = await run_in_threadpool(engine.predict, image, filename, confidence, task_id, "auto")
            results.append(result)
        total_detections = sum(int(item["detection_count"]) for item in results)
        total_inference = round(sum(float(item["inference_ms"]) for item in results), 1)
        batch_id = request.app.state.batch_store.put(results)
        public_results = []
        for index, item in enumerate(results):
            row = {key: value for key, value in item.items() if key not in {"annotated_png", "task_id"}}
            row["preview_url"] = f"/api/batch/{batch_id}/preview/{index}"
            public_results.append(row)
        system_total = round((time.perf_counter() - request_started) * 1000, 1)
        return JSONResponse(
            {
                "batch_id": batch_id,
                "image_count": len(results),
                "detection_count": total_detections,
                "inference_ms": total_inference,
                "system_total_ms": system_total,
                "results": public_results,
                "download_url": f"/api/batch/{batch_id}/download",
            }
        )
    except HTTPException as exc:
        return JSONResponse({"error": multipart_error(exc)}, status_code=400)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except (RuntimeError, OSError) as exc:
        return JSONResponse({"error": f"推理服务暂时不可用：{exc}"}, status_code=503)


async def batch_preview(request: Request) -> Response:
    item = request.app.state.batch_store.get(request.path_params["batch_id"])
    index = int(request.path_params["index"])
    if item is None or not 0 <= index < len(item["results"]):
        return JSONResponse({"error": "批量预览已过期或不存在。"}, status_code=404)
    return Response(item["results"][index]["annotated_png"], media_type="image/png")


async def batch_download(request: Request) -> Response:
    item = request.app.state.batch_store.get(request.path_params["batch_id"])
    if item is None:
        return JSONResponse({"error": "批量结果包已过期或不存在。"}, status_code=404)
    return Response(
        build_batch_zip(item["results"]),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="lingdong-agent-results.zip"'},
    )


async def not_found(_request: Request, _exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": "请求的资源不存在。"}, status_code=404)


def create_app(engine_provider: EngineProvider | None = None) -> Starlette:
    application = Starlette(
        debug=False,
        routes=[
            Route("/api/health", health, methods=["GET"]),
            Route("/api/capabilities", capabilities, methods=["GET"]),
            Route("/api/detect", detect, methods=["POST"]),
            Route("/api/batch", batch_detect, methods=["POST"]),
            Route("/api/batch/{batch_id:str}/preview/{index:int}", batch_preview, methods=["GET"]),
            Route("/api/batch/{batch_id:str}/download", batch_download, methods=["GET"]),
            Mount("/", app=StaticFiles(directory=STATIC_ROOT, html=True), name="static"),
        ],
        middleware=[Middleware(SecurityHeadersMiddleware)],
        exception_handlers={404: not_found},
    )
    application.state.engine_provider = engine_provider or default_engine_provider
    cache = WEB_SETTINGS.get("cache", {})
    application.state.batch_store = BatchResultStore(
        max_items=int(cache.get("max_items", 4)),
        ttl_seconds=int(cache.get("ttl_seconds", 1800)),
        max_bytes=int(cache.get("max_bytes", 512 * 1024 * 1024)),
    )
    return application


app = create_app()
