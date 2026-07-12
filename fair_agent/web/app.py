from __future__ import annotations

import base64
import json
import threading
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
    CLASS_NAMES,
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
    protocols: Dict[str, Dict[str, Any]] = {}
    class_ids = {name: class_id for class_id, name in CLASS_NAMES.items()}
    for item in manifest.get("incremental_models", []):
        protocol_id = str(item["protocol"])
        new_class = protocol_id.split("_new_", 1)[-1]
        if new_class not in class_ids:
            continue
        accepted = item.get("acceptance") == "passed"
        protocols[protocol_id] = {
            "id": protocol_id,
            "new_class": new_class,
            "global_class_id": class_ids[new_class],
            "weights": resolve_path(item["path"]),
            "new_map50": float(item["new_map50"]),
            "krr": float(item["krr"]),
            "available": accepted or not require_passed,
        }
    return {
        "detector_path": resolve_path(web.get("detector_weights", config["model"]["weights"])),
        "context_path": context_path,
        "device_index": str(config.get("runtime", {}).get("default_device", "0")),
        "predict": dict(web.get("predict", {})),
        "incremental_enabled": bool(web.get("incremental_demo", {}).get("enabled", True)),
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
                )
    return _engine


def public_result(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = {key: value for key, value in result.items() if key != "annotated_png"}
    payload["annotated_base64"] = base64.b64encode(result["annotated_png"]).decode("ascii")
    return payload


def parse_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("置信度参数无效。") from exc
    if not 0.01 <= confidence <= 0.80:
        raise ValueError("置信度必须位于0.01到0.80之间。")
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
                "classes": list(CLASS_NAMES.values()),
            }
        )
    except (RuntimeError, OSError, ValueError) as exc:
        return JSONResponse({"status": "error", "error": f"模型服务初始化失败：{exc}"}, status_code=503)


async def capabilities(_request: Request) -> JSONResponse:
    protocols = [
        {
            "id": item["id"],
            "new_class": item["new_class"],
            "new_map50": item["new_map50"],
            "krr": item["krr"],
            "available": item["available"],
        }
        for item in WEB_SETTINGS["protocols"].values()
    ]
    return JSONResponse(
        {
            "models": [
                {"id": "scene_sensor_net_v1", "name": "场景认知"},
                {"id": "unified_yolo11s_v1", "name": "统一目标检测"},
                {"id": "incremental_model_bank_v1", "name": "增量模型库"},
            ],
            "incremental_enabled": WEB_SETTINGS["incremental_enabled"],
            "protocols": protocols,
        }
    )


async def detect(request: Request) -> JSONResponse:
    try:
        async with request.form(max_files=1, max_fields=4, max_part_size=MAX_FILE_BYTES) as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise ValueError("请选择一张图像。")
            data = await upload.read()
            image, task_id = validate_image_bytes(data, upload.filename or "image")
            confidence = parse_confidence(form.get("confidence", 0.15))
            protocol_value = str(form.get("incremental_protocol", "")).strip()
            incremental_protocol = protocol_value or None
        provider: EngineProvider = request.app.state.engine_provider
        engine = await run_in_threadpool(provider)
        result = await run_in_threadpool(
            engine.predict,
            image,
            upload.filename or "image",
            confidence,
            task_id,
            incremental_protocol,
        )
        return JSONResponse(public_result(result))
    except HTTPException as exc:
        return JSONResponse({"error": multipart_error(exc)}, status_code=400)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except (RuntimeError, OSError) as exc:
        return JSONResponse({"error": f"推理服务暂时不可用：{exc}"}, status_code=503)


async def batch_detect(request: Request) -> Response:
    try:
        async with request.form(
            max_files=MAX_BATCH_FILES,
            max_fields=4,
            max_part_size=MAX_FILE_BYTES,
        ) as form:
            uploads = [item for item in form.getlist("files") if isinstance(item, UploadFile)]
            rows = [(item.filename or "image", await item.read()) for item in uploads]
            validated = validate_batch_uploads(rows)
            confidence = parse_confidence(form.get("confidence", 0.15))
        provider: EngineProvider = request.app.state.engine_provider
        engine = await run_in_threadpool(provider)
        results = []
        for filename, _data, image, task_id in validated:
            result = await run_in_threadpool(engine.predict, image, filename, confidence, task_id)
            results.append(result)
        archive = build_batch_zip(results)
        total_detections = sum(int(item["detection_count"]) for item in results)
        total_elapsed = round(sum(float(item["elapsed_ms"]) for item in results), 1)
        return Response(
            archive,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="agile-agent-results.zip"',
                "X-Image-Count": str(len(results)),
                "X-Detection-Count": str(total_detections),
                "X-Elapsed-Ms": str(total_elapsed),
            },
        )
    except HTTPException as exc:
        return JSONResponse({"error": multipart_error(exc)}, status_code=400)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except (RuntimeError, OSError) as exc:
        return JSONResponse({"error": f"推理服务暂时不可用：{exc}"}, status_code=503)


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
            Mount("/", app=StaticFiles(directory=STATIC_ROOT, html=True), name="static"),
        ],
        middleware=[Middleware(SecurityHeadersMiddleware)],
        exception_handlers={404: not_found},
    )
    application.state.engine_provider = engine_provider or default_engine_provider
    return application


app = create_app()
