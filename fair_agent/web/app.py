from __future__ import annotations

import base64
import threading
from pathlib import Path
from typing import Any, Callable, Dict

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


ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
DETECTOR_PATH = ROOT / "models" / "base" / "yolo11s_ir_sar_imgsz640.pt"
CONTEXT_PATH = ROOT / "models" / "context" / "scene_sensor_net.pt"
EngineProvider = Callable[[], WebInferenceEngine]

_engine: WebInferenceEngine | None = None
_engine_lock = threading.Lock()


def default_engine_provider() -> WebInferenceEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = WebInferenceEngine(DETECTOR_PATH, CONTEXT_PATH, device_index="0")
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
    queue = {"waiting": 0, "active": False, "completed": 0}
    if provider is default_engine_provider and _engine is not None:
        queue = _engine.queue_status()
    return JSONResponse(
        {
            "status": "ready",
            "device": "cuda:0",
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


async def detect(request: Request) -> JSONResponse:
    try:
        async with request.form(max_files=1, max_fields=4, max_part_size=MAX_FILE_BYTES) as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise ValueError("请选择一张图像。")
            data = await upload.read()
            image, task_id = validate_image_bytes(data, upload.filename or "image")
            confidence = parse_confidence(form.get("confidence", 0.15))
        provider: EngineProvider = request.app.state.engine_provider
        engine = await run_in_threadpool(provider)
        result = await run_in_threadpool(
            engine.predict,
            image,
            upload.filename or "image",
            confidence,
            task_id,
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
