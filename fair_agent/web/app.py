from __future__ import annotations

import json
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import yaml

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException, MultiPartParser, parse_options_header
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from fair_agent.modules.web_inference import (
    WebInferenceEngine,
    build_batch_zip,
    render_annotated_png,
    decode_batch_images,
    decode_image_bytes,
)
from fair_agent.core.config import inference_backend_options, load_config, resolve_path
from fair_agent.core.runtime_log import StructuredEventLog, new_trace_id
from fair_agent.modules.incremental_workbench import IncrementalBatchStore, TrainingJobManager


STATIC_ROOT = Path(__file__).resolve().parent / "static"
EngineProvider = Callable[[], WebInferenceEngine]
DETECTION_UPLOAD_SPOOL_MAX_BYTES = 2 * 1024 * 1024
DETECTION_FAST_MULTIPART_MAX_BYTES = DETECTION_UPLOAD_SPOOL_MAX_BYTES + 64 * 1024


class BatchResultStore:
    def __init__(self, max_items: int, ttl_seconds: int, max_bytes: int) -> None:
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self._items: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _result_bytes(results: list[Dict[str, Any]]) -> int:
        image_bytes = sum(len(item.get("source_bytes", b"")) for item in results)
        metadata = [
            {key: value for key, value in item.items() if key not in {"source_bytes", "annotated_png", "annotated_image"}}
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


def build_web_settings(
    config: Mapping[str, Any] | None = None,
    generation_channel: str | None = None,
    generation_id: str | None = None,
) -> Dict[str, Any]:
    config = dict(config or load_config())
    web = config["web"]
    inference = dict(config["inference"])
    routing = dict(config["routing"])
    backend_name = str(inference["backend"])
    backend_options = inference_backend_options(config)
    registry_path = resolve_path(web["functional_registry"])
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    context_entry = next(
        item for item in registry["models"] if item.get("function") == "context_perception"
    )
    context_path = resolve_path(context_entry["artifacts"][0]["path"])
    from fair_agent.modules.generation_management import active_generation_registry
    from fair_agent.modules.model_generations import (
        generation_settings,
        generation_web_settings,
        load_generation_registry,
    )

    loaded_registry = load_generation_registry(active_generation_registry(config))
    generation = (
        generation_settings(loaded_registry, generation_id)
        if generation_id is not None
        else generation_web_settings(
            loaded_registry,
            generation_channel or str(web["generation_channel"]),
        )
    )
    class_names = generation["class_names"]
    base_class_names = {
        int(global_id): class_names[int(global_id)]
        for global_id in generation["base_class_ids"]
    }
    protocols = generation["protocols"]
    if backend_name == "tensorrt_engine" and backend_options.get("precision") == "int8":
        backend_options["engines"] = {
            **dict(backend_options.get("engines") or {}),
            **dict(generation.get("engine_deployments") or {}),
        }
        validation_report = backend_options.get("validation_report")
        if backend_options.get("validated") is True and validation_report:
            validation_payload = json.loads(
                resolve_path(validation_report).read_text(encoding="utf-8")
            )
            if validation_payload.get("accepted") is not True:
                raise ValueError("TensorRT验收报告未通过，不能加载量化阈值。")
            calibrated_thresholds = {
                int(key): float(value)
                for key, value in validation_payload.get("threshold_calibration", {})
                .get("thresholds", {})
                .items()
            }
            for protocol in protocols.values():
                owned = [int(value) for value in protocol.get("global_class_ids", [])]
                thresholds = {
                    int(key): float(value)
                    for key, value in dict(
                        protocol.get("activation_thresholds") or {}
                    ).items()
                }
                for class_id in owned:
                    if class_id in calibrated_thresholds:
                        thresholds[class_id] = calibrated_thresholds[class_id]
                protocol["activation_thresholds"] = thresholds
                if len(owned) == 1:
                    protocol["activation_threshold"] = thresholds[owned[0]]
    return {
        "detector_path": generation["detector_path"],
        "context_path": context_path,
        "device_index": str(config.get("runtime", {}).get("default_device", "0")),
        "backend": backend_name,
        "predict": {
            "imgsz": int(inference["imgsz"]),
            "specialist_imgsz": int(inference["specialist_imgsz"]),
            "iou": float(inference["iou"]),
            "max_det": int(inference["max_det"]),
            "batch_size": int(inference["batch_size"]),
            "warmup_iterations": int(inference["warmup_iterations"]),
            "warmup_batch_size": int(inference["warmup_batch_size"]),
            "warmup_width": int(inference["warmup_width"]),
            "warmup_height": int(inference["warmup_height"]),
            "confidence_default": float(inference["confidence_default"]),
            "preload_specialists": bool(inference["preload_specialists"]),
            "quantize": inference["quantize"],
            "cudnn_benchmark": bool(inference["cudnn_benchmark"]),
            "compile": bool(inference["compile"]),
        },
        "incremental_enabled": bool(routing["incremental_enabled"]),
        "class_names": class_names,
        "active_class_ids": generation["active_class_ids"],
        "base_class_ids": list(base_class_names),
        "base_local_to_global": generation["base_local_to_global"],
        "generation_id": generation["generation_id"],
        "generation_name": generation["generation_name"],
        "generation_status": generation["generation_status"],
        "base_model_id": generation["base_model_id"],
        "base_model_name": generation["base_model_name"],
        "class_owners": generation["class_owners"],
        "routing": routing,
        "decoding": dict(config["decoding"]),
        "storage": dict(config["storage"]),
        "ui": dict(config["ui"]),
        "performance": dict(config["performance"]),
        "logging": dict(config["logging"]),
        "incremental_workbench": dict(config["incremental_workbench"]),
        "native_backend": backend_options,
        "confidence": {
            "min": float(inference["confidence_min"]),
            "max": float(inference["confidence_max"]),
            "default": float(inference["confidence_default"]),
        },
        "protocols": protocols,
    }


WEB_SETTINGS = build_web_settings()


class AtomicEngineProvider:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self._engine: WebInferenceEngine | None = None
        self._settings = build_web_settings(self.config)
        self._lock = threading.RLock()
        self._fallbacks: Dict[str, tuple[WebInferenceEngine | None, Dict[str, Any]]] = {}

    @staticmethod
    def _build_engine(settings: Mapping[str, Any]) -> WebInferenceEngine:
        return WebInferenceEngine(
            settings["detector_path"], settings["context_path"],
            device_index=settings["device_index"], predict_options=settings["predict"],
            incremental_protocols=settings["protocols"], class_names=settings["class_names"],
            base_class_ids=settings["base_class_ids"],
            base_local_to_global=settings.get("base_local_to_global"),
            routing_options=settings["routing"], generation_id=settings["generation_id"],
            base_model_id=settings["base_model_id"], class_owners=settings["class_owners"],
            backend_name=settings["backend"], native_options=settings["native_backend"],
        )

    def get(self) -> WebInferenceEngine:
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = self._build_engine(self._settings)
        return self._engine

    def settings(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._settings)

    def promote(
        self,
        candidate_id: str,
        manifest_path: str,
        shadow_engine: WebInferenceEngine,
        smoke: Mapping[str, Any],
    ) -> Dict[str, Any]:
        with self._lock:
            from fair_agent.modules.generation_management import promote_generation, rollback_generation

            previous_engine = self._engine
            previous_settings = dict(self._settings)
            previous_generation = str(previous_settings["generation_id"])
            promotion = promote_generation(self.config, candidate_id, manifest_path)
            try:
                settings = build_web_settings(self.config)
                if settings["generation_id"] != candidate_id:
                    raise RuntimeError("production注册表已切换，但运行时代际解析不一致。")
            except Exception:
                rollback_generation(self.config, previous_generation)
                self._engine = previous_engine
                self._settings = previous_settings
                raise
            self._fallbacks[previous_generation] = (previous_engine, previous_settings)
            self._engine = shadow_engine
            self._settings = settings
        return {**promotion, "shadow_smoke": dict(smoke), "runtime_swap": "atomic"}

    def rollback(self, target_id: str) -> Dict[str, Any]:
        from fair_agent.modules.generation_management import rollback_generation, shadow_load_generation

        with self._lock:
            current_generation = str(self._settings["generation_id"])
            if current_generation == target_id:
                return {"previous": current_generation, "production": target_id, "runtime_swap": "unchanged"}
            cached = self._fallbacks.get(target_id)
            if cached is None:
                engine, smoke = shadow_load_generation(self.config, target_id)
            else:
                engine, settings = cached
                smoke = {"generation_id": target_id, "source": "cached_runtime"}
            rollback = rollback_generation(self.config, target_id)
            if cached is None:
                settings = build_web_settings(self.config)
            if settings["generation_id"] != target_id:
                raise RuntimeError("注册表已回滚，但运行时代际解析不一致。")
            self._fallbacks[current_generation] = (self._engine, dict(self._settings))
            self._engine = engine
            self._settings = dict(settings)
            return {**rollback, "shadow_smoke": smoke, "runtime_swap": "atomic_rollback"}


_default_runtime_manager = AtomicEngineProvider(load_config())


def default_engine_provider() -> WebInferenceEngine:
    return _default_runtime_manager.get()


def public_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"annotated_png", "source_bytes"}
    }


def request_web_settings(request: Request) -> Dict[str, Any]:
    manager = getattr(request.app.state, "runtime_manager", None)
    if manager is not None:
        return manager.settings()
    return dict(getattr(request.app.state, "web_settings", WEB_SETTINGS))


def parse_confidence(value: Any, settings: Mapping[str, Any] | None = None) -> float:
    bounds = dict((settings or WEB_SETTINGS)["confidence"])
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("置信度参数无效。") from exc
    if not float(bounds["min"]) <= confidence <= float(bounds["max"]):
        raise ValueError(f"置信度必须位于{bounds['min']:.2f}到{bounds['max']:.2f}之间。")
    return confidence


async def parse_detection_form(request: Request):
    """Parse a detection upload while keeping contract-sized PNGs in memory."""
    parser = MultiPartParser(
        request.headers,
        request.stream(),
        max_files=float("inf"),
        max_fields=float("inf"),
    )
    # Starlette <=0.37 uses max_file_size; newer releases renamed the spool limit.
    parser.max_file_size = DETECTION_UPLOAD_SPOOL_MAX_BYTES
    parser.spool_max_size = DETECTION_UPLOAD_SPOOL_MAX_BYTES
    try:
        return await parser.parse()
    except MultiPartException as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def parse_small_multipart(body: bytes, content_type: str) -> tuple[bytes, str, Any]:
    """Extract the single detection file and confidence from a bounded body."""
    _disposition, parameters = parse_options_header(content_type)
    boundary = parameters.get(b"boundary")
    if not isinstance(boundary, bytes) or not boundary or len(boundary) > 200:
        raise ValueError("Multipart请求缺少有效boundary。")
    delimiter = b"--" + boundary
    if not body.startswith(delimiter + b"\r\n"):
        raise ValueError("Multipart请求起始boundary无效。")

    cursor = len(delimiter) + 2
    file_data: bytes | None = None
    filename = "image"
    confidence: Any = None
    while True:
        marker = body.find(b"\r\n" + delimiter, cursor)
        if marker < 0:
            raise ValueError("Multipart请求缺少结束boundary。")
        part = body[cursor:marker]
        header_block, separator, content = part.partition(b"\r\n\r\n")
        if not separator or len(header_block) > 16 * 1024:
            raise ValueError("Multipart分段头无效。")
        headers: dict[bytes, bytes] = {}
        for line in header_block.split(b"\r\n"):
            key, colon, value = line.partition(b":")
            if not colon:
                raise ValueError("Multipart分段头格式无效。")
            headers[key.strip().lower()] = value.strip()
        disposition, options = parse_options_header(headers.get(b"content-disposition", b""))
        if disposition != b"form-data" or b"name" not in options:
            raise ValueError("Multipart分段缺少Content-Disposition name。")
        field_name = options[b"name"].decode("utf-8", "replace")
        if b"filename" in options and field_name == "file" and file_data is None:
            file_data = content
            filename = options[b"filename"].decode("utf-8", "replace") or "image"
        elif field_name == "confidence":
            confidence = content.decode("ascii", "strict")

        boundary_end = marker + 2 + len(delimiter)
        suffix = body[boundary_end:boundary_end + 2]
        if suffix == b"--":
            trailing = body[boundary_end + 2:]
            if trailing not in (b"", b"\r\n"):
                raise ValueError("Multipart结束boundary后存在多余数据。")
            break
        if suffix != b"\r\n":
            raise ValueError("Multipart分段boundary无效。")
        cursor = boundary_end + 2

    if file_data is None:
        raise ValueError("请选择一张图像。")
    return file_data, filename, confidence


async def parse_detection_upload(request: Request) -> tuple[bytes, str, Any]:
    content_length = request.headers.get("content-length")
    content_type = request.headers.get("content-type", "")
    try:
        body_size = int(content_length) if content_length is not None else -1
    except ValueError:
        body_size = -1
    if (
        content_type.lower().startswith("multipart/form-data;")
        and 0 <= body_size <= DETECTION_FAST_MULTIPART_MAX_BYTES
    ):
        return parse_small_multipart(await request.body(), content_type)

    form = await parse_detection_form(request)
    try:
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise ValueError("请选择一张图像。")
        return (
            await upload.read(),
            upload.filename or "image",
            form.get("confidence"),
        )
    finally:
        await form.close()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        trace_id = request.headers.get("X-Request-ID") or new_trace_id("http")
        request.state.trace_id = trace_id
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            if request.url.path.startswith("/api/") and hasattr(request.app.state, "event_log"):
                request.app.state.event_log.append(
                    "http.request.failed", level="error", component="web", trace_id=trace_id,
                    message=str(exc), duration_ms=(time.perf_counter() - started) * 1000,
                    details={"method": request.method, "path": request.url.path},
                )
            raise
        response.headers["X-Request-ID"] = trace_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; font-src 'self'; object-src 'none'"
        )
        if request.url.path.startswith("/api/") and hasattr(request.app.state, "event_log"):
            request.app.state.event_log.append(
                "http.request.completed",
                level="warning" if status_code >= 400 else "info",
                component="web",
                trace_id=trace_id,
                duration_ms=(time.perf_counter() - started) * 1000,
                details={"method": request.method, "path": request.url.path, "status_code": status_code},
            )
        return response


async def health(request: Request) -> JSONResponse:
    provider: EngineProvider = request.app.state.engine_provider
    settings = request_web_settings(request)
    try:
        engine = await run_in_threadpool(provider)
        queue = engine.queue_status()
        return JSONResponse(
            {
                "status": "ready",
                "device": (
                    f'ascend:{settings["device_index"]}'
                    if settings["backend"] == "ascend_acl"
                    else f'cuda:{settings["device_index"]}'
                ),
                "backend": settings["backend"],
                "validation_candidate": bool(
                    settings.get("native_backend", {}).get("validation_candidate", False)
                ),
                "queue": queue,
                "generation_id": settings["generation_id"],
                "generation_name": settings["generation_name"],
                "classes": [
                    settings["class_names"][class_id]
                    for class_id in settings["active_class_ids"]
                ],
            }
        )
    except (RuntimeError, OSError, ValueError) as exc:
        return JSONResponse({"status": "error", "error": f"模型服务初始化失败：{exc}"}, status_code=503)


async def capabilities(request: Request) -> JSONResponse:
    settings = request_web_settings(request)
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
        for item in settings["protocols"].values()
    ]
    models = [
        {"id": "scene_sensor_net_v1", "name": "场景认知", "role": "context_perception"},
        {
            "id": settings["base_model_id"],
            "name": settings["base_model_name"],
            "role": "frozen_base",
        },
    ]
    models.extend(
        {
            "id": item["id"],
            "name": item.get("display_name") or f"{item['class_name']}增量专家",
            "role": "class_incremental_expert",
            "available": item["available"],
        }
        for item in settings["protocols"].values()
    )
    return JSONResponse(
        {
            "generation_id": settings["generation_id"],
            "generation_name": settings["generation_name"],
            "generation_status": settings["generation_status"],
            "active_classes": [
                settings["class_names"][class_id]
                for class_id in settings["active_class_ids"]
            ],
            "models": models,
            "incremental_enabled": settings["incremental_enabled"],
            "protocols": protocols,
        }
    )


def public_config_payload(settings: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    settings = dict(settings or WEB_SETTINGS)
    return {
        "confidence": dict(settings["confidence"]),
        "ui": dict(settings["ui"]),
        "incremental": {
            "max_archive_bytes": int(settings["incremental_workbench"]["max_archive_bytes"]),
            "max_archive_mb": int(settings["incremental_workbench"]["max_archive_bytes"]) // (1024 * 1024),
            "accepted_format": "ZIP",
            "preview_limit": int(settings["incremental_workbench"]["preview_limit"]),
            "job_log_tail_lines": int(settings["incremental_workbench"]["job_log_tail_lines"]),
            "poll_interval_ms": int(settings["incremental_workbench"]["poll_interval_ms"]),
        },
        "labels": {
            "classes": {str(key): value for key, value in settings["class_names"].items()},
            "sensors": {"ir": "红外", "sar": "SAR"},
            "scenes": {"air": "空域", "forest": "林地", "sea": "海域", "urban": "城市场景"},
        },
    }


async def public_config(request: Request) -> JSONResponse:
    return JSONResponse(public_config_payload(request_web_settings(request)))


async def detect(request: Request) -> JSONResponse:
    request_started = time.perf_counter()
    settings = request_web_settings(request)
    decoding = settings["decoding"]
    upload_started = time.perf_counter()
    try:
        data, filename, confidence_value = await parse_detection_upload(request)
        upload_ms = (time.perf_counter() - upload_started) * 1000
        confidence = parse_confidence(
            confidence_value if confidence_value is not None else settings["confidence"]["default"],
            settings,
        )
        provider: EngineProvider = request.app.state.engine_provider
        engine = await run_in_threadpool(provider)
        accepts_encoded = getattr(engine, "accepts_encoded", None)
        if callable(accepts_encoded) and accepts_encoded(data):
            decode_ms = 0.0
            result = await run_in_threadpool(
                engine.predict_encoded,
                data,
                filename,
                confidence,
                "auto",
            )
        else:
            decode_started = time.perf_counter()
            image = decode_image_bytes(
                data, filename, str(decoding["backend"])
            )
            decode_ms = (time.perf_counter() - decode_started) * 1000
            result = await run_in_threadpool(
                engine.predict,
                image,
                filename,
                confidence,
                "auto",
            )
        payload = public_result(result)
        payload.setdefault("timings", {}).update({
            "upload_parse_ms": round(upload_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "queue_wait_ms": float(result.get("queue_wait_ms", 0.0)),
        })
        payload["system_total_ms"] = round((time.perf_counter() - request_started) * 1000, 1)
        request.app.state.event_log.append(
            "inference.single.completed", component="inference", trace_id=request.state.trace_id,
            generation_id=result.get("agent", {}).get("decision", {}).get("generation_id", settings["generation_id"]),
            duration_ms=payload["system_total_ms"],
            details={
                "filename": filename, "detection_count": payload.get("detection_count", 0),
                "inference_ms": payload.get("inference_ms"), "context": payload.get("context"),
                "models_used": payload.get("agent", {}).get("models_used", []),
                "routing_decision": payload.get("agent", {}).get("decision", {}),
            },
        )
        return JSONResponse(payload)
    except HTTPException as exc:
        return JSONResponse({"error": str(exc.detail)}, status_code=400)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except (RuntimeError, OSError) as exc:
        return JSONResponse({"error": f"推理服务暂时不可用：{exc}"}, status_code=503)


async def batch_detect(request: Request) -> Response:
    request_started = time.perf_counter()
    settings = request_web_settings(request)
    decoding = settings["decoding"]
    upload_started = time.perf_counter()
    try:
        async with request.form(max_files=float("inf"), max_fields=float("inf")) as form:
            uploads = [item for item in form.getlist("files") if isinstance(item, UploadFile)]
            rows = [(item.filename or "image", await item.read()) for item in uploads]
            upload_ms = (time.perf_counter() - upload_started) * 1000
            confidence = parse_confidence(form.get("confidence", settings["confidence"]["default"]), settings)
        provider: EngineProvider = request.app.state.engine_provider
        engine = await run_in_threadpool(provider)
        accepts_encoded = getattr(engine, "accepts_encoded", None)
        predict_encoded_batch = getattr(engine, "predict_encoded_batch", None)
        if (
            rows
            and callable(accepts_encoded)
            and callable(predict_encoded_batch)
            and all(accepts_encoded(data) for _filename, data in rows)
        ):
            decode_ms = 0.0
            engine_started = time.perf_counter()
            completed_results = await run_in_threadpool(
                predict_encoded_batch,
                [(data, filename) for filename, data in rows],
                confidence,
                "auto",
            )
        else:
            decode_started = time.perf_counter()
            validated = decode_batch_images(
                rows,
                str(decoding["backend"]),
                int(decoding["workers"]),
            )
            decode_ms = (time.perf_counter() - decode_started) * 1000
            engine_started = time.perf_counter()
            completed_results = await run_in_threadpool(
                engine.predict_batch,
                [(image, filename) for filename, _data, image in validated],
                confidence,
                "auto",
            )
        if len(completed_results) != len(rows):
            raise RuntimeError("批量推理结果数量与输入不一致。")
        engine_ms = (time.perf_counter() - engine_started) * 1000
        for result, (_filename, source_bytes) in zip(completed_results, rows):
            result["source_bytes"] = source_bytes
        total_detections = sum(int(item["detection_count"]) for item in completed_results)
        total_inference = round(sum(float(item["inference_ms"]) for item in completed_results), 1)
        cache_started = time.perf_counter()
        batch_id = request.app.state.batch_store.put(completed_results)
        cache_ms = (time.perf_counter() - cache_started) * 1000
        public_results = []
        for index, item in enumerate(completed_results):
            row = {key: value for key, value in item.items() if key not in {"source_bytes", "annotated_png"}}
            row["preview_url"] = f"/api/batch/{batch_id}/preview/{index}"
            public_results.append(row)
        system_total = round((time.perf_counter() - request_started) * 1000, 1)
        request.app.state.event_log.append(
            "inference.batch.completed", component="inference", trace_id=request.state.trace_id,
            duration_ms=system_total, batch_id=batch_id,
            generation_id=(
                completed_results[0].get("agent", {}).get("decision", {}).get("generation_id", settings["generation_id"])
                if completed_results else settings["generation_id"]
            ),
            details={
                "image_count": len(completed_results), "detection_count": total_detections,
                "inference_ms": total_inference, "engine_ms": round(engine_ms, 3),
                "routing_decisions": [
                    {
                        "filename": item.get("filename"),
                        "models_used": item.get("agent", {}).get("models_used", []),
                        "decision": item.get("agent", {}).get("decision", {}),
                    }
                    for item in completed_results
                ],
            },
        )
        return JSONResponse(
            {
                "batch_id": batch_id,
                "image_count": len(completed_results),
                "detection_count": total_detections,
                "inference_ms": total_inference,
                "system_total_ms": system_total,
                "timings": {
                    "upload_parse_ms": round(upload_ms, 3),
                    "decode_ms": round(decode_ms, 3),
                    "queue_wait_ms": float(completed_results[0].get("queue_wait_ms", 0.0)) if completed_results else 0.0,
                    "batch_engine_ms": round(engine_ms, 3),
                    "cache_store_ms": round(cache_ms, 3),
                },
                "results": public_results,
                "download_url": f"/api/batch/{batch_id}/download",
            }
        )
    except HTTPException:
        return JSONResponse({"error": "无法读取请求中的图像。"}, status_code=400)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except (RuntimeError, OSError) as exc:
        return JSONResponse({"error": f"推理服务暂时不可用：{exc}"}, status_code=503)


async def batch_preview(request: Request) -> Response:
    item = request.app.state.batch_store.get(request.path_params["batch_id"])
    index = int(request.path_params["index"])
    if item is None or not 0 <= index < len(item["results"]):
        return JSONResponse({"error": "批量预览已过期或不存在。"}, status_code=404)
    render_started = time.perf_counter()
    result = item["results"][index]
    payload = render_annotated_png(result["source_bytes"], result.get("detections", []))
    return Response(
        payload,
        media_type="image/png",
        headers={"X-Render-Ms": f"{(time.perf_counter() - render_started) * 1000:.3f}"},
    )


async def batch_download(request: Request) -> Response:
    item = request.app.state.batch_store.get(request.path_params["batch_id"])
    if item is None:
        return JSONResponse({"error": "批量结果包已过期或不存在。"}, status_code=404)
    render_started = time.perf_counter()
    payload = build_batch_zip(item["results"])
    return Response(
        payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="lingdong-agent-results.zip"',
            "X-Render-Ms": f"{(time.perf_counter() - render_started) * 1000:.3f}",
        },
    )


def _incremental_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, KeyError):
        return JSONResponse({"error": "增量批次或训练任务不存在。"}, status_code=404)
    if isinstance(exc, (ValueError, IndexError, FileNotFoundError, zipfile.BadZipFile)):
        return JSONResponse({"error": str(exc) or "增量数据操作失败。"}, status_code=400)
    return JSONResponse({"error": f"增量工作台暂时不可用：{exc}"}, status_code=503)


async def incremental_batches(request: Request) -> JSONResponse:
    store: IncrementalBatchStore = request.app.state.incremental_store
    if request.method == "GET":
        return JSONResponse({"batches": await run_in_threadpool(store.list)})
    settings = request_web_settings(request)["incremental_workbench"]
    try:
        async with request.form(max_files=1, max_fields=3, max_part_size=int(settings["max_archive_bytes"])) as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise ValueError("请选择增量数据ZIP压缩包。")
            result = await run_in_threadpool(
                store.create_stream,
                upload.filename or "incremental.zip",
                upload.file,
                str(form.get("name") or ""),
                str(form.get("class_names") or "") or None,
            )
        return JSONResponse(result, status_code=201 if result["status"] == "AUDITED" else 422)
    except HTTPException:
        return JSONResponse({"error": "无法读取请求中的图像。"}, status_code=400)
    except Exception as exc:
        return _incremental_error(exc)


async def incremental_batch_detail(request: Request) -> JSONResponse:
    try:
        payload = await run_in_threadpool(request.app.state.incremental_store.get, request.path_params["batch_id"])
        return JSONResponse(payload)
    except Exception as exc:
        return _incremental_error(exc)


async def incremental_batch_classes(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        names = body.get("names") if isinstance(body, dict) else None
        if not isinstance(names, dict):
            raise ValueError("请求必须提供按源类别ID索引的names映射。")
        payload = await run_in_threadpool(
            request.app.state.incremental_store.rename_classes,
            request.path_params["batch_id"],
            names,
        )
        return JSONResponse(payload)
    except Exception as exc:
        return _incremental_error(exc)


async def incremental_batch_image(request: Request) -> Response:
    try:
        path = await run_in_threadpool(
            request.app.state.incremental_store.image_path,
            request.path_params["batch_id"],
            int(request.path_params["index"]),
        )
        return FileResponse(path)
    except Exception as exc:
        return _incremental_error(exc)


async def incremental_inject(request: Request) -> JSONResponse:
    try:
        payload = await run_in_threadpool(request.app.state.incremental_store.inject, request.path_params["batch_id"])
        return JSONResponse(payload)
    except Exception as exc:
        return _incremental_error(exc)


async def incremental_train(request: Request) -> JSONResponse:
    try:
        job = await run_in_threadpool(request.app.state.training_manager.start, request.path_params["batch_id"])
        return JSONResponse(job, status_code=202)
    except Exception as exc:
        return _incremental_error(exc)


async def incremental_jobs(request: Request) -> JSONResponse:
    batch_id = request.query_params.get("batch_id")
    return JSONResponse({"jobs": await run_in_threadpool(request.app.state.training_manager.list, batch_id)})


async def incremental_job_detail(request: Request) -> JSONResponse:
    batch_id = request.query_params.get("batch_id")
    if not batch_id:
        return JSONResponse({"error": "缺少batch_id。"}, status_code=400)
    try:
        job = await run_in_threadpool(request.app.state.training_manager.get, batch_id, request.path_params["job_id"])
        return JSONResponse(job)
    except Exception as exc:
        return _incremental_error(exc)


async def incremental_job_logs(request: Request) -> Response:
    batch_id = request.query_params.get("batch_id")
    if not batch_id:
        return JSONResponse({"error": "缺少batch_id。"}, status_code=400)
    try:
        text = await run_in_threadpool(
            request.app.state.training_manager.read_log,
            batch_id,
            request.path_params["job_id"],
            int(request.query_params.get("tail", str(request_web_settings(request)["incremental_workbench"]["job_log_tail_lines"]))),
        )
        return PlainTextResponse(text)
    except Exception as exc:
        return _incremental_error(exc)


async def incremental_job_cancel(request: Request) -> JSONResponse:
    batch_id = request.query_params.get("batch_id")
    if not batch_id:
        return JSONResponse({"error": "缺少batch_id。"}, status_code=400)
    try:
        job = await run_in_threadpool(request.app.state.training_manager.cancel, batch_id, request.path_params["job_id"])
        return JSONResponse(job)
    except Exception as exc:
        return _incremental_error(exc)


async def runtime_logs(request: Request) -> JSONResponse:
    batch_id = request.query_params.get("batch_id")
    if not batch_id:
        return JSONResponse({"error": "Web仅支持查询指定增量批次的操作记录；完整日志请使用CLI。"}, status_code=400)
    try:
        rows = await run_in_threadpool(
            request.app.state.event_log.query,
            limit=int(request.query_params.get("limit", "200")),
            level=request.query_params.get("level"),
            component=request.query_params.get("component"),
            trace_id=request.query_params.get("trace_id"),
            batch_id=batch_id,
            job_id=request.query_params.get("job_id"),
        )
        public_rows = [
            {
                key: row.get(key)
                for key in (
                    "timestamp", "level", "component", "event", "trace_id", "batch_id",
                    "job_id", "duration_ms", "message",
                )
                if row.get(key) is not None
            }
            for row in rows
        ]
        return JSONResponse({"events": public_rows})
    except (TypeError, ValueError) as exc:
        return JSONResponse({"error": f"日志查询参数无效：{exc}"}, status_code=400)


async def not_found(_request: Request, _exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": "请求的资源不存在。"}, status_code=404)


def create_app(
    engine_provider: EngineProvider | None = None,
    incremental_store: IncrementalBatchStore | None = None,
    training_manager: TrainingJobManager | None = None,
    event_log: StructuredEventLog | None = None,
    config: Mapping[str, Any] | None = None,
    runtime_manager: AtomicEngineProvider | None = None,
) -> Starlette:
    effective_config = dict(config or load_config())
    decoding = dict(effective_config["decoding"])
    if decoding["backend"] == "opencv":
        import cv2

        cv2.setNumThreads(int(decoding["opencv_threads"]))
    active_runtime = runtime_manager
    if engine_provider is None and active_runtime is None:
        active_runtime = _default_runtime_manager if config is None else AtomicEngineProvider(effective_config)
    settings = active_runtime.settings() if active_runtime is not None else build_web_settings(effective_config)
    application = Starlette(
        debug=False,
        routes=[
            Route("/api/health", health, methods=["GET"]),
            Route("/api/config/public", public_config, methods=["GET"]),
            Route("/api/capabilities", capabilities, methods=["GET"]),
            Route("/api/detect", detect, methods=["POST"]),
            Route("/api/batch", batch_detect, methods=["POST"]),
            Route("/api/batch/{batch_id:str}/preview/{index:int}", batch_preview, methods=["GET"]),
            Route("/api/batch/{batch_id:str}/download", batch_download, methods=["GET"]),
            Route("/api/incremental/batches", incremental_batches, methods=["GET", "POST"]),
            Route("/api/incremental/batches/{batch_id:str}", incremental_batch_detail, methods=["GET"]),
            Route("/api/incremental/batches/{batch_id:str}/classes", incremental_batch_classes, methods=["PATCH"]),
            Route("/api/incremental/batches/{batch_id:str}/images/{index:int}", incremental_batch_image, methods=["GET"]),
            Route("/api/incremental/batches/{batch_id:str}/inject", incremental_inject, methods=["POST"]),
            Route("/api/incremental/batches/{batch_id:str}/train", incremental_train, methods=["POST"]),
            Route("/api/incremental/jobs", incremental_jobs, methods=["GET"]),
            Route("/api/incremental/jobs/{job_id:str}", incremental_job_detail, methods=["GET"]),
            Route("/api/incremental/jobs/{job_id:str}/logs", incremental_job_logs, methods=["GET"]),
            Route("/api/incremental/jobs/{job_id:str}/cancel", incremental_job_cancel, methods=["POST"]),
            Route("/api/logs", runtime_logs, methods=["GET"]),
            Mount("/", app=StaticFiles(directory=STATIC_ROOT, html=True), name="static"),
        ],
        middleware=[Middleware(SecurityHeadersMiddleware)],
        exception_handlers={404: not_found},
    )
    application.state.runtime_manager = active_runtime
    application.state.engine_provider = engine_provider or active_runtime.get
    application.state.web_settings = settings
    log_settings = settings["logging"]
    logger = event_log or StructuredEventLog(
        root=log_settings["root"],
        max_file_bytes=int(log_settings["max_file_bytes"]),
        retained_files=int(log_settings["retained_files"]),
    )
    store = incremental_store or IncrementalBatchStore(
        settings["incremental_workbench"],
        logger,
        {
            class_id: settings["class_names"][class_id]
            for class_id in settings["active_class_ids"]
        },
        settings["class_names"],
    )
    application.state.event_log = logger
    application.state.incremental_store = store
    application.state.training_manager = training_manager or TrainingJobManager(
        store, settings["incremental_workbench"], logger, effective_config,
        active_runtime.promote if active_runtime is not None else None,
        active_runtime.rollback if active_runtime is not None else None,
    )
    cache = settings["storage"]
    application.state.batch_store = BatchResultStore(
        max_items=int(cache["max_items"]),
        ttl_seconds=int(cache["ttl_seconds"]),
        max_bytes=int(cache["max_bytes"]),
    )
    return application


app = create_app()
