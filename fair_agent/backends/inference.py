from __future__ import annotations

import ctypes
import io
import json
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from PIL import Image

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.hashes import sha256_file


@runtime_checkable
class InferenceBackend(Protocol):
    name: str

    def warmup(self, image: Image.Image) -> None: ...

    def predict(self, image: Image.Image, **options: Any) -> Any: ...

    def predict_batch(self, images: Sequence[Image.Image], **options: Any) -> Sequence[Any]: ...

    def timings(self) -> Mapping[str, float]: ...


class UltralyticsCudaBackend:
    name = "ultralytics_cuda"

    def __init__(self, weights: str | Path, device_index: str) -> None:
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("Ultralytics CUDA后端要求可用的NVIDIA GPU，不提供CPU回退。")
        if not str(device_index).isdigit() or int(device_index) >= torch.cuda.device_count():
            raise RuntimeError(f"GPU编号不可用：{device_index}")
        self.device_index = str(device_index)
        self.model = YOLO(str(resolve_path(weights)))
        self._last_timings: Mapping[str, float] = {}

    def _options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **{key: value for key, value in options.items() if value is not None},
            "device": self.device_index,
            "verbose": False,
        }

    def warmup(self, image: Image.Image) -> None:
        self.predict(image)

    def predict(self, image: Image.Image, **options: Any) -> Any:
        result = self.model.predict(source=image, **self._options(options))[0]
        self._last_timings = dict(getattr(result, "speed", None) or {})
        return result

    def predict_batch(self, images: Sequence[Image.Image], **options: Any) -> Sequence[Any]:
        batch_options = {**dict(options), "batch": len(images)}
        results = self.model.predict(source=list(images), **self._options(batch_options))
        self._last_timings = dict(getattr(results[-1], "speed", None) or {}) if results else {}
        return results

    def timings(self) -> Mapping[str, float]:
        return dict(self._last_timings)


class TensorRTEngineBackend:
    """Ultralytics result adapter backed by validated TensorRT engines."""

    name = "tensorrt_engine"

    def __init__(
        self,
        weights: str | Path,
        device_index: str,
        backend_options: Mapping[str, Any],
    ) -> None:
        if backend_options.get("validated") is not True:
            raise RuntimeError("TensorRT engine尚未通过完整精度与性能验收。")
        import tensorrt as trt
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT后端要求可用的NVIDIA GPU，不提供CPU回退。")
        if not str(device_index).isdigit() or int(device_index) >= torch.cuda.device_count():
            raise RuntimeError(f"GPU编号不可用：{device_index}")
        expected_version = str(backend_options["expected_version"])
        if str(trt.__version__) != expected_version:
            raise RuntimeError(
                f"TensorRT版本不匹配：engine={expected_version}, runtime={trt.__version__}"
            )
        capability = ".".join(str(value) for value in torch.cuda.get_device_capability(int(device_index)))
        if backend_options.get("require_exact_gpu") and capability != str(
            backend_options["expected_compute_capability"]
        ):
            raise RuntimeError(
                "GPU计算能力不匹配："
                f"engine={backend_options['expected_compute_capability']}, runtime={capability}"
            )

        source_key = rel_path(resolve_path(weights))
        entry = backend_options.get("engines", {}).get(source_key)
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"TensorRT配置未登记模型：{source_key}")
        engine_path = resolve_path(entry["path"])
        if not engine_path.is_file():
            raise RuntimeError(f"TensorRT engine不存在：{engine_path}")
        actual_sha256 = sha256_file(engine_path)
        if actual_sha256 != str(entry["sha256"]):
            raise RuntimeError(f"TensorRT engine哈希不匹配：{rel_path(engine_path)}")

        self.device_index = str(device_index)
        self.engine_path = engine_path
        self.expected_imgsz = int(entry["imgsz"])
        self.engine_min_batch_size = int(entry.get("min_batch_size", 1))
        self.engine_opt_batch_size = int(entry.get("opt_batch_size", entry["batch_size"]))
        self.engine_batch_size = int(entry["batch_size"])
        self.dynamic = bool(backend_options.get("dynamic"))
        self.model = YOLO(str(engine_path), task="detect")
        self._last_timings: Mapping[str, float] = {}

    def _options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        requested_imgsz = int(options.get("imgsz", self.expected_imgsz))
        if requested_imgsz != self.expected_imgsz:
            raise RuntimeError(
                f"TensorRT engine输入尺寸固定为{self.expected_imgsz}，收到{requested_imgsz}"
            )
        allowed = {"conf", "iou", "max_det"}
        return {
            **{key: value for key, value in options.items() if key in allowed and value is not None},
            "imgsz": self.expected_imgsz,
            "rect": self.dynamic,
            "device": self.device_index,
            "verbose": False,
        }

    def warmup(self, image: Image.Image) -> None:
        self.predict(image, imgsz=self.expected_imgsz)

    def predict(self, image: Image.Image, **options: Any) -> Any:
        started = time.perf_counter()
        result = self.model.predict(source=image, **self._options(options))[0]
        self._last_timings = {**dict(getattr(result, "speed", None) or {}), "backend_wall_ms": (time.perf_counter() - started) * 1000}
        return result

    def predict_batch(self, images: Sequence[Image.Image], **options: Any) -> Sequence[Any]:
        if len(images) > self.engine_batch_size:
            raise RuntimeError(
                f"TensorRT engine最大batch为{self.engine_batch_size}，收到{len(images)}"
            )
        if not images:
            return []
        if len(images) < self.engine_min_batch_size:
            raise RuntimeError(f"TensorRT engine最小batch为{self.engine_min_batch_size}，收到{len(images)}")
        started = time.perf_counter()
        results = self.model.predict(source=list(images), **self._options(options))
        self._last_timings = (
            {**dict(getattr(results[-1], "speed", None) or {}), "backend_wall_ms": (time.perf_counter() - started) * 1000}
            if results else {"backend_wall_ms": (time.perf_counter() - started) * 1000}
        )
        return results

    def timings(self) -> Mapping[str, float]:
        return dict(self._last_timings)


class _NativeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _NativeVector:
    def __init__(self, values: Sequence[float]) -> None:
        self.values = list(values)

    def __getitem__(self, _index: int) -> "_NativeVector":
        return self

    def tolist(self) -> list[float]:
        return list(self.values)


class _NativeBox:
    def __init__(self, row: Mapping[str, Any]) -> None:
        self.cls = _NativeScalar(float(row["class_id"]))
        self.conf = _NativeScalar(float(row["confidence"]))
        self.xyxy = _NativeVector(row["xyxy"])


class _NativeResult:
    def __init__(self, row: Mapping[str, Any]) -> None:
        self.boxes = [_NativeBox(item) for item in row.get("detections", [])]
        self.speed = dict(row.get("timings") or {})


class TensorRTNativeBackend:
    """ctypes binding for the versioned native TensorRT batch ABI."""

    name = "tensorrt_native"

    def __init__(self, native_options: Mapping[str, Any], weights: str | Path | None = None) -> None:
        if native_options.get("validated") is not True:
            raise RuntimeError("TensorRT原生后端尚未通过精度与性能验收。")
        library = resolve_path(native_options["library"])
        source_key = rel_path(resolve_path(weights)) if weights is not None else None
        engine_entry = native_options.get("engines", {}).get(source_key, {}) if source_key else {}
        detector_engine = resolve_path(engine_entry.get("path") or native_options.get("base_engine", ""))
        context_entry = native_options["context_engine"]
        context_engine = resolve_path(context_entry.get("path") if isinstance(context_entry, Mapping) else context_entry)
        missing = [str(path) for path in [library, detector_engine, context_engine] if not path.is_file()]
        if missing:
            raise RuntimeError("TensorRT原生资产缺失：" + ", ".join(missing))
        if engine_entry.get("sha256") and sha256_file(detector_engine) != str(engine_entry["sha256"]):
            raise RuntimeError(f"TensorRT原生检测engine哈希不匹配：{rel_path(detector_engine)}")
        try:
            self._library = ctypes.CDLL(str(library))
        except OSError as exc:
            raise RuntimeError(f"无法加载TensorRT原生库：{library}: {exc}") from exc
        required = {
            "agile_agent_backend_version", "agile_agent_create", "agile_agent_destroy",
            "agile_agent_warmup", "agile_agent_predict_batch", "agile_agent_free_result",
            "agile_agent_last_error",
        }
        missing_symbols = sorted(name for name in required if not hasattr(self._library, name))
        if missing_symbols:
            raise RuntimeError("TensorRT原生库ABI不完整：" + ", ".join(missing_symbols))
        self._library.agile_agent_backend_version.restype = ctypes.c_uint32
        if int(self._library.agile_agent_backend_version()) != 1:
            raise RuntimeError("TensorRT原生库ABI版本不受支持。")
        self._library.agile_agent_create.argtypes = [ctypes.c_char_p]
        self._library.agile_agent_create.restype = ctypes.c_void_p
        self._library.agile_agent_destroy.argtypes = [ctypes.c_void_p]
        self._library.agile_agent_warmup.argtypes = [ctypes.c_void_p]
        self._library.agile_agent_warmup.restype = ctypes.c_int
        self._library.agile_agent_predict_batch.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t, ctypes.c_char_p,
        ]
        self._library.agile_agent_predict_batch.restype = ctypes.c_void_p
        self._library.agile_agent_free_result.argtypes = [ctypes.c_void_p]
        self._library.agile_agent_last_error.argtypes = [ctypes.c_void_p]
        self._library.agile_agent_last_error.restype = ctypes.c_char_p
        create_payload = json.dumps({
            "detector_engine": str(detector_engine), "context_engine": str(context_engine),
            "precision": native_options["precision"],
        }).encode("utf-8")
        self._handle = self._library.agile_agent_create(create_payload)
        if not self._handle:
            raise RuntimeError(self._error("TensorRT原生后端初始化失败。"))
        self._last_timings: Mapping[str, float] = {}

    def _error(self, fallback: str) -> str:
        raw = self._library.agile_agent_last_error(getattr(self, "_handle", None))
        return raw.decode("utf-8", errors="replace") if raw else fallback

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._library.agile_agent_destroy(handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def warmup(self, image: Image.Image) -> None:
        if self._library.agile_agent_warmup(self._handle) != 0:
            raise RuntimeError(self._error("TensorRT原生后端预热失败。"))

    def predict(self, image: Image.Image, **options: Any) -> Any:
        return self.predict_batch([image], **options)[0]

    def predict_batch(self, images: Sequence[Image.Image], **options: Any) -> Sequence[Any]:
        encoded = []
        for image in images:
            output = io.BytesIO()
            image.save(output, format="PNG")
            encoded.append(output.getvalue())
        buffers = [ctypes.create_string_buffer(item) for item in encoded]
        pointers = (ctypes.c_void_p * len(buffers))(*[ctypes.cast(item, ctypes.c_void_p) for item in buffers])
        sizes = (ctypes.c_size_t * len(buffers))(*[len(item) for item in encoded])
        started = time.perf_counter()
        result_pointer = self._library.agile_agent_predict_batch(
            self._handle, pointers, sizes, len(buffers), json.dumps(options).encode("utf-8")
        )
        if not result_pointer:
            raise RuntimeError(self._error("TensorRT原生batch推理失败。"))
        try:
            payload = json.loads(ctypes.string_at(result_pointer).decode("utf-8"))
        finally:
            self._library.agile_agent_free_result(result_pointer)
        self._last_timings = {
            **dict(payload.get("timings") or {}),
            "backend_wall_ms": (time.perf_counter() - started) * 1000,
        }
        return [_NativeResult(row) for row in payload.get("results", [])]

    def timings(self) -> Mapping[str, float]:
        return dict(self._last_timings)


def create_backend(
    backend: str,
    weights: str | Path,
    device_index: str,
    native_options: Mapping[str, Any] | None = None,
) -> InferenceBackend:
    if backend == "ultralytics_cuda":
        return UltralyticsCudaBackend(weights, device_index)
    if backend == "tensorrt_engine":
        return TensorRTEngineBackend(weights, device_index, native_options or {})
    if backend == "tensorrt_native":
        return TensorRTNativeBackend(native_options or {}, weights)
    raise ValueError(f"未知推理后端：{backend}")
