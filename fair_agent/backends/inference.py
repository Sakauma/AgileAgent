from __future__ import annotations

import ctypes
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
        self.engine_batch_size = int(entry["batch_size"])
        self.model = YOLO(str(engine_path), task="detect")
        self._last_timings: Mapping[str, float] = {}

    def _options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        requested_imgsz = int(options.get("imgsz", self.expected_imgsz))
        if requested_imgsz != self.expected_imgsz:
            raise RuntimeError(
                f"TensorRT engine输入尺寸固定为{self.expected_imgsz}，收到{requested_imgsz}"
            )
        allowed = {"imgsz", "conf", "iou", "max_det"}
        return {
            **{key: value for key, value in options.items() if key in allowed and value is not None},
            "device": self.device_index,
            "verbose": False,
        }

    def warmup(self, image: Image.Image) -> None:
        self.predict(image, imgsz=self.expected_imgsz)

    def predict(self, image: Image.Image, **options: Any) -> Any:
        result = self.model.predict(source=image, **self._options(options))[0]
        self._last_timings = dict(getattr(result, "speed", None) or {})
        return result

    def predict_batch(self, images: Sequence[Image.Image], **options: Any) -> Sequence[Any]:
        if len(images) > self.engine_batch_size:
            raise RuntimeError(
                f"TensorRT engine batch固定为{self.engine_batch_size}，收到{len(images)}"
            )
        results = self.model.predict(source=list(images), **self._options(options))
        self._last_timings = dict(getattr(results[-1], "speed", None) or {}) if results else {}
        return results

    def timings(self) -> Mapping[str, float]:
        return dict(self._last_timings)


class TensorRTNativeBackend:
    """Strict loader for the native TensorRT ABI. It never falls back to CPU."""

    name = "tensorrt_native"

    def __init__(self, native_options: Mapping[str, Any]) -> None:
        if native_options.get("validated") is not True:
            raise RuntimeError("TensorRT原生后端尚未通过精度与性能验收。")
        library = resolve_path(native_options["library"])
        engines = [resolve_path(native_options["base_engine"]), resolve_path(native_options["context_engine"])]
        missing = [str(path) for path in [library, *engines] if not path.is_file()]
        if missing:
            raise RuntimeError("TensorRT原生资产缺失：" + ", ".join(missing))
        try:
            self._library = ctypes.CDLL(str(library))
        except OSError as exc:
            raise RuntimeError(f"无法加载TensorRT原生库：{library}: {exc}") from exc
        if not hasattr(self._library, "agile_agent_backend_version"):
            raise RuntimeError("TensorRT原生库ABI不兼容：缺少agile_agent_backend_version")
        if not hasattr(self._library, "agile_agent_create"):
            raise RuntimeError("TensorRT原生库ABI不完整：缺少agile_agent_create")
        raise RuntimeError("TensorRT原生推理绑定尚未在此Python版本中启用。")

    def warmup(self, image: Image.Image) -> None:
        raise RuntimeError("TensorRT原生后端尚未初始化。")

    def predict(self, image: Image.Image, **options: Any) -> Any:
        raise RuntimeError("TensorRT原生后端尚未初始化。")

    def predict_batch(self, images: Sequence[Image.Image], **options: Any) -> Sequence[Any]:
        raise RuntimeError("TensorRT原生后端尚未初始化。")

    def timings(self) -> Mapping[str, float]:
        return {}


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
        return TensorRTNativeBackend(native_options or {})
    raise ValueError(f"未知推理后端：{backend}")
