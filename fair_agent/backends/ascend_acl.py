from __future__ import annotations

import atexit
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.hashes import sha256_file


ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
DVPP_PIXEL_FORMAT_RGB888 = 12
DVPP_CHANNEL_MODE_VPC = 1
DVPP_CHANNEL_MODE_PNGD = 8

ACL_DTYPES = {
    0: np.dtype(np.float32),
    1: np.dtype(np.float16),
    2: np.dtype(np.int8),
    3: np.dtype(np.int32),
    4: np.dtype(np.uint8),
    6: np.dtype(np.int16),
    7: np.dtype(np.uint16),
    8: np.dtype(np.uint32),
    9: np.dtype(np.int64),
    10: np.dtype(np.uint64),
    11: np.dtype(np.float64),
    12: np.dtype(np.bool_),
}


def _require(call_result: Any, operation: str) -> None:
    result = call_result[-1] if isinstance(call_result, tuple) else call_result
    if int(result) != ACL_SUCCESS:
        raise RuntimeError(f"{operation}失败，ACL错误码：{result}")


def _value(call_result: Any, operation: str) -> Any:
    if not isinstance(call_result, tuple) or len(call_result) < 2:
        raise RuntimeError(f"{operation}返回格式异常：{call_result!r}")
    *values, result = call_result
    _require(result, operation)
    return values[0] if len(values) == 1 else tuple(values)


def _dims(call_result: Any, operation: str) -> dict[str, Any]:
    value = _value(call_result, operation)
    if not isinstance(value, dict) or "dims" not in value:
        raise RuntimeError(f"{operation}维度格式异常：{value!r}")
    return value


def _dtype(code: int) -> np.dtype[Any]:
    if code not in ACL_DTYPES:
        raise RuntimeError(f"暂不支持ACL数据类型：{code}")
    return ACL_DTYPES[code]


def _numpy_pointer(acl: Any, array: np.ndarray) -> int:
    pointer = acl.util.numpy_to_ptr(array)
    if not isinstance(pointer, int):
        raise RuntimeError(f"acl.util.numpy_to_ptr返回格式异常：{pointer!r}")
    return pointer


class AscendAclRuntime:
    _instance: "AscendAclRuntime | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, device_id: int) -> None:
        try:
            import acl
        except ImportError as exc:
            raise RuntimeError(
                "无法导入PyACL，请先加载现有CANN的set_env.sh；禁止回退到CPU模型。"
            ) from exc
        self.acl = acl
        self.device_id = int(device_id)
        self.lock = threading.RLock()
        self.models: weakref.WeakSet[AscendAclModel] = weakref.WeakSet()
        self.preprocessors: weakref.WeakSet[AscendEncodedPreprocessor] = (
            weakref.WeakSet()
        )
        self.closed = False
        _require(acl.init(), "acl.init")
        try:
            _require(acl.rt.set_device(self.device_id), "acl.rt.set_device")
            self.context = _value(
                acl.rt.create_context(self.device_id), "acl.rt.create_context"
            )
        except Exception:
            acl.finalize()
            raise
        atexit.register(self.close)

    @classmethod
    def acquire(cls, device_id: int) -> "AscendAclRuntime":
        with cls._instance_lock:
            if cls._instance is None or cls._instance.closed:
                cls._instance = cls(device_id)
            elif cls._instance.device_id != int(device_id):
                raise RuntimeError(
                    f"进程已绑定Ascend设备{cls._instance.device_id}，不能切换到{device_id}"
                )
            return cls._instance

    def activate(self) -> None:
        if self.closed:
            raise RuntimeError("Ascend ACL Runtime已经关闭")
        _require(self.acl.rt.set_context(self.context), "acl.rt.set_context")

    def register(self, model: "AscendAclModel") -> None:
        self.models.add(model)

    def unregister(self, model: "AscendAclModel") -> None:
        self.models.discard(model)

    def register_preprocessor(self, preprocessor: "AscendEncodedPreprocessor") -> None:
        self.preprocessors.add(preprocessor)

    def unregister_preprocessor(self, preprocessor: "AscendEncodedPreprocessor") -> None:
        self.preprocessors.discard(preprocessor)

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            # Models own datasets, buffers and model descriptors tied to this
            # context.  Release them before destroying the shared context even
            # when shutdown is initiated by the Runtime atexit callback.
            for preprocessor in list(self.preprocessors):
                preprocessor.close()
            for model in list(self.models):
                model.close()
            self.closed = True
            try:
                self.acl.rt.destroy_context(self.context)
            finally:
                try:
                    self.acl.rt.reset_device(self.device_id)
                finally:
                    self.acl.finalize()


class AscendAclModel:
    def __init__(
        self,
        runtime: AscendAclRuntime,
        path: Path,
        execution_mode: str = "synchronous",
    ) -> None:
        self.runtime = runtime
        self.path = path
        self.execution_mode = str(execution_mode)
        if self.execution_mode not in {"synchronous", "async_stream"}:
            raise RuntimeError(f"Ascend执行模式非法：{self.execution_mode}")
        self.closed = False
        self.execution_lock = threading.Lock()
        self.model_id: int | None = None
        self.desc: Any = None
        self.stream: Any = None
        self.timing_start_event: Any = None
        self.timing_end_event: Any = None
        self.preloaded_runs = 0
        self.input_dataset: Any = None
        self.output_dataset: Any = None
        self.device_buffers: list[int] = []
        self.data_buffers: list[Any] = []
        self.output_contracts: list[dict[str, Any]] = []
        self.output_hosts: list[np.ndarray] = []
        with runtime.lock:
            runtime.activate()
            try:
                self.model_id = int(
                    _value(
                        runtime.acl.mdl.load_from_file(str(path)),
                        "acl.mdl.load_from_file",
                    )
                )
                self.desc = runtime.acl.mdl.create_desc()
                if self.desc is None:
                    raise RuntimeError("acl.mdl.create_desc返回空")
                _require(
                    runtime.acl.mdl.get_desc(self.desc, self.model_id),
                    "acl.mdl.get_desc",
                )
                self._create_datasets()
                if self.execution_mode == "async_stream":
                    self.stream = _value(
                        runtime.acl.rt.create_stream(), "acl.rt.create_stream"
                    )
                    self.timing_start_event = _value(
                        runtime.acl.rt.create_event(), "acl.rt.create_event(start)"
                    )
                    self.timing_end_event = _value(
                        runtime.acl.rt.create_event(), "acl.rt.create_event(end)"
                    )
                runtime.register(self)
            except Exception:
                self.close()
                raise

    def _create_datasets(self) -> None:
        acl = self.runtime.acl
        if int(acl.mdl.get_num_inputs(self.desc)) != 1:
            raise RuntimeError(f"{self.path.name}必须是单输入静态模型")
        input_dims = _dims(acl.mdl.get_input_dims(self.desc, 0), "acl.mdl.get_input_dims")
        self.input_shape = tuple(int(value) for value in input_dims["dims"])
        self.input_dtype = _dtype(int(acl.mdl.get_input_data_type(self.desc, 0)))
        self.input_size = int(acl.mdl.get_input_size_by_index(self.desc, 0))
        self.input_dataset = acl.mdl.create_dataset()
        if self.input_dataset is None:
            raise RuntimeError("acl.mdl.create_dataset(input)返回空")
        input_device = int(
            _value(
                acl.rt.malloc(self.input_size, ACL_MEM_MALLOC_HUGE_FIRST),
                "acl.rt.malloc(input)",
            )
        )
        self.device_buffers.append(input_device)
        self.input_device = input_device
        input_buffer = acl.create_data_buffer(input_device, self.input_size)
        if input_buffer is None:
            raise RuntimeError("acl.create_data_buffer(input)返回空")
        self.data_buffers.append(input_buffer)
        _require(
            acl.mdl.add_dataset_buffer(self.input_dataset, input_buffer),
            "acl.mdl.add_dataset_buffer(input)",
        )

        self.output_dataset = acl.mdl.create_dataset()
        if self.output_dataset is None:
            raise RuntimeError("acl.mdl.create_dataset(output)返回空")
        for index in range(int(acl.mdl.get_num_outputs(self.desc))):
            dims = _dims(
                acl.mdl.get_output_dims(self.desc, index),
                f"acl.mdl.get_output_dims({index})",
            )
            size = int(acl.mdl.get_output_size_by_index(self.desc, index))
            data_type = _dtype(int(acl.mdl.get_output_data_type(self.desc, index)))
            device = int(
                _value(
                    acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST),
                    f"acl.rt.malloc(output {index})",
                )
            )
            self.device_buffers.append(device)
            buffer = acl.create_data_buffer(device, size)
            if buffer is None:
                raise RuntimeError(f"acl.create_data_buffer(output {index})返回空")
            self.data_buffers.append(buffer)
            _require(
                acl.mdl.add_dataset_buffer(self.output_dataset, buffer),
                f"acl.mdl.add_dataset_buffer(output {index})",
            )
            self.output_contracts.append(
                {
                    "index": index,
                    "shape": tuple(int(value) for value in dims["dims"]),
                    "dtype": data_type,
                    "size": size,
                    "device": device,
                }
            )
            # The web engine serializes requests and each backend consumes its
            # raw output before the next invocation.  Reusing one host buffer
            # per output avoids allocating a byte array and copying it again
            # for every image while retaining the exact output dtype/shape.
            self.output_hosts.append(np.empty(size, dtype=np.uint8))

    def _copy_outputs(self) -> list[np.ndarray]:
        acl = self.runtime.acl
        outputs = []
        for contract, host in zip(self.output_contracts, self.output_hosts):
            _require(
                acl.rt.memcpy(
                    _numpy_pointer(acl, host),
                    host.nbytes,
                    contract["device"],
                    contract["size"],
                    ACL_MEMCPY_DEVICE_TO_HOST,
                ),
                f"acl.rt.memcpy(output {contract['index']})",
            )
            outputs.append(
                host.view(contract["dtype"]).reshape(contract["shape"])
            )
        return outputs

    def execute(self, array: np.ndarray) -> tuple[list[np.ndarray], float]:
        batch = np.ascontiguousarray(array, dtype=self.input_dtype)
        if tuple(batch.shape) != self.input_shape:
            raise RuntimeError(
                f"{self.path.name}输入shape错误：{batch.shape} != {self.input_shape}"
            )
        if batch.nbytes != self.input_size:
            raise RuntimeError(
                f"{self.path.name}输入字节数错误：{batch.nbytes} != {self.input_size}"
            )
        acl = self.runtime.acl
        execution_guard = (
            self.runtime.lock
            if self.execution_mode == "synchronous"
            else self.execution_lock
        )
        with execution_guard:
            self.runtime.activate()
            _require(
                acl.rt.memcpy(
                    self.device_buffers[0],
                    self.input_size,
                    _numpy_pointer(acl, batch),
                    batch.nbytes,
                    ACL_MEMCPY_HOST_TO_DEVICE,
                ),
                "acl.rt.memcpy(input)",
            )
            started = time.perf_counter_ns()
            if self.execution_mode == "async_stream":
                _require(
                    acl.mdl.execute_async(
                        self.model_id,
                        self.input_dataset,
                        self.output_dataset,
                        self.stream,
                    ),
                    "acl.mdl.execute_async",
                )
                _require(
                    acl.rt.synchronize_stream(self.stream),
                    "acl.rt.synchronize_stream",
                )
            else:
                _require(
                    acl.mdl.execute(
                        self.model_id, self.input_dataset, self.output_dataset
                    ),
                    "acl.mdl.execute",
                )
            inference_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            outputs = self._copy_outputs()
        return outputs, inference_ms

    def execute_preloaded(self, ready_event: Any) -> tuple[list[np.ndarray], float]:
        if self.execution_mode != "async_stream" or self.stream is None:
            raise RuntimeError("Ascend设备预处理输入要求async_stream执行模式")
        acl = self.runtime.acl
        with self.execution_lock:
            self.runtime.activate()
            if self.preloaded_runs:
                _require(
                    acl.rt.reset_event(self.timing_start_event, self.stream),
                    "acl.rt.reset_event(start)",
                )
                _require(
                    acl.rt.reset_event(self.timing_end_event, self.stream),
                    "acl.rt.reset_event(end)",
                )
            _require(
                acl.rt.stream_wait_event(self.stream, ready_event),
                "acl.rt.stream_wait_event(input)",
            )
            _require(
                acl.rt.record_event(self.timing_start_event, self.stream),
                "acl.rt.record_event(start)",
            )
            _require(
                acl.mdl.execute_async(
                    self.model_id,
                    self.input_dataset,
                    self.output_dataset,
                    self.stream,
                ),
                "acl.mdl.execute_async(preloaded)",
            )
            _require(
                acl.rt.record_event(self.timing_end_event, self.stream),
                "acl.rt.record_event(end)",
            )
            self.preloaded_runs += 1
            _require(
                acl.rt.synchronize_stream(self.stream),
                "acl.rt.synchronize_stream(preloaded)",
            )
            inference_ms = float(
                _value(
                    acl.rt.event_elapsed_time(
                        self.timing_start_event, self.timing_end_event
                    ),
                    "acl.rt.event_elapsed_time(model)",
                )
            )
            outputs = self._copy_outputs()
        return outputs, inference_ms

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        acl = self.runtime.acl
        with self.runtime.lock:
            if self.runtime.closed:
                return
            self.runtime.activate()
            for buffer in reversed(self.data_buffers):
                acl.destroy_data_buffer(buffer)
            if self.input_dataset is not None:
                acl.mdl.destroy_dataset(self.input_dataset)
            if self.output_dataset is not None:
                acl.mdl.destroy_dataset(self.output_dataset)
            if self.timing_end_event is not None:
                acl.rt.destroy_event(self.timing_end_event)
            if self.timing_start_event is not None:
                acl.rt.destroy_event(self.timing_start_event)
            if self.stream is not None:
                acl.rt.destroy_stream(self.stream)
            for device in reversed(self.device_buffers):
                acl.rt.free(device)
            if self.desc is not None:
                acl.mdl.destroy_desc(self.desc)
            if self.model_id is not None:
                acl.mdl.unload(self.model_id)
            self.runtime.unregister(self)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _validated_model_entry(
    options: Mapping[str, Any], weights: str | Path
) -> Mapping[str, Any]:
    source_key = rel_path(resolve_path(weights))
    entry = options.get("models", {}).get(source_key)
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"Ascend配置未登记模型：{source_key}")
    return entry


def _load_model(options: Mapping[str, Any], entry: Mapping[str, Any]) -> AscendAclModel:
    from fair_agent.modules.ascend_release import require_ascend_runtime_artifacts

    require_ascend_runtime_artifacts(options)
    path = resolve_path(entry["path"])
    if not path.is_file():
        raise RuntimeError(f"Ascend OM不存在：{path}")
    digest = str(entry.get("sha256") or "")
    if len(digest) != 64 or sha256_file(path) != digest:
        raise RuntimeError(f"Ascend OM哈希缺失或不匹配：{path}")
    runtime = AscendAclRuntime.acquire(int(options.get("device_id", 0)))
    return AscendAclModel(
        runtime,
        path,
        execution_mode=str(options.get("execution_mode", "synchronous")),
    )


def _align(value: int, alignment: int) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


def validate_dvpp_scene_resize_stages(value: Any) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or len(value) > 4:
        raise ValueError("Ascend Scene DVPP多级resize必须是最多4级的尺寸列表")
    stages = []
    for index, stage in enumerate(value):
        if not isinstance(stage, (list, tuple)) or len(stage) != 2:
            raise ValueError(f"Ascend Scene DVPP resize第{index}级必须包含宽和高")
        width, height = stage
        if any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 16
            or item > 4096
            or item % 2
            for item in (width, height)
        ):
            raise ValueError(
                f"Ascend Scene DVPP resize第{index}级宽高必须是16-4096范围内的偶数"
            )
        stages.append((int(width), int(height)))
    return tuple(stages)


class AscendEncodedPreprocessor:
    """Decode one fixed production PNG into the three resident AIPP inputs."""

    source_width = 640
    source_height = 512

    def __init__(
        self,
        runtime: AscendAclRuntime,
        base_model: AscendAclModel,
        specialist_model: AscendAclModel,
        context_model: AscendAclModel,
        scene_resize_stages: Any = None,
    ) -> None:
        contracts = (
            (base_model, (1, 736, 896, 3), "base"),
            (specialist_model, (1, 512, 640, 3), "specialist"),
            (context_model, (1, 160, 160, 3), "context"),
        )
        for model, shape, name in contracts:
            if model.runtime is not runtime:
                raise RuntimeError(f"Ascend {name}模型没有共享同一ACL Runtime")
            if model.execution_mode != "async_stream":
                raise RuntimeError(f"Ascend {name}设备预处理要求async_stream")
            if model.input_dtype != np.dtype(np.uint8) or model.input_shape != shape:
                raise RuntimeError(
                    f"Ascend {name}设备预处理输入契约错误："
                    f"shape={model.input_shape}, dtype={model.input_dtype}"
                )
        self.runtime = runtime
        self.base_model = base_model
        self.specialist_model = specialist_model
        self.context_model = context_model
        self.scene_resize_stages = validate_dvpp_scene_resize_stages(
            scene_resize_stages
        )
        self.lock = threading.Lock()
        self.closed = False
        self.stream: Any = None
        self.base_ready_event: Any = None
        self.specialist_ready_event: Any = None
        self.context_ready_event: Any = None
        self.prepared_runs = 0
        self.channel: Any = None
        self.descriptors: list[Any] = []
        self.resize_configs: list[Any] = []
        self.scene_intermediate_descs: list[Any] = []
        self.roi: Any = None
        self.dvpp_buffers: list[int] = []
        with runtime.lock:
            runtime.activate()
            try:
                self._create()
                runtime.register_preprocessor(self)
            except Exception:
                self.close()
                raise

    @staticmethod
    def accepts(data: bytes) -> bool:
        if len(data) < 26 or data[:8] != b"\x89PNG\r\n\x1a\n":
            return False
        if data[12:16] != b"IHDR":
            return False
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        bit_depth = int(data[24])
        color_type = int(data[25])
        return (
            width == AscendEncodedPreprocessor.source_width
            and height == AscendEncodedPreprocessor.source_height
            and bit_depth == 8
            and color_type in {2, 6}
        )

    def _picture_desc(
        self, pointer: int, width: int, height: int
    ) -> tuple[Any, int, int, int]:
        acl = self.runtime.acl
        width_stride = _align(width, 16) * 3
        height_stride = _align(height, 2)
        size = width_stride * height_stride
        desc = acl.media.dvpp_create_pic_desc()
        if desc is None:
            raise RuntimeError("acl.media.dvpp_create_pic_desc返回空")
        self.descriptors.append(desc)
        for result, operation in (
            (acl.media.dvpp_set_pic_desc_data(desc, pointer), "set data"),
            (acl.media.dvpp_set_pic_desc_size(desc, size), "set size"),
            (
                acl.media.dvpp_set_pic_desc_format(
                    desc, DVPP_PIXEL_FORMAT_RGB888
                ),
                "set format",
            ),
            (acl.media.dvpp_set_pic_desc_width(desc, width), "set width"),
            (acl.media.dvpp_set_pic_desc_height(desc, height), "set height"),
            (
                acl.media.dvpp_set_pic_desc_width_stride(desc, width_stride),
                "set width stride",
            ),
            (
                acl.media.dvpp_set_pic_desc_height_stride(desc, height_stride),
                "set height stride",
            ),
        ):
            _require(result, f"acl.media.dvpp_set_pic_desc_{operation}")
        return desc, width_stride, height_stride, size

    def _dvpp_buffer(self, size: int, operation: str) -> int:
        pointer = int(
            _value(self.runtime.acl.media.dvpp_malloc(size), operation)
        )
        self.dvpp_buffers.append(pointer)
        return pointer

    def _resize_config(self) -> Any:
        acl = self.runtime.acl
        config = acl.media.dvpp_create_resize_config()
        if config is None:
            raise RuntimeError("acl.media.dvpp_create_resize_config返回空")
        self.resize_configs.append(config)
        _require(
            acl.media.dvpp_set_resize_config_interpolation(config, 0),
            "acl.media.dvpp_set_resize_config_interpolation",
        )
        return config

    def _create(self) -> None:
        acl = self.runtime.acl
        self.stream = _value(acl.rt.create_stream(), "acl.rt.create_stream(DVPP)")
        self.base_ready_event = _value(
            acl.rt.create_event(), "acl.rt.create_event(base input)"
        )
        self.specialist_ready_event = _value(
            acl.rt.create_event(), "acl.rt.create_event(specialist input)"
        )
        self.context_ready_event = _value(
            acl.rt.create_event(), "acl.rt.create_event(context input)"
        )
        self.channel = acl.media.dvpp_create_channel_desc()
        if self.channel is None:
            raise RuntimeError("acl.media.dvpp_create_channel_desc返回空")
        _require(
            acl.media.dvpp_set_channel_desc_mode(
                self.channel, DVPP_CHANNEL_MODE_VPC | DVPP_CHANNEL_MODE_PNGD
            ),
            "acl.media.dvpp_set_channel_desc_mode",
        )
        _require(
            acl.media.dvpp_create_channel(self.channel),
            "acl.media.dvpp_create_channel",
        )

        self.decoded_desc, self.decoded_width_stride, _, decoded_size = (
            self._picture_desc(
                self.specialist_model.input_device,
                self.source_width,
                self.source_height,
            )
        )
        if decoded_size != self.specialist_model.input_size:
            raise RuntimeError("Ascend增量OM输入大小与RGB888 PNGD输出不一致")

        base_resize_size = _align(896, 16) * 3 * _align(717, 2)
        self.base_resize_pointer = self._dvpp_buffer(
            base_resize_size, "acl.media.dvpp_malloc(base resize)"
        )
        self.base_resize_desc, self.base_width_stride, _, _ = self._picture_desc(
            self.base_resize_pointer, 896, 717
        )
        self.base_desc, base_width_stride, _, base_size = self._picture_desc(
            self.base_model.input_device, 896, 736
        )
        if base_size != self.base_model.input_size:
            raise RuntimeError("Ascend基础OM输入大小与RGB888 VPC输出不一致")
        if base_width_stride != self.base_width_stride:
            raise RuntimeError("Ascend基础resize与letterbox stride不一致")
        _require(
            acl.rt.memset(
                self.base_model.input_device,
                self.base_model.input_size,
                114,
                self.base_model.input_size,
            ),
            "acl.rt.memset(base letterbox)",
        )

        for index, (width, height) in enumerate(self.scene_resize_stages):
            intermediate_size = _align(width, 16) * 3 * _align(height, 2)
            intermediate_pointer = self._dvpp_buffer(
                intermediate_size,
                f"acl.media.dvpp_malloc(scene intermediate {index})",
            )
            intermediate_desc, _, _, _ = self._picture_desc(
                intermediate_pointer, width, height
            )
            self.scene_intermediate_descs.append(intermediate_desc)

        scene_resize_size = _align(176, 16) * 3 * _align(176, 2)
        self.scene_resize_pointer = self._dvpp_buffer(
            scene_resize_size, "acl.media.dvpp_malloc(scene resize)"
        )
        self.scene_resize_desc, _, _, _ = self._picture_desc(
            self.scene_resize_pointer, 176, 176
        )
        self.scene_desc, _, _, scene_size = self._picture_desc(
            self.context_model.input_device, 160, 160
        )
        if scene_size != self.context_model.input_size:
            raise RuntimeError("Ascend场景OM输入大小与RGB888 VPC输出不一致")

        self.base_resize_config = self._resize_config()
        self.scene_resize_config = self._resize_config()
        self.roi = acl.media.dvpp_create_roi_config(8, 167, 8, 167)
        if self.roi is None:
            raise RuntimeError("acl.media.dvpp_create_roi_config返回空")

    def prepare(self, data: bytes) -> float:
        if not self.accepts(data):
            raise ValueError("Ascend DVPP仅接受固定640x512的8位RGB/RGBA PNG")
        acl = self.runtime.acl
        encoded = np.frombuffer(data, dtype=np.uint8)
        encoded_pointer = _numpy_pointer(acl, encoded)
        with self.lock:
            self.runtime.activate()
            width, height, _components = _value(
                acl.media.dvpp_png_get_image_info(
                    encoded_pointer, encoded.nbytes
                ),
                "acl.media.dvpp_png_get_image_info",
            )
            if (int(width), int(height)) != (self.source_width, self.source_height):
                raise ValueError("Ascend DVPP PNG尺寸与固定生产契约不一致")
            predicted_size = int(
                _value(
                    acl.media.dvpp_png_predict_dec_size(
                        encoded_pointer,
                        encoded.nbytes,
                        DVPP_PIXEL_FORMAT_RGB888,
                    ),
                    "acl.media.dvpp_png_predict_dec_size",
                )
            )
            if predicted_size != self.specialist_model.input_size:
                raise ValueError("Ascend DVPP PNG解码大小与增量OM输入不一致")
            started = time.perf_counter_ns()
            if self.prepared_runs:
                for event, name in (
                    (self.base_ready_event, "base"),
                    (self.specialist_ready_event, "specialist"),
                    (self.context_ready_event, "context"),
                ):
                    _require(
                        acl.rt.reset_event(event, self.stream),
                        f"acl.rt.reset_event({name})",
                    )
            _require(
                acl.media.dvpp_png_decode_async(
                    self.channel,
                    encoded_pointer,
                    encoded.nbytes,
                    self.decoded_desc,
                    self.stream,
                ),
                "acl.media.dvpp_png_decode_async",
            )
            _require(
                acl.rt.record_event(self.specialist_ready_event, self.stream),
                "acl.rt.record_event(specialist input)",
            )
            _require(
                acl.media.dvpp_vpc_resize_async(
                    self.channel,
                    self.decoded_desc,
                    self.base_resize_desc,
                    self.base_resize_config,
                    self.stream,
                ),
                "acl.media.dvpp_vpc_resize_async(base)",
            )
            _require(
                acl.rt.memcpy_async(
                    self.base_model.input_device + 9 * self.base_width_stride,
                    self.base_width_stride * 717,
                    self.base_resize_pointer,
                    self.base_width_stride * 717,
                    ACL_MEMCPY_DEVICE_TO_DEVICE,
                    self.stream,
                ),
                "acl.rt.memcpy_async(base letterbox)",
            )
            _require(
                acl.rt.record_event(self.base_ready_event, self.stream),
                "acl.rt.record_event(base input)",
            )
            scene_source = self.decoded_desc
            for index, intermediate_desc in enumerate(
                self.scene_intermediate_descs
            ):
                _require(
                    acl.media.dvpp_vpc_resize_async(
                        self.channel,
                        scene_source,
                        intermediate_desc,
                        self.scene_resize_config,
                        self.stream,
                    ),
                    f"acl.media.dvpp_vpc_resize_async(scene intermediate {index})",
                )
                scene_source = intermediate_desc
            _require(
                acl.media.dvpp_vpc_resize_async(
                    self.channel,
                    scene_source,
                    self.scene_resize_desc,
                    self.scene_resize_config,
                    self.stream,
                ),
                "acl.media.dvpp_vpc_resize_async(scene final)",
            )
            _require(
                acl.media.dvpp_vpc_crop_async(
                    self.channel,
                    self.scene_resize_desc,
                    self.scene_desc,
                    self.roi,
                    self.stream,
                ),
                "acl.media.dvpp_vpc_crop_async(scene)",
            )
            _require(
                acl.rt.record_event(self.context_ready_event, self.stream),
                "acl.rt.record_event(context input)",
            )
            self.prepared_runs += 1
            return (time.perf_counter_ns() - started) / 1_000_000.0

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        acl = self.runtime.acl
        with self.runtime.lock:
            if self.runtime.closed:
                return
            self.runtime.activate()
            if self.roi is not None:
                acl.media.dvpp_destroy_roi_config(self.roi)
            for config in reversed(self.resize_configs):
                acl.media.dvpp_destroy_resize_config(config)
            for desc in reversed(self.descriptors):
                acl.media.dvpp_destroy_pic_desc(desc)
            for pointer in reversed(self.dvpp_buffers):
                acl.media.dvpp_free(pointer)
            if self.channel is not None:
                acl.media.dvpp_destroy_channel(self.channel)
                acl.media.dvpp_destroy_channel_desc(self.channel)
            for event in (
                self.context_ready_event,
                self.specialist_ready_event,
                self.base_ready_event,
            ):
                if event is not None:
                    acl.rt.destroy_event(event)
            if self.stream is not None:
                acl.rt.destroy_stream(self.stream)
            self.runtime.unregister_preprocessor(self)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def detector_tensor(
    image: Image.Image,
    input_height: int,
    input_width: int,
    rgb_array: np.ndarray | None = None,
    input_mode: str = "nchw_float32",
) -> tuple[np.ndarray, dict[str, float | int]]:
    import cv2

    if rgb_array is None:
        rgb_image = image if image.mode == "RGB" else image.convert("RGB")
        rgb = np.ascontiguousarray(np.asarray(rgb_image))
    else:
        rgb = np.ascontiguousarray(rgb_array)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(
                f"Ascend共享RGB输入格式错误：shape={rgb.shape}, dtype={rgb.dtype}"
            )
    original_height, original_width = map(int, rgb.shape[:2])
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    resized = (
        cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        if (resized_width, resized_height) != (original_width, original_height)
        else rgb
    )
    horizontal = input_width - resized_width
    vertical = input_height - resized_height
    left = int(round(horizontal / 2.0 - 0.1))
    right = int(round(horizontal / 2.0 + 0.1))
    top = int(round(vertical / 2.0 - 0.1))
    bottom = int(round(vertical / 2.0 + 0.1))
    canvas = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if input_mode == "nhwc_uint8_aipp":
        tensor = np.ascontiguousarray(canvas, dtype=np.uint8)[None, ...]
    elif input_mode == "nchw_float32":
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)
        np.divide(tensor, np.float32(255.0), out=tensor)
        tensor = tensor[None, ...]
    else:
        raise RuntimeError(f"Ascend检测输入模式非法：{input_mode}")
    return tensor, {
        "original_height": original_height,
        "original_width": original_width,
        "scale": float(scale),
        "pad_left": left,
        "pad_top": top,
    }


def _box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    top_left = np.maximum(one[:2], many[:, :2])
    bottom_right = np.minimum(one[2:], many[:, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=1)
    one_area = max(0.0, float((one[2] - one[0]) * (one[3] - one[1])))
    many_area = np.maximum(many[:, 2] - many[:, 0], 0.0) * np.maximum(
        many[:, 3] - many[:, 1], 0.0
    )
    return intersection / np.maximum(one_area + many_area - intersection, 1e-9)


def yolo_detections(
    raw: np.ndarray,
    info: Mapping[str, float | int],
    confidence: float,
    iou: float,
    max_det: int,
) -> list[dict[str, Any]]:
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] < 5:
        raise RuntimeError(f"YOLO输出契约错误：{raw.shape}")
    prediction = raw[0]
    scores = prediction[4:].T
    anchor_ids, class_ids = np.where(scores > float(confidence))
    if not len(anchor_ids):
        return []
    xywh = prediction[:4, anchor_ids].T
    boxes = np.empty_like(xywh, dtype=np.float32)
    boxes[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2.0
    boxes[:, 2:] = xywh[:, :2] + xywh[:, 2:] / 2.0
    confidences = scores[anchor_ids, class_ids]
    # Ultralytics multi-label NMS ranks all class candidates globally and only
    # suppresses boxes inside the same class.  Keeping the global confidence
    # order matters when max_det truncates a busy image.
    candidates = np.argsort(-confidences, kind="stable")
    kept: list[int] = []
    while len(candidates) and len(kept) < int(max_det):
        current = int(candidates[0])
        kept.append(current)
        if len(candidates) == 1:
            break
        remaining = candidates[1:]
        same_class = class_ids[remaining] == class_ids[current]
        suppressed = np.zeros(len(remaining), dtype=np.bool_)
        if np.any(same_class):
            suppressed[same_class] = (
                _box_iou(boxes[current], boxes[remaining[same_class]]) > float(iou)
            )
        candidates = remaining[~suppressed]
    scale = float(info["scale"])
    original_width = float(info["original_width"])
    original_height = float(info["original_height"])
    rows = []
    for index in kept[: int(max_det)]:
        x1, y1, x2, y2 = boxes[index]
        restored = [
            min(max((float(x1) - float(info["pad_left"])) / scale, 0.0), original_width),
            min(max((float(y1) - float(info["pad_top"])) / scale, 0.0), original_height),
            min(max((float(x2) - float(info["pad_left"])) / scale, 0.0), original_width),
            min(max((float(y2) - float(info["pad_top"])) / scale, 0.0), original_height),
        ]
        rows.append(
            {
                "class_id": int(class_ids[index]),
                "confidence": float(confidences[index]),
                "xyxy": restored,
            }
        )
    return rows


class _ArrayView:
    def __init__(self, values: Any) -> None:
        self.values = values

    def detach(self) -> "_ArrayView":
        return self

    def cpu(self) -> "_ArrayView":
        return self

    def tolist(self) -> Any:
        return self.values


class AscendBoxes:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)
        self.xyxy = _ArrayView([row["xyxy"] for row in rows])
        self.conf = _ArrayView([float(row["confidence"]) for row in rows])
        self.cls = _ArrayView([float(row["class_id"]) for row in rows])

    def __len__(self) -> int:
        return len(self.rows)


class AscendResult:
    def __init__(self, rows: Sequence[Mapping[str, Any]], timings: Mapping[str, float]) -> None:
        # Keep the postprocessed mappings available to the Agent layer.  The
        # boxes adapter remains for Ultralytics compatibility, but routing no
        # longer has to rebuild these rows through three temporary lists.
        self.records = tuple(rows)
        self.boxes = AscendBoxes(self.records)
        self.speed = dict(timings)


class AscendAclBackend:
    name = "ascend_acl"

    def __init__(
        self, options: Mapping[str, Any], weights: str | Path | None = None
    ) -> None:
        if weights is None:
            raise RuntimeError("Ascend检测后端缺少模型标识")
        entry = _validated_model_entry(options, weights)
        self.model = _load_model(options, entry)
        if (
            self.model.input_dtype == np.dtype(np.uint8)
            and len(self.model.input_shape) == 4
            and self.model.input_shape[0] == 1
            and self.model.input_shape[3] == 3
        ):
            self.input_mode = "nhwc_uint8_aipp"
            self.expected_height = int(self.model.input_shape[1])
            self.expected_width = int(self.model.input_shape[2])
        elif (
            self.model.input_dtype == np.dtype(np.float32)
            and len(self.model.input_shape) == 4
            and self.model.input_shape[0] == 1
            and self.model.input_shape[1] == 3
        ):
            self.input_mode = "nchw_float32"
            self.expected_height = int(self.model.input_shape[2])
            self.expected_width = int(self.model.input_shape[3])
        else:
            raise RuntimeError(
                f"Ascend检测OM输入契约不受支持："
                f"shape={self.model.input_shape}, dtype={self.model.input_dtype}"
            )
        self._last_timings: dict[str, float] = {}

    def predict(self, image: Image.Image, **options: Any) -> AscendResult:
        preprocess_started = time.perf_counter_ns()
        batch, info = detector_tensor(
            image,
            self.expected_height,
            self.expected_width,
            options.get("_ascend_rgb_array"),
            self.input_mode,
        )
        preprocess_ms = (time.perf_counter_ns() - preprocess_started) / 1_000_000.0
        outputs, inference_ms = self.model.execute(batch)
        postprocess_started = time.perf_counter_ns()
        rows = yolo_detections(
            outputs[0],
            info,
            float(options.get("conf", 0.5)),
            float(options.get("iou", 0.7)),
            int(options.get("max_det", 300)),
        )
        postprocess_ms = (
            time.perf_counter_ns() - postprocess_started
        ) / 1_000_000.0
        self._last_timings = {
            "preprocess": preprocess_ms,
            "inference": inference_ms,
            "postprocess": postprocess_ms,
        }
        return AscendResult(rows, self._last_timings)

    def predict_preloaded(
        self,
        info: Mapping[str, float | int],
        ready_event: Any,
        **options: Any,
    ) -> AscendResult:
        outputs, inference_ms = self.model.execute_preloaded(ready_event)
        postprocess_started = time.perf_counter_ns()
        rows = yolo_detections(
            outputs[0],
            info,
            float(options.get("conf", 0.5)),
            float(options.get("iou", 0.7)),
            int(options.get("max_det", 300)),
        )
        postprocess_ms = (
            time.perf_counter_ns() - postprocess_started
        ) / 1_000_000.0
        self._last_timings = {
            "preprocess": 0.0,
            "inference": inference_ms,
            "postprocess": postprocess_ms,
        }
        return AscendResult(rows, self._last_timings)

    def predict_batch(
        self, images: Sequence[Image.Image], **options: Any
    ) -> Sequence[AscendResult]:
        rgb_arrays = options.pop("_ascend_rgb_arrays", None)
        if rgb_arrays is None:
            return [self.predict(image, **options) for image in images]
        if len(rgb_arrays) != len(images):
            raise RuntimeError("Ascend批量共享RGB输入数量不匹配")
        return [
            self.predict(image, _ascend_rgb_array=rgb, **options)
            for image, rgb in zip(images, rgb_arrays)
        ]

    def warmup(self, image: Image.Image) -> None:
        self.predict(image)

    def timings(self) -> Mapping[str, float]:
        return dict(self._last_timings)

    def close(self) -> None:
        self.model.close()


def context_tensor(
    image: Image.Image,
    image_size: int,
    input_mode: str = "nchw_float32",
) -> np.ndarray:
    resize_size = int(round(image_size * 1.1))
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    resized = rgb_image.resize(
        (resize_size, resize_size), Image.Resampling.BILINEAR
    )
    offset = (resize_size - image_size) // 2
    cropped = resized.crop((offset, offset, offset + image_size, offset + image_size))
    cropped_array = np.asarray(cropped)
    if input_mode == "nhwc_uint8_aipp":
        return np.ascontiguousarray(cropped_array, dtype=np.uint8)[None, ...]
    if input_mode == "nchw_float32":
        array = np.asarray(cropped_array, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.25
        return np.ascontiguousarray(
            array.transpose(2, 0, 1), dtype=np.float32
        )[None, ...]
    raise RuntimeError(f"Ascend上下文输入模式非法：{input_mode}")


class AscendAclContextModel:
    def __init__(self, options: Mapping[str, Any]) -> None:
        entry = options.get("context_model")
        if not isinstance(entry, Mapping):
            raise RuntimeError("Ascend配置缺少context_model")
        self.model = _load_model(options, entry)
        if (
            self.model.input_dtype == np.dtype(np.uint8)
            and len(self.model.input_shape) == 4
            and self.model.input_shape[0] == 1
            and self.model.input_shape[3] == 3
        ):
            self.input_mode = "nhwc_uint8_aipp"
            input_height = int(self.model.input_shape[1])
            input_width = int(self.model.input_shape[2])
        elif (
            self.model.input_dtype == np.dtype(np.float32)
            and len(self.model.input_shape) == 4
            and self.model.input_shape[0] == 1
            and self.model.input_shape[1] == 3
        ):
            self.input_mode = "nchw_float32"
            input_height = int(self.model.input_shape[2])
            input_width = int(self.model.input_shape[3])
        else:
            raise RuntimeError(
                f"Ascend上下文OM输入契约不受支持："
                f"shape={self.model.input_shape}, dtype={self.model.input_dtype}"
            )
        if input_height != input_width:
            raise RuntimeError(
                f"Ascend上下文OM必须使用方形输入：{self.model.input_shape}"
            )
        self.image_size = input_height

    def predict(self, image: Image.Image) -> dict[str, Any]:
        outputs, inference_ms = self.model.execute(
            context_tensor(
                image,
                self.image_size,
                self.input_mode,
            )
        )
        return self._result(outputs, inference_ms)

    def predict_preloaded(self, ready_event: Any) -> dict[str, Any]:
        outputs, inference_ms = self.model.execute_preloaded(ready_event)
        return self._result(outputs, inference_ms)

    @staticmethod
    def _result(outputs: Sequence[np.ndarray], inference_ms: float) -> dict[str, Any]:
        sensor_prob = _softmax(outputs[0])[0]
        scene_prob = _softmax(outputs[1])[0]
        sensor_names = ["ir", "sar"]
        scene_names = ["air", "forest", "sea", "urban"]
        sensor_id = int(sensor_prob.argmax())
        scene_id = int(scene_prob.argmax())
        return {
            "sensor": sensor_names[sensor_id],
            "sensor_confidence": float(sensor_prob[sensor_id]),
            "sensor_probabilities": {
                name: float(sensor_prob[index])
                for index, name in enumerate(sensor_names)
            },
            "scene": scene_names[scene_id],
            "scene_confidence": float(scene_prob[scene_id]),
            "scene_probabilities": {
                name: float(scene_prob[index])
                for index, name in enumerate(scene_names)
            },
            "_inference_ms": inference_ms,
        }

    def predict_batch(self, images: Sequence[Image.Image]) -> list[dict[str, Any]]:
        return [self.predict(image) for image in images]

    def close(self) -> None:
        self.model.close()


def load_ascend_context_model(
    options: Mapping[str, Any]
) -> tuple[AscendAclContextModel, dict[str, Any]]:
    model = AscendAclContextModel(options)
    return model, {"preprocessing": {"image_size": model.image_size}}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)
