from __future__ import annotations

import atexit
import ctypes
import math
import threading
import time
import warnings
import weakref
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from fair_agent.core.config import registry_source_key, resolve_path
from fair_agent.core.hashes import sha256_file


ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
DVPP_PIXEL_FORMAT_RGB888 = 12
DVPP_CHANNEL_MODE_VPC = 1
DVPP_CHANNEL_MODE_PNGD = 8
ASCEND_MODEL_ROLES = ("scene", "base", "specialist")
ASCEND_STREAM_PRIORITY_LABELS = ("high", "normal", "low")

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


def _pinned_numpy_buffer(acl: Any, size: int, operation: str) -> tuple[int, Any, np.ndarray]:
    """Allocate non-owning NumPy storage over an ACL pinned host buffer.

    ``acl.util.ptr_to_numpy`` installs an owning base object on CANN 7.0.RC1 and
    therefore double-frees memory that is also released with ``free_host``.
    A ctypes view has no such ownership transfer; the ACL allocation remains
    the sole owner and is explicitly released during model shutdown.
    """

    pointer = int(_value(acl.rt.malloc_host(int(size)), operation))
    owner = (ctypes.c_uint8 * int(size)).from_address(pointer)
    return pointer, owner, np.ctypeslib.as_array(owner)


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
        self.stream_priority_status: dict[str, Any] | None = None
        self._stream_priority_request: tuple[tuple[str, str], ...] | None = None
        self._stream_priority_warning_emitted = False
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

    def resolve_stream_priorities(
        self, requested: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Resolve a real device priority range or explicitly fall back.

        CANN 7.0.RC1 on the target exposes ``create_stream_with_config`` but
        does not expose a PyACL priority-range query; direct board probes show
        that only priority 0 is accepted.  A create function alone is not a
        meaningful high/normal/low capability, so configured candidates use
        ordinary streams unless the runtime can report and successfully probe
        three distinct priority levels.
        """

        if requested is None:
            return {
                "requested": False,
                "supported": False,
                "reason": "not_requested",
                "values": {},
            }
        normalized = {str(key): str(value) for key, value in requested.items()}
        if set(normalized) != set(ASCEND_MODEL_ROLES) or any(
            value not in ASCEND_STREAM_PRIORITY_LABELS
            for value in normalized.values()
        ):
            raise RuntimeError("Ascend stream优先级配置非法")
        request_key = tuple(sorted(normalized.items()))
        with self.lock:
            if self._stream_priority_request is not None:
                if self._stream_priority_request != request_key:
                    raise RuntimeError("同一ACL Runtime不能混用不同stream优先级配置")
                assert self.stream_priority_status is not None
                return dict(self.stream_priority_status)

            self._stream_priority_request = request_key
            self.activate()
            create_with_config = getattr(
                self.acl.rt, "create_stream_with_config", None
            )
            get_priority_range = getattr(
                self.acl.rt, "get_stream_priority_range", None
            )
            reason = ""
            priority_values: dict[str, int] = {}
            if not callable(create_with_config):
                reason = "create_stream_with_config_unavailable"
            elif not callable(get_priority_range):
                reason = "priority_range_api_unavailable"
            else:
                range_result = get_priority_range()
                if not isinstance(range_result, tuple) or len(range_result) != 3:
                    reason = "priority_range_result_invalid"
                else:
                    first, second, result = range_result
                    if int(result) != ACL_SUCCESS:
                        reason = f"priority_range_failed:{int(result)}"
                    else:
                        minimum, maximum = sorted((int(first), int(second)))
                        midpoint = (minimum + maximum + 1) // 2
                        priority_values = {
                            "high": minimum,
                            "normal": midpoint,
                            "low": maximum,
                        }
                        if len(set(priority_values.values())) != 3:
                            reason = "priority_range_has_fewer_than_three_levels"

            created: list[Any] = []
            if not reason:
                try:
                    for value in dict.fromkeys(priority_values.values()):
                        stream_result = create_with_config(int(value), 0)
                        if (
                            not isinstance(stream_result, tuple)
                            or len(stream_result) < 2
                            or int(stream_result[-1]) != ACL_SUCCESS
                        ):
                            error = (
                                stream_result[-1]
                                if isinstance(stream_result, tuple)
                                and stream_result
                                else "invalid_result"
                            )
                            reason = f"priority_probe_failed:{error}"
                            break
                        created.append(stream_result[0])
                finally:
                    for stream in reversed(created):
                        _require(
                            self.acl.rt.destroy_stream(stream),
                            "acl.rt.destroy_stream(priority probe)",
                        )

            supported = not reason
            self.stream_priority_status = {
                "requested": True,
                "supported": supported,
                "reason": "supported" if supported else reason,
                "requested_labels": normalized,
                "values": priority_values if supported else {},
            }
            if not supported and not self._stream_priority_warning_emitted:
                warnings.warn(
                    "Ascend stream priority不受目标PyACL支持，使用普通stream并跳过优先级候选："
                    f"{reason}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._stream_priority_warning_emitted = True
            return dict(self.stream_priority_status)

    def create_model_stream(
        self,
        role: str,
        requested: Mapping[str, Any] | None,
    ) -> Any:
        if role not in ASCEND_MODEL_ROLES:
            raise RuntimeError(f"Ascend模型stream角色非法：{role}")
        status = self.resolve_stream_priorities(requested)
        if status["supported"]:
            label = str(status["requested_labels"][role])
            priority = int(status["values"][label])
            return _value(
                self.acl.rt.create_stream_with_config(priority, 0),
                f"acl.rt.create_stream_with_config({role}={label}:{priority})",
            )
        return _value(self.acl.rt.create_stream(), "acl.rt.create_stream")

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


class AscendAclExecutionHandle:
    """One in-flight model execution with exactly one final device wait."""

    def __init__(
        self,
        model: "AscendAclModel",
        submit_ms: float,
        input_reference: np.ndarray | None,
        preloaded: bool,
    ) -> None:
        self.model = model
        self.submit_ms = float(submit_ms)
        # Pageable async copies may retain the source until the stream reaches
        # the H2D operation. Pinned mode instead retains its resident staging
        # allocation through the model lifetime.
        self.input_reference = input_reference
        self.preloaded = bool(preloaded)
        self._lock = threading.Lock()
        self._completed = False
        self._result: tuple[list[np.ndarray], dict[str, Any]] | None = None
        self._error: BaseException | None = None

    def result(self) -> tuple[list[np.ndarray], dict[str, Any]]:
        with self._lock:
            if self._completed:
                if self._error is not None:
                    raise self._error
                assert self._result is not None
                return self._result
            acl = self.model.runtime.acl
            wait_started = time.perf_counter_ns()
            try:
                with self.model.runtime.lock:
                    self.model.runtime.activate()
                    _require(
                        acl.rt.synchronize_event(self.model.output_copy_end_event),
                        "acl.rt.synchronize_event(model completion)",
                    )
                    copied_indices = list(range(len(self.model.output_contracts)))
                    wait_ms = (time.perf_counter_ns() - wait_started) / 1_000_000.0
                    input_copy_ms = 0.0
                    if self.model.detailed_event_timing and not self.preloaded:
                        input_copy_ms = self.model._event_elapsed(
                            self.model.input_copy_start_event,
                            self.model.inference_start_event,
                            "input copy",
                        )
                    inference_ms = self.model._event_elapsed(
                        self.model.inference_start_event,
                        self.model.inference_end_event,
                        "model",
                    )
                    output_copy_ms = 0.0
                    if self.model.detailed_event_timing:
                        output_copy_ms = self.model._event_elapsed(
                            self.model.inference_end_event,
                            self.model.output_copy_end_event,
                            "output copy",
                        )
                outputs = self.model._output_views()
                self._result = (
                    outputs,
                    {
                        "submit_ms": self.submit_ms,
                        "wait_ms": wait_ms,
                        "input_copy_ms": input_copy_ms,
                        "inference_ms": inference_ms,
                        "output_copy_ms": output_copy_ms,
                        "output_copy_mode": "all",
                        "copied_output_indices": tuple(copied_indices),
                    },
                )
            except BaseException as exc:
                # A failed event wait or timing query must not leave a stream
                # using resident host buffers while close() tears them down.
                try:
                    with self.model.runtime.lock:
                        if not self.model.runtime.closed:
                            self.model.runtime.activate()
                            acl.rt.synchronize_stream(self.model.stream)
                except Exception:
                    pass
                self._error = exc
            finally:
                self.input_reference = None
                self._completed = True
                self.model._finish_handle(self)
            if self._error is not None:
                raise self._error
            assert self._result is not None
            return self._result


class AscendAclModel:
    def __init__(
        self,
        runtime: AscendAclRuntime,
        path: Path,
        execution_mode: str = "synchronous",
        memory_mode: str = "pageable",
        schedule_mode: str = "threaded_execute",
        detailed_event_timing: bool = True,
        stream_role: str = "base",
        stream_priorities: Mapping[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.path = path
        self.execution_mode = str(execution_mode)
        if self.execution_mode not in {"synchronous", "async_stream"}:
            raise RuntimeError(f"Ascend执行模式非法：{self.execution_mode}")
        self.memory_mode = str(memory_mode)
        if self.memory_mode not in {"pageable", "pinned"}:
            raise RuntimeError(f"Ascend内存模式非法：{self.memory_mode}")
        if self.memory_mode == "pinned" and self.execution_mode != "async_stream":
            raise RuntimeError("Ascend锁页内存要求async_stream执行模式")
        self.schedule_mode = str(schedule_mode)
        if self.schedule_mode not in {"threaded_execute", "unified_enqueue"}:
            raise RuntimeError(f"Ascend调度模式非法：{self.schedule_mode}")
        if not isinstance(detailed_event_timing, bool):
            raise RuntimeError("Ascend详细event计时开关必须是布尔值")
        self.detailed_event_timing = detailed_event_timing
        self.stream_role = str(stream_role)
        if self.stream_role not in ASCEND_MODEL_ROLES:
            raise RuntimeError(f"Ascend模型stream角色非法：{self.stream_role}")
        self.stream_priority_label = (
            str(stream_priorities[self.stream_role])
            if stream_priorities is not None
            else "normal"
        )
        self.stream_priority_supported = False
        self.stream_priority_value: int | None = None
        self.closed = False
        self.accepting_submissions = True
        self.close_lock = threading.Lock()
        self.execution_lock = threading.Lock()
        self.execution_condition = threading.Condition(self.execution_lock)
        self._submission_in_progress = False
        self._outstanding: AscendAclExecutionHandle | None = None
        self.model_id: int | None = None
        self.desc: Any = None
        self.stream: Any = None
        self.input_copy_start_event: Any = None
        self.inference_start_event: Any = None
        self.inference_end_event: Any = None
        self.output_copy_end_event: Any = None
        self.async_runs = 0
        self.threaded_runs = 0
        self.input_dataset: Any = None
        self.output_dataset: Any = None
        self.device_buffers: list[int] = []
        self.data_buffers: list[Any] = []
        self.output_contracts: list[dict[str, Any]] = []
        self.output_hosts: list[np.ndarray] = []
        self.host_pointers: list[int] = []
        self.host_owners: list[Any] = []
        self.input_host_pointer: int | None = None
        self.input_host: np.ndarray | None = None
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
                    self.stream = runtime.create_model_stream(
                        self.stream_role,
                        stream_priorities,
                    )
                    priority_status = runtime.resolve_stream_priorities(
                        stream_priorities
                    )
                    self.stream_priority_supported = bool(
                        priority_status["supported"]
                    )
                    if self.stream_priority_supported:
                        self.stream_priority_value = int(
                            priority_status["values"][self.stream_priority_label]
                        )
                    if self.detailed_event_timing:
                        self.input_copy_start_event = self._create_event(
                            "input copy start"
                        )
                    self.inference_start_event = self._create_event("inference start")
                    self.inference_end_event = self._create_event("inference end")
                    self.output_copy_end_event = self._create_event("output copy end")
                runtime.register(self)
            except Exception:
                self.close()
                raise

    def _create_event(self, name: str) -> Any:
        return _value(
            self.runtime.acl.rt.create_event(), f"acl.rt.create_event({name})"
        )

    def _create_datasets(self) -> None:
        acl = self.runtime.acl
        if int(acl.mdl.get_num_inputs(self.desc)) != 1:
            raise RuntimeError(f"{self.path.name}必须是单输入静态模型")
        input_dims = _dims(acl.mdl.get_input_dims(self.desc, 0), "acl.mdl.get_input_dims")
        self.input_shape = tuple(int(value) for value in input_dims["dims"])
        self.input_dtype = _dtype(int(acl.mdl.get_input_data_type(self.desc, 0)))
        self.input_size = int(acl.mdl.get_input_size_by_index(self.desc, 0))
        if self.memory_mode == "pinned":
            pointer, owner, host = _pinned_numpy_buffer(
                acl, self.input_size, "acl.rt.malloc_host(input)"
            )
            self.input_host_pointer = pointer
            self.input_host = host
            self.host_pointers.append(pointer)
            self.host_owners.append(owner)
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
            contract = {
                "index": index,
                "shape": tuple(int(value) for value in dims["dims"]),
                "dtype": data_type,
                "size": size,
                "device": device,
            }
            self.output_contracts.append(contract)
            # The web engine serializes requests and consumes raw output before
            # the next invocation. Reusing resident host buffers avoids both
            # allocation and an additional post-copy on every image.
            if self.memory_mode == "pinned":
                pointer, owner, host = _pinned_numpy_buffer(
                    acl, size, f"acl.rt.malloc_host(output {index})"
                )
                self.host_pointers.append(pointer)
                self.host_owners.append(owner)
                contract["host_pointer"] = pointer
                self.output_hosts.append(host)
            else:
                self.output_hosts.append(np.empty(size, dtype=np.uint8))

    def _output_views(self) -> list[np.ndarray]:
        return [
            host.view(contract["dtype"]).reshape(contract["shape"])
            for contract, host in zip(self.output_contracts, self.output_hosts)
        ]

    def _copy_output_indices_synchronous(
        self, indices: Sequence[int]
    ) -> list[np.ndarray]:
        acl = self.runtime.acl
        for index in indices:
            contract = self.output_contracts[int(index)]
            host = self.output_hosts[int(index)]
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
        return self._output_views()

    def _enqueue_output_indices(self, indices: Sequence[int]) -> None:
        acl = self.runtime.acl
        for index in indices:
            contract = self.output_contracts[int(index)]
            host = self.output_hosts[int(index)]
            host_pointer = int(
                contract.get("host_pointer") or _numpy_pointer(acl, host)
            )
            _require(
                acl.rt.memcpy_async(
                    host_pointer,
                    host.nbytes,
                    contract["device"],
                    contract["size"],
                    ACL_MEMCPY_DEVICE_TO_HOST,
                    self.stream,
                ),
                f"acl.rt.memcpy_async(output {contract['index']})",
            )

    def _copy_outputs_synchronous(self) -> list[np.ndarray]:
        return self._copy_output_indices_synchronous(
            tuple(range(len(self.output_contracts)))
        )

    def _event_elapsed(self, start: Any, end: Any, name: str) -> float:
        return float(
            _value(
                self.runtime.acl.rt.event_elapsed_time(start, end),
                f"acl.rt.event_elapsed_time({name})",
            )
        )

    def _finish_handle(self, handle: AscendAclExecutionHandle) -> None:
        with self.execution_condition:
            if self._outstanding is handle:
                self._outstanding = None
            self.execution_condition.notify_all()

    def _claim_submission(self) -> None:
        with self.execution_condition:
            if self.closed or not self.accepting_submissions:
                raise RuntimeError(f"{self.path.name}已经关闭，拒绝Ascend提交")
            if self._submission_in_progress or self._outstanding is not None:
                raise RuntimeError(f"{self.path.name}已有未完成的Ascend执行")
            self._submission_in_progress = True

    def _reset_async_events(self) -> None:
        if not self.async_runs:
            return
        acl = self.runtime.acl
        for event, name in (
            (self.input_copy_start_event, "input copy start"),
            (self.inference_start_event, "inference start"),
            (self.inference_end_event, "inference end"),
            (self.output_copy_end_event, "output copy end"),
        ):
            if event is None:
                continue
            _require(
                acl.rt.reset_event(event, self.stream),
                f"acl.rt.reset_event({name})",
            )

    def _enqueue(
        self,
        batch: np.ndarray | None,
        ready_event: Any | None,
    ) -> AscendAclExecutionHandle:
        if self.execution_mode != "async_stream" or self.stream is None:
            raise RuntimeError("Ascend异步提交要求async_stream执行模式")
        self._claim_submission()
        acl = self.runtime.acl
        submit_started = time.perf_counter_ns()
        input_reference: np.ndarray | None = None
        try:
            with self.runtime.lock:
                self.runtime.activate()
                self._reset_async_events()
                if ready_event is None:
                    assert batch is not None
                    if self.memory_mode == "pinned":
                        assert self.input_host is not None
                        np.copyto(
                            self.input_host,
                            batch.view(np.uint8).reshape(-1),
                        )
                        source_pointer = int(self.input_host_pointer)
                    else:
                        input_reference = batch
                        source_pointer = _numpy_pointer(acl, batch)
                    if self.detailed_event_timing:
                        _require(
                            acl.rt.record_event(
                                self.input_copy_start_event, self.stream
                            ),
                            "acl.rt.record_event(input copy start)",
                        )
                    _require(
                        acl.rt.memcpy_async(
                            self.input_device,
                            self.input_size,
                            source_pointer,
                            self.input_size,
                            ACL_MEMCPY_HOST_TO_DEVICE,
                            self.stream,
                        ),
                        "acl.rt.memcpy_async(input)",
                    )
                else:
                    _require(
                        acl.rt.stream_wait_event(self.stream, ready_event),
                        "acl.rt.stream_wait_event(input)",
                    )
                    if self.detailed_event_timing:
                        # Keep a well-defined zero-length input-copy interval
                        # for preloaded device inputs without another wait.
                        _require(
                            acl.rt.record_event(
                                self.input_copy_start_event, self.stream
                            ),
                            "acl.rt.record_event(preloaded input)",
                        )
                _require(
                    acl.rt.record_event(self.inference_start_event, self.stream),
                    "acl.rt.record_event(inference start)",
                )
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
                    acl.rt.record_event(self.inference_end_event, self.stream),
                    "acl.rt.record_event(inference end)",
                )
                self._enqueue_output_indices(tuple(range(len(self.output_contracts))))
                _require(
                    acl.rt.record_event(self.output_copy_end_event, self.stream),
                    "acl.rt.record_event(output copy end)",
                )
                self.async_runs += 1
            handle = AscendAclExecutionHandle(
                self,
                (time.perf_counter_ns() - submit_started) / 1_000_000.0,
                input_reference,
                ready_event is not None,
            )
            with self.execution_condition:
                if self._outstanding is not None:
                    raise RuntimeError(f"{self.path.name}Ascend提交状态冲突")
                self._outstanding = handle
                self._submission_in_progress = False
                self.execution_condition.notify_all()
            return handle
        except BaseException:
            try:
                with self.runtime.lock:
                    if not self.runtime.closed and self.stream is not None:
                        self.runtime.activate()
                        acl.rt.synchronize_stream(self.stream)
            except Exception:
                pass
            with self.execution_condition:
                self._submission_in_progress = False
                self.execution_condition.notify_all()
            raise

    def submit(self, array: np.ndarray) -> AscendAclExecutionHandle:
        batch = np.ascontiguousarray(array, dtype=self.input_dtype)
        if tuple(batch.shape) != self.input_shape:
            raise RuntimeError(
                f"{self.path.name}输入shape错误：{batch.shape} != {self.input_shape}"
            )
        if batch.nbytes != self.input_size:
            raise RuntimeError(
                f"{self.path.name}输入字节数错误：{batch.nbytes} != {self.input_size}"
            )
        return self._enqueue(batch, None)

    def submit_preloaded(self, ready_event: Any) -> AscendAclExecutionHandle:
        return self._enqueue(None, ready_event)

    def _execute_threaded_async(
        self,
        batch: np.ndarray | None,
        ready_event: Any | None,
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        """Execute one async stream and copy all outputs synchronously."""

        if self.execution_mode != "async_stream" or self.stream is None:
            raise RuntimeError("Ascend threaded_execute要求async_stream执行模式")
        self._claim_submission()
        acl = self.runtime.acl
        submit_started = time.perf_counter_ns()
        try:
            with self.execution_lock:
                self.runtime.activate()
                if self.threaded_runs:
                    for event, name in (
                        (self.inference_start_event, "inference start"),
                        (self.inference_end_event, "inference end"),
                    ):
                        _require(
                            acl.rt.reset_event(event, self.stream),
                            f"acl.rt.reset_event({name})",
                        )
                input_copy_ms = 0.0
                if ready_event is None:
                    assert batch is not None
                    input_copy_started = time.perf_counter_ns()
                    source = batch
                    source_pointer = _numpy_pointer(acl, source)
                    if self.memory_mode == "pinned":
                        assert self.input_host is not None
                        np.copyto(
                            self.input_host,
                            batch.view(np.uint8).reshape(-1),
                        )
                        source = self.input_host
                        source_pointer = int(self.input_host_pointer)
                    _require(
                        acl.rt.memcpy(
                            self.input_device,
                            self.input_size,
                            source_pointer,
                            self.input_size,
                            ACL_MEMCPY_HOST_TO_DEVICE,
                        ),
                        "acl.rt.memcpy(input threaded_execute)",
                    )
                    input_copy_ms = (
                        time.perf_counter_ns() - input_copy_started
                    ) / 1_000_000.0
                else:
                    _require(
                        acl.rt.stream_wait_event(self.stream, ready_event),
                        "acl.rt.stream_wait_event(input threaded_execute)",
                    )
                _require(
                    acl.rt.record_event(self.inference_start_event, self.stream),
                    "acl.rt.record_event(inference start threaded_execute)",
                )
                _require(
                    acl.mdl.execute_async(
                        self.model_id,
                        self.input_dataset,
                        self.output_dataset,
                        self.stream,
                    ),
                    "acl.mdl.execute_async(threaded_execute)",
                )
                _require(
                    acl.rt.record_event(self.inference_end_event, self.stream),
                    "acl.rt.record_event(inference end threaded_execute)",
                )
                submit_ms = (
                    time.perf_counter_ns() - submit_started
                ) / 1_000_000.0
                wait_started = time.perf_counter_ns()
                _require(
                    acl.rt.synchronize_stream(self.stream),
                    "acl.rt.synchronize_stream(threaded_execute)",
                )
                wait_ms = (
                    time.perf_counter_ns() - wait_started
                ) / 1_000_000.0
                inference_ms = self._event_elapsed(
                    self.inference_start_event,
                    self.inference_end_event,
                    "model threaded_execute",
                )
                output_copy_started = time.perf_counter_ns()
                copied_indices = list(range(len(self.output_contracts)))
                outputs = self._copy_outputs_synchronous()
                output_copy_ms = (
                    time.perf_counter_ns() - output_copy_started
                ) / 1_000_000.0
                self.threaded_runs += 1
            return outputs, {
                "submit_ms": submit_ms,
                "wait_ms": wait_ms,
                "input_copy_ms": input_copy_ms,
                "inference_ms": inference_ms,
                "output_copy_ms": output_copy_ms,
                "output_copy_mode": "all",
                "copied_output_indices": tuple(copied_indices),
            }
        except BaseException:
            try:
                if not self.runtime.closed and self.stream is not None:
                    self.runtime.activate()
                    acl.rt.synchronize_stream(self.stream)
            except Exception:
                pass
            raise
        finally:
            with self.execution_condition:
                self._submission_in_progress = False
                self.execution_condition.notify_all()

    def execute_threaded(
        self,
        array: np.ndarray,
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        batch = np.ascontiguousarray(array, dtype=self.input_dtype)
        if tuple(batch.shape) != self.input_shape:
            raise RuntimeError(
                f"{self.path.name}输入shape错误：{batch.shape} != {self.input_shape}"
            )
        if batch.nbytes != self.input_size:
            raise RuntimeError(
                f"{self.path.name}输入字节数错误：{batch.nbytes} != {self.input_size}"
            )
        return self._execute_threaded_async(batch, None)

    def execute_preloaded_threaded(
        self,
        ready_event: Any,
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        return self._execute_threaded_async(None, ready_event)

    def execute(self, array: np.ndarray) -> tuple[list[np.ndarray], float]:
        if self.execution_mode == "async_stream":
            if self.schedule_mode == "threaded_execute":
                outputs, timings = self.execute_threaded(array)
            else:
                outputs, timings = self.submit(array).result()
            return outputs, timings["inference_ms"]
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
            _require(
                acl.mdl.execute(
                    self.model_id, self.input_dataset, self.output_dataset
                ),
                "acl.mdl.execute",
            )
            inference_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            outputs = self._copy_outputs_synchronous()
        return outputs, inference_ms

    def execute_preloaded(self, ready_event: Any) -> tuple[list[np.ndarray], float]:
        if self.schedule_mode == "threaded_execute":
            outputs, timings = self.execute_preloaded_threaded(ready_event)
        else:
            outputs, timings = self.submit_preloaded(ready_event).result()
        return outputs, timings["inference_ms"]

    def close(self) -> None:
        with self.close_lock:
            if self.closed:
                return
            with self.execution_condition:
                self.accepting_submissions = False
                while self._submission_in_progress:
                    self.execution_condition.wait()
                outstanding = self._outstanding
            outstanding_error: BaseException | None = None
            if outstanding is not None:
                try:
                    outstanding.result()
                except BaseException as exc:
                    outstanding_error = exc
            acl = self.runtime.acl
            with self.runtime.lock:
                if self.runtime.closed:
                    self.closed = True
                    return
                self.runtime.activate()
                for event in (
                    self.output_copy_end_event,
                    self.inference_end_event,
                    self.inference_start_event,
                    self.input_copy_start_event,
                ):
                    if event is not None:
                        acl.rt.destroy_event(event)
                if self.stream is not None:
                    acl.rt.destroy_stream(self.stream)
                for buffer in reversed(self.data_buffers):
                    acl.destroy_data_buffer(buffer)
                if self.input_dataset is not None:
                    acl.mdl.destroy_dataset(self.input_dataset)
                if self.output_dataset is not None:
                    acl.mdl.destroy_dataset(self.output_dataset)
                for device in reversed(self.device_buffers):
                    acl.rt.free(device)
                for pointer in reversed(self.host_pointers):
                    acl.rt.free_host(pointer)
                self.host_pointers.clear()
                self.host_owners.clear()
                self.input_host = None
                self.output_hosts.clear()
                if self.desc is not None:
                    acl.mdl.destroy_desc(self.desc)
                if self.model_id is not None:
                    acl.mdl.unload(self.model_id)
                self.runtime.unregister(self)
                self.closed = True
            if outstanding_error is not None:
                raise outstanding_error

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _validated_model_entry(
    options: Mapping[str, Any], weights: str | Path
) -> Mapping[str, Any]:
    source_key = registry_source_key(weights)
    entry = options.get("models", {}).get(source_key)
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"Ascend配置未登记模型：{source_key}")
    return entry


def _load_model(
    options: Mapping[str, Any],
    entry: Mapping[str, Any],
    stream_role: str,
) -> AscendAclModel:
    from fair_agent.modules.ascend_release import require_ascend_runtime_artifacts

    require_ascend_runtime_artifacts(options)
    path = resolve_path(entry["path"])
    if not path.is_file():
        raise RuntimeError(f"Ascend OM不存在：{path}")
    digest = str(entry.get("sha256") or "")
    if len(digest) != 64 or sha256_file(path) != digest:
        raise RuntimeError(f"Ascend OM哈希缺失或不匹配：{path}")
    detailed_event_timing = options.get("detailed_event_timing", True)
    if not isinstance(detailed_event_timing, bool):
        raise RuntimeError("Ascend detailed_event_timing必须是布尔值")
    runtime = AscendAclRuntime.acquire(int(options.get("device_id", 0)))
    stream_priorities = options.get("stream_priorities")
    if stream_priorities is not None and not isinstance(stream_priorities, Mapping):
        raise RuntimeError("Ascend stream_priorities必须是映射")
    runtime.resolve_stream_priorities(stream_priorities)
    return AscendAclModel(
        runtime,
        path,
        execution_mode=str(options.get("execution_mode", "synchronous")),
        memory_mode=str(options.get("memory_mode", "pageable")),
        schedule_mode=str(options.get("schedule_mode", "threaded_execute")),
        detailed_event_timing=detailed_event_timing,
        stream_role=stream_role,
        stream_priorities=stream_priorities,
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
    """Decode one fixed production PNG into the resident AIPP inputs."""

    source_width = 640
    source_height = 512

    def __init__(
        self,
        runtime: AscendAclRuntime,
        base_model: AscendAclModel,
        specialist_model: AscendAclModel | None,
        context_model: AscendAclModel,
        scene_resize_stages: Any = None,
        max_encoded_bytes: int = 2 * 1024 * 1024,
        prepare_context: bool = True,
    ) -> None:
        contracts = [(base_model, tuple(base_model.input_shape), "base")]
        if specialist_model is not None:
            contracts.append(
                (specialist_model, tuple(specialist_model.input_shape), "specialist")
            )
        contracts.append((context_model, (1, 160, 160, 3), "context"))
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
        detector_shape = (1, 608, 736, 3)
        if tuple(base_model.input_shape) != detector_shape:
            raise RuntimeError(
                "Ascend310B v2 Base DVPP输入必须为1x608x736x3："
                f"{base_model.input_shape}"
            )
        specialist_shape = (
            tuple(specialist_model.input_shape)
            if specialist_model is not None
            else None
        )
        if specialist_shape not in {None, detector_shape}:
            raise RuntimeError(
                "Ascend310B v2 Specialist DVPP输入必须与Base相同："
                f"{specialist_shape}"
            )
        self.runtime = runtime
        self.base_model = base_model
        self.specialist_model = specialist_model
        self.context_model = context_model
        self.base_height = int(base_model.input_shape[1])
        self.base_width = int(base_model.input_shape[2])
        self.specialist_clone_from_base = (
            specialist_model is not None
            and specialist_shape == tuple(base_model.input_shape)
        )
        self.prepare_context = bool(prepare_context)
        if len({model.memory_mode for model, _shape, _name in contracts}) != 1:
            raise RuntimeError("Ascend DVPP三个模型必须使用相同内存模式")
        self.memory_mode = base_model.memory_mode
        if len({model.schedule_mode for model, _shape, _name in contracts}) != 1:
            raise RuntimeError("Ascend DVPP三个模型必须使用相同调度模式")
        self.schedule_mode = base_model.schedule_mode
        if len(
            {model.detailed_event_timing for model, _shape, _name in contracts}
        ) != 1:
            raise RuntimeError("Ascend DVPP三个模型必须使用相同event计时模式")
        self.detailed_event_timing = base_model.detailed_event_timing
        self.max_encoded_bytes = int(max_encoded_bytes)
        if self.max_encoded_bytes <= 0:
            raise RuntimeError("Ascend编码输入上限必须为正数")
        self.scene_resize_stages = validate_dvpp_scene_resize_stages(
            scene_resize_stages
        )
        self.lock = threading.Lock()
        self.closed = False
        self.stream: Any = None
        self.base_ready_event: Any = None
        self.specialist_ready_event: Any = None
        self.context_ready_event: Any = None
        self.timing_start_event: Any = None
        self.prepared_runs = 0
        self.channel: Any = None
        self.descriptors: list[Any] = []
        self.resize_configs: list[Any] = []
        self.scene_intermediate_descs: list[Any] = []
        self.roi: Any = None
        self.dvpp_buffers: list[int] = []
        self.encoded_host_pointer: int | None = None
        self.encoded_host_owner: Any = None
        self.encoded_host: np.ndarray | None = None
        self.encoded_host_capacity = 0
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
            and color_type in {0, 2, 6}
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
        self.specialist_ready_event = (
            _value(acl.rt.create_event(), "acl.rt.create_event(specialist input)")
            if self.specialist_model is not None
            else None
        )
        self.context_ready_event = _value(
            acl.rt.create_event(), "acl.rt.create_event(context input)"
        )
        if self.detailed_event_timing:
            self.timing_start_event = _value(
                acl.rt.create_event(), "acl.rt.create_event(DVPP start)"
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

        expected_decoded_size = (
            _align(self.source_width, 16)
            * 3
            * _align(self.source_height, 2)
        )
        decoded_pointer = self._dvpp_buffer(
            expected_decoded_size,
            "acl.media.dvpp_malloc(decoded input)",
        )
        self.decoded_desc, self.decoded_width_stride, _, decoded_size = (
            self._picture_desc(
                decoded_pointer,
                self.source_width,
                self.source_height,
            )
        )
        if decoded_size != expected_decoded_size:
            raise RuntimeError("Ascend RGB888 PNGD输出大小计算不一致")
        self.decoded_size = decoded_size

        base_scale = min(
            self.base_width / self.source_width,
            self.base_height / self.source_height,
        )
        self.base_resize_width = int(round(self.source_width * base_scale))
        self.base_resize_height = int(round(self.source_height * base_scale))
        if self.base_resize_width != self.base_width:
            raise RuntimeError("Ascend DVPP当前只支持无水平padding的检测输入")
        self.base_pad_top = (self.base_height - self.base_resize_height) // 2
        if self.base_pad_top < 0:
            raise RuntimeError("Ascend Base letterbox padding非法")
        base_resize_size = (
            _align(self.base_resize_width, 16)
            * 3
            * _align(self.base_resize_height, 2)
        )
        self.base_resize_pointer = self._dvpp_buffer(
            base_resize_size, "acl.media.dvpp_malloc(base resize)"
        )
        self.base_resize_desc, self.base_width_stride, _, _ = self._picture_desc(
            self.base_resize_pointer,
            self.base_resize_width,
            self.base_resize_height,
        )
        self.base_desc, base_width_stride, _, base_size = self._picture_desc(
            self.base_model.input_device, self.base_width, self.base_height
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

    def _ensure_encoded_host(self, size: int) -> tuple[int, np.ndarray]:
        if size > self.max_encoded_bytes:
            raise ValueError(
                f"Ascend编码PNG超过锁页staging上限：{size} > {self.max_encoded_bytes}"
            )
        if self.memory_mode != "pinned":
            raise RuntimeError("Ascend锁页encoded staging仅在pinned模式分配")
        if size > self.encoded_host_capacity:
            capacity = max(size, min(self.max_encoded_bytes, max(64 * 1024, self.encoded_host_capacity * 2)))
            if self.encoded_host_pointer is not None:
                _require(
                    self.runtime.acl.rt.synchronize_stream(self.stream),
                    "acl.rt.synchronize_stream(encoded staging grow)",
                )
                _require(
                    self.runtime.acl.rt.free_host(self.encoded_host_pointer),
                    "acl.rt.free_host(encoded staging grow)",
                )
            pointer, owner, host = _pinned_numpy_buffer(
                self.runtime.acl,
                capacity,
                "acl.rt.malloc_host(encoded staging)",
            )
            self.encoded_host_pointer = pointer
            self.encoded_host_owner = owner
            self.encoded_host = host
            self.encoded_host_capacity = capacity
        assert self.encoded_host_pointer is not None and self.encoded_host is not None
        return self.encoded_host_pointer, self.encoded_host[:size]

    def prepare(self, data: bytes) -> float:
        if not self.accepts(data):
            raise ValueError("Ascend DVPP仅接受固定640x512的8位灰度/RGB/RGBA PNG")
        if self.closed:
            raise RuntimeError("Ascend DVPP编码预处理器已经关闭")
        if len(data) > self.max_encoded_bytes:
            raise ValueError(
                f"Ascend编码PNG超过上传上限：{len(data)} > {self.max_encoded_bytes}"
            )
        acl = self.runtime.acl
        with self.lock:
            self.runtime.activate()
            if self.memory_mode == "pinned":
                encoded_pointer, encoded = self._ensure_encoded_host(len(data))
                np.copyto(encoded, np.frombuffer(data, dtype=np.uint8))
            else:
                encoded = np.frombuffer(data, dtype=np.uint8)
                encoded_pointer = _numpy_pointer(acl, encoded)
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
            if predicted_size != self.decoded_size:
                raise ValueError("Ascend DVPP PNG解码大小与固定RGB输出不一致")
            started = time.perf_counter_ns()
            if self.prepared_runs:
                reset_events = [(self.base_ready_event, "base")]
                if self.prepare_context:
                    reset_events.append((self.context_ready_event, "context"))
                if self.specialist_ready_event is not None:
                    reset_events.insert(
                        1,
                        (self.specialist_ready_event, "specialist"),
                    )
                if self.detailed_event_timing:
                    reset_events.insert(0, (self.timing_start_event, "DVPP start"))
                for event, name in reset_events:
                    _require(
                        acl.rt.reset_event(event, self.stream),
                        f"acl.rt.reset_event({name})",
                    )
            if self.detailed_event_timing:
                _require(
                    acl.rt.record_event(self.timing_start_event, self.stream),
                    "acl.rt.record_event(DVPP start)",
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
                    self.base_model.input_device
                    + self.base_pad_top * self.base_width_stride,
                    self.base_width_stride * self.base_resize_height,
                    self.base_resize_pointer,
                    self.base_width_stride * self.base_resize_height,
                    ACL_MEMCPY_DEVICE_TO_DEVICE,
                    self.stream,
                ),
                "acl.rt.memcpy_async(base letterbox)",
            )
            _require(
                acl.rt.record_event(self.base_ready_event, self.stream),
                "acl.rt.record_event(base input)",
            )
            if self.specialist_clone_from_base:
                assert self.specialist_model is not None
                _require(
                    acl.rt.memcpy_async(
                        self.specialist_model.input_device,
                        self.specialist_model.input_size,
                        self.base_model.input_device,
                        self.base_model.input_size,
                        ACL_MEMCPY_DEVICE_TO_DEVICE,
                        self.stream,
                    ),
                    "acl.rt.memcpy_async(specialist shared letterbox)",
                )
                _require(
                    acl.rt.record_event(self.specialist_ready_event, self.stream),
                    "acl.rt.record_event(specialist shared input)",
                )
            if self.prepare_context:
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

    def device_ms(self) -> float:
        if not self.prepared_runs or not self.detailed_event_timing:
            return 0.0
        with self.runtime.lock:
            self.runtime.activate()
            # CANN 7.0.RC1 does not make a cross-stream dependency sufficient
            # for event_elapsed_time(): the producing event must be explicitly
            # synchronized on the host first, even after a dependent model
            # completion event has finished. At this point all model handles
            # have already completed, so this confirms timestamp visibility
            # without extending the device critical path.
            completion_event = (
                self.context_ready_event
                if self.prepare_context
                else self.base_ready_event
            )
            _require(
                self.runtime.acl.rt.synchronize_event(completion_event),
                "acl.rt.synchronize_event(DVPP completion)",
            )
            return float(
                _value(
                    self.runtime.acl.rt.event_elapsed_time(
                        self.timing_start_event, completion_event
                    ),
                    "acl.rt.event_elapsed_time(DVPP)",
                )
            )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        acl = self.runtime.acl
        with self.runtime.lock:
            if self.runtime.closed:
                return
            self.runtime.activate()
            if self.stream is not None:
                _require(
                    acl.rt.synchronize_stream(self.stream),
                    "acl.rt.synchronize_stream(DVPP close)",
                )
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
                self.timing_start_event,
            ):
                if event is not None:
                    acl.rt.destroy_event(event)
            if self.stream is not None:
                acl.rt.destroy_stream(self.stream)
            if self.encoded_host_pointer is not None:
                acl.rt.free_host(self.encoded_host_pointer)
                self.encoded_host_pointer = None
                self.encoded_host_owner = None
                self.encoded_host = None
                self.encoded_host_capacity = 0
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


def _restore_detection_rows(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    info: Mapping[str, float | int],
) -> list[dict[str, Any]]:
    scale = float(info["scale"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"Ascend检测坐标缩放非法：{scale}")
    original_width = float(info["original_width"])
    original_height = float(info["original_height"])
    rows = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        restored = [
            min(max((float(x1) - float(info["pad_left"])) / scale, 0.0), original_width),
            min(max((float(y1) - float(info["pad_top"])) / scale, 0.0), original_height),
            min(max((float(x2) - float(info["pad_left"])) / scale, 0.0), original_width),
            min(max((float(y2) - float(info["pad_top"])) / scale, 0.0), original_height),
        ]
        rows.append(
            {
                "class_id": int(class_id),
                "confidence": float(score),
                "xyxy": restored,
            }
        )
    return rows


def yolo26_e2e_v1_records(
    outputs: Sequence[np.ndarray],
    info: Mapping[str, float | int],
    confidence: float,
    max_det: int,
    *,
    contract_max_det: int,
    class_count: int,
) -> list[dict[str, Any]]:
    """Parse Ultralytics YOLO26 end-to-end detections without Host NMS.

    YOLO26 exports a single fixed ``[1, max_det, 6]`` tensor whose rows are
    ``xyxy, score, class_id`` in letterboxed input coordinates.  Selection and
    duplicate suppression are already part of the exported graph.
    """

    if len(outputs) != 1:
        raise RuntimeError(f"yolo26_e2e_v1输出数量错误：{len(outputs)} != 1")
    detections = np.asarray(outputs[0])
    expected_shape = (1, int(contract_max_det), 6)
    if tuple(detections.shape) != expected_shape:
        raise RuntimeError(
            f"yolo26_e2e_v1输出shape错误：{tuple(detections.shape)} != {expected_shape}"
        )
    if detections.dtype != np.dtype(np.float32):
        raise RuntimeError(
            f"yolo26_e2e_v1输出dtype错误：{detections.dtype} != float32"
        )
    if int(contract_max_det) <= 0 or int(class_count) <= 0:
        raise RuntimeError("yolo26_e2e_v1 max_det/class_count契约非法")
    requested_max_det = int(max_det)
    if requested_max_det <= 0 or requested_max_det > int(contract_max_det):
        raise RuntimeError(
            "yolo26_e2e_v1 max_det超出设备契约："
            f"{requested_max_det} not in [1, {contract_max_det}]"
        )

    rows = detections[0]
    boxes = rows[:, :4]
    scores = rows[:, 4]
    class_values = rows[:, 5]
    if not (
        np.all(np.isfinite(boxes))
        and np.all(np.isfinite(scores))
        and np.all(np.isfinite(class_values))
    ):
        raise RuntimeError("yolo26_e2e_v1包含非有限检测值")
    rounded_classes = np.rint(class_values)
    if np.any(np.abs(class_values - rounded_classes) > 1e-4):
        raise RuntimeError("yolo26_e2e_v1包含非整数类别ID")
    class_ids = rounded_classes.astype(np.int32, copy=False)
    if np.any(class_ids < 0) or np.any(class_ids >= int(class_count)):
        raise RuntimeError("yolo26_e2e_v1包含越界类别ID")

    selected = np.flatnonzero(scores > float(confidence))[:requested_max_det]
    if len(selected) > 1 and np.any(scores[selected][1:] > scores[selected][:-1]):
        raise RuntimeError("yolo26_e2e_v1有效记录没有按置信度降序输出")
    return _restore_detection_rows(
        boxes[selected], scores[selected], class_ids[selected], info
    )


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


class AscendDetectorExecutionHandle:
    def __init__(
        self,
        backend: "AscendAclBackend",
        model_handle: AscendAclExecutionHandle,
        info: Mapping[str, float | int],
        options: Mapping[str, Any],
        preprocess_ms: float,
    ) -> None:
        self.backend = backend
        self.model_handle = model_handle
        self.info = dict(info)
        self.options = dict(options)
        self.preprocess_ms = float(preprocess_ms)
        self._result: AscendResult | None = None
        self._lock = threading.Lock()

    def result(self) -> AscendResult:
        with self._lock:
            if self._result is not None:
                return self._result
            outputs, execution = self.model_handle.result()
            self._result = self.backend._result_from_outputs(
                outputs,
                self.info,
                self.options,
                self.preprocess_ms,
                execution,
            )
            return self._result


class AscendAclBackend:
    name = "ascend_acl"

    def __init__(
        self, options: Mapping[str, Any], weights: str | Path | None = None
    ) -> None:
        if weights is None:
            raise RuntimeError("Ascend检测后端缺少模型标识")
        entry = _validated_model_entry(options, weights)
        stream_role = str(options.get("_stream_role", "base"))
        if stream_role not in {"base", "specialist"}:
            raise RuntimeError(f"Ascend检测模型stream角色非法：{stream_role}")
        self.model = _load_model(options, entry, stream_role)
        self.output_contract = str(entry.get("output_contract") or "")
        if self.output_contract != "yolo26_e2e_v1":
            raise RuntimeError(
                f"Ascend v2 检测输出契约必须是yolo26_e2e_v1：{self.output_contract}"
            )
        self.contract_max_det = int(entry.get("max_det", 0))
        self.class_count = int(entry.get("class_count", 0))
        expected = (((1, self.contract_max_det, 6), np.dtype(np.float32)),)
        actual = tuple(
            (tuple(contract["shape"]), contract["dtype"])
            for contract in self.model.output_contracts
        )
        if self.contract_max_det <= 0 or self.class_count <= 0 or actual != expected:
            raise RuntimeError(
                f"yolo26_e2e_v1 OM输出契约错误：{actual} != {expected}"
            )
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

    def _result_from_outputs(
        self,
        outputs: Sequence[np.ndarray],
        info: Mapping[str, float | int],
        options: Mapping[str, Any],
        preprocess_ms: float,
        execution: Mapping[str, float],
    ) -> AscendResult:
        postprocess_started = time.perf_counter_ns()
        confidence = float(options.get("conf", 0.5))
        max_det = int(options.get("max_det", 300))
        rows = yolo26_e2e_v1_records(
            outputs,
            info,
            confidence,
            max_det,
            contract_max_det=self.contract_max_det,
            class_count=self.class_count,
        )
        postprocess_ms = (
            time.perf_counter_ns() - postprocess_started
        ) / 1_000_000.0
        timings = {
            "preprocess": float(preprocess_ms),
            "inference": float(execution["inference_ms"]),
            "postprocess": postprocess_ms,
        }
        if "submit_ms" in execution:
            timings.update(
                {
                    "ascend_submit": float(execution["submit_ms"]),
                    "ascend_wait": float(execution["wait_ms"]),
                    "ascend_input_copy": float(execution["input_copy_ms"]),
                    "ascend_output_copy": float(execution["output_copy_ms"]),
                }
            )
        self._last_timings = timings
        return AscendResult(rows, timings)

    def _detector_input(
        self, image: Image.Image, options: Mapping[str, Any]
    ) -> tuple[np.ndarray, dict[str, float | int], float]:
        preprocess_started = time.perf_counter_ns()
        batch, info = detector_tensor(
            image,
            self.expected_height,
            self.expected_width,
            options.get("_ascend_rgb_array"),
            self.input_mode,
        )
        preprocess_ms = (
            time.perf_counter_ns() - preprocess_started
        ) / 1_000_000.0
        return batch, info, preprocess_ms

    def predict(self, image: Image.Image, **options: Any) -> AscendResult:
        if (
            self.model.execution_mode == "async_stream"
            and self.model.schedule_mode == "unified_enqueue"
        ):
            return self.submit(image, **options).result()
        preprocess_started = time.perf_counter_ns()
        batch, info = detector_tensor(
            image,
            self.expected_height,
            self.expected_width,
            options.get("_ascend_rgb_array"),
            self.input_mode,
        )
        preprocess_ms = (time.perf_counter_ns() - preprocess_started) / 1_000_000.0
        if self.model.execution_mode == "async_stream":
            outputs, execution = self.model.execute_threaded(batch)
        else:
            outputs, inference_ms = self.model.execute(batch)
            execution = {"inference_ms": inference_ms}
        return self._result_from_outputs(
            outputs, info, options, preprocess_ms, execution
        )

    def submit(self, image: Image.Image, **options: Any) -> AscendDetectorExecutionHandle:
        preprocess_started = time.perf_counter_ns()
        batch, info = detector_tensor(
            image,
            self.expected_height,
            self.expected_width,
            options.get("_ascend_rgb_array"),
            self.input_mode,
        )
        preprocess_ms = (
            time.perf_counter_ns() - preprocess_started
        ) / 1_000_000.0
        return AscendDetectorExecutionHandle(
            self,
            self.model.submit(batch),
            info,
            options,
            preprocess_ms,
        )

    def predict_preloaded(
        self,
        info: Mapping[str, float | int],
        ready_event: Any,
        **options: Any,
    ) -> AscendResult:
        if self.model.schedule_mode == "unified_enqueue":
            return self.submit_preloaded(info, ready_event, **options).result()
        outputs, execution = self.model.execute_preloaded_threaded(ready_event)
        return self._result_from_outputs(outputs, info, options, 0.0, execution)

    def submit_preloaded(
        self,
        info: Mapping[str, float | int],
        ready_event: Any,
        **options: Any,
    ) -> AscendDetectorExecutionHandle:
        return AscendDetectorExecutionHandle(
            self,
            self.model.submit_preloaded(ready_event),
            info,
            options,
            0.0,
        )

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
        self.model = _load_model(options, entry, "scene")
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
        if (
            self.model.execution_mode == "async_stream"
            and self.model.schedule_mode == "unified_enqueue"
        ):
            return self.submit(image).result()
        batch = context_tensor(image, self.image_size, self.input_mode)
        if self.model.execution_mode == "async_stream":
            outputs, timings = self.model.execute_threaded(batch)
            return self._result(
                outputs,
                timings["inference_ms"],
                self._ascend_timings(timings),
            )
        outputs, inference_ms = self.model.execute(batch)
        return self._result(outputs, inference_ms)

    def submit(self, image: Image.Image) -> "AscendContextExecutionHandle":
        return AscendContextExecutionHandle(
            self,
            self.model.submit(
                context_tensor(
                    image,
                    self.image_size,
                    self.input_mode,
                )
            ),
        )

    def predict_preloaded(self, ready_event: Any) -> dict[str, Any]:
        if self.model.schedule_mode == "unified_enqueue":
            return self.submit_preloaded(ready_event).result()
        outputs, timings = self.model.execute_preloaded_threaded(ready_event)
        return self._result(
            outputs,
            timings["inference_ms"],
            self._ascend_timings(timings),
        )

    def submit_preloaded(self, ready_event: Any) -> "AscendContextExecutionHandle":
        return AscendContextExecutionHandle(
            self,
            self.model.submit_preloaded(ready_event),
        )

    @staticmethod
    def _ascend_timings(timings: Mapping[str, float]) -> dict[str, float]:
        return {
            "ascend_submit": float(timings["submit_ms"]),
            "ascend_wait": float(timings["wait_ms"]),
            "ascend_input_copy": float(timings["input_copy_ms"]),
            "ascend_output_copy": float(timings["output_copy_ms"]),
        }

    @staticmethod
    def _result(
        outputs: Sequence[np.ndarray],
        inference_ms: float,
        ascend_timings: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        sensor_prob = _softmax(outputs[0])[0]
        scene_prob = _softmax(outputs[1])[0]
        sensor_names = ["ir", "sar"]
        scene_names = ["air", "forest", "sea", "urban"]
        sensor_id = int(sensor_prob.argmax())
        scene_id = int(scene_prob.argmax())
        result = {
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
        if ascend_timings is not None:
            result["_ascend_timings"] = dict(ascend_timings)
        return result

    def predict_batch(self, images: Sequence[Image.Image]) -> list[dict[str, Any]]:
        return [self.predict(image) for image in images]

    def close(self) -> None:
        self.model.close()


class AscendContextExecutionHandle:
    def __init__(
        self,
        context_model: AscendAclContextModel,
        model_handle: AscendAclExecutionHandle,
    ) -> None:
        self.context_model = context_model
        self.model_handle = model_handle
        self._result: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def result(self) -> dict[str, Any]:
        with self._lock:
            if self._result is not None:
                return self._result
            outputs, timings = self.model_handle.result()
            self._result = self.context_model._result(
                outputs,
                timings["inference_ms"],
                {
                    "ascend_submit": timings["submit_ms"],
                    "ascend_wait": timings["wait_ms"],
                    "ascend_input_copy": timings["input_copy_ms"],
                    "ascend_output_copy": timings["output_copy_ms"],
                },
            )
            return self._result


def load_ascend_context_model(
    options: Mapping[str, Any]
) -> tuple[AscendAclContextModel, dict[str, Any]]:
    model = AscendAclContextModel(options)
    return model, {"preprocessing": {"image_size": model.image_size}}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)
