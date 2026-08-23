from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fair_agent.backends.ascend_acl import (
    ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEMCPY_HOST_TO_DEVICE,
    AscendAclModel,
    AscendAclRuntime,
)


class FakeUtil:
    @staticmethod
    def numpy_to_ptr(array: np.ndarray) -> int:
        return int(array.ctypes.data)


class FakeRt:
    def __init__(self, operations: list[str]) -> None:
        self.operations = operations
        self.devices: dict[int, bytearray] = {}
        self.hosts: dict[int, Any] = {}
        self.event_times: dict[int, float] = {}
        self.next_device = 0x1000
        self.next_stream = 1
        self.next_event = 1
        self.clock = 0.0
        self.synchronize_event_calls = 0
        self.synchronize_stream_calls = 0
        self.fail_output_copy = False
        self.fail_synchronize_event = False

    def malloc(self, size: int, _policy: int) -> tuple[int, int]:
        pointer = self.next_device
        self.next_device += 0x1000
        self.devices[pointer] = bytearray(size)
        self.operations.append("malloc_device")
        return pointer, 0

    def free(self, pointer: int) -> int:
        self.operations.append("free_device")
        self.devices.pop(pointer)
        return 0

    def malloc_host(self, size: int) -> tuple[int, int]:
        owner = ctypes.create_string_buffer(size)
        pointer = ctypes.addressof(owner)
        self.hosts[pointer] = owner
        self.operations.append("malloc_host")
        return pointer, 0

    def free_host(self, pointer: int) -> int:
        self.operations.append("free_host")
        self.hosts.pop(pointer)
        return 0

    def create_stream(self) -> tuple[int, int]:
        stream = self.next_stream
        self.next_stream += 1
        self.operations.append("create_stream")
        return stream, 0

    def create_stream_with_config(
        self, _priority: int, _flag: int
    ) -> tuple[int, int]:
        stream = self.next_stream
        self.next_stream += 1
        self.operations.append("create_stream_with_config")
        return stream, 0

    def destroy_stream(self, _stream: int) -> int:
        self.operations.append("destroy_stream")
        return 0

    def create_event(self) -> tuple[int, int]:
        event = self.next_event
        self.next_event += 1
        self.operations.append("create_event")
        return event, 0

    def destroy_event(self, _event: int) -> int:
        self.operations.append("destroy_event")
        return 0

    def record_event(self, event: int, _stream: int) -> int:
        self.clock += 1.0
        self.event_times[event] = self.clock
        self.operations.append(f"record_event:{event}")
        return 0

    def reset_event(self, event: int, _stream: int) -> int:
        self.operations.append(f"reset_event:{event}")
        return 0

    def stream_wait_event(self, _stream: int, _event: Any) -> int:
        self.operations.append("stream_wait_event")
        return 0

    def event_elapsed_time(self, start: int, end: int) -> tuple[float, int]:
        self.operations.append("event_elapsed_time")
        return self.event_times[end] - self.event_times[start], 0

    def synchronize_event(self, _event: int) -> int:
        self.synchronize_event_calls += 1
        self.operations.append("synchronize_event")
        if self.fail_synchronize_event:
            self.fail_synchronize_event = False
            return 97
        return 0

    def synchronize_stream(self, _stream: int) -> int:
        self.synchronize_stream_calls += 1
        self.operations.append("synchronize_stream")
        return 0

    def memcpy_async(
        self,
        destination: int,
        _destination_size: int,
        source: int,
        size: int,
        kind: int,
        _stream: int,
    ) -> int:
        if kind == ACL_MEMCPY_DEVICE_TO_HOST and self.fail_output_copy:
            self.fail_output_copy = False
            self.operations.append("memcpy_async_d2h_failed")
            return 91
        if kind == ACL_MEMCPY_HOST_TO_DEVICE:
            self.devices[destination][:size] = ctypes.string_at(source, size)
            self.operations.append("memcpy_async_h2d")
        elif kind == ACL_MEMCPY_DEVICE_TO_HOST:
            ctypes.memmove(destination, bytes(self.devices[source][:size]), size)
            self.operations.append("memcpy_async_d2h")
        else:
            raise AssertionError(f"unexpected memcpy kind: {kind}")
        return 0

    def memcpy(
        self,
        destination: int,
        destination_size: int,
        source: int,
        size: int,
        kind: int,
    ) -> int:
        if kind == ACL_MEMCPY_HOST_TO_DEVICE:
            self.devices[destination][:size] = ctypes.string_at(source, size)
            self.operations.append("memcpy_h2d")
        elif kind == ACL_MEMCPY_DEVICE_TO_HOST:
            ctypes.memmove(destination, bytes(self.devices[source][:size]), size)
            self.operations.append("memcpy_d2h")
        else:
            raise AssertionError(f"unexpected memcpy kind: {kind}")
        return 0


class FakeMdl:
    def __init__(
        self,
        rt: FakeRt,
        operations: list[str],
        output_specs: list[tuple[list[int], int]] | None = None,
    ) -> None:
        self.rt = rt
        self.operations = operations
        self.output_specs = output_specs or [([1, 4], 4)]
        self.output_payloads: dict[int, bytes] = {}

    def load_from_file(self, _path: str) -> tuple[int, int]:
        self.operations.append("load_model")
        return 1, 0

    @staticmethod
    def create_desc() -> object:
        return object()

    @staticmethod
    def get_desc(_desc: object, _model_id: int) -> int:
        return 0

    @staticmethod
    def get_num_inputs(_desc: object) -> int:
        return 1

    @staticmethod
    def get_input_dims(_desc: object, _index: int) -> tuple[dict[str, Any], int]:
        return {"dims": [1, 4]}, 0

    @staticmethod
    def get_input_data_type(_desc: object, _index: int) -> int:
        return 4

    @staticmethod
    def get_input_size_by_index(_desc: object, _index: int) -> int:
        return 4

    def get_num_outputs(self, _desc: object) -> int:
        return len(self.output_specs)

    def get_output_dims(
        self, _desc: object, index: int
    ) -> tuple[dict[str, Any], int]:
        return {"dims": self.output_specs[index][0]}, 0

    def get_output_data_type(self, _desc: object, index: int) -> int:
        return self.output_specs[index][1]

    def get_output_size_by_index(self, _desc: object, index: int) -> int:
        shape, dtype = self.output_specs[index]
        itemsize = {0: 4, 3: 4, 4: 1}[dtype]
        return int(np.prod(shape)) * itemsize

    @staticmethod
    def create_dataset() -> list[tuple[int, int]]:
        return []

    @staticmethod
    def add_dataset_buffer(
        dataset: list[tuple[int, int]], buffer: tuple[int, int]
    ) -> int:
        dataset.append(buffer)
        return 0

    def execute_async(
        self,
        _model_id: int,
        input_dataset: list[tuple[int, int]],
        output_dataset: list[tuple[int, int]],
        _stream: int,
    ) -> int:
        source, source_size = input_dataset[0]
        for index, (destination, destination_size) in enumerate(output_dataset):
            payload = self.output_payloads.get(index)
            if payload is None:
                copied = min(source_size, destination_size)
                self.rt.devices[destination][:copied] = self.rt.devices[source][:copied]
            else:
                copied = min(len(payload), destination_size)
                self.rt.devices[destination][:copied] = payload[:copied]
        self.operations.append("execute_async")
        return 0

    @staticmethod
    def execute(
        _model_id: int,
        _input_dataset: list[tuple[int, int]],
        _output_dataset: list[tuple[int, int]],
    ) -> int:
        return 0

    def destroy_dataset(self, _dataset: list[tuple[int, int]]) -> int:
        self.operations.append("destroy_dataset")
        return 0

    def destroy_desc(self, _desc: object) -> int:
        self.operations.append("destroy_desc")
        return 0

    def unload(self, _model_id: int) -> int:
        self.operations.append("unload_model")
        return 0


class FakeAcl:
    def __init__(
        self,
        output_specs: list[tuple[list[int], int]] | None = None,
    ) -> None:
        self.operations: list[str] = []
        self.rt = FakeRt(self.operations)
        self.mdl = FakeMdl(self.rt, self.operations, output_specs)
        self.util = FakeUtil()

    def create_data_buffer(self, pointer: int, size: int) -> tuple[int, int]:
        self.operations.append("create_data_buffer")
        return pointer, size

    def destroy_data_buffer(self, _buffer: tuple[int, int]) -> int:
        self.operations.append("destroy_data_buffer")
        return 0


class FakeRuntime:
    def __init__(
        self,
        output_specs: list[tuple[list[int], int]] | None = None,
    ) -> None:
        self.acl = FakeAcl(output_specs)
        self.lock = threading.RLock()
        self.closed = False
        self.models: set[AscendAclModel] = set()
        self.stream_priority_status: dict[str, Any] | None = None
        self._stream_priority_request: tuple[tuple[str, str], ...] | None = None
        self._stream_priority_warning_emitted = False

    resolve_stream_priorities = AscendAclRuntime.resolve_stream_priorities
    create_model_stream = AscendAclRuntime.create_model_stream

    def activate(self) -> None:
        self.acl.operations.append("activate")

    def register(self, model: AscendAclModel) -> None:
        self.models.add(model)

    def unregister(self, model: AscendAclModel) -> None:
        self.models.discard(model)


def create_model(
    *,
    memory_mode: str = "pinned",
    schedule_mode: str = "unified_enqueue",
    detailed_event_timing: bool = True,
    stream_role: str = "base",
    stream_priorities: dict[str, str] | None = None,
    output_specs: list[tuple[list[int], int]] | None = None,
) -> tuple[FakeRuntime, AscendAclModel]:
    runtime = FakeRuntime(output_specs)
    model = AscendAclModel(
        runtime,  # type: ignore[arg-type]
        Path("fake.om"),
        execution_mode="async_stream",
        memory_mode=memory_mode,
        schedule_mode=schedule_mode,
        detailed_event_timing=detailed_event_timing,
        stream_role=stream_role,
        stream_priorities=stream_priorities,
    )
    return runtime, model


def test_pinned_async_execution_enqueues_all_work_before_single_wait() -> None:
    runtime, model = create_model()
    handle = model.submit(np.asarray([[1, 2, 3, 4]], dtype=np.uint8))

    assert runtime.acl.rt.synchronize_event_calls == 0
    assert runtime.acl.operations.index("memcpy_async_h2d") < runtime.acl.operations.index(
        "execute_async"
    ) < runtime.acl.operations.index("memcpy_async_d2h")
    with pytest.raises(RuntimeError, match="未完成"):
        model.submit(np.asarray([[5, 6, 7, 8]], dtype=np.uint8))

    outputs, timings = handle.result()
    assert outputs[0].tolist() == [[1, 2, 3, 4]]
    assert timings["input_copy_ms"] == 1.0
    assert timings["inference_ms"] == 1.0
    assert timings["output_copy_ms"] == 1.0
    assert runtime.acl.rt.synchronize_event_calls == 1
    assert handle.result()[0][0].tolist() == [[1, 2, 3, 4]]
    assert runtime.acl.rt.synchronize_event_calls == 1

    model.close()
    assert runtime.acl.operations.count("malloc_host") == 2
    assert runtime.acl.operations.count("free_host") == 2
    assert runtime.acl.operations.index("synchronize_event") < runtime.acl.operations.index(
        "destroy_stream"
    ) < runtime.acl.operations.index("free_host")
    assert runtime.acl.operations.index("free_host") < runtime.acl.operations.index(
        "unload_model"
    )


def test_preloaded_execution_waits_on_ready_event_without_h2d() -> None:
    runtime, model = create_model()
    runtime.acl.rt.devices[model.input_device][:] = b"\x09\x08\x07\x06"

    operation_start = len(runtime.acl.operations)
    outputs, timings = model.submit_preloaded("dvpp-ready").result()
    operations = runtime.acl.operations[operation_start:]

    assert "stream_wait_event" in operations
    assert "memcpy_async_h2d" not in operations
    assert outputs[0].tolist() == [[9, 8, 7, 6]]
    assert timings["input_copy_ms"] == 0.0
    model.close()


def test_threaded_execute_reproduces_stream_wait_then_synchronous_d2h() -> None:
    runtime, model = create_model(schedule_mode="threaded_execute")
    runtime.acl.rt.devices[model.input_device][:] = b"\x04\x03\x02\x01"

    operation_start = len(runtime.acl.operations)
    outputs, timings = model.execute_preloaded_threaded("dvpp-ready")
    operations = runtime.acl.operations[operation_start:]

    assert outputs[0].tolist() == [[4, 3, 2, 1]]
    assert operations.index("stream_wait_event") < operations.index(
        "execute_async"
    ) < operations.index("synchronize_stream") < operations.index("memcpy_d2h")
    assert "memcpy_async_d2h" not in operations
    assert runtime.acl.rt.synchronize_event_calls == 0
    assert timings["input_copy_ms"] == 0.0
    assert timings["inference_ms"] == 1.0
    model.close()


def test_compact_event_timing_keeps_model_time_and_skips_copy_queries() -> None:
    runtime, model = create_model(detailed_event_timing=False)
    handle = model.submit(np.asarray([[1, 2, 3, 4]], dtype=np.uint8))

    outputs, timings = handle.result()

    assert outputs[0].tolist() == [[1, 2, 3, 4]]
    assert timings["input_copy_ms"] == 0.0
    assert timings["inference_ms"] == 1.0
    assert timings["output_copy_ms"] == 0.0
    assert runtime.acl.operations.count("create_event") == 3
    assert runtime.acl.operations.count("event_elapsed_time") == 1
    model.close()


def test_priority_request_falls_back_when_range_api_is_unavailable() -> None:
    priorities = {
        "scene": "normal",
        "base": "high",
        "specialist": "normal",
    }
    with pytest.warns(RuntimeWarning, match="跳过优先级候选"):
        runtime, model = create_model(stream_priorities=priorities)

    assert runtime.stream_priority_status == {
        "requested": True,
        "supported": False,
        "reason": "priority_range_api_unavailable",
        "requested_labels": priorities,
        "values": {},
    }
    assert model.stream_priority_label == "high"
    assert model.stream_priority_supported is False
    assert model.stream_priority_value is None
    assert runtime.acl.operations.count("create_stream") == 1
    assert "create_stream_with_config" not in runtime.acl.operations
    model.close()


def test_priority_request_maps_reported_range_before_model_stream_creation() -> None:
    runtime = FakeRuntime()
    runtime.acl.rt.get_stream_priority_range = lambda: (0, 7, 0)
    priorities = {
        "scene": "normal",
        "base": "normal",
        "specialist": "high",
    }
    model = AscendAclModel(
        runtime,  # type: ignore[arg-type]
        Path("fake.om"),
        execution_mode="async_stream",
        memory_mode="pageable",
        schedule_mode="threaded_execute",
        detailed_event_timing=False,
        stream_role="specialist",
        stream_priorities=priorities,
    )

    assert runtime.stream_priority_status == {
        "requested": True,
        "supported": True,
        "reason": "supported",
        "requested_labels": priorities,
        "values": {"high": 0, "normal": 4, "low": 7},
    }
    assert model.stream_priority_supported is True
    assert model.stream_priority_value == 0
    assert runtime.acl.operations.count("create_stream_with_config") == 4
    assert runtime.acl.operations.count("destroy_stream") == 3
    assert "create_stream" not in runtime.acl.operations
    model.close()


def test_enqueue_failure_drains_stream_and_allows_next_submission() -> None:
    runtime, model = create_model()
    runtime.acl.rt.fail_output_copy = True

    with pytest.raises(RuntimeError, match="91"):
        model.submit(np.asarray([[1, 2, 3, 4]], dtype=np.uint8))
    assert runtime.acl.rt.synchronize_stream_calls == 1

    outputs, _timings = model.submit(
        np.asarray([[5, 6, 7, 8]], dtype=np.uint8)
    ).result()
    assert outputs[0].tolist() == [[5, 6, 7, 8]]
    model.close()


def test_final_wait_failure_recovers_stream_before_releasing_host_memory() -> None:
    runtime, model = create_model()
    runtime.acl.rt.fail_synchronize_event = True
    handle = model.submit(np.asarray([[1, 2, 3, 4]], dtype=np.uint8))

    with pytest.raises(RuntimeError, match="97"):
        handle.result()
    assert runtime.acl.rt.synchronize_stream_calls == 1

    model.close()
    assert runtime.acl.operations.index("synchronize_stream") < runtime.acl.operations.index(
        "free_host"
    )
