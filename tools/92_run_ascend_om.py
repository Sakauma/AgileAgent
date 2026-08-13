#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

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


def require(result: int, operation: str) -> None:
    if result != ACL_SUCCESS:
        raise RuntimeError(f"{operation}失败，ACL错误码：{result}")


def require_status(call_result: Any, operation: str) -> None:
    result = call_result[-1] if isinstance(call_result, tuple) else call_result
    require(int(result), operation)


def result_value(call_result: Any, operation: str) -> Any:
    if not isinstance(call_result, tuple) or len(call_result) < 2:
        raise RuntimeError(f"{operation}返回格式异常：{call_result!r}")
    *values, result = call_result
    require(int(result), operation)
    return values[0] if len(values) == 1 else tuple(values)


def dims_value(call_result: Any, operation: str) -> dict[str, Any]:
    value = result_value(call_result, operation)
    if not isinstance(value, dict) or "dims" not in value:
        raise RuntimeError(f"{operation}维度格式异常：{value!r}")
    return value


def dtype_for(code: int) -> np.dtype[Any]:
    try:
        return ACL_DTYPES[code]
    except KeyError as exc:
        raise RuntimeError(f"暂不支持ACL数据类型：{code}") from exc


def load_input(path: Path, expected_shape: tuple[int, ...], expected_dtype: np.dtype[Any]) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if tuple(array.shape) != expected_shape:
        raise ValueError(f"输入shape不匹配：期望{expected_shape}，实际{array.shape}")
    if array.dtype != expected_dtype:
        array = array.astype(expected_dtype)
    return np.ascontiguousarray(array)


def numpy_pointer(acl: Any, array: np.ndarray) -> int:
    pointer = acl.util.numpy_to_ptr(array)
    if not isinstance(pointer, int):
        raise RuntimeError(f"acl.util.numpy_to_ptr返回格式异常：{pointer!r}")
    return pointer


def main() -> int:
    parser = argparse.ArgumentParser(description="在Ascend设备上执行一个静态shape OM。")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("warmup必须>=0，runs必须>=1")

    import acl

    model_path = args.model.resolve()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    initialized = False
    device_set = False
    context = None
    model_id = None
    desc = None
    input_dataset = None
    output_dataset = None
    device_buffers: list[int] = []
    data_buffers: list[Any] = []

    try:
        require_status(acl.init(), "acl.init")
        initialized = True
        require_status(acl.rt.set_device(args.device), "acl.rt.set_device")
        device_set = True
        context = result_value(acl.rt.create_context(args.device), "acl.rt.create_context")
        model_id = result_value(acl.mdl.load_from_file(str(model_path)), "acl.mdl.load_from_file")
        desc = acl.mdl.create_desc()
        if desc is None:
            raise RuntimeError("acl.mdl.create_desc返回空")
        require_status(acl.mdl.get_desc(desc, model_id), "acl.mdl.get_desc")

        input_count = int(acl.mdl.get_num_inputs(desc))
        output_count = int(acl.mdl.get_num_outputs(desc))
        if input_count != 1:
            raise RuntimeError(f"当前runner要求单输入，模型实际为{input_count}")

        input_dims = dims_value(acl.mdl.get_input_dims(desc, 0), "acl.mdl.get_input_dims")
        input_shape = tuple(int(value) for value in input_dims["dims"])
        input_dtype_code = int(acl.mdl.get_input_data_type(desc, 0))
        input_dtype = dtype_for(input_dtype_code)
        input_size = int(acl.mdl.get_input_size_by_index(desc, 0))
        input_array = load_input(input_path, input_shape, input_dtype)
        if input_array.nbytes != input_size:
            raise RuntimeError(f"输入字节数不匹配：期望{input_size}，实际{input_array.nbytes}")

        input_dataset = acl.mdl.create_dataset()
        if input_dataset is None:
            raise RuntimeError("acl.mdl.create_dataset(input)返回空")
        input_device = result_value(
            acl.rt.malloc(input_size, ACL_MEM_MALLOC_HUGE_FIRST), "acl.rt.malloc(input)"
        )
        device_buffers.append(input_device)
        require_status(
            acl.rt.memcpy(
                input_device,
                input_size,
                numpy_pointer(acl, input_array),
                input_array.nbytes,
                ACL_MEMCPY_HOST_TO_DEVICE,
            ),
            "acl.rt.memcpy(input)",
        )
        input_buffer = acl.create_data_buffer(input_device, input_size)
        if input_buffer is None:
            raise RuntimeError("acl.create_data_buffer(input)返回空")
        data_buffers.append(input_buffer)
        require_status(
            acl.mdl.add_dataset_buffer(input_dataset, input_buffer),
            "acl.mdl.add_dataset_buffer(input)",
        )

        output_dataset = acl.mdl.create_dataset()
        if output_dataset is None:
            raise RuntimeError("acl.mdl.create_dataset(output)返回空")
        output_contracts: list[dict[str, Any]] = []
        output_device_buffers: list[int] = []
        for index in range(output_count):
            output_dims = dims_value(acl.mdl.get_output_dims(desc, index), f"acl.mdl.get_output_dims({index})")
            output_shape = tuple(int(value) for value in output_dims["dims"])
            output_dtype_code = int(acl.mdl.get_output_data_type(desc, index))
            output_dtype = dtype_for(output_dtype_code)
            output_size = int(acl.mdl.get_output_size_by_index(desc, index))
            output_device = result_value(
                acl.rt.malloc(output_size, ACL_MEM_MALLOC_HUGE_FIRST),
                f"acl.rt.malloc(output {index})",
            )
            device_buffers.append(output_device)
            output_device_buffers.append(output_device)
            output_buffer = acl.create_data_buffer(output_device, output_size)
            if output_buffer is None:
                raise RuntimeError(f"acl.create_data_buffer(output {index})返回空")
            data_buffers.append(output_buffer)
            require_status(
                acl.mdl.add_dataset_buffer(output_dataset, output_buffer),
                f"acl.mdl.add_dataset_buffer(output {index})",
            )
            output_contracts.append(
                {
                    "index": index,
                    "name": output_dims.get("name", f"output_{index}"),
                    "shape": output_shape,
                    "dtype_code": output_dtype_code,
                    "dtype": output_dtype,
                    "size": output_size,
                }
            )

        for _ in range(args.warmup):
            require_status(
                acl.mdl.execute(model_id, input_dataset, output_dataset),
                "acl.mdl.execute(warmup)",
            )

        timings_ms: list[float] = []
        for _ in range(args.runs):
            started = time.perf_counter_ns()
            require_status(
                acl.mdl.execute(model_id, input_dataset, output_dataset),
                "acl.mdl.execute",
            )
            timings_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)

        output_records: list[dict[str, Any]] = []
        for contract, output_device in zip(output_contracts, output_device_buffers):
            host = np.empty(contract["size"], dtype=np.uint8)
            require_status(
                acl.rt.memcpy(
                    numpy_pointer(acl, host),
                    host.nbytes,
                    output_device,
                    contract["size"],
                    ACL_MEMCPY_DEVICE_TO_HOST,
                ),
                f"acl.rt.memcpy(output {contract['index']})",
            )
            output = host.view(contract["dtype"]).reshape(contract["shape"]).copy()
            output_path = output_dir / f"output_{contract['index']}.npy"
            np.save(output_path, output, allow_pickle=False)
            output_records.append(
                {
                    "index": contract["index"],
                    "name": contract["name"],
                    "shape": list(contract["shape"]),
                    "dtype": str(contract["dtype"]),
                    "bytes": contract["size"],
                    "path": str(output_path),
                    "finite": bool(np.isfinite(output).all()),
                    "min": float(np.nanmin(output)),
                    "max": float(np.nanmax(output)),
                    "mean": float(np.nanmean(output)),
                }
            )

        report = {
            "model": str(model_path),
            "input": {
                "path": str(input_path),
                "name": input_dims.get("name", "input_0"),
                "shape": list(input_shape),
                "dtype": str(input_dtype),
                "bytes": input_size,
            },
            "outputs": output_records,
            "warmup": args.warmup,
            "runs": args.runs,
            "timings_ms": timings_ms,
            "mean_ms": float(np.mean(timings_ms)),
            "p95_ms": float(np.percentile(timings_ms, 95)),
            "passed": all(record["finite"] for record in output_records),
        }
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    finally:
        for data_buffer in reversed(data_buffers):
            acl.destroy_data_buffer(data_buffer)
        if input_dataset is not None:
            acl.mdl.destroy_dataset(input_dataset)
        if output_dataset is not None:
            acl.mdl.destroy_dataset(output_dataset)
        for device_buffer in reversed(device_buffers):
            acl.rt.free(device_buffer)
        if desc is not None:
            acl.mdl.destroy_desc(desc)
        if model_id is not None:
            acl.mdl.unload(model_id)
        if context is not None:
            acl.rt.destroy_context(context)
        if device_set:
            acl.rt.reset_device(args.device)
        if initialized:
            acl.finalize()


if __name__ == "__main__":
    raise SystemExit(main())
