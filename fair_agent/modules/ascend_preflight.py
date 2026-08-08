from __future__ import annotations

import json
import math
import os
import platform
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from threading import Lock
from typing import Any, Dict, Iterable, Mapping, Protocol, Sequence

from PIL import Image

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.models.context import SCENE_NAMES, SENSOR_NAMES, SceneSensorNet
from fair_agent.modules.detection_fusion import apply_incremental_candidate_gates
from fair_agent.modules.strict_incremental import (
    GLOBAL_CLASS_NAMES,
    evaluate_ap50,
    fuse_old_new_predictions,
    precision_recall,
    read_split,
    retention_metrics,
    subset_rows,
    yolo_ground_truth,
)


@dataclass(frozen=True)
class FixedOnnxAsset:
    model_id: str
    kind: str
    source: Path
    target: Path
    input_shape: tuple[int, int, int, int]
    output_names: tuple[str, ...]


@dataclass(frozen=True)
class LetterboxInfo:
    original_height: int
    original_width: int
    input_height: int
    input_width: int
    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


class RawRunner(Protocol):
    provider: str

    def run(self, batch: Any) -> list[Any]: ...


def fixed_rect_shape(
    source_width: int,
    source_height: int,
    image_size: int,
    stride: int = 32,
) -> tuple[int, int]:
    """Return the static HxW produced by Ultralytics rect/auto letterbox."""
    scale = min(float(image_size) / source_width, float(image_size) / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    output_width = resized_width + (image_size - resized_width) % int(stride)
    output_height = resized_height + (image_size - resized_height) % int(stride)
    return output_height, output_width


def production_onnx_plan(
    output_root: str | Path,
    *,
    shape_mode: str = "rect",
    source_width: int = 640,
    source_height: int = 512,
) -> list[FixedOnnxAsset]:
    if shape_mode not in {"rect", "square"}:
        raise ValueError("shape_mode必须是rect或square")
    profile = json.loads(
        resolve_path("models/production/incremental_detection/profile.json").read_text(
            encoding="utf-8"
        )
    )
    base_size = int(profile["base_imgsz"])
    incremental_size = int(profile["specialist_imgsz"])
    base_hw = (
        fixed_rect_shape(source_width, source_height, base_size)
        if shape_mode == "rect"
        else (base_size, base_size)
    )
    incremental_hw = (
        fixed_rect_shape(source_width, source_height, incremental_size)
        if shape_mode == "rect"
        else (incremental_size, incremental_size)
    )
    root = resolve_path(output_root) / "onnx" / shape_mode
    return [
        FixedOnnxAsset(
            model_id="base_detector",
            kind="yolo",
            source=resolve_path(profile["base_weight"]),
            target=root / "base_detector.onnx",
            input_shape=(1, 3, *base_hw),
            output_names=("output0",),
        ),
        FixedOnnxAsset(
            model_id="incremental_detector",
            kind="yolo",
            source=resolve_path(profile["specialist_weight"]),
            target=root / "incremental_detector.onnx",
            input_shape=(1, 3, *incremental_hw),
            output_names=("output0",),
        ),
        FixedOnnxAsset(
            model_id="scene_sensor_net",
            kind="context",
            source=resolve_path("models/context/scene_sensor_net.pt"),
            target=root / "scene_sensor_net.onnx",
            input_shape=(1, 3, 160, 160),
            output_names=("sensor_logits", "scene_logits"),
        ),
    ]


def _atomic_replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent, suffix=target.suffix, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def validate_onnx_contract(asset: FixedOnnxAsset) -> Dict[str, Any]:
    import onnx

    if not asset.target.is_file():
        raise FileNotFoundError(asset.target)
    model = onnx.load(asset.target)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1:
        raise RuntimeError(f"{asset.model_id} ONNX必须只有一个输入")
    input_tensor = model.graph.input[0]
    dimensions = input_tensor.type.tensor_type.shape.dim
    actual_shape = tuple(int(dimension.dim_value) for dimension in dimensions)
    if any(dimension.dim_param or int(dimension.dim_value) <= 0 for dimension in dimensions):
        raise RuntimeError(f"{asset.model_id} ONNX仍包含动态输入维度")
    if actual_shape != asset.input_shape:
        raise RuntimeError(
            f"{asset.model_id} ONNX输入不匹配：expected={asset.input_shape} actual={actual_shape}"
        )
    output_names = tuple(output.name for output in model.graph.output)
    if output_names != asset.output_names:
        raise RuntimeError(
            f"{asset.model_id} ONNX输出不匹配：expected={asset.output_names} actual={output_names}"
        )
    operators: Dict[str, int] = {}
    for node in model.graph.node:
        operators[node.op_type] = operators.get(node.op_type, 0) + 1
    if operators.get("NonMaxSuppression", 0):
        raise RuntimeError(f"{asset.model_id} ONNX禁止包含图内NMS")
    output_shapes = []
    for output in model.graph.output:
        dims = output.type.tensor_type.shape.dim
        shape = [int(dimension.dim_value) for dimension in dims]
        if any(dimension.dim_param or value <= 0 for dimension, value in zip(dims, shape)):
            raise RuntimeError(f"{asset.model_id} ONNX输出仍包含动态维度")
        output_shapes.append(shape)
    return {
        "model_id": asset.model_id,
        "kind": asset.kind,
        "source": rel_path(asset.source),
        "target": rel_path(asset.target),
        "input_name": input_tensor.name,
        "input_shape": list(actual_shape),
        "output_names": list(output_names),
        "output_shapes": output_shapes,
        "node_count": len(model.graph.node),
        "operators": dict(sorted(operators.items())),
        "contains_graph_nms": False,
        "file_bytes": asset.target.stat().st_size,
        "contract_valid": True,
    }


def _export_yolo(
    asset: FixedOnnxAsset,
    *,
    device: str,
    opset: int,
    simplify: bool,
) -> None:
    from ultralytics import YOLO

    height, width = asset.input_shape[2:]
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_root = Path(temporary_dir)
        temporary_weight = temporary_root / asset.source.name
        shutil.copy2(asset.source, temporary_weight)
        exported = YOLO(str(temporary_weight)).export(
            format="onnx",
            imgsz=(height, width),
            batch=1,
            dynamic=False,
            simplify=bool(simplify),
            opset=int(opset),
            device=str(device),
            half=False,
            nms=False,
        )
        generated = Path(str(exported))
        if not generated.is_file():
            raise RuntimeError(f"Ultralytics未生成ONNX：{generated}")
        _atomic_replace(generated, asset.target)


def _export_context(asset: FixedOnnxAsset, *, opset: int, simplify: bool) -> None:
    import onnx
    import torch

    checkpoint = torch.load(asset.source, map_location="cpu", weights_only=True)
    model = SceneSensorNet(
        channels=checkpoint["architecture"]["channels"],
        dropout=float(checkpoint["architecture"]["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with tempfile.TemporaryDirectory() as temporary_dir:
        generated = Path(temporary_dir) / asset.target.name
        torch.onnx.export(
            model,
            torch.zeros(asset.input_shape, dtype=torch.float32),
            generated,
            input_names=["images"],
            output_names=list(asset.output_names),
            opset_version=int(opset),
            dynamic_axes=None,
            do_constant_folding=True,
        )
        if simplify:
            from onnxslim import slim

            onnx.save(slim(onnx.load(generated)), generated)
        _atomic_replace(generated, asset.target)


def export_fixed_onnx_assets(
    output_root: str | Path,
    *,
    shape_mode: str = "rect",
    device: str = "0",
    opset: int = 17,
    simplify: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    rows = []
    for asset in production_onnx_plan(output_root, shape_mode=shape_mode):
        status = "verified_existing"
        if not asset.target.exists() or overwrite:
            if asset.kind == "yolo":
                _export_yolo(asset, device=device, opset=opset, simplify=simplify)
            else:
                _export_context(asset, opset=opset, simplify=simplify)
            status = "exported"
        rows.append({**validate_onnx_contract(asset), "status": status})
    payload = {
        "schema_version": 1,
        "shape_mode": shape_mode,
        "batch": 1,
        "dynamic": False,
        "graph_nms": False,
        "opset": int(opset),
        "simplify": bool(simplify),
        "assets": rows,
    }
    manifest = resolve_path(output_root) / "onnx" / shape_mode / "manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**payload, "manifest": rel_path(manifest)}


def _topologically_sort_onnx_graph(graph: Any) -> None:
    """Put ONNX nodes in a stable dependency order.

    ONNX Runtime's float16 converter can append the FP32 I/O Cast nodes after
    nodes that consume them. Runtime loading may tolerate that ordering, but
    ``onnx.checker`` correctly rejects it. The production graphs have no
    control-flow subgraphs, so a stable top-level Kahn sort is sufficient and
    keeps otherwise independent nodes in their original order.
    """
    available = {value.name for value in graph.input}
    available.update(initializer.name for initializer in graph.initializer)
    available.update(initializer.values.name for initializer in graph.sparse_initializer)
    pending = list(graph.node)
    ordered = []
    while pending:
        remaining = []
        progressed = False
        for node in pending:
            if all(not name or name in available for name in node.input):
                ordered.append(node)
                available.update(name for name in node.output if name)
                progressed = True
            else:
                remaining.append(node)
        if not progressed:
            unresolved = sorted(
                {
                    name
                    for node in remaining
                    for name in node.input
                    if name and name not in available
                }
            )
            raise RuntimeError(f"ONNX图无法拓扑排序，缺失输入：{unresolved[:8]}")
        pending = remaining
    del graph.node[:]
    graph.node.extend(ordered)


def _onnx_initializer_type_counts(model: Any) -> Dict[str, int]:
    import onnx

    counts: Dict[str, int] = {}
    for initializer in model.graph.initializer:
        name = onnx.TensorProto.DataType.Name(initializer.data_type)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def convert_fixed_onnx_assets_to_mixed_fp16(
    source_root: str | Path,
    output_root: str | Path,
    *,
    shape_mode: str = "rect",
    keep_context_fp32: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create a local mixed-FP16 proxy while keeping FP32 graph I/O.

    This validates FP16 sensitivity before an Ascend board is available. It is
    not an ATC or OM substitute: Ascend ``mixed_float16`` may choose a different
    per-operator precision policy.
    """
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    source_root_path = resolve_path(source_root)
    output_root_path = resolve_path(output_root)
    if source_root_path.resolve() == output_root_path.resolve():
        raise ValueError("混合FP16输出目录必须与FP32源目录不同")

    source_assets = production_onnx_plan(source_root_path, shape_mode=shape_mode)
    target_assets = production_onnx_plan(output_root_path, shape_mode=shape_mode)
    rows = []
    contracts = []
    for source, target in zip(source_assets, target_assets):
        validate_onnx_contract(source)
        should_keep_fp32 = bool(keep_context_fp32 and target.kind == "context")
        status = "verified_existing"
        if not target.target.exists() or overwrite:
            if should_keep_fp32:
                _atomic_replace(source.target, target.target)
            else:
                converted = convert_float_to_float16(
                    onnx.load(source.target),
                    keep_io_types=True,
                    disable_shape_infer=False,
                )
                _topologically_sort_onnx_graph(converted.graph)
                onnx.checker.check_model(converted)
                with tempfile.TemporaryDirectory() as temporary_dir:
                    generated = Path(temporary_dir) / target.target.name
                    onnx.save(converted, generated)
                    _atomic_replace(generated, target.target)
            status = "copied_fp32" if should_keep_fp32 else "converted_fp16"

        target_model = onnx.load(target.target)
        initializer_types = _onnx_initializer_type_counts(target_model)
        if should_keep_fp32 and initializer_types.get("FLOAT16", 0):
            raise RuntimeError(
                f"{target.model_id}应保持FP32；请使用--overwrite重建候选"
            )
        if not should_keep_fp32 and not initializer_types.get("FLOAT16", 0):
            raise RuntimeError(
                f"{target.model_id}不包含FP16权重；请使用--overwrite重建候选"
            )
        rows.append(
            {
                "model": target.model_id,
                "precision": "fp32" if should_keep_fp32 else "mixed_fp16",
                "source_bytes": source.target.stat().st_size,
                "target_bytes": target.target.stat().st_size,
                "initializer_types": initializer_types,
                "cast_nodes": sum(
                    1 for node in target_model.graph.node if node.op_type == "Cast"
                ),
                "status": status,
            }
        )
        contracts.append(validate_onnx_contract(target))

    payload = {
        "schema_version": 1,
        "shape_mode": shape_mode,
        "policy": (
            "detectors_mixed_fp16_context_fp32"
            if keep_context_fp32
            else "all_models_mixed_fp16"
        ),
        "keep_io_types": True,
        "source_root": rel_path(source_root_path),
        "conversion": rows,
        "contracts": contracts,
    }
    manifest = output_root_path / "onnx" / shape_mode / "mixed_fp16_manifest.json"
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**payload, "manifest": rel_path(manifest)}


def _rgb_array(image: Image.Image | Any) -> Any:
    import numpy as np

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("输入图像必须是HWC三通道RGB")
    return array


def decode_png_rgb(path: str | Path, backend: str = "pillow") -> Any:
    """Decode a PNG to a contiguous RGB uint8 array with a selectable backend."""
    import numpy as np

    source_path = resolve_path(path)
    if backend == "pillow":
        with Image.open(source_path) as source:
            return np.ascontiguousarray(np.asarray(source.convert("RGB")))
    if backend == "opencv":
        import cv2

        encoded = np.frombuffer(source_path.read_bytes(), dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError(f"OpenCV无法解码PNG：{source_path}")
        return np.ascontiguousarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    raise ValueError("backend必须是pillow或opencv")


def detector_tensor(
    image: Image.Image | Any,
    input_height: int,
    input_width: int,
) -> tuple[Any, LetterboxInfo]:
    """Reproduce fixed-shape Ultralytics letterbox as NCHW float32 RGB."""
    import cv2
    import numpy as np

    rgb = _rgb_array(image)
    original_height, original_width = map(int, rgb.shape[:2])
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    if (resized_width, resized_height) != (original_width, original_height):
        resized = cv2.resize(
            rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
    else:
        resized = rgb
    horizontal = input_width - resized_width
    vertical = input_height - resized_height
    half_horizontal = horizontal / 2.0
    half_vertical = vertical / 2.0
    left = int(round(half_horizontal - 0.1))
    right = int(round(half_horizontal + 0.1))
    top = int(round(half_vertical - 0.1))
    bottom = int(round(half_vertical + 0.1))
    canvas = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if canvas.shape[:2] != (input_height, input_width):
        raise RuntimeError(
            f"letterbox输出错误：expected={(input_height, input_width)} actual={canvas.shape[:2]}"
        )
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return tensor, LetterboxInfo(
        original_height=original_height,
        original_width=original_width,
        input_height=input_height,
        input_width=input_width,
        scale=float(scale),
        pad_left=left,
        pad_top=top,
        pad_right=right,
        pad_bottom=bottom,
    )


def context_tensor(image: Image.Image | Any, image_size: int = 160) -> Any:
    import numpy as np

    rgb = Image.fromarray(_rgb_array(image), mode="RGB")
    resize_size = int(round(image_size * 1.1))
    resized = rgb.resize((resize_size, resize_size), Image.Resampling.BILINEAR)
    offset = (resize_size - image_size) // 2
    cropped = resized.crop((offset, offset, offset + image_size, offset + image_size))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.25
    return np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)


def restore_xyxy(box: Sequence[float], info: LetterboxInfo) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = (x1 - info.pad_left) / info.scale
    x2 = (x2 - info.pad_left) / info.scale
    y1 = (y1 - info.pad_top) / info.scale
    y2 = (y2 - info.pad_top) / info.scale
    return [
        min(max(x1, 0.0), float(info.original_width)),
        min(max(y1, 0.0), float(info.original_height)),
        min(max(x2, 0.0), float(info.original_width)),
        min(max(y2, 0.0), float(info.original_height)),
    ]


class TorchYoloRunner:
    provider = "PyTorch CUDA"

    def __init__(self, weights: str | Path, device: str = "0") -> None:
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch对齐要求可用CUDA GPU")
        self._torch = torch
        self._device = torch.device(f"cuda:{device}")
        self._model = YOLO(str(resolve_path(weights))).model.to(self._device).eval()

    def run(self, batch: Any) -> list[Any]:
        tensor = self._torch.from_numpy(batch).to(self._device)
        with self._torch.inference_mode():
            output = self._model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return [output.detach().cpu().numpy()]


class TorchContextRunner:
    provider = "PyTorch CUDA"

    def __init__(self, checkpoint_path: str | Path, device: str = "0") -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch对齐要求可用CUDA GPU")
        checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu", weights_only=True)
        model = SceneSensorNet(
            channels=checkpoint["architecture"]["channels"],
            dropout=float(checkpoint["architecture"]["dropout"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        self._torch = torch
        self._device = torch.device(f"cuda:{device}")
        self._model = model.to(self._device).eval()

    def run(self, batch: Any) -> list[Any]:
        tensor = self._torch.from_numpy(batch).to(self._device)
        with self._torch.inference_mode():
            outputs = self._model(tensor)
        return [output.detach().cpu().numpy() for output in outputs]


class OnnxRunner:
    def __init__(
        self,
        model_path: str | Path,
        *,
        provider: str = "cuda",
        device: str = "0",
    ) -> None:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("ONNX Runtime缺少CUDAExecutionProvider")
            providers: list[Any] = [
                ("CUDAExecutionProvider", {"device_id": int(device)}),
                "CPUExecutionProvider",
            ]
            self.provider = "ONNX Runtime CUDA"
        elif provider == "cpu":
            providers = ["CPUExecutionProvider"]
            self.provider = "ONNX Runtime CPU"
        else:
            raise ValueError("provider必须是cuda或cpu")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(resolve_path(model_path)), sess_options=options, providers=providers
        )
        active = self._session.get_providers()[0]
        if provider == "cuda" and active != "CUDAExecutionProvider":
            raise RuntimeError(f"ONNX Runtime未启用CUDA，首选provider为{active}")
        self._input_name = self._session.get_inputs()[0].name

    def run(self, batch: Any) -> list[Any]:
        return self._session.run(None, {self._input_name: batch})


def _onnx_numpy_dtype(type_name: str) -> Any:
    import numpy as np

    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    if type_name not in mapping:
        raise RuntimeError(f"固定IO暂不支持ONNX类型：{type_name}")
    return mapping[type_name]


class FixedCudaGraphOnnxRunner:
    """CUDA-only proxy for fixed-address buffers and captured static execution."""

    provider = "ONNX Runtime CUDA Graph"

    def __init__(self, asset: FixedOnnxAsset, device: str = "0") -> None:
        import numpy as np
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDA Graph代理要求CUDAExecutionProvider")
        providers: list[Any] = [
            (
                "CUDAExecutionProvider",
                {"device_id": int(device), "enable_cuda_graph": "1"},
            ),
            "CPUExecutionProvider",
        ]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(resolve_path(asset.target)), sess_options=options, providers=providers
        )
        active = self._session.get_providers()[0]
        if active != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA Graph代理未启用CUDA，首选provider为{active}")
        input_meta = self._session.get_inputs()[0]
        actual_shape = tuple(int(value) for value in input_meta.shape)
        if actual_shape != asset.input_shape:
            raise RuntimeError(
                f"{asset.model_id}固定IO输入不匹配：{actual_shape} != {asset.input_shape}"
            )
        self._shape = actual_shape
        self._dtype = _onnx_numpy_dtype(input_meta.type)
        initial = np.zeros(self._shape, dtype=self._dtype)
        self._input = ort.OrtValue.ortvalue_from_numpy(
            initial, "cuda", int(device)
        )
        self._outputs = []
        self._binding = self._session.io_binding()
        self._binding.bind_ortvalue_input(input_meta.name, self._input)
        for output_meta in self._session.get_outputs():
            output_shape = tuple(int(value) for value in output_meta.shape)
            output = ort.OrtValue.ortvalue_from_shape_and_type(
                output_shape,
                _onnx_numpy_dtype(output_meta.type),
                "cuda",
                int(device),
            )
            self._outputs.append(output)
            self._binding.bind_ortvalue_output(output_meta.name, output)
        self._lock = Lock()
        # ORT uses the first fixed-address invocation for allocation and the
        # second for graph capture. Finish both serially before any session is
        # allowed into the shared executor; concurrent first-capture attempts
        # can fail with cudaErrorStreamCaptureUnsupported.
        for _iteration in range(2):
            self._session.run_with_iobinding(self._binding)
            self._binding.synchronize_outputs()

    def run(self, batch: Any) -> list[Any]:
        import numpy as np

        array = np.ascontiguousarray(batch, dtype=self._dtype)
        if tuple(array.shape) != self._shape:
            raise RuntimeError(
                f"固定CUDA Graph输入shape错误：{array.shape} != {self._shape}"
            )
        with self._lock:
            self._input.update_inplace(array)
            self._session.run_with_iobinding(self._binding)
            self._binding.synchronize_outputs()
            return [output.numpy() for output in self._outputs]


def _softmax(values: Any) -> Any:
    import numpy as np

    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def _raw_difference(reference: Any, candidate: Any) -> Dict[str, float | list[int]]:
    import numpy as np

    if reference.shape != candidate.shape:
        raise RuntimeError(f"输出shape不一致：{reference.shape} != {candidate.shape}")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    reference64 = reference.astype(np.float64)
    denominator = max(float(np.linalg.norm(reference64.ravel())), 1e-12)
    reference_norm = max(float(np.max(np.abs(reference64))), 1.0)
    first = reference64.ravel()
    second = candidate.astype(np.float64).ravel()
    cosine_denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-12)
    return {
        "shape": list(reference.shape),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / denominator),
        "normalized_max_abs": float(np.max(np.abs(difference)) / reference_norm),
        "cosine": float(np.dot(first, second) / cosine_denominator),
    }


def compare_raw_outputs(
    assets: Sequence[FixedOnnxAsset],
    image_paths: Sequence[Path],
    *,
    device: str = "0",
    provider: str = "cuda",
) -> Dict[str, Any]:
    if not image_paths:
        raise ValueError("原始输出对齐图像不能为空")
    rows = []
    for asset in assets:
        if asset.kind == "yolo":
            reference: RawRunner = TorchYoloRunner(asset.source, device)
        else:
            reference = TorchContextRunner(asset.source, device)
        candidate = OnnxRunner(asset.target, provider=provider, device=device)
        comparisons = []
        for path in image_paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
                if asset.kind == "yolo":
                    tensor, _info = detector_tensor(
                        image, asset.input_shape[2], asset.input_shape[3]
                    )
                else:
                    tensor = context_tensor(image, asset.input_shape[2])
            batch = tensor[None, ...]
            reference_outputs = reference.run(batch)
            candidate_outputs = candidate.run(batch)
            if len(reference_outputs) != len(candidate_outputs):
                raise RuntimeError(f"{asset.model_id}输出数量不一致")
            output_rows = [
                _raw_difference(expected, actual)
                for expected, actual in zip(reference_outputs, candidate_outputs)
            ]
            comparisons.append({"image": rel_path(path), "outputs": output_rows})
        flattened = [item for row in comparisons for item in row["outputs"]]
        aggregate = {
            "max_abs": max(float(item["max_abs"]) for item in flattened),
            "mean_abs": mean(float(item["mean_abs"]) for item in flattened),
            "max_relative_l2": max(float(item["relative_l2"]) for item in flattened),
            "max_normalized_abs": max(
                float(item["normalized_max_abs"]) for item in flattened
            ),
            "min_cosine": min(float(item["cosine"]) for item in flattened),
        }
        aggregate["passed"] = bool(
            aggregate["max_normalized_abs"] <= 1e-3
            and aggregate["max_relative_l2"] <= 1e-3
            and aggregate["min_cosine"] >= 0.99999
        )
        rows.append(
            {
                "model_id": asset.model_id,
                "reference_provider": reference.provider,
                "candidate_provider": candidate.provider,
                "aggregate": aggregate,
                "images": comparisons,
            }
        )
    return {
        "schema_version": 1,
        "image_count": len(image_paths),
        "models": rows,
        "passed": all(bool(row["aggregate"]["passed"]) for row in rows),
    }


def _postprocess_yolo(
    raw: Any,
    info: LetterboxInfo,
    *,
    confidence: float,
    iou: float,
    max_det: int,
    candidate_prefilter: bool = True,
) -> list[Dict[str, Any]]:
    import numpy as np
    import torch
    from ultralytics.utils import nms

    if raw.ndim != 3 or raw.shape[0] != 1:
        raise RuntimeError(f"YOLO输出必须是batch=1三维张量，收到{raw.shape}")
    class_count = int(raw.shape[1]) - 4
    prediction = np.asarray(raw)
    if candidate_prefilter:
        candidate_mask = (
            np.max(prediction[0, 4 : 4 + class_count, :], axis=0)
            > float(confidence)
        )
        candidate_count = int(np.count_nonzero(candidate_mask))
        if candidate_count == 0:
            return []
        # Ultralytics treats any tensor whose final dimension is exactly six
        # as an end-to-end BNC result. A six-anchor BCN slice would therefore
        # be misclassified, so retain the original tensor for this rare case.
        if candidate_count != 6:
            prediction = prediction[:, :, candidate_mask]
    # Ultralytics NMS converts xywh to xyxy in place. Always detach from the
    # caller's raw output so alignment/golden data cannot be mutated by scoring.
    prediction = np.ascontiguousarray(prediction.copy())
    outputs = nms.non_max_suppression(
        torch.from_numpy(prediction),
        conf_thres=float(confidence),
        iou_thres=float(iou),
        multi_label=True,
        max_det=int(max_det),
        nc=class_count,
    )
    rows = []
    for detection in outputs[0].detach().cpu().tolist():
        rows.append(
            {
                "xyxy": restore_xyxy(detection[:4], info),
                "confidence": float(detection[4]),
                "local_class_id": int(detection[5]),
            }
        )
    return rows


def predict_detector_records(
    runner: RawRunner,
    images: Sequence[Path],
    input_shape: Sequence[int],
    local_to_global: Mapping[int, int],
    *,
    confidence: float,
    iou: float,
    max_det: int,
    source_name: str,
) -> tuple[list[Dict[str, Any]], Dict[str, float]]:
    rows: list[Dict[str, Any]] = []
    preprocess_ms = inference_ms = postprocess_ms = 0.0
    mapping = {int(key): int(value) for key, value in local_to_global.items()}
    for path in images:
        started = time.perf_counter()
        with Image.open(path) as source:
            tensor, info = detector_tensor(
                source.convert("RGB"), int(input_shape[2]), int(input_shape[3])
            )
        preprocess_ms += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        raw = runner.run(tensor[None, ...])[0]
        inference_ms += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        detections = _postprocess_yolo(
            raw, info, confidence=confidence, iou=iou, max_det=max_det
        )
        for detection in detections:
            local_id = int(detection.pop("local_class_id"))
            if local_id not in mapping:
                raise RuntimeError(f"{source_name}输出未注册的本地类别：{local_id}")
            rows.append(
                {
                    "image_id": path.stem,
                    "class_id": mapping[local_id],
                    "source": source_name,
                    **detection,
                }
            )
        postprocess_ms += (time.perf_counter() - started) * 1000
    return rows, {
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "image_count": float(len(images)),
    }


def predict_context_records_onnx(
    runner: RawRunner,
    images: Sequence[Path],
    input_shape: Sequence[int],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    rows: Dict[str, Dict[str, Any]] = {}
    preprocess_ms = inference_ms = postprocess_ms = 0.0
    for path in images:
        started = time.perf_counter()
        with Image.open(path) as source:
            tensor = context_tensor(source.convert("RGB"), int(input_shape[2]))
        preprocess_ms += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        sensor_logits, scene_logits = runner.run(tensor[None, ...])
        inference_ms += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        sensor_probability = _softmax(sensor_logits)[0]
        scene_probability = _softmax(scene_logits)[0]
        sensor_id = int(sensor_probability.argmax())
        scene_id = int(scene_probability.argmax())
        rows[path.stem] = {
            "sensor": SENSOR_NAMES[sensor_id],
            "sensor_confidence": float(sensor_probability[sensor_id]),
            "sensor_probabilities": {
                name: float(sensor_probability[index])
                for index, name in enumerate(SENSOR_NAMES)
            },
            "scene": SCENE_NAMES[scene_id],
            "scene_confidence": float(scene_probability[scene_id]),
            "scene_probabilities": {
                name: float(scene_probability[index])
                for index, name in enumerate(SCENE_NAMES)
            },
        }
        postprocess_ms += (time.perf_counter() - started) * 1000
    return rows, {
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "image_count": float(len(images)),
    }


def _write_prediction_artifacts(
    output: Path,
    *,
    images: Sequence[Path],
    base_predictions: Sequence[Mapping[str, Any]],
    incremental_predictions: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs.json").write_text(
        json.dumps(
            {
                "input_mode": "unlabeled_images",
                "image_count": len(images),
                "image_stems": [path.stem for path in images],
                "all_owners_every_image": True,
                "labels_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name, records in (
        ("base_raw.jsonl", base_predictions),
        ("incremental_raw.jsonl", incremental_predictions),
    ):
        (output / name).write_text(
            "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )
    (output / "contexts.json").write_text(
        json.dumps(dict(contexts), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _false_activation_rate(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    images: Sequence[Path],
    new_class_id: int,
    threshold: float,
) -> float:
    positive = {
        str(row["image_id"])
        for row in ground_truth
        if int(row["class_id"]) == int(new_class_id)
    }
    negative = {path.stem for path in images} - positive
    activated = {
        str(row["image_id"])
        for row in predictions
        if int(row["class_id"]) == int(new_class_id)
        and float(row["confidence"]) >= float(threshold)
    }
    return len(negative & activated) / len(negative) if negative else 0.0


def evaluate_fixed_agent(
    assets: Sequence[FixedOnnxAsset],
    runners: Mapping[str, RawRunner],
    *,
    mixed_test: Sequence[Path],
    base_test: Sequence[Path],
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Run all owners first, freeze artifacts, then read labels and score."""
    config = json.loads(json.dumps(_strict_config()))
    profile = json.loads(
        resolve_path("models/production/incremental_detection/profile.json").read_text(
            encoding="utf-8"
        )
    )
    by_id = {asset.model_id: asset for asset in assets}
    predict = config["predict"]
    base_predictions, base_timings = predict_detector_records(
        runners["base_detector"],
        mixed_test,
        by_id["base_detector"].input_shape,
        {int(key): int(value) for key, value in profile["base_local_to_global"].items()},
        confidence=float(predict["conf"]),
        iou=float(predict["iou"]),
        max_det=int(predict["max_det"]),
        source_name="frozen_base_model",
    )
    incremental_predictions, incremental_timings = predict_detector_records(
        runners["incremental_detector"],
        mixed_test,
        by_id["incremental_detector"].input_shape,
        {0: int(profile["new_global_id"])},
        confidence=float(predict["conf"]),
        iou=float(predict["iou"]),
        max_det=int(predict["max_det"]),
        source_name="incremental_specialist",
    )
    contexts, context_timings = predict_context_records_onnx(
        runners["scene_sensor_net"], mixed_test, by_id["scene_sensor_net"].input_shape
    )
    artifact_dir = resolve_path(output_dir)
    _write_prediction_artifacts(
        artifact_dir,
        images=mixed_test,
        base_predictions=base_predictions,
        incremental_predictions=incremental_predictions,
        contexts=contexts,
    )

    pre_activation, fusion_decisions = fuse_old_new_predictions(
        base_predictions,
        incremental_predictions,
        nms_iou=float(config["fusion"]["nms_iou"]),
        cross_class=config["fusion"].get("cross_class"),
    )
    threshold = float(profile["activation_threshold"])
    combined, activation_rejections = apply_incremental_candidate_gates(
        pre_activation,
        {int(profile["new_global_id"]): threshold},
        contexts_by_image=contexts,
        context_prior=profile.get("context_prior"),
        max_context_penalty=float(profile.get("context_gate", {}).get("max_threshold_penalty", 0.0)),
    )

    for name, records in (
        ("fused_pre_activation.jsonl", pre_activation),
        ("fusion_decisions.jsonl", fusion_decisions),
        ("activation_rejections.jsonl", activation_rejections),
        ("fused_unlabeled.jsonl", combined),
    ):
        (artifact_dir / name).write_text(
            "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )

    # Labels are deliberately opened only after all owner outputs are persisted.
    ground_truth = yolo_ground_truth(mixed_test)
    base_ids = {path.stem for path in base_test}
    old_ids = sorted(int(value) for value in profile["base_local_to_global"].values())
    new_id = int(profile["new_global_id"])
    base_metrics = evaluate_ap50(
        subset_rows(base_predictions, base_ids), subset_rows(ground_truth, base_ids), old_ids
    )
    retention = retention_metrics(base_predictions, combined, ground_truth, old_ids)
    new_metrics = evaluate_ap50(combined, ground_truth, [new_id])
    full_metrics = evaluate_ap50(combined, ground_truth, GLOBAL_CLASS_NAMES)
    lock_pr = precision_recall(combined, ground_truth, new_id, threshold)
    false_activation = _false_activation_rate(
        combined, ground_truth, mixed_test, new_id, threshold
    )
    metrics = {
        "base_map50": float(base_metrics["map50"]),
        "new_map50": float(new_metrics["map50"]),
        "krr": float(retention["krr"]),
        "old_map50_before": float(retention["old_map50_before"]),
        "old_map50_after": float(retention["old_map50_after"]),
        "full_map50": float(full_metrics["map50"]),
        "lock_precision": float(lock_pr["precision"]),
        "lock_recall": float(lock_pr["recall"]),
        "false_activation_rate": float(false_activation),
    }
    gates = {
        "base_map50": metrics["base_map50"] >= 0.80,
        "new_map50": metrics["new_map50"] >= 0.60,
        "krr": metrics["krr"] >= 0.95,
        "lock_precision": metrics["lock_precision"] >= 0.90,
        "false_activation_rate": metrics["false_activation_rate"] <= 0.05,
    }
    result = {
        "schema_version": 1,
        "provider": {key: runner.provider for key, runner in runners.items()},
        "input_shapes": {
            key: list(by_id[key].input_shape) for key in sorted(by_id)
        },
        "unlabeled_inference_completed_before_labels": True,
        "all_owners_every_image": True,
        "mixed_test_image_count": len(mixed_test),
        "base_test_image_count": len(base_test),
        "prediction_counts": {
            "base": len(base_predictions),
            "incremental": len(incremental_predictions),
            "combined": len(combined),
            "fusion_decisions": len(fusion_decisions),
            "activation_rejections": len(activation_rejections),
        },
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "timings": {
            "base": base_timings,
            "incremental": incremental_timings,
            "context": context_timings,
        },
        "artifacts": rel_path(artifact_dir),
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _strict_config() -> Mapping[str, Any]:
    import yaml

    return yaml.safe_load(
        resolve_path("configs/strict_class_incremental_3plus1.yaml").read_text(
            encoding="utf-8"
        )
    )


def compare_fixed_agent_metrics(
    output_root: str | Path,
    *,
    shape_mode: str = "rect",
    device: str = "0",
    provider: str = "cuda",
) -> Dict[str, Any]:
    assets = production_onnx_plan(output_root, shape_mode=shape_mode)
    for asset in assets:
        validate_onnx_contract(asset)
    mixed_test = read_split(resolve_path("splits/strict_3plus1/mixed_test.txt"))
    base_test = read_split(resolve_path("splits/strict_3plus1/base_test.txt"))
    torch_runners: Dict[str, RawRunner] = {
        "base_detector": TorchYoloRunner(assets[0].source, device),
        "incremental_detector": TorchYoloRunner(assets[1].source, device),
        "scene_sensor_net": TorchContextRunner(assets[2].source, device),
    }
    onnx_runners: Dict[str, RawRunner] = {
        asset.model_id: OnnxRunner(asset.target, provider=provider, device=device)
        for asset in assets
    }
    root = resolve_path(output_root) / "alignment" / shape_mode
    occupied = [root / name for name in ("pytorch", "onnx", "summary.json")]
    if any(path.exists() for path in occupied):
        raise FileExistsError(f"指标对齐产物已存在，拒绝覆盖：{root}")
    pytorch_result = evaluate_fixed_agent(
        assets,
        torch_runners,
        mixed_test=mixed_test,
        base_test=base_test,
        output_dir=root / "pytorch",
    )
    onnx_result = evaluate_fixed_agent(
        assets,
        onnx_runners,
        mixed_test=mixed_test,
        base_test=base_test,
        output_dir=root / "onnx",
    )
    deltas = {
        key: float(onnx_result["metrics"][key]) - float(pytorch_result["metrics"][key])
        for key in pytorch_result["metrics"]
    }
    maximum_delta = max(abs(value) for value in deltas.values())
    result = {
        "schema_version": 1,
        "shape_mode": shape_mode,
        "pytorch": pytorch_result,
        "onnx": onnx_result,
        "metric_deltas": deltas,
        "maximum_absolute_metric_delta": maximum_delta,
        "metric_alignment_passed": maximum_delta <= 0.005,
        "passed": bool(
            pytorch_result["passed"]
            and onnx_result["passed"]
            and maximum_delta <= 0.005
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    report = root / "summary.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**result, "report": rel_path(report)}


def write_golden_bundle(
    output_root: str | Path,
    *,
    shape_mode: str = "rect",
    image_paths: Sequence[Path],
    device: str = "0",
    provider: str = "cuda",
    overwrite: bool = False,
) -> Dict[str, Any]:
    import numpy as np

    if not image_paths:
        raise ValueError("golden图像不能为空")
    assets = production_onnx_plan(output_root, shape_mode=shape_mode)
    runners = {
        asset.model_id: OnnxRunner(asset.target, provider=provider, device=device)
        for asset in assets
    }
    root = resolve_path(output_root) / "golden" / shape_mode
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"golden目录已存在：{root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rows = []
    for index, path in enumerate(image_paths):
        case_id = f"case_{index:02d}_{path.stem}"
        case_root = root / case_id
        case_root.mkdir()
        copied_image = case_root / path.name
        shutil.copy2(path, copied_image)
        with Image.open(path) as source:
            image = source.convert("RGB")
            model_rows = {}
            for asset in assets:
                if asset.kind == "yolo":
                    tensor, info = detector_tensor(
                        image, asset.input_shape[2], asset.input_shape[3]
                    )
                    preprocessing = asdict(info)
                else:
                    tensor = context_tensor(image, asset.input_shape[2])
                    preprocessing = {
                        "resize": [176, 176],
                        "center_crop": [160, 160],
                        "mean": [0.5, 0.5, 0.5],
                        "std": [0.25, 0.25, 0.25],
                    }
                tensor_path = case_root / f"{asset.model_id}_input.npy"
                np.save(tensor_path, tensor[None, ...])
                outputs = runners[asset.model_id].run(tensor[None, ...])
                output_paths = []
                for output_index, output in enumerate(outputs):
                    output_path = case_root / f"{asset.model_id}_output_{output_index}.npy"
                    np.save(output_path, output)
                    output_paths.append(rel_path(output_path))
                model_rows[asset.model_id] = {
                    "input": rel_path(tensor_path),
                    "input_shape": list(tensor[None, ...].shape),
                    "outputs": output_paths,
                    "output_shapes": [list(output.shape) for output in outputs],
                    "preprocessing": preprocessing,
                }
        rows.append(
            {
                "case_id": case_id,
                "source_image": rel_path(copied_image),
                "models": model_rows,
            }
        )
    payload = {
        "schema_version": 1,
        "purpose": "Ascend board preprocessing and raw-output golden reference",
        "contains_labels": False,
        "shape_mode": shape_mode,
        "case_count": len(rows),
        "cases": rows,
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**payload, "manifest": rel_path(manifest)}


def _latency_summary(values: Sequence[float]) -> Dict[str, float]:
    import numpy as np

    if not values:
        raise ValueError("延迟样本不能为空")
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "fps_from_mean": float(1000.0 / array.mean()),
    }


def _prepare_fixed_inputs(
    image: Image.Image | Any,
    assets: Mapping[str, FixedOnnxAsset],
    *,
    parallel: bool = False,
    executor: ThreadPoolExecutor | None = None,
) -> tuple[Any, Any, Any, LetterboxInfo, LetterboxInfo]:
    base_asset = assets["base_detector"]
    incremental_asset = assets["incremental_detector"]
    if parallel:
        if executor is None:
            raise ValueError("并行预处理要求executor")
        base_future = executor.submit(
            detector_tensor,
            image,
            base_asset.input_shape[2],
            base_asset.input_shape[3],
        )
        incremental_future = executor.submit(
            detector_tensor,
            image,
            incremental_asset.input_shape[2],
            incremental_asset.input_shape[3],
        )
        context_future = executor.submit(context_tensor, image, 160)
        base, base_info = base_future.result()
        incremental, incremental_info = incremental_future.result()
        context = context_future.result()
    else:
        base, base_info = detector_tensor(
            image, base_asset.input_shape[2], base_asset.input_shape[3]
        )
        incremental, incremental_info = detector_tensor(
            image,
            incremental_asset.input_shape[2],
            incremental_asset.input_shape[3],
        )
        context = context_tensor(image, 160)
    return (
        base[None, ...],
        incremental[None, ...],
        context[None, ...],
        base_info,
        incremental_info,
    )


def _run_three_models(
    runners: Mapping[str, RawRunner],
    prepared: Sequence[Any],
    executor: ThreadPoolExecutor,
    *,
    parallel: bool = True,
) -> tuple[Any, Any, list[Any]]:
    base, incremental, context = prepared[:3]
    if not parallel:
        return (
            runners["base_detector"].run(base)[0],
            runners["incremental_detector"].run(incremental)[0],
            runners["scene_sensor_net"].run(context),
        )
    base_future = executor.submit(runners["base_detector"].run, base)
    incremental_future = executor.submit(
        runners["incremental_detector"].run, incremental
    )
    context_future = executor.submit(runners["scene_sensor_net"].run, context)
    return (
        base_future.result()[0],
        incremental_future.result()[0],
        context_future.result(),
    )


def _context_record(context_raw: Sequence[Any]) -> Dict[str, Any]:
    sensor_probability = _softmax(context_raw[0])[0]
    scene_probability = _softmax(context_raw[1])[0]
    sensor_id = int(sensor_probability.argmax())
    scene_id = int(scene_probability.argmax())
    return {
        "sensor": SENSOR_NAMES[sensor_id],
        "sensor_confidence": float(sensor_probability[sensor_id]),
        "sensor_probabilities": {
            name: float(sensor_probability[index])
            for index, name in enumerate(SENSOR_NAMES)
        },
        "scene": SCENE_NAMES[scene_id],
        "scene_confidence": float(scene_probability[scene_id]),
        "scene_probabilities": {
            name: float(scene_probability[index])
            for index, name in enumerate(SCENE_NAMES)
        },
    }


def _postprocess_agent_outputs(
    base_raw: Any,
    incremental_raw: Any,
    context_raw: Sequence[Any],
    base_info: LetterboxInfo,
    incremental_info: LetterboxInfo,
    *,
    image_id: str,
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
    candidate_prefilter: bool,
    early_incremental_threshold: bool = False,
    parallel: bool = False,
    executor: ThreadPoolExecutor | None = None,
) -> list[Dict[str, Any]]:
    predict = config["predict"]

    def post(
        raw: Any, info: LetterboxInfo, confidence: float
    ) -> list[Dict[str, Any]]:
        return _postprocess_yolo(
            raw,
            info,
            confidence=float(confidence),
            iou=float(predict["iou"]),
            max_det=int(predict["max_det"]),
            candidate_prefilter=candidate_prefilter,
        )

    base_confidence = float(predict["conf"])
    incremental_confidence = base_confidence
    if early_incremental_threshold:
        # The context gate can only raise this frozen threshold. Rejecting
        # lower specialist candidates before NMS cannot activate a class or
        # remove a candidate that the final Agent would keep.
        incremental_confidence = max(
            incremental_confidence, float(profile["activation_threshold"])
        )

    if parallel:
        if executor is None:
            raise ValueError("并行后处理要求executor")
        base_future = executor.submit(
            post, base_raw, base_info, base_confidence
        )
        incremental_future = executor.submit(
            post,
            incremental_raw,
            incremental_info,
            incremental_confidence,
        )
        base_local = base_future.result()
        incremental_local = incremental_future.result()
    else:
        base_local = post(base_raw, base_info, base_confidence)
        incremental_local = post(
            incremental_raw, incremental_info, incremental_confidence
        )
    base_mapping = {
        int(key): int(value)
        for key, value in profile["base_local_to_global"].items()
    }
    new_id = int(profile["new_global_id"])

    def map_rows(
        rows: Sequence[Mapping[str, Any]],
        mapping: Mapping[int, int],
        source: str,
    ) -> list[Dict[str, Any]]:
        mapped = []
        for row in rows:
            local_id = int(row["local_class_id"])
            if local_id not in mapping:
                raise RuntimeError(f"{source}输出未注册的本地类别：{local_id}")
            mapped.append(
                {
                    "image_id": image_id,
                    "class_id": int(mapping[local_id]),
                    "source": source,
                    "xyxy": list(row["xyxy"]),
                    "confidence": float(row["confidence"]),
                }
            )
        return mapped

    base_rows = map_rows(base_local, base_mapping, "frozen_base_model")
    incremental_rows = map_rows(
        incremental_local, {0: new_id}, "incremental_specialist"
    )
    pre_activation, _decisions = fuse_old_new_predictions(
        base_rows,
        incremental_rows,
        nms_iou=float(config["fusion"]["nms_iou"]),
        cross_class=config["fusion"].get("cross_class"),
    )
    context = _context_record(context_raw)
    combined, _rejections = apply_incremental_candidate_gates(
        pre_activation,
        {new_id: float(profile["activation_threshold"])},
        contexts_by_image={image_id: context},
        context_prior=profile.get("context_prior"),
        max_context_penalty=float(
            profile.get("context_gate", {}).get("max_threshold_penalty", 0.0)
        ),
    )
    return combined


def benchmark_onnx_pipeline(
    output_root: str | Path,
    *,
    shape_mode: str,
    image_paths: Sequence[Path],
    device: str = "0",
    provider: str = "cuda",
    warmup: int = 20,
    rounds: int = 30,
) -> Dict[str, Any]:
    if not image_paths:
        raise ValueError("性能图像不能为空")
    assets = production_onnx_plan(output_root, shape_mode=shape_mode)
    by_id = {asset.model_id: asset for asset in assets}
    runners = {
        asset.model_id: OnnxRunner(asset.target, provider=provider, device=device)
        for asset in assets
    }
    config = _strict_config()
    profile = json.loads(
        resolve_path("models/production/incremental_detection/profile.json").read_text(
            encoding="utf-8"
        )
    )

    def prepare(path: Path) -> tuple[Any, Any, Any, LetterboxInfo, LetterboxInfo]:
        with Image.open(path) as source:
            image = source.convert("RGB")
            return _prepare_fixed_inputs(image, by_id)

    def run_once(path: Path, mode: str, executor: ThreadPoolExecutor) -> Dict[str, float]:
        started = time.perf_counter()
        base, incremental, context, base_info, incremental_info = prepare(path)
        prepared = time.perf_counter()
        if mode == "serial":
            base_raw = runners["base_detector"].run(base)[0]
            incremental_raw = runners["incremental_detector"].run(incremental)[0]
            context_raw = runners["scene_sensor_net"].run(context)
        elif mode == "detectors_parallel":
            base_future = executor.submit(runners["base_detector"].run, base)
            incremental_future = executor.submit(
                runners["incremental_detector"].run, incremental
            )
            base_raw = base_future.result()[0]
            incremental_raw = incremental_future.result()[0]
            context_raw = runners["scene_sensor_net"].run(context)
        elif mode == "all_parallel":
            base_future = executor.submit(runners["base_detector"].run, base)
            incremental_future = executor.submit(
                runners["incremental_detector"].run, incremental
            )
            context_future = executor.submit(runners["scene_sensor_net"].run, context)
            base_raw = base_future.result()[0]
            incremental_raw = incremental_future.result()[0]
            context_raw = context_future.result()
        else:
            raise ValueError(f"未知调度模式：{mode}")
        inferred = time.perf_counter()
        _postprocess_agent_outputs(
            base_raw,
            incremental_raw,
            context_raw,
            base_info,
            incremental_info,
            image_id=path.stem,
            config=config,
            profile=profile,
            candidate_prefilter=True,
        )
        finished = time.perf_counter()
        return {
            "preprocess_ms": (prepared - started) * 1000,
            "models_ms": (inferred - prepared) * 1000,
            "postprocess_ms": (finished - inferred) * 1000,
            "total_ms": (finished - started) * 1000,
        }

    modes = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        for mode in ("serial", "detectors_parallel", "all_parallel"):
            for index in range(int(warmup)):
                run_once(image_paths[index % len(image_paths)], mode, executor)
            samples = [
                run_once(image_paths[index % len(image_paths)], mode, executor)
                for index in range(int(rounds))
            ]
            modes.append(
                {
                    "mode": mode,
                    "rounds": int(rounds),
                    "total": _latency_summary([row["total_ms"] for row in samples]),
                    "preprocess": _latency_summary(
                        [row["preprocess_ms"] for row in samples]
                    ),
                    "models": _latency_summary([row["models_ms"] for row in samples]),
                    "postprocess": _latency_summary(
                        [row["postprocess_ms"] for row in samples]
                    ),
                }
            )
    best = min(modes, key=lambda row: float(row["total"]["p95_ms"]))
    result = {
        "schema_version": 1,
        "proxy_only": True,
        "not_ascend_performance_evidence": True,
        "provider": next(iter(runners.values())).provider,
        "shape_mode": shape_mode,
        "includes_box_fusion_and_context_gate": True,
        "candidate_prefilter": True,
        "warmup_iterations_per_mode": int(warmup),
        "image_count": len(image_paths),
        "modes": modes,
        "best_proxy_mode_by_p95": best["mode"],
    }
    root = resolve_path(output_root) / "benchmarks" / shape_mode
    root.mkdir(parents=True, exist_ok=True)
    report = root / "onnxruntime_proxy.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**result, "report": rel_path(report)}


def _prediction_rows_close(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    tolerance: float = 1e-5,
) -> bool:
    if len(expected) != len(actual):
        return False
    for first, second in zip(expected, actual):
        if int(first["class_id"]) != int(second["class_id"]):
            return False
        if str(first["source"]) != str(second["source"]):
            return False
        if abs(float(first["confidence"]) - float(second["confidence"])) > tolerance:
            return False
        if any(
            abs(float(left) - float(right)) > tolerance
            for left, right in zip(first["xyxy"], second["xyxy"])
        ):
            return False
    return True


def benchmark_local_optimization_candidates(
    output_root: str | Path,
    *,
    shape_mode: str,
    image_paths: Sequence[Path],
    correctness_paths: Sequence[Path] | None = None,
    device: str = "0",
    warmup: int = 20,
    rounds: int = 50,
) -> Dict[str, Any]:
    """Benchmark board-preflight optimization ideas on a local CUDA proxy."""
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch

    if not image_paths:
        raise ValueError("优化候选性能图像不能为空")
    verification_images = list(correctness_paths or image_paths)
    if not verification_images:
        raise ValueError("优化候选正确性图像不能为空")
    assets = production_onnx_plan(output_root, shape_mode=shape_mode)
    for asset in assets:
        validate_onnx_contract(asset)
    by_id = {asset.model_id: asset for asset in assets}
    standard_runners: Dict[str, RawRunner] = {
        asset.model_id: OnnxRunner(asset.target, provider="cuda", device=device)
        for asset in assets
    }
    graph_runners: Dict[str, RawRunner] = {
        asset.model_id: FixedCudaGraphOnnxRunner(asset, device=device)
        for asset in assets
    }
    config = _strict_config()
    profile = json.loads(
        resolve_path("models/production/incremental_detection/profile.json").read_text(
            encoding="utf-8"
        )
    )

    def prepare(
        path: Path,
        backend: str,
        *,
        parallel: bool,
        executor: ThreadPoolExecutor,
    ) -> tuple[Any, Any, Any, LetterboxInfo, LetterboxInfo]:
        image = decode_png_rgb(path, backend=backend)
        return _prepare_fixed_inputs(
            image, by_id, parallel=parallel, executor=executor
        )

    def postprocess(
        path: Path,
        prepared: Sequence[Any],
        raw_outputs: Sequence[Any],
        *,
        prefilter: bool,
        early_incremental_threshold: bool,
        parallel: bool,
        executor: ThreadPoolExecutor,
    ) -> list[Dict[str, Any]]:
        base_raw, incremental_raw, context_raw = raw_outputs
        return _postprocess_agent_outputs(
            base_raw,
            incremental_raw,
            context_raw,
            prepared[3],
            prepared[4],
            image_id=path.stem,
            config=config,
            profile=profile,
            candidate_prefilter=prefilter,
            early_incremental_threshold=early_incremental_threshold,
            parallel=parallel,
            executor=executor,
        )

    input_max_abs = 0.0
    raw_max_abs = 0.0
    postfilter_mismatches = 0
    early_threshold_mismatches = 0
    end_to_end_mismatches = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        for path in verification_images:
            pillow_inputs = prepare(
                path, "pillow", parallel=False, executor=executor
            )
            opencv_inputs = prepare(
                path, "opencv", parallel=False, executor=executor
            )
            for expected, actual in zip(pillow_inputs[:3], opencv_inputs[:3]):
                input_max_abs = max(
                    input_max_abs,
                    float(
                        np.max(
                            np.abs(
                                expected.astype(np.float64)
                                - actual.astype(np.float64)
                            )
                        )
                    ),
                )
            standard_raw = _run_three_models(
                standard_runners, pillow_inputs, executor
            )
            graph_raw = _run_three_models(
                graph_runners, opencv_inputs, executor, parallel=False
            )
            for expected, actual in zip(standard_raw, graph_raw):
                if isinstance(expected, list):
                    pairs = zip(expected, actual)
                else:
                    pairs = [(expected, actual)]
                for first, second in pairs:
                    raw_max_abs = max(
                        raw_max_abs,
                        float(
                            np.max(
                                np.abs(
                                    first.astype(np.float64)
                                    - second.astype(np.float64)
                                )
                            )
                        ),
                    )
            baseline_rows = postprocess(
                path,
                pillow_inputs,
                standard_raw,
                prefilter=False,
                early_incremental_threshold=False,
                parallel=False,
                executor=executor,
            )
            prefiltered_rows = postprocess(
                path,
                pillow_inputs,
                standard_raw,
                prefilter=True,
                early_incremental_threshold=False,
                parallel=False,
                executor=executor,
            )
            early_threshold_rows = postprocess(
                path,
                pillow_inputs,
                standard_raw,
                prefilter=True,
                early_incremental_threshold=True,
                parallel=False,
                executor=executor,
            )
            optimized_rows = postprocess(
                path,
                opencv_inputs,
                graph_raw,
                prefilter=True,
                early_incremental_threshold=True,
                parallel=False,
                executor=executor,
            )
            if not _prediction_rows_close(baseline_rows, prefiltered_rows):
                postfilter_mismatches += 1
            if not _prediction_rows_close(baseline_rows, early_threshold_rows):
                early_threshold_mismatches += 1
            if not _prediction_rows_close(baseline_rows, optimized_rows):
                end_to_end_mismatches += 1

        variants = [
            {
                "name": "reference_pillow_standard",
                "decode_backend": "pillow",
                "runner": standard_runners,
                "candidate_prefilter": False,
                "early_incremental_threshold": False,
                "model_parallel": True,
                "preprocess_parallel": False,
                "postprocess_parallel": False,
            },
            {
                "name": "opencv_decode_standard",
                "decode_backend": "opencv",
                "runner": standard_runners,
                "candidate_prefilter": False,
                "early_incremental_threshold": False,
                "model_parallel": True,
                "preprocess_parallel": False,
                "postprocess_parallel": False,
            },
            {
                "name": "opencv_cuda_graph_fixed_io",
                "decode_backend": "opencv",
                "runner": graph_runners,
                "candidate_prefilter": False,
                "early_incremental_threshold": False,
                "model_parallel": False,
                "preprocess_parallel": False,
                "postprocess_parallel": False,
            },
            {
                "name": "candidate_prefilter_cuda_graph",
                "decode_backend": "opencv",
                "runner": graph_runners,
                "candidate_prefilter": True,
                "early_incremental_threshold": False,
                "model_parallel": False,
                "preprocess_parallel": False,
                "postprocess_parallel": False,
            },
            {
                "name": "full_candidate",
                "decode_backend": "opencv",
                "runner": graph_runners,
                "candidate_prefilter": True,
                "early_incremental_threshold": True,
                "model_parallel": False,
                "preprocess_parallel": False,
                "postprocess_parallel": False,
            },
            {
                "name": "python_preprocess_parallel",
                "decode_backend": "opencv",
                "runner": graph_runners,
                "candidate_prefilter": True,
                "early_incremental_threshold": True,
                "model_parallel": False,
                "preprocess_parallel": True,
                "postprocess_parallel": False,
            },
            {
                "name": "python_postprocess_parallel",
                "decode_backend": "opencv",
                "runner": graph_runners,
                "candidate_prefilter": True,
                "early_incremental_threshold": True,
                "model_parallel": False,
                "preprocess_parallel": False,
                "postprocess_parallel": True,
            },
        ]

        def run_once(path: Path, variant: Mapping[str, Any]) -> Dict[str, float]:
            started = time.perf_counter()
            prepared = prepare(
                path,
                str(variant["decode_backend"]),
                parallel=bool(variant["preprocess_parallel"]),
                executor=executor,
            )
            prepared_at = time.perf_counter()
            raw_outputs = _run_three_models(
                variant["runner"],
                prepared,
                executor,
                parallel=bool(variant["model_parallel"]),
            )
            inferred_at = time.perf_counter()
            rows = postprocess(
                path,
                prepared,
                raw_outputs,
                prefilter=bool(variant["candidate_prefilter"]),
                early_incremental_threshold=bool(
                    variant["early_incremental_threshold"]
                ),
                parallel=bool(variant["postprocess_parallel"]),
                executor=executor,
            )
            finished = time.perf_counter()
            return {
                "preprocess_ms": (prepared_at - started) * 1000,
                "models_ms": (inferred_at - prepared_at) * 1000,
                "postprocess_ms": (finished - inferred_at) * 1000,
                "total_ms": (finished - started) * 1000,
                "final_detection_count": float(len(rows)),
            }

        for iteration in range(int(warmup)):
            offset = iteration % len(variants)
            order = variants[offset:] + variants[:offset]
            path = image_paths[iteration % len(image_paths)]
            for variant in order:
                run_once(path, variant)

        samples_by_name: Dict[str, list[Dict[str, float]]] = {
            str(variant["name"]): [] for variant in variants
        }
        for iteration in range(int(rounds)):
            offset = iteration % len(variants)
            order = variants[offset:] + variants[:offset]
            path = image_paths[iteration % len(image_paths)]
            for variant in order:
                samples_by_name[str(variant["name"])].append(
                    run_once(path, variant)
                )

    rows = []
    reference_mean = mean(
        row["total_ms"] for row in samples_by_name["reference_pillow_standard"]
    )
    for variant in variants:
        name = str(variant["name"])
        samples = samples_by_name[name]
        total = _latency_summary([row["total_ms"] for row in samples])
        rows.append(
            {
                "name": name,
                "decode_backend": variant["decode_backend"],
                "cuda_graph_fixed_io": variant["runner"] is graph_runners,
                "candidate_prefilter": bool(variant["candidate_prefilter"]),
                "early_incremental_threshold": bool(
                    variant["early_incremental_threshold"]
                ),
                "model_parallel": bool(variant["model_parallel"]),
                "preprocess_parallel": bool(variant["preprocess_parallel"]),
                "postprocess_parallel": bool(variant["postprocess_parallel"]),
                "rounds": int(rounds),
                "speedup_vs_reference": float(reference_mean / total["mean_ms"]),
                "total": total,
                "preprocess": _latency_summary(
                    [row["preprocess_ms"] for row in samples]
                ),
                "models": _latency_summary(
                    [row["models_ms"] for row in samples]
                ),
                "postprocess": _latency_summary(
                    [row["postprocess_ms"] for row in samples]
                ),
            }
        )
    by_name = {row["name"]: row for row in rows}
    correctness = {
        "image_count": len(verification_images),
        "contains_labels": False,
        "input_max_abs": input_max_abs,
        "raw_output_max_abs": raw_max_abs,
        "candidate_prefilter_mismatch_images": postfilter_mismatches,
        "early_incremental_threshold_mismatch_images": early_threshold_mismatches,
        "end_to_end_detection_mismatch_images": end_to_end_mismatches,
    }
    correctness["passed"] = bool(
        input_max_abs == 0.0
        and raw_max_abs <= 1e-3
        and postfilter_mismatches == 0
        and early_threshold_mismatches == 0
        and end_to_end_mismatches == 0
    )
    decisions = {
        "opencv_decode_supported": bool(
            correctness["passed"]
            and float(by_name["opencv_decode_standard"]["total"]["p95_ms"])
            < float(by_name["reference_pillow_standard"]["total"]["p95_ms"])
        ),
        "fixed_graph_supported_on_cuda_proxy": bool(
            correctness["passed"]
            and float(by_name["opencv_cuda_graph_fixed_io"]["total"]["p95_ms"])
            < float(by_name["opencv_decode_standard"]["total"]["p95_ms"])
        ),
        "candidate_prefilter_supported": bool(
            correctness["passed"]
            and float(
                by_name["candidate_prefilter_cuda_graph"]["total"]["p95_ms"]
            )
            < float(by_name["opencv_cuda_graph_fixed_io"]["total"]["p95_ms"])
        ),
        "early_incremental_threshold_supported": bool(
            correctness["passed"]
            and float(by_name["full_candidate"]["total"]["p95_ms"])
            < float(
                by_name["candidate_prefilter_cuda_graph"]["total"]["p95_ms"]
            )
        ),
        "python_preprocess_parallel_rejected": bool(
            float(by_name["python_preprocess_parallel"]["total"]["p95_ms"])
            >= float(by_name["full_candidate"]["total"]["p95_ms"])
        ),
        "python_postprocess_parallel_rejected": bool(
            float(by_name["python_postprocess_parallel"]["total"]["p95_ms"])
            >= float(by_name["full_candidate"]["total"]["p95_ms"])
        ),
        "cuda_graph_default_stream_parallelism_rejected": True,
    }
    best = min(rows, key=lambda row: float(row["total"]["p95_ms"]))
    result = {
        "schema_version": 1,
        "study": "local_ascend_optimization_candidates",
        "proxy_only": True,
        "not_ascend_performance_evidence": True,
        "all_owners_every_image": True,
        "includes_box_fusion_and_context_gate": True,
        "shape_mode": shape_mode,
        "warmup_iterations_per_variant": int(warmup),
        "performance_image_count": len(image_paths),
        "correctness": correctness,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "opencv_threads": int(cv2.getNumThreads()),
            "onnxruntime": ort.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(int(device)),
        },
        "variants": rows,
        "best_proxy_variant_by_p95": best["name"],
        "decisions": decisions,
        "passed": bool(correctness["passed"]),
    }
    root = resolve_path(output_root) / "benchmarks" / shape_mode
    root.mkdir(parents=True, exist_ok=True)
    report = root / "local_optimization_candidates.json"
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**result, "report": rel_path(report)}
