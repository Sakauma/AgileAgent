from __future__ import annotations

import ast
import collections
import fnmatch
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml
from PIL import Image

from fair_agent.core.config import config_sha256, rel_path, resolve_path, write_config
from fair_agent.core.hashes import sha256_file


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_images(path: str | Path) -> list[Path]:
    split = resolve_path(path)
    if not split.is_file():
        raise FileNotFoundError(f"INT8代表性数据划分不存在：{split}")
    images = [resolve_path(line.strip()) for line in split.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not images or any(not image.is_file() for image in images):
        raise ValueError("INT8代表性数据划分为空或包含缺失图像。")
    return images


def _optimal_detector_shape(images: Sequence[Path], image_size: int) -> tuple[int, int]:
    shapes: collections.Counter[tuple[int, int]] = collections.Counter()
    for path in images:
        with Image.open(path) as source:
            width, height = source.size
        scale = min(image_size / width, image_size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        output_width = resized_width + (image_size - resized_width) % 32
        output_height = resized_height + (image_size - resized_height) % 32
        shapes[(output_height, output_width)] += 1
    if not shapes:
        raise ValueError("无法从空校准集推导TensorRT最优空间尺寸。")
    return sorted(shapes, key=lambda shape: (-shapes[shape], shape))[0]


def _label_classes(image: Path) -> set[int]:
    label = image.with_suffix(".txt")
    if not label.is_file():
        raise FileNotFoundError(f"INT8校准图像缺少标签：{label}")
    return {
        int(line.split()[0])
        for line in label.read_text(encoding="utf-8").splitlines()
        if line.strip() and len(line.split()) == 5
    }


def _registered_model(config: Mapping[str, Any], source: str) -> Mapping[str, Any] | None:
    from fair_agent.modules.generation_management import active_generation_registry
    from fair_agent.modules.model_generations import load_generation_registry

    registry = load_generation_registry(active_generation_registry(config))
    normalized = rel_path(resolve_path(source))
    return next(
        (model for model in registry["models_by_id"].values() if rel_path(model["resolved_path"]) == normalized),
        None,
    )


def _select_representative_images(
    images: Sequence[Path],
    class_ids: set[int],
    settings: Mapping[str, Any],
) -> list[Path]:
    seed = int(settings["seed"])
    maximum = int(settings["max_images"])
    minimum_per_class = int(settings["minimum_images_per_class"])
    rows = [(path, _label_classes(path)) for path in images]
    if class_ids:
        rows = [(path, labels) for path, labels in rows if labels and labels <= class_ids]
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{rel_path(row[0])}".encode("utf-8")).hexdigest(),
    )
    selected: list[Path] = []
    selected_set: set[Path] = set()
    for class_id in sorted(class_ids):
        candidates = [path for path, labels in ordered if class_id in labels and path not in selected_set]
        needed = minimum_per_class - sum(class_id in _label_classes(path) for path in selected)
        if needed > len(candidates):
            raise ValueError(f"INT8校准类别{class_id}不足{minimum_per_class}张代表性图像。")
        for path in candidates[:max(0, needed)]:
            selected.append(path)
            selected_set.add(path)
    if len(selected) > maximum:
        raise ValueError("INT8逐类最小覆盖超过max_images，请调整校准配置。")
    for path, _labels in ordered:
        if len(selected) >= maximum:
            break
        if path not in selected_set:
            selected.append(path)
            selected_set.add(path)
    if not selected:
        raise ValueError("没有满足类别所有权约束的INT8代表性图像。")
    return selected


def prepare_calibration_manifest(
    config: Mapping[str, Any],
    source: str,
    images: Sequence[Path],
    class_ids: Sequence[int],
    scope: str,
    forbidden_stems: Sequence[str] = (),
    preprocessing: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    backend = config["tensorrt_backend"]
    settings = backend["int8_calibration"]
    if settings["enabled"] is not True:
        raise ValueError("INT8校准未启用。")
    selected = _select_representative_images(images, set(map(int, class_ids)), settings)
    forbidden = {str(value).casefold() for value in forbidden_stems}
    leaked = sorted(path.stem for path in selected if path.stem.casefold() in forbidden)
    if leaked:
        raise ValueError(f"INT8校准集与封存lock重叠：{leaked[:10]}")
    rows = []
    for image in selected:
        label = image.with_suffix(".txt")
        rows.append({
            "image": rel_path(image),
            "image_sha256": sha256_file(image),
            "label": rel_path(label),
            "label_sha256": sha256_file(label),
            "class_ids": sorted(_label_classes(image)),
        })
    fingerprint_payload = {
        "source": rel_path(resolve_path(source)),
        "scope": scope,
        "class_ids": sorted(map(int, class_ids)),
        "batch_size": int(settings["batch_size"]),
        "seed": int(settings["seed"]),
        "preprocessing": dict(preprocessing or {}),
        "files": rows,
    }
    fingerprint = _canonical_hash(fingerprint_payload)
    root = resolve_path(settings["cache_root"])
    manifest_path = root / "manifests" / f"{Path(source).stem}-{fingerprint[:16]}.json"
    cache_path = root / "cache" / f"{fingerprint}.cache"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        **fingerprint_payload,
        "calibration_cache": rel_path(cache_path),
        "lock_content_read": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError("INT8校准manifest发生不可解释的指纹冲突。")
    else:
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
    return {
        "manifest": rel_path(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "fingerprint": fingerprint,
        "cache": rel_path(cache_path),
        "batch_size": int(settings["batch_size"]),
        "image_count": len(rows),
        "images": [resolve_path(row["image"]) for row in rows],
        "scope": scope,
    }


def _static_calibration(config: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
    settings = config["tensorrt_backend"]["int8_calibration"]
    images = _split_images(settings["representative_split"])
    lock_stems = [path.stem for path in _split_images(config["generation"]["recheck_lock_split"])]
    model = _registered_model(config, str(row["source"])) if row["kind"] == "yolo" else None
    class_ids = sorted(model["owns_classes"]) if model else []
    scope = f"registered_model:{model['id']}" if model else "context_perception"
    return prepare_calibration_manifest(
        config,
        str(row["source"]),
        images,
        class_ids,
        scope,
        lock_stems,
        preprocessing={
            "mode": "context_center_crop" if row["kind"] == "context" else "detector_letterbox",
            "imgsz": int(row["imgsz"]),
            "opt_height": int(row.get("opt_height", row["imgsz"])),
            "opt_width": int(row.get("opt_width", row["imgsz"])),
            "dynamic_spatial": bool(config["tensorrt_backend"]["dynamic"]),
            "minimum_spatial_size": int(config["tensorrt_backend"]["minimum_spatial_size"]),
            "tensorrt_version": str(config["tensorrt_backend"]["expected_version"]),
            "compute_capability": str(config["tensorrt_backend"]["expected_compute_capability"]),
            "mixed_precision": dict(config["tensorrt_backend"]["mixed_precision"]),
        },
    )


def export_plan(config: Mapping[str, Any]) -> list[Dict[str, Any]]:
    backend = config["tensorrt_backend"]
    rows = [
        {
            "kind": "yolo",
            "source": source,
            "target": str(entry["path"]),
            "imgsz": int(entry["imgsz"]),
            "opt_height": int(entry["opt_height"]),
            "opt_width": int(entry["opt_width"]),
            "min_batch_size": int(entry["min_batch_size"]),
            "opt_batch_size": int(entry["opt_batch_size"]),
            "batch_size": int(entry["batch_size"]),
            "expected_sha256": entry.get("sha256"),
        }
        for source, entry in backend["engines"].items()
    ]
    context = backend["context_engine"]
    rows.append({
        "kind": "context",
        "source": str(backend["export"]["context_checkpoint"]),
        "target": str(context["path"]),
        "imgsz": int(context["imgsz"]),
        "min_batch_size": int(context["min_batch_size"]),
        "opt_batch_size": int(context["opt_batch_size"]),
        "batch_size": int(context["batch_size"]),
        "expected_sha256": context.get("sha256"),
    })
    return rows


def _verified_existing(row: Mapping[str, Any]) -> Dict[str, Any] | None:
    target = resolve_path(row["target"])
    if not target.is_file():
        return None
    if not row.get("expected_sha256"):
        raise RuntimeError(
            f"目标engine已存在但设备配置尚未登记哈希，拒绝采用或覆盖：{rel_path(target)}"
        )
    digest = sha256_file(target)
    if digest != row["expected_sha256"]:
        raise RuntimeError(f"已有engine哈希不匹配，拒绝覆盖：{rel_path(target)}")
    return {**dict(row), "status": "verified", "sha256": digest}


def _replace_export(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=target.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(target)


def _onnx_metadata(path: Path) -> Dict[str, Any]:
    import onnx

    model = onnx.load(path)
    metadata: Dict[str, Any] = {}
    for item in model.metadata_props:
        raw = item.value
        try:
            metadata[item.key] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            try:
                metadata[item.key] = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                metadata[item.key] = raw
    return metadata


def _int8_calibrator(
    trt: Any,
    calibration: Mapping[str, Any],
    image_shape: int | tuple[int, int],
    device_index: str,
    mode: str,
) -> Any:
    import numpy as np
    import torch

    target_height, target_width = (
        (int(image_shape), int(image_shape))
        if isinstance(image_shape, int)
        else (int(image_shape[0]), int(image_shape[1]))
    )

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self.paths = [resolve_path(path) for path in calibration["images"]]
            self.batch_size = int(calibration["batch_size"])
            self.cache_path = resolve_path(calibration["cache"])
            self.offset = 0
            self.device = torch.device(f"cuda:{device_index}")
            self.device_batch: Any | None = None

        def get_batch_size(self) -> int:
            return self.batch_size

        @staticmethod
        def detector_array(path: Path) -> Any:
            with Image.open(path) as source:
                image = source.convert("RGB")
                scale = min(target_width / image.width, target_height / image.height)
                target = (
                    max(1, int(round(image.width * scale))),
                    max(1, int(round(image.height * scale))),
                )
                resized = image.resize(target, Image.Resampling.BILINEAR)
                canvas = Image.new("RGB", (target_width, target_height), (114, 114, 114))
                canvas.paste(resized, ((target_width - target[0]) // 2, (target_height - target[1]) // 2))
                array = np.asarray(canvas, dtype=np.float32) / 255.0
            return np.transpose(array, (2, 0, 1))

        @staticmethod
        def context_array(path: Path) -> Any:
            if target_height != target_width:
                raise ValueError("场景模型INT8校准只支持方形输入。")
            resize_size = int(round(target_height * 1.1))
            crop = (resize_size - target_height) // 2
            with Image.open(path) as source:
                resized = source.convert("RGB").resize(
                    (resize_size, resize_size), Image.Resampling.BILINEAR
                )
                image = resized.crop((crop, crop, crop + target_height, crop + target_height))
                array = np.asarray(image, dtype=np.float32) / 255.0
            array = (array - 0.5) / 0.25
            return np.transpose(array, (2, 0, 1))

        def get_batch(self, _names: Sequence[str]) -> list[int] | None:
            if self.offset >= len(self.paths):
                return None
            paths = self.paths[self.offset:self.offset + self.batch_size]
            self.offset += len(paths)
            while len(paths) < self.batch_size:
                paths.append(paths[-1])
            transform = self.context_array if mode == "context" else self.detector_array
            host = np.stack([transform(path) for path in paths]).astype(np.float32, copy=False)
            self.device_batch = torch.from_numpy(host).to(self.device, non_blocking=False).contiguous()
            return [int(self.device_batch.data_ptr())]

        def read_calibration_cache(self) -> bytes | None:
            return self.cache_path.read_bytes() if self.cache_path.is_file() else None

        def write_calibration_cache(self, cache: bytes) -> None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(".tmp")
            temporary.write_bytes(cache)
            os.replace(temporary, self.cache_path)

    return EntropyCalibrator()


def _apply_mixed_precision(
    network: Any,
    builder_config: Any,
    backend: Mapping[str, Any],
    trt: Any,
) -> Dict[str, Any]:
    settings = backend["mixed_precision"]
    report: Dict[str, Any] = {
        "enabled": bool(settings["enabled"]),
        "constraint": str(settings["constraint"]),
        "patterns": list(settings["fp16_layer_patterns"]),
        "matched_layer_count": 0,
        "matched_layers": [],
    }
    if backend["precision"] != "int8" or not report["enabled"]:
        return report
    patterns = report["patterns"]
    float_types = {trt.float16, trt.float32}
    precision_layer_types = {
        trt.LayerType.ACTIVATION,
        trt.LayerType.CONVOLUTION,
        trt.LayerType.DECONVOLUTION,
        trt.LayerType.ELEMENTWISE,
        trt.LayerType.MATRIX_MULTIPLY,
        trt.LayerType.NORMALIZATION,
        trt.LayerType.PARAMETRIC_RELU,
        trt.LayerType.POOLING,
        trt.LayerType.SOFTMAX,
    }
    matched = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if not any(fnmatch.fnmatchcase(str(layer.name), pattern) for pattern in patterns):
            continue
        if layer.type not in precision_layer_types:
            continue
        float_outputs = []
        for output_index in range(layer.num_outputs):
            output = layer.get_output(output_index)
            if (
                output is not None
                and output.dtype in float_types
                and not bool(output.is_shape_tensor)
            ):
                float_outputs.append((output_index, output))
        if not float_outputs:
            continue
        layer.precision = trt.float16
        for output_index, _output in float_outputs:
            layer.set_output_type(output_index, trt.float16)
        matched.append({
            "index": index,
            "name": str(layer.name),
            "type": str(layer.type),
            "outputs": [str(output.name) for _output_index, output in float_outputs],
        })
    minimum = int(settings["minimum_matched_layers"])
    if len(matched) < minimum:
        raise RuntimeError(
            f"混合精度FP16层仅命中{len(matched)}层，低于配置要求{minimum}层。"
        )
    constraint = (
        trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS
        if settings["constraint"] == "obey"
        else trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS
    )
    builder_config.set_flag(constraint)
    report["matched_layer_count"] = len(matched)
    report["matched_layers"] = matched
    return report


def _build_detector_engine(
    onnx_path: Path,
    output_path: Path,
    row: Mapping[str, Any],
    backend: Mapping[str, Any],
) -> Dict[str, Any]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        details = "; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError("YOLO ONNX解析失败：" + details)
    if network.num_inputs != 1:
        raise RuntimeError("YOLO TensorRT导出要求单一图像输入。")
    image_size = int(row["imgsz"])
    optimum_height = int(row["opt_height"])
    optimum_width = int(row["opt_width"])
    minimum = int(row["min_batch_size"])
    optimum = int(row["opt_batch_size"])
    maximum = int(row["batch_size"])
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    minimum_spatial = (
        int(backend["minimum_spatial_size"])
        if bool(backend["dynamic"])
        else image_size
    )
    profile.set_shape(
        input_tensor.name,
        (minimum, 3, minimum_spatial, minimum_spatial),
        (optimum, 3, optimum_height, optimum_width),
        (maximum, 3, image_size, image_size),
    )
    builder_config = builder.create_builder_config()
    builder_config.add_optimization_profile(profile)
    builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(float(backend["workspace_gib"]) * 1024 ** 3),
    )
    calibrator = None
    if backend["precision"] in {"fp16", "int8"}:
        builder_config.set_flag(trt.BuilderFlag.FP16)
    if backend["precision"] == "int8":
        calibration = row.get("calibration")
        if not isinstance(calibration, Mapping):
            raise ValueError("INT8 YOLO导出缺少代表性数据校准manifest。")
        builder_config.set_flag(trt.BuilderFlag.INT8)
        calibrator = _int8_calibrator(
            trt,
            calibration,
            (optimum_height, optimum_width),
            str(backend["export"]["device"]),
            "detector",
        )
        builder_config.int8_calibrator = calibrator
        calibration_profile = builder.create_optimization_profile()
        calibration_shape = (optimum, 3, optimum_height, optimum_width)
        calibration_profile.set_shape(
            input_tensor.name, calibration_shape, calibration_shape, calibration_shape
        )
        if not builder_config.set_calibration_profile(calibration_profile):
            raise RuntimeError("TensorRT INT8校准profile注册失败。")
    precision_report = _apply_mixed_precision(network, builder_config, backend, trt)
    serialized = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise RuntimeError("YOLO TensorRT engine构建失败。")
    metadata = _onnx_metadata(onnx_path)
    metadata["batch"] = maximum
    metadata["mixed_precision"] = {
        key: value for key, value in precision_report.items() if key != "matched_layers"
    }
    encoded_metadata = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    with output_path.open("wb") as output:
        output.write(len(encoded_metadata).to_bytes(4, byteorder="little", signed=True))
        output.write(encoded_metadata)
        output.write(bytes(serialized))
    return precision_report


def _export_yolo(row: Mapping[str, Any], backend: Mapping[str, Any]) -> str:
    from ultralytics import YOLO

    source = resolve_path(row["source"])
    if not source.is_file():
        raise FileNotFoundError(source)
    before = sha256_file(source)
    export = backend["export"]
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_root = Path(temporary_dir)
        temporary_weights = temporary_root / source.name
        shutil.copy2(source, temporary_weights)
        result = YOLO(str(temporary_weights)).export(
            format="onnx",
            imgsz=int(row["imgsz"]),
            batch=int(row["opt_batch_size"]),
            dynamic=True,
            simplify=bool(export["simplify"]),
            opset=int(export["onnx_opset"]),
            device=str(export["device"]),
        )
        onnx_path = Path(str(result))
        if not onnx_path.is_file():
            raise RuntimeError(f"Ultralytics未生成YOLO ONNX：{onnx_path}")
        generated = temporary_root / "model.engine"
        precision_report = _build_detector_engine(onnx_path, generated, row, backend)
        target = resolve_path(row["target"])
        digest = _replace_export(generated, target)
        precision_payload = {
            "schema_version": 1,
            "engine": rel_path(target),
            "engine_sha256": digest,
            "source": rel_path(source),
            **precision_report,
        }
        report_path = target.with_suffix(target.suffix + ".precision.json")
        temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary_report.write_text(
            json.dumps(precision_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_report, report_path)
    if sha256_file(source) != before:
        raise RuntimeError(f"导出过程修改了冻结权重：{rel_path(source)}")
    return digest


def _export_context(row: Mapping[str, Any], backend: Mapping[str, Any]) -> str:
    import onnx
    import tensorrt as trt
    import torch

    from fair_agent.models.context import SceneSensorNet

    source = resolve_path(row["source"])
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    model = SceneSensorNet(
        channels=checkpoint["architecture"]["channels"],
        dropout=float(checkpoint["architecture"]["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    export = backend["export"]
    image_size = int(row["imgsz"])
    min_batch = int(row["min_batch_size"])
    opt_batch = int(row["opt_batch_size"])
    max_batch = int(row["batch_size"])
    with tempfile.TemporaryDirectory() as temporary_dir:
        onnx_path = Path(temporary_dir) / "scene_sensor_net.onnx"
        torch.onnx.export(
            model,
            torch.zeros((opt_batch, 3, image_size, image_size)),
            onnx_path,
            input_names=["images"],
            output_names=["sensor_logits", "scene_logits"],
            dynamic_axes={
                "images": {0: "batch"},
                "sensor_logits": {0: "batch"},
                "scene_logits": {0: "batch"},
            },
            opset_version=int(export["onnx_opset"]),
        )
        if export["simplify"]:
            from onnxslim import slim

            onnx.save(slim(onnx.load(onnx_path)), onnx_path)
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx_path.read_bytes()):
            details = "; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise RuntimeError("Scene-SensorNet ONNX解析失败：" + details)
        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(float(backend["workspace_gib"]) * 1024 ** 3),
        )
        calibrator = None
        if backend["precision"] in {"fp16", "int8"}:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        if backend["precision"] == "int8":
            calibration = row.get("calibration")
            if not isinstance(calibration, Mapping):
                raise ValueError("INT8场景模型导出缺少代表性数据校准manifest。")
            builder_config.set_flag(trt.BuilderFlag.INT8)
            calibrator = _int8_calibrator(
                trt,
                calibration,
                image_size,
                str(export["device"]),
                "context",
            )
            builder_config.int8_calibrator = calibrator
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "images",
            (min_batch, 3, image_size, image_size),
            (opt_batch, 3, image_size, image_size),
            (max_batch, 3, image_size, image_size),
        )
        builder_config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, builder_config)
        if serialized is None:
            raise RuntimeError("Scene-SensorNet TensorRT engine构建失败。")
        generated = Path(temporary_dir) / "scene_sensor_net.engine"
        generated.write_bytes(bytes(serialized))
        return _replace_export(generated, resolve_path(row["target"]))


def export_or_verify_engines(config: Mapping[str, Any], verify_only: bool = False) -> Dict[str, Any]:
    backend = config["tensorrt_backend"]
    plan = export_plan(config)
    if verify_only and any(not row.get("expected_sha256") for row in plan):
        raise ValueError("只读校验要求设备配置中的所有TensorRT engine均已登记真实SHA256。")
    rows = []
    for row in plan:
        existing = _verified_existing(row)
        if existing is not None:
            rows.append(existing)
            continue
        if verify_only:
            raise FileNotFoundError(resolve_path(row["target"]))
        if backend["validated"] is not False:
            raise RuntimeError("生成新engine前必须使用validated: false的设备专用配置。")
        if not backend["export"]["overwrite"] and resolve_path(row["target"]).exists():
            raise RuntimeError(f"目标engine已存在且禁止覆盖：{row['target']}")
        runtime_row = dict(row)
        if backend["precision"] == "int8":
            runtime_row["calibration"] = _static_calibration(config, runtime_row)
        digest = (
            _export_yolo(runtime_row, backend)
            if runtime_row["kind"] == "yolo"
            else _export_context(runtime_row, backend)
        )
        public_calibration = None
        if "calibration" in runtime_row:
            public_calibration = {
                key: value for key, value in runtime_row["calibration"].items() if key != "images"
            }
        rows.append({
            **row,
            "status": "exported",
            "sha256": digest,
            "calibration": public_calibration,
        })
    return {"engine_count": len(rows), "engines": rows}


def export_incremental_int8_engine(
    config: Mapping[str, Any],
    weights: str | Path,
    generation_id: str,
    image_paths: Sequence[Path],
    class_ids: Sequence[int],
    forbidden_stems: Sequence[str],
) -> Dict[str, Any]:
    backend = dict(config["tensorrt_backend"])
    if backend["precision"] != "int8":
        raise ValueError("当前设备配置不是INT8 TensorRT模式。")
    calibration_settings = backend["int8_calibration"]
    maximum_batch = int(config["inference"]["batch_size"])
    optimum_batch = min(int(calibration_settings["batch_size"]), maximum_batch)
    target = (
        resolve_path(calibration_settings["cache_root"])
        / "engines"
        / f"{generation_id}.engine"
    )
    if target.exists():
        raise FileExistsError(f"增量INT8 engine已存在：{target}")
    optimum_height, optimum_width = _optimal_detector_shape(
        image_paths, int(config["inference"]["specialist_imgsz"])
    )
    row: Dict[str, Any] = {
        "kind": "yolo",
        "source": rel_path(resolve_path(weights)),
        "target": rel_path(target),
        "imgsz": int(config["inference"]["specialist_imgsz"]),
        "opt_height": optimum_height,
        "opt_width": optimum_width,
        "min_batch_size": 1,
        "opt_batch_size": optimum_batch,
        "batch_size": maximum_batch,
        "expected_sha256": None,
    }
    row["calibration"] = prepare_calibration_manifest(
        config,
        row["source"],
        image_paths,
        class_ids,
        f"incremental_generation:{generation_id}",
        forbidden_stems,
        preprocessing={
            "mode": "detector_letterbox",
            "imgsz": int(row["imgsz"]),
            "opt_height": optimum_height,
            "opt_width": optimum_width,
            "dynamic_spatial": bool(backend["dynamic"]),
            "minimum_spatial_size": int(backend["minimum_spatial_size"]),
            "tensorrt_version": str(backend["expected_version"]),
            "compute_capability": str(backend["expected_compute_capability"]),
            "mixed_precision": dict(backend["mixed_precision"]),
        },
    )
    digest = _export_yolo(row, backend)
    precision_report = target.with_suffix(target.suffix + ".precision.json")
    return {
        "backend": "tensorrt_engine",
        "precision": "int8",
        "path": rel_path(target),
        "sha256": digest,
        "imgsz": row["imgsz"],
        "opt_height": row["opt_height"],
        "opt_width": row["opt_width"],
        "min_batch_size": row["min_batch_size"],
        "opt_batch_size": row["opt_batch_size"],
        "batch_size": row["batch_size"],
        "calibration_manifest": row["calibration"]["manifest"],
        "calibration_manifest_sha256": row["calibration"]["manifest_sha256"],
        "calibration_fingerprint": row["calibration"]["fingerprint"],
        "calibration_image_count": row["calibration"]["image_count"],
        "precision_report": rel_path(precision_report),
        "precision_report_sha256": sha256_file(precision_report),
        "lock_content_read": False,
    }


def write_export_hashes(config_path: str | Path, result: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify exported files and atomically persist their digests to the device profile."""
    path = resolve_path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"设备配置必须是YAML映射：{path}")
    backend = payload.get("tensorrt_backend")
    if not isinstance(backend, dict) or backend.get("validated") is not False:
        raise ValueError("仅允许向validated: false的设备专用配置回填engine哈希。")

    exported = [row for row in result.get("engines", []) if row.get("status") == "exported"]
    if not exported:
        return {
            "updated": False,
            "config": rel_path(path),
            "config_sha256": config_sha256(payload),
        }

    for row in exported:
        target = resolve_path(str(row["target"]))
        if not target.is_file():
            raise FileNotFoundError(f"待登记的engine不存在：{target}")
        digest = sha256_file(target)
        if digest != row.get("sha256"):
            raise RuntimeError(f"engine落盘哈希与导出结果不一致：{rel_path(target)}")
        if row["kind"] == "context":
            backend["context_engine"]["sha256"] = digest
        else:
            source = str(row["source"])
            if source not in backend["engines"]:
                raise KeyError(f"设备配置中不存在导出源：{source}")
            backend["engines"][source]["sha256"] = digest

    written = write_config(path, payload, "tensorrt-export:record-engine-hashes")
    stored = yaml.safe_load(written.read_text(encoding="utf-8"))
    return {
        "updated": True,
        "config": rel_path(written),
        "config_sha256": config_sha256(stored),
        "engine_hashes_recorded": len(exported),
    }
