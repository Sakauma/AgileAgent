from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from fair_agent.core.config import config_sha256, rel_path, resolve_path, write_config
from fair_agent.core.hashes import sha256_file


def export_plan(config: Mapping[str, Any]) -> list[Dict[str, Any]]:
    backend = config["tensorrt_backend"]
    rows = [
        {
            "kind": "yolo",
            "source": source,
            "target": str(entry["path"]),
            "imgsz": int(entry["imgsz"]),
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


def _export_yolo(row: Mapping[str, Any], backend: Mapping[str, Any]) -> str:
    from ultralytics import YOLO

    source = resolve_path(row["source"])
    if not source.is_file():
        raise FileNotFoundError(source)
    before = sha256_file(source)
    export = backend["export"]
    result = YOLO(str(source)).export(
        format="engine",
        imgsz=int(row["imgsz"]),
        batch=int(row["batch_size"]),
        half=backend["precision"] == "fp16",
        dynamic=bool(backend["dynamic"]),
        workspace=float(backend["workspace_gib"]),
        simplify=bool(export["simplify"]),
        opset=int(export["onnx_opset"]),
        device=str(export["device"]),
    )
    generated = Path(str(result))
    if not generated.is_file():
        raise RuntimeError(f"Ultralytics未生成TensorRT engine：{generated}")
    if sha256_file(source) != before:
        raise RuntimeError(f"导出过程修改了冻结权重：{rel_path(source)}")
    return _replace_export(generated, resolve_path(row["target"]))


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
    min_batch = int(export["context_min_batch"])
    opt_batch = int(export["context_opt_batch"])
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
        if backend["precision"] == "fp16":
            builder_config.set_flag(trt.BuilderFlag.FP16)
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
        digest = _export_yolo(row, backend) if row["kind"] == "yolo" else _export_context(row, backend)
        rows.append({**row, "status": "exported", "sha256": digest})
    return {"engine_count": len(rows), "engines": rows}


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
