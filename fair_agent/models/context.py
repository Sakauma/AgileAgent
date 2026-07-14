from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from PIL import Image
from torch import nn

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.hashes import sha256_file


SENSOR_NAMES = ["ir", "sar"]
SCENE_NAMES = ["air", "forest", "sea", "urban"]


def validate_context_input_shape(
    shape: Sequence[int], image_size: int, max_batch_size: int
) -> int:
    if len(shape) != 4:
        raise RuntimeError("Scene-SensorNet TensorRT输入必须是NCHW四维张量。")
    batch_size = int(shape[0])
    if not 1 <= batch_size <= int(max_batch_size):
        raise RuntimeError(
            f"Scene-SensorNet TensorRT batch范围为1-{max_batch_size}，收到{batch_size}"
        )
    if tuple(int(value) for value in shape[1:]) != (3, int(image_size), int(image_size)):
        raise RuntimeError(f"Scene-SensorNet TensorRT输入尺寸必须为{image_size}")
    return batch_size


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SceneSensorNet(nn.Module):
    def __init__(self, channels: Sequence[int] = (24, 48, 96, 160), dropout: float = 0.2) -> None:
        super().__init__()
        blocks = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(ConvBlock(in_channels, int(out_channels)))
            in_channels = int(out_channels)
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.sensor_head = nn.Linear(in_channels, len(SENSOR_NAMES))
        self.scene_head = nn.Linear(in_channels, len(SCENE_NAMES))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.pool(self.features(images)).flatten(1)
        features = self.dropout(features)
        return self.sensor_head(features), self.scene_head(features)


class TensorRTSceneSensorNet:
    def __init__(self, engine_config: Dict[str, Any], device: str | torch.device) -> None:
        import tensorrt as trt

        self.device = require_cuda_device(device)
        engine_path = resolve_path(engine_config["path"])
        if not engine_path.is_file() or sha256_file(engine_path) != str(engine_config["sha256"]):
            raise RuntimeError(f"Scene-SensorNet TensorRT engine缺失或哈希错误：{rel_path(engine_path)}")
        self.image_size = int(engine_config["imgsz"])
        self.max_batch_size = int(engine_config["batch_size"])
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("Scene-SensorNet TensorRT engine反序列化失败。")
        self.context = self.engine.create_execution_context()
        self.tensor_names = {
            self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)
        }
        required = {"images", "sensor_logits", "scene_logits"}
        if not required.issubset(self.tensor_names):
            raise RuntimeError("Scene-SensorNet TensorRT engine输入输出契约不匹配。")

    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = validate_context_input_shape(
            images.shape, self.image_size, self.max_batch_size
        )
        images = images.contiguous()
        sensor_logits = torch.empty((batch_size, len(SENSOR_NAMES)), device=self.device)
        scene_logits = torch.empty((batch_size, len(SCENE_NAMES)), device=self.device)
        if not self.context.set_input_shape("images", tuple(images.shape)):
            raise RuntimeError("Scene-SensorNet TensorRT动态输入设置失败。")
        self.context.set_tensor_address("images", images.data_ptr())
        self.context.set_tensor_address("sensor_logits", sensor_logits.data_ptr())
        self.context.set_tensor_address("scene_logits", scene_logits.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("Scene-SensorNet TensorRT执行失败。")
        return sensor_logits, scene_logits


def evaluation_transform(image_size: int) -> Any:
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("Scene-SensorNet 缺少 torchvision，请先完成智能体环境配置。") from exc
    resize_size = int(round(image_size * 1.1))
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        ]
    )


def require_cuda_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("Scene-SensorNet 仅支持 NVIDIA GPU，不提供 CPU 执行路径。")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无法使用 CUDA。")
    if resolved.index is not None and resolved.index >= torch.cuda.device_count():
        raise RuntimeError(f"GPU 编号不可用：{resolved.index}")
    return resolved


def load_context_model(path: str | Path, device: str | torch.device) -> tuple[SceneSensorNet, Dict[str, Any]]:
    resolved_device = require_cuda_device(device)
    checkpoint = torch.load(Path(path), map_location=resolved_device, weights_only=True)
    model = SceneSensorNet(
        channels=checkpoint["architecture"]["channels"],
        dropout=float(checkpoint["architecture"]["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(resolved_device).eval()
    return model, checkpoint


def load_tensorrt_context_model(
    path: str | Path,
    engine_config: Dict[str, Any],
    device: str | torch.device,
) -> tuple[TensorRTSceneSensorNet, Dict[str, Any]]:
    resolved_device = require_cuda_device(device)
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = TensorRTSceneSensorNet(engine_config, resolved_device)
    if int(checkpoint["preprocessing"]["image_size"]) != model.image_size:
        raise RuntimeError("Scene-SensorNet checkpoint与TensorRT engine输入尺寸不一致。")
    return model, checkpoint


@torch.inference_mode()
def predict_context(
    model: SceneSensorNet,
    checkpoint: Dict[str, Any],
    image: Image.Image,
    device: str | torch.device,
    stream: torch.cuda.Stream | None = None,
) -> Dict[str, Any]:
    resolved_device = require_cuda_device(device)
    image_size = int(checkpoint["preprocessing"]["image_size"])
    tensor = evaluation_transform(image_size)(image.convert("RGB")).unsqueeze(0)
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        tensor = tensor.to(resolved_device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        sensor_logits, scene_logits = model(tensor)
        end_event.record()
    end_event.synchronize()
    sensor_prob = sensor_logits.softmax(dim=1)[0]
    scene_prob = scene_logits.softmax(dim=1)[0]
    sensor_id = int(sensor_prob.argmax())
    scene_id = int(scene_prob.argmax())
    return {
        "sensor": SENSOR_NAMES[sensor_id],
        "sensor_confidence": float(sensor_prob[sensor_id]),
        "sensor_probabilities": {
            name: float(sensor_prob[index]) for index, name in enumerate(SENSOR_NAMES)
        },
        "scene": SCENE_NAMES[scene_id],
        "scene_confidence": float(scene_prob[scene_id]),
        "scene_probabilities": {
            name: float(scene_prob[index]) for index, name in enumerate(SCENE_NAMES)
        },
        "_inference_ms": float(start_event.elapsed_time(end_event)),
    }


@torch.inference_mode()
def predict_context_batch(
    model: SceneSensorNet,
    checkpoint: Dict[str, Any],
    images: Sequence[Image.Image],
    device: str | torch.device,
    stream: torch.cuda.Stream | None = None,
) -> list[Dict[str, Any]]:
    resolved_device = require_cuda_device(device)
    if not images:
        return []
    image_size = int(checkpoint["preprocessing"]["image_size"])
    transform = evaluation_transform(image_size)
    tensor = torch.stack([transform(image.convert("RGB")) for image in images])
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        tensor = tensor.to(resolved_device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        sensor_logits, scene_logits = model(tensor)
        end_event.record()
    end_event.synchronize()
    inference_per_image = float(start_event.elapsed_time(end_event)) / len(images)
    sensor_probs = sensor_logits.softmax(dim=1)
    scene_probs = scene_logits.softmax(dim=1)
    results = []
    for sensor_prob, scene_prob in zip(sensor_probs, scene_probs):
        sensor_id = int(sensor_prob.argmax())
        scene_id = int(scene_prob.argmax())
        results.append({
            "sensor": SENSOR_NAMES[sensor_id],
            "sensor_confidence": float(sensor_prob[sensor_id]),
            "sensor_probabilities": {
                name: float(sensor_prob[index]) for index, name in enumerate(SENSOR_NAMES)
            },
            "scene": SCENE_NAMES[scene_id],
            "scene_confidence": float(scene_prob[scene_id]),
            "scene_probabilities": {
                name: float(scene_prob[index]) for index, name in enumerate(SCENE_NAMES)
            },
            "_inference_ms": inference_per_image,
        })
    return results


@torch.inference_mode()
def evaluate_context_paths(
    model: SceneSensorNet,
    checkpoint: Dict[str, Any],
    paths: Sequence[Path],
    device: str | torch.device,
    batch_size: int = 64,
) -> Dict[str, float | int]:
    resolved_device = require_cuda_device(device)
    if not paths:
        raise ValueError("上下文模型评估集不能为空。")
    transform = evaluation_transform(int(checkpoint["preprocessing"]["image_size"]))
    sensor_correct = scene_correct = joint_correct = total = 0
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        tensors = []
        sensor_targets = []
        scene_targets = []
        for path in batch_paths:
            parts = path.stem.split("_")
            if len(parts) != 5:
                raise ValueError(f"无法从文件名解析上下文标签：{path.name}")
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
            sensor_targets.append(SENSOR_NAMES.index(parts[0]))
            scene_targets.append(SCENE_NAMES.index(parts[3]))
        images = torch.stack(tensors).to(resolved_device)
        sensor_target = torch.tensor(sensor_targets, device=resolved_device)
        scene_target = torch.tensor(scene_targets, device=resolved_device)
        sensor_logits, scene_logits = model(images)
        sensor_pred = sensor_logits.argmax(dim=1)
        scene_pred = scene_logits.argmax(dim=1)
        sensor_correct += int((sensor_pred == sensor_target).sum())
        scene_correct += int((scene_pred == scene_target).sum())
        joint_correct += int(((sensor_pred == sensor_target) & (scene_pred == scene_target)).sum())
        total += len(batch_paths)
    return {
        "image_count": total,
        "sensor_accuracy": sensor_correct / total,
        "scene_accuracy": scene_correct / total,
        "joint_accuracy": joint_correct / total,
    }
