from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from PIL import Image
from torch import nn


SENSOR_NAMES = ["ir", "sar"]
SCENE_NAMES = ["air", "forest", "sea", "urban"]


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


@torch.inference_mode()
def predict_context(model: SceneSensorNet, checkpoint: Dict[str, Any], image: Image.Image, device: str | torch.device) -> Dict[str, Any]:
    resolved_device = require_cuda_device(device)
    image_size = int(checkpoint["preprocessing"]["image_size"])
    tensor = evaluation_transform(image_size)(image.convert("RGB")).unsqueeze(0).to(resolved_device)
    sensor_logits, scene_logits = model(tensor)
    sensor_prob = sensor_logits.softmax(dim=1)[0]
    scene_prob = scene_logits.softmax(dim=1)[0]
    sensor_id = int(sensor_prob.argmax())
    scene_id = int(scene_prob.argmax())
    return {
        "sensor": SENSOR_NAMES[sensor_id],
        "sensor_confidence": float(sensor_prob[sensor_id]),
        "scene": SCENE_NAMES[scene_id],
        "scene_confidence": float(scene_prob[scene_id]),
    }


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
