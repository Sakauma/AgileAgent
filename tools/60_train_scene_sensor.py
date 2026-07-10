#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.hashes import sha256_file
from fair_agent.models.context import SCENE_NAMES, SENSOR_NAMES, SceneSensorNet, evaluation_transform


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def parse_targets(path: Path) -> tuple[int, int]:
    parts = path.stem.split("_")
    if len(parts) != 5:
        raise ValueError(f"无法从文件名解析上下文标签：{path.name}")
    return SENSOR_NAMES.index(parts[0]), SCENE_NAMES.index(parts[3])


def read_split(path: Path) -> list[Path]:
    images = [resolve(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [str(image) for image in images if not image.exists()]
    if missing:
        raise FileNotFoundError(f"划分中存在缺失图像：{missing[:3]}")
    return images


class ContextDataset(Dataset):
    def __init__(self, images: Sequence[Path], transform: transforms.Compose) -> None:
        self.images = list(images)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        path = self.images[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        sensor, scene = parse_targets(path)
        return tensor, sensor, scene


def training_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.85, 1.15), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.18, contrast=0.18)], p=0.6),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        ]
    )


def class_weights(images: Iterable[Path], target_index: int, class_count: int, device: torch.device) -> torch.Tensor:
    counts = Counter(parse_targets(path)[target_index] for path in images)
    total = sum(counts.values())
    return torch.tensor([total / (class_count * counts[index]) for index in range(class_count)], dtype=torch.float32, device=device)


@torch.inference_mode()
def evaluate(model: SceneSensorNet, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    sensor_correct = scene_correct = joint_correct = total = 0
    sensor_confusion = [[0 for _ in SENSOR_NAMES] for _ in SENSOR_NAMES]
    scene_confusion = [[0 for _ in SCENE_NAMES] for _ in SCENE_NAMES]
    for images, sensor_target, scene_target in loader:
        images = images.to(device, non_blocking=True)
        sensor_target = sensor_target.to(device, non_blocking=True)
        scene_target = scene_target.to(device, non_blocking=True)
        sensor_logits, scene_logits = model(images)
        sensor_pred = sensor_logits.argmax(dim=1)
        scene_pred = scene_logits.argmax(dim=1)
        sensor_correct += int((sensor_pred == sensor_target).sum())
        scene_correct += int((scene_pred == scene_target).sum())
        joint_correct += int(((sensor_pred == sensor_target) & (scene_pred == scene_target)).sum())
        total += int(images.shape[0])
        for truth, prediction in zip(sensor_target.tolist(), sensor_pred.tolist()):
            sensor_confusion[truth][prediction] += 1
        for truth, prediction in zip(scene_target.tolist(), scene_pred.tolist()):
            scene_confusion[truth][prediction] += 1
    return {
        "image_count": total,
        "sensor_accuracy": sensor_correct / total,
        "scene_accuracy": scene_correct / total,
        "joint_accuracy": joint_correct / total,
        "sensor_confusion": sensor_confusion,
        "scene_confusion": scene_confusion,
    }


def markdown_report(metrics: Dict[str, Any]) -> str:
    lines = [
        "# Scene-SensorNet 训练报告",
        "",
        f"- 生成时间：`{metrics['created_at']}`",
        f"- 参数量：`{metrics['parameter_count']}`",
        f"- 权重 SHA256：`{metrics['weights_sha256']}`",
        f"- 验收：`{metrics['acceptance']['passed']}`",
        "",
        "| split | images | sensor accuracy | scene accuracy | joint accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ["dev", "lock"]:
        row = metrics[split]
        lines.append(f"| {split} | {row['image_count']} | {row['sensor_accuracy']:.4f} | {row['scene_accuracy']:.4f} | {row['joint_accuracy']:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="训练传感器与场景多任务认知模型。")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "scene_sensor_model.yaml")
    parser.add_argument("--force", action="store_true", help="允许覆盖已有认知模型产物。")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device_index = str(config["train"]["device"])
    if not torch.cuda.is_available() or int(device_index) >= torch.cuda.device_count():
        raise SystemExit("Scene-SensorNet 训练要求可用的 NVIDIA GPU。")
    device = torch.device(f"cuda:{device_index}")

    output_weights = resolve(config["output"]["weights"])
    output_metrics = resolve(config["output"]["metrics"])
    output_report = resolve(config["output"]["report"])
    if not args.force and (output_weights.exists() or output_metrics.exists()):
        raise SystemExit("认知模型产物已存在；如需覆盖请显式传入 --force。")

    train_images = read_split(resolve(config["data"]["train"]))
    dev_images = read_split(resolve(config["data"]["dev"]))
    lock_images = read_split(resolve(config["data"]["lock"]))
    image_size = int(config["data"]["image_size"])
    batch_size = int(config["train"]["batch"])
    workers = int(config["data"]["num_workers"])
    generator = torch.Generator().manual_seed(seed)

    loaders = {
        "train": DataLoader(ContextDataset(train_images, training_transform(image_size)), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, generator=generator, persistent_workers=workers > 0),
        "dev": DataLoader(ContextDataset(dev_images, evaluation_transform(image_size)), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=workers > 0),
        "lock": DataLoader(ContextDataset(lock_images, evaluation_transform(image_size)), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=workers > 0),
    }

    model_cfg = config["model"]
    model = SceneSensorNet(channels=model_cfg["channels"], dropout=float(model_cfg["dropout"])).to(device)
    train_cfg = config["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]), eta_min=float(train_cfg["min_learning_rate"]))
    sensor_loss = nn.CrossEntropyLoss(weight=class_weights(train_images, 0, len(SENSOR_NAMES), device), label_smoothing=float(train_cfg["label_smoothing"]))
    scene_loss = nn.CrossEntropyLoss(weight=class_weights(train_images, 1, len(SCENE_NAMES), device), label_smoothing=float(train_cfg["label_smoothing"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg["amp"]))

    best_score = -1.0
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience = 0
    history = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for images, sensor_target, scene_target in loaders["train"]:
            images = images.to(device, non_blocking=True)
            sensor_target = sensor_target.to(device, non_blocking=True)
            scene_target = scene_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(train_cfg["amp"])):
                sensor_logits, scene_logits = model(images)
                loss = float(train_cfg["sensor_loss_weight"]) * sensor_loss(sensor_logits, sensor_target) + float(train_cfg["scene_loss_weight"]) * scene_loss(scene_logits, scene_target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss) * images.shape[0]
            total += int(images.shape[0])
        scheduler.step()
        dev = evaluate(model, loaders["dev"], device)
        score = 0.35 * dev["sensor_accuracy"] + 0.65 * dev["scene_accuracy"]
        history.append({"epoch": epoch, "train_loss": running_loss / total, "learning_rate": scheduler.get_last_lr()[0], **{f"dev_{key}": value for key, value in dev.items() if key.endswith("accuracy")}})
        print(json.dumps(history[-1], ensure_ascii=False))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
        if patience >= int(train_cfg["patience"]):
            break

    if best_state is None:
        raise RuntimeError("训练未生成有效权重。")
    model.load_state_dict(best_state)
    model.to(device)
    dev_metrics = evaluate(model, loaders["dev"], device)
    lock_metrics = evaluate(model, loaders["lock"], device)

    output_weights.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "model_id": "scene_sensor_net_v1",
        "architecture": {"channels": list(model_cfg["channels"]), "dropout": float(model_cfg["dropout"])},
        "preprocessing": {"image_size": image_size, "mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
        "sensor_names": SENSOR_NAMES,
        "scene_names": SCENE_NAMES,
        "state_dict": best_state,
    }
    torch.save(checkpoint, output_weights)

    acceptance_cfg = config["acceptance"]
    checks = {
        "lock_sensor_accuracy": lock_metrics["sensor_accuracy"] >= float(acceptance_cfg["min_lock_sensor_accuracy"]),
        "lock_scene_accuracy": lock_metrics["scene_accuracy"] >= float(acceptance_cfg["min_lock_scene_accuracy"]),
        "lock_joint_accuracy": lock_metrics["joint_accuracy"] >= float(acceptance_cfg["min_lock_joint_accuracy"]),
    }
    metrics = {
        "schema_version": 1,
        "model_id": "scene_sensor_net_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "weights": output_weights.relative_to(ROOT).as_posix(),
        "weights_sha256": sha256_file(output_weights),
        "train_image_count": len(train_images),
        "dev": dev_metrics,
        "lock": lock_metrics,
        "acceptance": {"thresholds": acceptance_cfg, "checks": checks, "passed": all(checks.values())},
        "history": history,
    }
    output_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(markdown_report(metrics), encoding="utf-8")
    print(json.dumps({"weights": str(output_weights), "metrics": str(output_metrics), "acceptance": metrics["acceptance"]}, ensure_ascii=False, indent=2))
    return 0 if metrics["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
