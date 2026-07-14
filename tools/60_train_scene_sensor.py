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
    active_count = sum(counts[index] > 0 for index in range(class_count))
    if total == 0 or active_count == 0:
        raise ValueError("上下文训练集为空。")
    return torch.tensor(
        [total / (active_count * counts[index]) if counts[index] else 0.0 for index in range(class_count)],
        dtype=torch.float32,
        device=device,
    )


def target_distribution(images: Iterable[Path], target_index: int, names: Sequence[str]) -> Dict[str, int]:
    counts = Counter(parse_targets(path)[target_index] for path in images)
    return {name: int(counts[index]) for index, name in enumerate(names)}


def register_row_mask(parameter: torch.Tensor, allowed_rows: Iterable[int]) -> Any:
    rows = sorted(set(int(index) for index in allowed_rows))

    def mask_gradient(gradient: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(gradient)
        mask[rows] = 1
        return gradient * mask

    return parameter.register_hook(mask_gradient)


def restore_linear_rows(
    layer: nn.Linear,
    row_ids: Sequence[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    if not row_ids:
        return
    with torch.no_grad():
        layer.weight[row_ids].copy_(weight)
        layer.bias[row_ids].copy_(bias)


def linear_row_drift(
    layer: nn.Linear,
    row_ids: Sequence[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> float:
    if not row_ids:
        return 0.0
    weight_delta = (layer.weight[row_ids].detach().cpu() - weight.cpu()).abs().max()
    bias_delta = (layer.bias[row_ids].detach().cpu() - bias.cpu()).abs().max()
    return max(float(weight_delta), float(bias_delta))


def context_preflight(config: Dict[str, Any]) -> Dict[str, Any]:
    errors: list[str] = []
    checks: Dict[str, Any] = {}
    try:
        base_train = read_split(resolve(config["data"]["train"]))
        base_dev = read_split(resolve(config["data"]["dev"]))
        incremental_train = read_split(resolve(config["data"]["incremental_train"]))
        incremental_dev = read_split(resolve(config["data"]["incremental_dev"]))
    except (KeyError, OSError, ValueError) as exc:
        return {"ready": False, "checks": checks, "errors": [f"上下文划分不可读：{exc}"]}
    base_distribution = target_distribution(base_train, 1, SCENE_NAMES)
    incremental_distribution = target_distribution(incremental_train, 1, SCENE_NAMES)
    base_scenes = {name for name, count in base_distribution.items() if count > 0}
    incremental_scenes = {name for name, count in incremental_distribution.items() if count > 0}
    overlap = sorted({path.stem for path in base_train + base_dev} & {path.stem for path in incremental_train + incremental_dev})
    if overlap:
        errors.append(f"基础与增量上下文数据存在重复 stem：{overlap[:5]}")
    if not incremental_scenes:
        errors.append("增量上下文训练集没有场景样本")
    if base_scenes & incremental_scenes:
        errors.append("增量上下文数据混入基础场景")
    if base_scenes | incremental_scenes != set(SCENE_NAMES):
        errors.append("基础与增量上下文数据未覆盖全部场景")
    device_index = int(config["train"]["device"])
    cuda_ready = torch.cuda.is_available() and device_index < torch.cuda.device_count()
    if not cuda_ready:
        errors.append(f"上下文训练 GPU 不可用：{device_index}")
    output_conflicts = [
        str(resolve(path))
        for path in config.get("output", {}).values()
        if resolve(path).exists()
    ]
    if output_conflicts:
        errors.append("上下文输出已存在，拒绝覆盖")
    checks.update(
        {
            "base_train_count": len(base_train),
            "base_dev_count": len(base_dev),
            "incremental_train_count": len(incremental_train),
            "incremental_dev_count": len(incremental_dev),
            "base_scene_distribution": base_distribution,
            "incremental_scene_distribution": incremental_distribution,
            "data_overlap": overlap,
            "device": device_index,
            "cuda_ready": cuda_ready,
            "output_conflicts": output_conflicts,
        }
    )
    return {"ready": not errors, "checks": checks, "errors": errors}


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
        f"- 增量场景：`{metrics.get('incremental_context', {}).get('new_scenes', [])}`",
        f"- 旧场景行最大漂移：`{metrics.get('incremental_context', {}).get('old_scene_row_drift')}`",
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
    parser.add_argument("--check-only", action="store_true", help="只检查增量上下文数据、GPU 与输出冲突。")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.check_only:
        result = context_preflight(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ready"] else 1

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
    incremental_enabled = bool(config["data"].get("incremental_train") and config["data"].get("incremental_dev"))
    incremental_train_images = (
        read_split(resolve(config["data"]["incremental_train"])) if incremental_enabled else []
    )
    incremental_dev_images = (
        read_split(resolve(config["data"]["incremental_dev"])) if incremental_enabled else []
    )
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
    if incremental_enabled:
        base_stems = {path.stem for path in train_images + dev_images}
        incremental_stems = {path.stem for path in incremental_train_images + incremental_dev_images}
        overlap = sorted(base_stems & incremental_stems)
        if overlap:
            raise ValueError(f"基础与增量上下文数据存在重复 stem：{overlap[:5]}")
        loaders["incremental_train"] = DataLoader(
            ContextDataset(incremental_train_images, training_transform(image_size)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(seed + 1),
            persistent_workers=workers > 0,
        )
        loaders["incremental_dev"] = DataLoader(
            ContextDataset(incremental_dev_images, evaluation_transform(image_size)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
        loaders["combined_dev"] = DataLoader(
            ContextDataset(dev_images + incremental_dev_images, evaluation_transform(image_size)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )

    model_cfg = config["model"]
    model = SceneSensorNet(channels=model_cfg["channels"], dropout=float(model_cfg["dropout"])).to(device)
    train_cfg = config["train"]
    base_scene_distribution = target_distribution(train_images, 1, SCENE_NAMES)
    base_scene_ids = [index for index, name in enumerate(SCENE_NAMES) if base_scene_distribution[name] > 0]
    missing_scene_ids = [index for index in range(len(SCENE_NAMES)) if index not in base_scene_ids]
    missing_weight = model.scene_head.weight[missing_scene_ids].detach().clone()
    missing_bias = model.scene_head.bias[missing_scene_ids].detach().clone()
    base_mask_handles = [
        register_row_mask(model.scene_head.weight, base_scene_ids),
        register_row_mask(model.scene_head.bias, base_scene_ids),
    ]
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
            restore_linear_rows(model.scene_head, missing_scene_ids, missing_weight, missing_bias)
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
    for handle in base_mask_handles:
        handle.remove()
    model.load_state_dict(best_state)
    model.to(device)
    base_dev_metrics = evaluate(model, loaders["dev"], device)
    incremental_history = []
    incremental_best_epoch = 0
    incremental_dev_metrics: Dict[str, Any] | None = None
    old_scene_row_drift = 0.0
    new_scene_names: list[str] = []
    if incremental_enabled:
        incremental_cfg = config.get("incremental_train", {})
        incremental_scene_distribution = target_distribution(incremental_train_images, 1, SCENE_NAMES)
        new_scene_ids = [
            index
            for index, name in enumerate(SCENE_NAMES)
            if incremental_scene_distribution[name] > 0 and index not in base_scene_ids
        ]
        repeated_scene_ids = [
            index
            for index, name in enumerate(SCENE_NAMES)
            if incremental_scene_distribution[name] > 0 and index in base_scene_ids
        ]
        if not new_scene_ids:
            raise ValueError("增量上下文数据没有提供基础阶段缺失的新场景。")
        if repeated_scene_ids:
            raise ValueError(
                "增量上下文数据混入基础场景：" + ", ".join(SCENE_NAMES[index] for index in repeated_scene_ids)
            )
        new_scene_names = [SCENE_NAMES[index] for index in new_scene_ids]
        old_scene_ids = [index for index in range(len(SCENE_NAMES)) if index not in new_scene_ids]
        old_scene_weight = model.scene_head.weight[old_scene_ids].detach().clone()
        old_scene_bias = model.scene_head.bias[old_scene_ids].detach().clone()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.scene_head.weight.requires_grad_(True)
        model.scene_head.bias.requires_grad_(True)
        incremental_mask_handles = [
            register_row_mask(model.scene_head.weight, new_scene_ids),
            register_row_mask(model.scene_head.bias, new_scene_ids),
        ]
        incremental_optimizer = torch.optim.AdamW(
            [model.scene_head.weight, model.scene_head.bias],
            lr=float(incremental_cfg["learning_rate"]),
            weight_decay=0.0,
        )
        incremental_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            incremental_optimizer,
            T_max=int(incremental_cfg["epochs"]),
            eta_min=float(incremental_cfg["min_learning_rate"]),
        )
        incremental_loss = nn.CrossEntropyLoss(label_smoothing=0.0)
        incremental_scaler = torch.amp.GradScaler("cuda", enabled=bool(incremental_cfg["amp"]))
        incremental_best_score = -1.0
        incremental_best_state: Dict[str, torch.Tensor] | None = None
        incremental_patience = 0
        for epoch in range(1, int(incremental_cfg["epochs"]) + 1):
            model.eval()
            model.scene_head.train()
            running_loss = 0.0
            total = 0
            for images, _sensor_target, scene_target in loaders["incremental_train"]:
                images = images.to(device, non_blocking=True)
                scene_target = scene_target.to(device, non_blocking=True)
                incremental_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=bool(incremental_cfg["amp"])):
                    _sensor_logits, scene_logits = model(images)
                    loss = incremental_loss(scene_logits, scene_target)
                incremental_scaler.scale(loss).backward()
                incremental_scaler.step(incremental_optimizer)
                incremental_scaler.update()
                restore_linear_rows(
                    model.scene_head,
                    old_scene_ids,
                    old_scene_weight,
                    old_scene_bias,
                )
                running_loss += float(loss) * images.shape[0]
                total += int(images.shape[0])
            incremental_scheduler.step()
            current = evaluate(model, loaders["incremental_dev"], device)
            score = float(current["scene_accuracy"])
            incremental_history.append(
                {
                    "epoch": epoch,
                    "train_loss": running_loss / total,
                    "learning_rate": incremental_scheduler.get_last_lr()[0],
                    **{f"dev_{key}": value for key, value in current.items() if key.endswith("accuracy")},
                }
            )
            print(json.dumps({"phase": "incremental_context", **incremental_history[-1]}, ensure_ascii=False))
            if score > incremental_best_score:
                incremental_best_score = score
                incremental_best_epoch = epoch
                incremental_best_state = copy.deepcopy(
                    {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
                )
                incremental_patience = 0
            else:
                incremental_patience += 1
            if incremental_patience >= int(incremental_cfg["patience"]):
                break
        for handle in incremental_mask_handles:
            handle.remove()
        if incremental_best_state is None:
            raise RuntimeError("增量上下文阶段未生成有效权重。")
        model.load_state_dict(incremental_best_state)
        model.to(device)
        old_scene_row_drift = linear_row_drift(
            model.scene_head,
            old_scene_ids,
            old_scene_weight,
            old_scene_bias,
        )
        incremental_dev_metrics = evaluate(model, loaders["incremental_dev"], device)
        dev_metrics = evaluate(model, loaders["combined_dev"], device)
    else:
        dev_metrics = base_dev_metrics
    lock_metrics = evaluate(model, loaders["lock"], device)

    output_weights.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 2 if incremental_enabled else 1,
        "model_id": "scene_sensor_net_v1",
        "architecture": {"channels": list(model_cfg["channels"]), "dropout": float(model_cfg["dropout"])},
        "preprocessing": {"image_size": image_size, "mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
        "sensor_names": SENSOR_NAMES,
        "scene_names": SCENE_NAMES,
        "adaptation": {
            "mode": "incremental_scene_head_rows" if incremental_enabled else "joint_supervised",
            "new_scenes": new_scene_names,
            "learning_data_scope": "incremental_dataset_only" if incremental_enabled else "full_training_split",
        },
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
    }
    torch.save(checkpoint, output_weights)

    acceptance_cfg = config["acceptance"]
    checks = {
        "lock_sensor_accuracy": lock_metrics["sensor_accuracy"] >= float(acceptance_cfg["min_lock_sensor_accuracy"]),
        "lock_scene_accuracy": lock_metrics["scene_accuracy"] >= float(acceptance_cfg["min_lock_scene_accuracy"]),
        "lock_joint_accuracy": lock_metrics["joint_accuracy"] >= float(acceptance_cfg["min_lock_joint_accuracy"]),
    }
    if incremental_enabled:
        checks.update(
            {
                "incremental_scene_accuracy": bool(incremental_dev_metrics)
                and incremental_dev_metrics["scene_accuracy"]
                >= float(acceptance_cfg["min_incremental_scene_accuracy"]),
                "old_scene_row_isolation": old_scene_row_drift
                <= float(acceptance_cfg["max_old_scene_row_drift"]),
                "incremental_data_disjoint": not ({path.stem for path in train_images + dev_images} & {path.stem for path in incremental_train_images + incremental_dev_images}),
            }
        )
    metrics = {
        "schema_version": 1,
        "model_id": "scene_sensor_net_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "best_epoch": best_epoch,
        "incremental_best_epoch": incremental_best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "weights": output_weights.relative_to(ROOT).as_posix(),
        "weights_sha256": sha256_file(output_weights),
        "train_image_count": len(train_images),
        "train_scene_distribution": base_scene_distribution,
        "base_dev": base_dev_metrics,
        "dev": dev_metrics,
        "lock": lock_metrics,
        "incremental_context": {
            "enabled": incremental_enabled,
            "learning_data_scope": "incremental_dataset_only" if incremental_enabled else None,
            "old_raw_image_count": 0 if incremental_enabled else None,
            "train_image_count": len(incremental_train_images),
            "dev_image_count": len(incremental_dev_images),
            "new_scenes": new_scene_names,
            "old_scene_row_drift": old_scene_row_drift,
            "dev": incremental_dev_metrics,
            "history": incremental_history,
        },
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
