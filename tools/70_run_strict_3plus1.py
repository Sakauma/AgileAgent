#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.runtime_log import StructuredEventLog, event_log_from_config, mirror_state_event
from fair_agent.modules.strict_incremental import (
    GLOBAL_CLASS_NAMES,
    bootstrap_metrics,
    build_protocol_dataset,
    calibrate_threshold,
    class_aware_nms,
    evaluate_ap50,
    image_class_ids,
    load_yaml,
    materialize_lock_data,
    precision_recall,
    read_split,
    retention_metrics,
    sha256_file,
    source_label,
    subset_rows,
    validate_protocol_spec,
    yolo_ground_truth,
)
from fair_agent.modules.incremental_methods import (
    build_duet_checkpoint,
    configure_duet_specialist,
    configure_yolo_iod_lite_student,
    old_classification_row_drift,
    restore_protected_old_rows,
    shared_parameter_relative_drift,
)


def _audit_event(
    config: Mapping[str, Any],
    protocol_id: str,
    state: str,
    status: str = "completed",
    **details: Any,
) -> None:
    configured = config.get("experiment_audit", {}).get("events")
    if configured:
        path = Path(str(configured)).resolve()
    else:
        run_id = str(config.get("_active_run_id") or "unknown")
        path = resolve_path(config["paths"]["report_root"]) / run_id / f"{protocol_id}.events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": datetime.now().astimezone().isoformat(),
        "monotonic_ns": time.monotonic_ns(),
        "protocol": protocol_id,
        "state": state,
        "status": status,
        **details,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    global_logging = config.get("experiment_audit", {}).get("global_logging")
    if global_logging:
        logger = StructuredEventLog(
            global_logging["root"],
            int(global_logging["max_file_bytes"]),
            int(global_logging["retained_files"]),
        )
    else:
        logger = event_log_from_config()
    mirror_state_event(
        logger,
        state,
        status=status,
        experiment_id=str(
            config.get("experiment_audit", {}).get("experiment_id")
            or config.get("_experiment_id")
            or config.get("experiment", {}).get("id")
            or "strict-class-incremental"
        ),
        run_id=str(
            config.get("experiment_audit", {}).get("run_id")
            or config.get("_active_run_id")
            or "unknown"
        ),
        protocol_id=protocol_id,
        details=details,
    )


def training_preflight(config: Mapping[str, Any], run_id: str) -> Dict[str, Any]:
    """Read-only validation for a training-only server checkout."""
    errors: list[str] = []
    warnings: list[str] = []
    checks: Dict[str, Any] = {}

    model_path = resolve_path(str(config.get("model") or ""))
    checks["initial_model"] = {"path": rel_path(model_path), "exists": model_path.is_file()}
    if not model_path.is_file():
        errors.append(f"初始化权重不存在：{rel_path(model_path)}")
    shared_base_value = config.get("paths", {}).get("shared_base_checkpoint")
    if shared_base_value:
        shared_base = resolve_path(shared_base_value)
        checks["shared_base_checkpoint"] = {
            "path": rel_path(shared_base),
            "exists": shared_base.is_file(),
        }
        if not shared_base.is_file():
            errors.append(f"共享三类基础权重不存在：{rel_path(shared_base)}")

    source_splits = config.get("paths", {}).get("source_splits", {})
    if set(source_splits) != {"train", "val", "lock"}:
        errors.append("paths.source_splits 必须完整声明 train、val、lock")
        split_rows: Dict[str, list[Path]] = {}
    else:
        split_rows = {}
        for split_name, split_path in source_splits.items():
            try:
                rows = read_split(split_path)
                missing = [rel_path(path) for path in rows if not path.is_file()]
                missing_labels = [rel_path(path) for path in rows if path.is_file() and not source_label(path).is_file()]
            except (OSError, ValueError) as exc:
                errors.append(f"{split_name} 划分不可读：{exc}")
                continue
            if not rows:
                errors.append(f"{split_name} 划分为空")
            if missing:
                errors.append(f"{split_name} 缺少 {len(missing)} 张图像")
            if missing_labels:
                errors.append(f"{split_name} 缺少 {len(missing_labels)} 个标签")
            split_rows[split_name] = rows

    split_counts = {name: len(rows) for name, rows in split_rows.items()}
    intersections: Dict[str, list[str]] = {}
    if set(split_rows) == {"train", "val", "lock"}:
        stems = {name: {path.stem for path in rows} for name, rows in split_rows.items()}
        intersections = {
            "train_val": sorted(stems["train"] & stems["val"]),
            "train_lock": sorted(stems["train"] & stems["lock"]),
            "val_lock": sorted(stems["val"] & stems["lock"]),
        }
        if any(intersections.values()):
            errors.append("train、val、lock 存在重复图像 stem")
    checks["splits"] = {"counts": split_counts, "intersections": intersections}

    protocol_checks = []
    for protocol in config.get("protocols", []):
        protocol_id = str(protocol.get("id") or "unknown")
        item: Dict[str, Any] = {"id": protocol_id}
        try:
            validate_protocol_spec(protocol)
            adaptation_mode = str(
                protocol.get(
                    "adaptation_mode",
                    config.get("adaptation", {}).get("mode", "frozen_base_plus_new_specialist"),
                )
            )
            item["adaptation_mode"] = adaptation_mode
            if adaptation_mode in {"duet_yolo11s", "yolo_iod_lite"}:
                if not protocol.get("build_unified_student"):
                    raise ValueError(f"{adaptation_mode} 必须生成四类 student 数据视图")
                if adaptation_mode not in config.get("methods", {}):
                    raise ValueError(f"缺少 methods.{adaptation_mode} 配置")
            new_id = int(protocol["new_global_id"])
            counts = {
                split_name: sum(new_id in image_class_ids(image) for image in rows)
                for split_name, rows in split_rows.items()
            }
            item["incremental_image_counts"] = counts
            expected = protocol.get("expected_incremental_counts", {})
            for split_name in ("train", "val"):
                if split_name in expected and counts.get(split_name) != int(expected[split_name]):
                    errors.append(
                        f"{protocol_id} {split_name} 新增类样本数不符："
                        f"expected={expected[split_name]} actual={counts.get(split_name)}"
                    )
            if "lock_positive" in expected and counts.get("lock") != int(expected["lock_positive"]):
                errors.append(
                    f"{protocol_id} lock 新增类样本数不符："
                    f"expected={expected['lock_positive']} actual={counts.get('lock')}"
                )
            item["valid"] = True
        except (KeyError, OSError, TypeError, ValueError) as exc:
            item["valid"] = False
            errors.append(f"{protocol_id} 协议检查失败：{exc}")
        protocol_checks.append(item)
    if not protocol_checks:
        errors.append("未声明任何增量协议")
    checks["protocols"] = protocol_checks

    ultralytics_ready = importlib.util.find_spec("ultralytics") is not None
    checks["ultralytics"] = ultralytics_ready
    if not ultralytics_ready:
        errors.append("当前 Python 环境缺少 ultralytics")
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
        device_names = [torch.cuda.get_device_name(index) for index in range(device_count)]
        checks["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": device_count,
            "devices": device_names,
        }
        if not cuda_available or device_count < 1:
            errors.append("当前 Python 环境无法使用 NVIDIA GPU")
        requested = {
            int(protocol.get("preferred_device", 0))
            for protocol in config.get("protocols", [])
        }
        unavailable = sorted(index for index in requested if index < 0 or index >= device_count)
        if unavailable:
            errors.append(f"配置引用了不可用 GPU：{unavailable}")
    except (ImportError, RuntimeError) as exc:
        checks["torch"] = {"cuda_available": False, "error": str(exc)}
        errors.append(f"PyTorch/CUDA 检查失败：{exc}")

    conflicts = []
    paths = config.get("paths", {})
    for key in ("dataset_root", "run_root", "report_root"):
        value = paths.get(key)
        if value:
            candidate = resolve_path(value) / run_id
            if candidate.exists():
                conflicts.append(rel_path(candidate))
    freeze_root = paths.get("freeze_root")
    if freeze_root:
        for protocol in config.get("protocols", []):
            candidate = resolve_path(freeze_root) / str(protocol.get("id")) / run_id
            if candidate.exists():
                conflicts.append(rel_path(candidate))
    checks["output_conflicts"] = conflicts
    if conflicts:
        errors.append("run_id 已存在产物，拒绝覆盖：" + ", ".join(conflicts))

    if config.get("runtime", {}).get("parallel") and len(config.get("protocols", [])) == 1:
        warnings.append("当前仅有一个协议，parallel 配置不会产生并行训练")
    return {
        "schema_version": 1,
        "mode": "training_preflight",
        "run_id": run_id,
        "ready": not errors,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }


def resolve_device_assignments(
    config: Mapping[str, Any], device_count: int
) -> tuple[Dict[str, str], int]:
    protocols = list(config.get("protocols", []))
    assignments = {
        str(protocol["id"]): str(protocol.get("preferred_device", "0"))
        for protocol in protocols
    }
    unavailable = sorted(
        {int(device) for device in assignments.values() if int(device) < 0 or int(device) >= device_count}
    )
    if unavailable:
        raise RuntimeError(f"配置引用了不可用 GPU：{unavailable}")
    parallel = bool(config.get("runtime", {}).get("parallel", False))
    max_workers = min(len(protocols), device_count) if parallel else 1
    return assignments, max(1, max_workers)


def protocol_by_id(config: Mapping[str, Any], protocol_id: str) -> Dict[str, Any]:
    protocol = next((item for item in config["protocols"] if item["id"] == protocol_id), None)
    if protocol is None:
        raise ValueError(f"未知严格增量协议：{protocol_id}")
    return dict(protocol)


def best_weight(model: Any, train_result: Any) -> Path:
    trainer = model.trainer
    for candidate in [getattr(trainer, "best", None), getattr(trainer, "last", None)]:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    save_dir = Path(getattr(train_result, "save_dir", trainer.save_dir))
    for name in ("best.pt", "last.pt"):
        candidate = save_dir / "weights" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("训练结束但未找到 best.pt 或 last.pt")


def training_history(model: Any, phase: str, requested_epochs: int) -> Dict[str, Any]:
    trainer = model.trainer
    results_path = Path(trainer.save_dir) / "results.csv"
    rows: list[Dict[str, Any]] = []
    if results_path.exists():
        with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                row: Dict[str, Any] = {}
                for key, value in raw.items():
                    name = str(key).strip()
                    text = str(value).strip()
                    try:
                        row[name] = float(text)
                    except ValueError:
                        row[name] = text
                rows.append(row)
    metric_key = next(
        (key for key in (rows[0] if rows else {}) if "mAP50(B)" in key),
        None,
    )
    best_row = max(rows, key=lambda row: float(row.get(metric_key, float("-inf")))) if metric_key else None
    completed_epochs = len(rows)
    return {
        "phase": phase,
        "results_csv": rel_path(results_path) if results_path.exists() else None,
        "requested_epochs": int(requested_epochs),
        "completed_epochs": completed_epochs,
        "best_epoch": int(float(best_row.get("epoch", 0))) if best_row else None,
        "best_metric": metric_key,
        "best_metric_value": float(best_row[metric_key]) if best_row and metric_key else None,
        "stopped_early": completed_epochs < int(requested_epochs),
        "stop_reason": "early_stopping_or_interruption" if completed_epochs < int(requested_epochs) else "epoch_budget_reached",
        "epochs": rows,
    }


def _trainer_model(trainer: Any) -> Any:
    return trainer.model.module if hasattr(trainer.model, "module") else trainer.model


def configure_expanded_student(
    trainer: Any,
    teacher_weight: Path,
    base_local_to_global: Mapping[int, int],
    new_global_id: int,
    teacher_model: Any | None = None,
) -> None:
    import torch
    from ultralytics import YOLO

    student = _trainer_model(trainer)
    teacher = teacher_model if teacher_model is not None else YOLO(str(teacher_weight)).model
    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    mapping = {int(local): int(global_id) for local, global_id in base_local_to_global.items()}
    with torch.no_grad():
        for key, teacher_value in teacher_state.items():
            if key not in student_state:
                continue
            student_value = student_state[key]
            if student_value.shape == teacher_value.shape:
                student_value.copy_(teacher_value)
            elif (
                ".cv3." in key
                and teacher_value.ndim >= 1
                and teacher_value.shape[0] == len(mapping)
                and student_value.shape[0] == len(GLOBAL_CLASS_NAMES)
                and student_value.shape[1:] == teacher_value.shape[1:]
            ):
                for local_id, global_id in mapping.items():
                    student_value[global_id].copy_(teacher_value[local_id])
    student.load_state_dict(student_state)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    old_rows = []
    head = student.model[-1]
    branches = getattr(head, "cv3", None)
    if branches is None:
        raise RuntimeError("YOLO 检测头不支持分类通道隔离")
    for branch in branches:
        classifier = branch[-1]
        if classifier.out_channels != len(GLOBAL_CLASS_NAMES):
            raise RuntimeError(f"四类学生检测头宽度错误：{classifier.out_channels}")
        classifier.weight.requires_grad_(True)
        classifier.bias.requires_grad_(True)
        old_ids = sorted(set(GLOBAL_CLASS_NAMES) - {new_global_id})
        with torch.no_grad():
            torch.nn.init.normal_(classifier.weight[new_global_id], mean=0.0, std=0.01)
            classifier.bias[new_global_id].fill_(-4.5951198501)
        old_rows.append((classifier, old_ids, classifier.weight[old_ids].detach().clone(), classifier.bias[old_ids].detach().clone()))

        def mask_gradient(gradient: Any, class_id: int = new_global_id) -> Any:
            mask = torch.zeros_like(gradient)
            mask[class_id] = 1
            return gradient * mask

        classifier.weight.register_hook(mask_gradient)
        classifier.bias.register_hook(mask_gradient)
    trainer._clean_incremental_old_rows = old_rows
    restore_expanded_student(trainer)
    ema = getattr(getattr(trainer, "ema", None), "ema", None)
    if ema is None:
        raise RuntimeError("Ultralytics EMA 尚未初始化，无法保证冻结权重一致性")
    ema.load_state_dict(student.state_dict())
    trainer.ema.updates = 0
    for parameter in ema.parameters():
        parameter.requires_grad_(False)


def restore_expanded_student(trainer: Any) -> None:
    import torch

    model = _trainer_model(trainer)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    with torch.no_grad():
        for classifier, old_ids, weight, bias in getattr(trainer, "_clean_incremental_old_rows", []):
            classifier.weight[old_ids].copy_(weight)
            classifier.bias[old_ids].copy_(bias)


def expanded_student_old_drift(
    model_cls: Any,
    teacher_weight: Path,
    student_weight: Path,
    base_local_to_global: Mapping[int, int],
) -> float:
    teacher = model_cls(str(teacher_weight)).model.state_dict()
    student = model_cls(str(student_weight)).model.state_dict()
    mapping = {int(local): int(global_id) for local, global_id in base_local_to_global.items()}
    maximum = 0.0
    for key, before in teacher.items():
        if key not in student:
            continue
        after = student[key]
        if before.shape == after.shape:
            difference = (after - before).abs()
        elif (
            ".cv3." in key
            and before.ndim >= 1
            and before.shape[0] == len(mapping)
            and after.shape[0] == len(GLOBAL_CLASS_NAMES)
            and before.shape[1:] == after.shape[1:]
        ):
            difference = max((after[global_id] - before[local_id]).abs().max() for local_id, global_id in mapping.items())
            maximum = max(maximum, float(difference.item()))
            continue
        else:
            continue
        if difference.numel():
            maximum = max(maximum, float(difference.max().item()))
    return maximum


def train_arguments(
    config: Mapping[str, Any],
    phase: str,
    dataset: Path,
    project: Path,
    name: str,
    device: str,
) -> Dict[str, Any]:
    common = dict(config["common"])
    common.update(dict(config[phase]))
    common.update({
        "data": str(dataset),
        "project": str(project),
        "name": name,
        "device": device,
        "workers": int(config["runtime"].get("workers", 8)),
        "seed": int(config["seed"]),
        "exist_ok": False,
    })
    return common


def method_train_arguments(
    config: Mapping[str, Any],
    method: str,
    stage: str,
    dataset: Path,
    project: Path,
    name: str,
    device: str,
) -> Dict[str, Any]:
    method_settings = config.get("methods", {}).get(method)
    if not isinstance(method_settings, Mapping):
        raise ValueError(f"缺少增量方法配置：methods.{method}")
    stage_settings = method_settings.get(stage)
    if not isinstance(stage_settings, Mapping):
        raise ValueError(f"缺少增量训练阶段配置：methods.{method}.{stage}")
    arguments = dict(config["common"])
    arguments.update(dict(stage_settings))
    arguments.update(
        {
            "data": str(dataset),
            "project": str(project),
            "name": name,
            "device": device,
            "workers": int(config["runtime"].get("workers", 8)),
            "seed": int(config["seed"]),
            "exist_ok": False,
        }
    )
    return arguments


def evaluation_predictor_class() -> type:
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import nms, ops

    class EvaluationDetectionPredictor(DetectionPredictor):
        def postprocess(self, preds: Any, img: Any, orig_imgs: Any, **kwargs: Any) -> list[Any]:
            outputs = nms.non_max_suppression(
                preds,
                self.args.conf,
                kwargs.pop("iou", self.args.iou),
                self.args.classes,
                self.args.agnostic_nms,
                multi_label=True,
                max_det=self.args.max_det,
                nc=0 if self.args.task == "detect" else len(self.model.names),
                end2end=getattr(self.model, "end2end", False),
                rotated=self.args.task == "obb",
            )
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
            return self.construct_results(outputs, img, orig_imgs, **kwargs)

    return EvaluationDetectionPredictor


def predict_records(
    model: Any,
    images: Sequence[Path],
    local_to_global: Mapping[int, int],
    config: Mapping[str, Any],
    device: str,
    source_name: str,
) -> tuple[list[Dict[str, Any]], float]:
    predict = dict(config["predict"])
    predict_args = {
        "imgsz": int(config["common"]["imgsz"]),
        "conf": float(predict["conf"]),
        "iou": float(predict["iou"]),
        "max_det": int(predict["max_det"]),
        "rect": bool(predict.get("rect", True)),
        "device": device,
        "verbose": False,
    }
    evaluation_batch = int(predict.get("evaluation_batch", predict["batch"]))
    if model.__class__.__module__.startswith("ultralytics."):
        predict_args["predictor"] = evaluation_predictor_class()
    expected_paths = {path.resolve() for path in images}
    parents = {path.resolve().parent for path in images}
    directory_source: Path | None = None
    if len(parents) == 1 and next(iter(parents)).is_dir():
        parent = next(iter(parents))
        directory_images = {
            path.resolve()
            for path in parent.iterdir()
            if path.is_file() and path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
        }
        if directory_images == expected_paths:
            directory_source = parent
    if directory_source is not None:
        results = model.predict(
            source=str(directory_source),
            batch=evaluation_batch,
            **predict_args,
        )
    else:
        results = []
        for path in images:
            image_results = model.predict(source=str(path), batch=1, **predict_args)
            if len(image_results) != 1:
                raise RuntimeError(
                    f"单图预测数量不一致：image={path.stem} actual={len(image_results)}"
                )
            results.extend(image_results)
    if len(results) != len(images):
        raise RuntimeError(f"预测数量不一致：expected={len(images)} actual={len(results)}")
    expected_stems = [path.stem for path in images]
    result_stems = [Path(str(getattr(result, "path", ""))).stem for result in results]
    if any(not stem for stem in result_stems):
        raise RuntimeError("Ultralytics 预测结果缺少可识别的 result.path")
    if Counter(result_stems) != Counter(expected_stems):
        missing = sorted((Counter(expected_stems) - Counter(result_stems)).elements())
        unexpected = sorted((Counter(result_stems) - Counter(expected_stems)).elements())
        raise RuntimeError(
            "预测图像标识不一致："
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    rows = []
    inference_ms = 0.0
    mapping = {int(key): int(value) for key, value in local_to_global.items()}
    for result, image_id in zip(results, result_stems):
        inference_ms += float((getattr(result, "speed", None) or {}).get("inference", 0.0))
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        coordinates = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().tolist()
        for xyxy, confidence, local_id_value in zip(coordinates, confidences, class_ids):
            local_id = int(local_id_value)
            if local_id not in mapping:
                raise RuntimeError(f"模型输出未注册的本地类别：{local_id}")
            rows.append({
                "image_id": image_id,
                "class_id": mapping[local_id],
                "confidence": float(confidence),
                "xyxy": [float(value) for value in xyxy],
                "source": source_name,
            })
    return rows, inference_ms


def remap_ground_truth(rows: Sequence[Mapping[str, Any]], mapping: Mapping[int, int]) -> list[Dict[str, Any]]:
    normalized = {int(key): int(value) for key, value in mapping.items()}
    return [{**row, "class_id": normalized[int(row["class_id"])]} for row in rows]


def recording_validator_class() -> type:
    from ultralytics.models.yolo.detect.val import DetectionValidator

    class RecordingDetectionValidator(DetectionValidator):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.exported_predictions: list[Dict[str, Any]] = []
            self.exported_image_ids: list[str] = []

        def update_metrics(self, preds: list[Dict[str, Any]], batch: Dict[str, Any]) -> None:
            for index, pred in enumerate(preds):
                prepared_batch = self._prepare_batch(index, batch)
                prepared_pred = self._prepare_pred(
                    {
                        "bboxes": pred["bboxes"].clone(),
                        "conf": pred["conf"].clone(),
                        "cls": pred["cls"].clone(),
                    }
                )
                scaled = self.scale_preds(prepared_pred, prepared_batch)
                image_id = Path(prepared_batch["im_file"]).stem
                self.exported_image_ids.append(image_id)
                for box, confidence, class_id in zip(
                    scaled["bboxes"].detach().cpu().tolist(),
                    scaled["conf"].detach().cpu().tolist(),
                    scaled["cls"].detach().cpu().tolist(),
                ):
                    self.exported_predictions.append(
                        {
                            "image_id": image_id,
                            "class_id": int(class_id),
                            "confidence": float(confidence),
                            "xyxy": [float(value) for value in box],
                        }
                    )
            super().update_metrics(preds, batch)

        def get_stats(self) -> Dict[str, Any]:
            stats = super().get_stats()
            self.metrics.prediction_records = self.exported_predictions
            self.metrics.evaluated_image_ids = self.exported_image_ids
            return stats

    return RecordingDetectionValidator


def validator_records(
    model: Any,
    dataset: Path,
    split: str,
    images: Sequence[Path],
    local_to_global: Mapping[int, int],
    config: Mapping[str, Any],
    device: str,
    source_name: str,
    project: Path,
    name: str,
    artifact_path: Path,
) -> tuple[list[Dict[str, Any]], float, float]:
    predict = config["predict"]
    metrics = model.val(
        validator=recording_validator_class(),
        data=str(dataset),
        split=split,
        imgsz=int(config["common"]["imgsz"]),
        batch=int(predict.get("evaluation_batch", predict["batch"])),
        conf=float(predict["conf"]),
        iou=float(predict["iou"]),
        max_det=int(predict["max_det"]),
        rect=bool(predict.get("rect", True)),
        workers=int(config["runtime"].get("workers", 8)),
        device=device,
        project=str(project),
        name=name,
        plots=False,
        verbose=False,
        exist_ok=False,
    )
    evaluated_ids = list(getattr(metrics, "evaluated_image_ids", []))
    expected_ids = [path.stem for path in images]
    if Counter(evaluated_ids) != Counter(expected_ids):
        raise RuntimeError(
            f"Validator 图像标识不一致：expected={len(expected_ids)} actual={len(evaluated_ids)}"
        )
    mapping = {int(key): int(value) for key, value in local_to_global.items()}
    rows = []
    for raw in getattr(metrics, "prediction_records", []):
        local_id = int(raw["class_id"])
        if local_id not in mapping:
            raise RuntimeError(f"Validator 输出未注册的本地类别：{local_id}")
        rows.append(
            {
                **raw,
                "class_id": mapping[local_id],
                "source": source_name,
            }
        )
    if artifact_path.exists():
        raise FileExistsError(f"拒绝覆盖逐框评测记录：{artifact_path}")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    speed = getattr(metrics, "speed", {}) or {}
    inference_ms = float(speed.get("inference", 0.0)) * len(images)
    return rows, inference_ms, float(metrics.box.map50)


def image_false_activation_rate(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    images: Sequence[Path],
    new_class_id: int,
    threshold: float,
) -> Dict[str, Any]:
    positive_images = {
        str(row["image_id"]) for row in ground_truth if int(row["class_id"]) == new_class_id
    }
    negative_images = {path.stem for path in images} - positive_images
    activated = {
        str(row["image_id"])
        for row in predictions
        if int(row["class_id"]) == new_class_id and float(row["confidence"]) >= threshold
    }
    false_images = negative_images & activated
    return {
        "negative_image_count": len(negative_images),
        "false_activation_image_count": len(false_images),
        "false_activation_rate": len(false_images) / len(negative_images) if negative_images else 0.0,
        "false_activation_stems": sorted(false_images),
    }


def sensor_metrics(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    lock_images: Sequence[Path],
    new_class_id: int,
) -> Dict[str, Any]:
    result = {}
    for sensor in ("ir", "sar"):
        image_ids = {path.stem for path in lock_images if path.stem.startswith(f"{sensor}_")}
        gt = subset_rows(ground_truth, image_ids)
        target_count = sum(int(row["class_id"]) == new_class_id for row in gt)
        if target_count == 0:
            result[sensor] = {"status": "not_applicable", "target_count": 0, "map50": None}
        else:
            metrics = evaluate_ap50(subset_rows(predictions, image_ids), gt, [new_class_id])
            result[sensor] = {"status": "evaluated", "target_count": target_count, "map50": metrics["map50"]}
    return result


def write_protocol_report(report_dir: Path, result: Mapping[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    fixed_outputs = [
        report_dir / "metrics.json",
        report_dir / "metrics.csv",
        report_dir / "report.md",
    ]
    conflicts = [path.name for path in fixed_outputs if path.exists()]
    if conflicts:
        raise FileExistsError(f"拒绝覆盖固定复核报告：{conflicts}")
    (report_dir / "metrics.json").write_text(
        json.dumps(dict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flat = {
        "protocol": result["protocol"],
        "old_map50_before": result["old_map50_before"],
        "old_map50_after": result["old_map50_after"],
        "new_map50": result["new_map50"],
        "full_map50": result["full_map50"],
        "krr": result["krr"],
        "calibration_threshold": result["calibration"]["selected"]["threshold"],
        "calibration_precision": result["calibration"]["selected"]["precision"],
        "calibration_recall": result["calibration"]["selected"]["recall"],
        "evaluator_error": result["evaluator_error"],
        "accepted": result["accepted"],
    }
    with (report_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    inference_line = (
        f"- 四类学生纯推理总用时：{result['student_inference_ms_total']:.1f} ms"
        if result.get("student_inference_ms_total") is not None
        else f"- 基础/specialist 纯推理总用时：{result['base_inference_ms_total']:.1f} / {result['specialist_inference_ms_total']:.1f} ms"
    )
    lines = [
        f"# {result['protocol']} 严格 3+1 类别增量报告",
        "",
        f"- 基础类别：{', '.join(result['base_classes'])}",
        f"- 新增类别：{result['new_class']}",
        f"- 增量方法：{result['adaptation_mode']}",
        f"- 旧类 mAP50 before/after：{result['old_map50_before']:.5f} / {result['old_map50_after']:.5f}",
        f"- New-mAP50：{result['new_map50']:.5f}",
        f"- 四类组合 mAP50：{result['full_map50']:.5f}",
        f"- KRR：{result['krr']:.5f}",
        f"- 校准阈值：{result['calibration']['selected']['threshold']:.2f}",
        f"- 校准 precision/recall：{result['calibration']['selected']['precision']:.5f} / {result['calibration']['selected']['recall']:.5f}",
        f"- lock precision/recall：{result['lock_deployment_metrics']['precision']:.5f} / {result['lock_deployment_metrics']['recall']:.5f}",
        f"- lock 图像误激活率：{result['false_activation']['false_activation_rate']:.5f}",
        f"- 自定义评测误差：{result['evaluator_error']:.6f}",
        f"- 共享参数相对漂移：{result.get('shared_parameter_relative_drift', 0.0):.6f}",
        f"- bootstrap New-mAP50 95% CI：[{result['bootstrap']['new_map50']['ci95_low']:.5f}, {result['bootstrap']['new_map50']['ci95_high']:.5f}]",
        f"- bootstrap 四类 mAP50 95% CI：[{result['bootstrap']['full_map50']['ci95_low']:.5f}, {result['bootstrap']['full_map50']['ci95_high']:.5f}]",
        inference_line,
        f"- 结论：{'通过' if result['accepted'] else '未通过'}",
        "",
        "## 新增类传感器分组",
        "",
        "| 传感器 | 目标数 | mAP50 |",
        "|---|---:|---:|",
        *[
            f"| {sensor.upper()} | {metrics['target_count']} | "
            f"{metrics['map50']:.5f} |" if metrics.get("map50") is not None
            else f"| {sensor.upper()} | {metrics['target_count']} | N/A |"
            for sensor, metrics in result["sensor_metrics"].items()
        ],
        "",
        "lock-val 仅在两份权重和校准阈值冻结后读取，本报告结果不得用于本 run 调参。",
    ]
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def freeze_profile(
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    run_id: str,
    base_weight: Path,
    specialist_weight: Path,
    calibration: Mapping[str, Any],
    report_dir: Path,
) -> Dict[str, Any]:
    profile_root = resolve_path(config["paths"]["freeze_root"]) / protocol["id"]
    version_dir = profile_root / run_id
    if version_dir.exists():
        raise FileExistsError(f"拒绝覆盖实验冻结目录：{version_dir}")
    version_dir.mkdir(parents=True)
    frozen_base = version_dir / "base.pt"
    frozen_specialist = version_dir / "specialist.pt"
    frozen_calibration = version_dir / "calibration.json"
    shutil.copy2(base_weight, frozen_base)
    shutil.copy2(specialist_weight, frozen_specialist)
    frozen_calibration.write_text(json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(report_dir / "metrics.json", version_dir / "metrics.json")
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    local_to_global = {int(key): int(value) for key, value in protocol["base_local_to_global"].items()}
    profile = {
        "schema_version": 1,
        "profile_id": protocol["id"],
        "run_id": run_id,
        "acceptance": "passed",
        "incremental_mode": "class_incremental",
        "base_weight": rel_path(frozen_base),
        "base_sha256": sha256_file(frozen_base),
        "specialist_weight": rel_path(frozen_specialist),
        "specialist_sha256": sha256_file(frozen_specialist),
        "base_local_to_global": local_to_global,
        "base_local_names": {local: GLOBAL_CLASS_NAMES[global_id] for local, global_id in local_to_global.items()},
        "class_names": GLOBAL_CLASS_NAMES,
        "new_class": protocol["new_class"],
        "new_global_id": int(protocol["new_global_id"]),
        "activation_threshold": float(calibration["selected"]["threshold"]),
        "calibration_source": rel_path(frozen_calibration),
        "metrics_source": rel_path(version_dir / "metrics.json"),
        "evidence_level": "verified",
        "new_map50": float(metrics["new_map50"]),
        "krr": float(metrics["krr"]),
        "lock_precision": metrics["lock_deployment_metrics"]["precision"],
        "lock_recall": metrics["lock_deployment_metrics"]["recall"],
        "lock_false_activation_rate": metrics["false_activation"]["false_activation_rate"],
    }
    (version_dir / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pointer = profile_root / "active.json"
    temporary = profile_root / "active.json.tmp"
    temporary.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(pointer)
    registry_path = profile_root.parent / "registry.json"
    registry = {"schema_version": 1, "verified_profiles": []}
    if registry_path.exists():
        loaded = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            registry = loaded
    entries = {
        str(item["profile_id"]): item
        for item in registry.get("verified_profiles", [])
        if isinstance(item, dict) and item.get("profile_id")
    }
    entries[profile["profile_id"]] = {
        "profile_id": profile["profile_id"],
        "run_id": profile["run_id"],
        "incremental_mode": profile["incremental_mode"],
        "new_class": profile["new_class"],
        "new_global_id": profile["new_global_id"],
        "active_profile": rel_path(pointer),
        "new_map50": profile["new_map50"],
        "krr": profile["krr"],
        "activation_threshold": profile["activation_threshold"],
        "lock_false_activation_rate": profile["lock_false_activation_rate"],
    }
    registry["verified_profiles"] = [entries[key] for key in sorted(entries)]
    registry_tmp = registry_path.with_suffix(".json.tmp")
    registry_tmp.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    registry_tmp.replace(registry_path)
    return profile


def freeze_student_profile(
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    run_id: str,
    base_weight: Path,
    student_weight: Path,
    calibration: Mapping[str, Any],
    report_dir: Path,
) -> Dict[str, Any]:
    profile_root = resolve_path(config["paths"]["freeze_root"]) / protocol["id"]
    version_dir = profile_root / run_id
    if version_dir.exists():
        raise FileExistsError(f"拒绝覆盖单模型增量冻结目录：{version_dir}")
    version_dir.mkdir(parents=True)
    frozen_teacher = version_dir / "teacher_base.pt"
    frozen_student = version_dir / "student_4class.pt"
    frozen_calibration = version_dir / "calibration.json"
    shutil.copy2(base_weight, frozen_teacher)
    shutil.copy2(student_weight, frozen_student)
    shutil.copy2(report_dir / "metrics.json", version_dir / "metrics.json")
    frozen_calibration.write_text(json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    profile = {
        "schema_version": 2,
        "profile_id": protocol["id"],
        "run_id": run_id,
        "acceptance": "passed",
        "incremental_mode": "class_incremental",
        "adaptation_mode": str(metrics.get("adaptation_mode", "expanded_single_student")),
        "new_channel_initialization": (
            "task_model_head_row"
            if metrics.get("adaptation_mode") in {"duet_yolo11s", "yolo_iod_lite"}
            else "deterministic_random_reset"
        ),
        "deployment": "single_detector",
        "teacher_weight": rel_path(frozen_teacher),
        "teacher_sha256": sha256_file(frozen_teacher),
        "model_weight": rel_path(frozen_student),
        "model_sha256": sha256_file(frozen_student),
        "class_names": GLOBAL_CLASS_NAMES,
        "new_class": protocol["new_class"],
        "new_global_id": int(protocol["new_global_id"]),
        "activation_threshold": float(calibration["selected"]["threshold"]),
        "calibration_source": rel_path(frozen_calibration),
        "metrics_source": rel_path(version_dir / "metrics.json"),
        "evidence_level": "verified",
        "new_map50": float(metrics["new_map50"]),
        "krr": float(metrics["krr"]),
        "full_map50": float(metrics["full_map50"]),
        "old_channel_max_abs_drift": float(metrics["old_channel_max_abs_drift"]),
    }
    (version_dir / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pointer = profile_root / "active.json"
    temporary = pointer.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(pointer)
    return profile


def run_protocol(
    config_path: str,
    run_id: str,
    protocol_id: str,
    device: str,
    recheck: bool = False,
) -> Dict[str, Any]:
    from ultralytics import YOLO

    config = load_yaml(config_path)
    config["_active_run_id"] = run_id
    config["_experiment_id"] = str(
        config.get("experiment", {}).get("id") or Path(config_path).stem
    )
    protocol = protocol_by_id(config, protocol_id)
    _audit_event(config, protocol_id, "CREATED", status="running", device=device)
    for key, value in config["runtime"].get("env", {}).items():
        os.environ[str(key)] = str(value)
    dataset_dir = resolve_path(config["paths"]["dataset_root"]) / run_id / protocol_id
    run_dir = resolve_path(config["paths"]["run_root"]) / run_id / protocol_id
    report_dir = resolve_path(config["paths"]["report_root"]) / run_id / protocol_id
    if recheck:
        report_dir = report_dir / "recheck"
    report_dir.mkdir(parents=True, exist_ok=False)
    base_dataset = dataset_dir / "base" / "dataset.yaml"
    incremental_dataset = dataset_dir / "incremental" / "dataset.yaml"
    student_dataset = dataset_dir / "student" / "dataset.yaml"
    adaptation_mode = str(
        protocol.get(
            "adaptation_mode",
            config.get("adaptation", {}).get("mode", "frozen_base_plus_new_specialist"),
        )
    )
    unified_modes = {"expanded_single_student", "duet_yolo11s", "yolo_iod_lite"}
    unified_student = adaptation_mode in unified_modes
    method_audit: Dict[str, Any] = {"method": adaptation_mode}
    base_history: Dict[str, Any] = {"phase": "base", "reused": bool(config.get("paths", {}).get("shared_base_checkpoint"))}
    incremental_histories: list[Dict[str, Any]] = []

    started = time.monotonic()
    if recheck:
        configured_base = config.get("paths", {}).get("shared_base_checkpoint")
        base_weight = resolve_path(configured_base) if configured_base else run_dir / "base" / "weights" / "best.pt"
        if adaptation_mode == "duet_yolo11s":
            incremental_weight = run_dir / "duet_final" / "weights" / "best.pt"
        elif adaptation_mode == "yolo_iod_lite":
            incremental_weight = run_dir / "yolo_iod_student" / "weights" / "best.pt"
        else:
            incremental_weight = run_dir / ("student" if unified_student else "specialist") / "weights" / "best.pt"
        if not base_weight.exists() or not incremental_weight.exists():
            raise FileNotFoundError(f"复核权重不完整：{protocol_id}")
        base_hash_before = sha256_file(base_weight)
    else:
        configured_base = config.get("paths", {}).get("shared_base_checkpoint")
        if configured_base:
            base_weight = resolve_path(configured_base)
            if not base_weight.is_file():
                raise FileNotFoundError(f"共享三类基础权重不存在：{base_weight}")
            method_audit["base_checkpoint_reused"] = True
        else:
            base_model = YOLO(str(config["model"]))
            base_train_result = base_model.train(
                **train_arguments(config, "base_train", base_dataset, run_dir, "base", device)
            )
            base_weight = best_weight(base_model, base_train_result)
            base_history = training_history(
                base_model,
                "base",
                int(config["base_train"]["epochs"]),
            )
            method_audit["base_checkpoint_reused"] = False
        _audit_event(
            config,
            protocol_id,
            "BASE_TRAINED",
            weight=rel_path(base_weight),
            sha256=sha256_file(base_weight),
            training_history=base_history,
        )
        base_hash_before = sha256_file(base_weight)
        _audit_event(config, protocol_id, "BASE_FROZEN", sha256=base_hash_before)
        mapping = {int(key): int(value) for key, value in protocol["base_local_to_global"].items()}
        new_id = int(protocol["new_global_id"])
        if adaptation_mode == "duet_yolo11s":
            method_settings = dict(config["methods"][adaptation_mode])
            current_model = YOLO(str(config["model"]))
            current_model.add_callback(
                "on_pretrain_routine_end",
                lambda trainer: configure_duet_specialist(
                    trainer,
                    base_weight=base_weight,
                    reference_weight=resolve_path(config["model"]),
                    settings=method_settings,
                ),
            )
            current_result = current_model.train(
                **method_train_arguments(
                    config,
                    adaptation_mode,
                    "current_task_train",
                    incremental_dataset,
                    run_dir,
                    "duet_current",
                    device,
                )
            )
            current_weight = best_weight(current_model, current_result)
            incremental_histories.append(training_history(
                current_model,
                "duet_current",
                int(config["methods"][adaptation_mode]["current_task_train"]["epochs"]),
            ))
            final_weight = run_dir / "duet_final" / "weights" / "best.pt"
            method_audit.update(
                build_duet_checkpoint(
                    resolve_path(config["model"]),
                    base_weight,
                    current_weight,
                    final_weight,
                    class_names=GLOBAL_CLASS_NAMES,
                    base_local_to_global=mapping,
                    new_global_id=new_id,
                    alpha_old=float(method_settings["task_arithmetic"]["alpha_old"]),
                    alpha_new=float(method_settings["task_arithmetic"]["alpha_new"]),
                    shared_key_exclude=tuple(method_settings.get("shared_key_exclude", ["model.23"])),
                )
            )
            trainer_audit = getattr(current_model.trainer, "_incremental_method_audit", {})
            method_audit.update(dict(trainer_audit))
            criterion = getattr(_trainer_model(current_model.trainer), "criterion", None)
            method_audit["last_auxiliary_losses"] = dict(
                getattr(criterion, "last_auxiliary", {})
            )
            method_audit["current_task_weight"] = rel_path(current_weight)
            incremental_weight = final_weight
        elif adaptation_mode == "yolo_iod_lite":
            method_settings = dict(config["methods"][adaptation_mode])
            current_model = YOLO(str(config["model"]))
            current_result = current_model.train(
                **method_train_arguments(
                    config,
                    adaptation_mode,
                    "current_teacher_train",
                    incremental_dataset,
                    run_dir,
                    "yolo_iod_current_teacher",
                    device,
                )
            )
            current_weight = best_weight(current_model, current_result)
            incremental_histories.append(training_history(
                current_model,
                "yolo_iod_current_teacher",
                int(config["methods"][adaptation_mode]["current_teacher_train"]["epochs"]),
            ))
            student_model = YOLO(str(config.get("adaptation", {}).get("student_init", config["model"])))
            student_model.add_callback(
                "on_pretrain_routine_end",
                lambda trainer: configure_yolo_iod_lite_student(
                    trainer,
                    base_weight=base_weight,
                    current_teacher_weight=current_weight,
                    reference_weight=resolve_path(config["model"]),
                    base_local_to_global=mapping,
                    new_global_id=new_id,
                    settings=method_settings,
                ),
            )
            student_model.add_callback("on_train_batch_start", restore_protected_old_rows)
            student_model.add_callback("on_train_batch_end", restore_protected_old_rows)
            student_model.add_callback("on_train_epoch_end", restore_protected_old_rows)
            student_result = student_model.train(
                **method_train_arguments(
                    config,
                    adaptation_mode,
                    "student_train",
                    student_dataset,
                    run_dir,
                    "yolo_iod_student",
                    device,
                )
            )
            incremental_weight = best_weight(student_model, student_result)
            incremental_histories.append(training_history(
                student_model,
                "yolo_iod_student",
                int(config["methods"][adaptation_mode]["student_train"]["epochs"]),
            ))
            method_audit.update(
                dict(getattr(student_model.trainer, "_incremental_method_audit", {}))
            )
            criterion = getattr(_trainer_model(student_model.trainer), "criterion", None)
            method_audit["last_auxiliary_losses"] = dict(
                getattr(criterion, "last_auxiliary", {})
            )
            method_audit["current_teacher_weight"] = rel_path(current_weight)
        elif unified_student:
            student_model = YOLO(str(config.get("adaptation", {}).get("student_init", config["model"])))
            student_model.add_callback(
                "on_pretrain_routine_end",
                lambda trainer: configure_expanded_student(trainer, base_weight, mapping, new_id),
            )
            student_model.add_callback("on_train_batch_start", restore_expanded_student)
            student_model.add_callback("on_train_batch_end", restore_expanded_student)
            student_model.add_callback("on_train_epoch_end", restore_expanded_student)
            student_train_result = student_model.train(
                **train_arguments(config, "student_train", student_dataset, run_dir, "student", device)
            )
            incremental_weight = best_weight(student_model, student_train_result)
            incremental_histories.append(training_history(
                student_model,
                "incremental_student",
                int(config["student_train"]["epochs"]),
            ))
        else:
            specialist_model = YOLO(str(base_weight))
            specialist_train_result = specialist_model.train(
                **train_arguments(config, "incremental_train", incremental_dataset, run_dir, "specialist", device)
            )
            incremental_weight = best_weight(specialist_model, specialist_train_result)
            incremental_histories.append(training_history(
                specialist_model,
                "incremental_specialist",
                int(config["incremental_train"]["epochs"]),
            ))
        _audit_event(
            config,
            protocol_id,
            "INCREMENT_TRAINED",
            weight=rel_path(incremental_weight),
            sha256=sha256_file(incremental_weight),
            training_histories=incremental_histories,
        )
        _audit_event(config, protocol_id, "INCREMENT_FROZEN", sha256=sha256_file(incremental_weight))
    base_hash_after = sha256_file(base_weight)
    base_weight_drift = 0.0 if base_hash_before == base_hash_after else 1.0

    new_id = int(protocol["new_global_id"])
    incremental_phase = "student" if unified_student else "incremental"
    incremental_val = read_split(dataset_dir / incremental_phase / "splits" / "val.txt")
    incremental_predictor = YOLO(str(incremental_weight))
    incremental_mapping = {class_id: class_id for class_id in GLOBAL_CLASS_NAMES} if unified_student else {0: new_id}
    dev_predictions, _dev_ms, _dev_reference_map50 = validator_records(
        incremental_predictor,
        student_dataset if unified_student else incremental_dataset,
        "val",
        incremental_val,
        incremental_mapping,
        config,
        device,
        "incremental_student",
        report_dir / "ultralytics",
        "dev_incremental",
        report_dir / "predictions" / "dev_incremental.jsonl",
    )
    dev_ground_truth = yolo_ground_truth(incremental_val)
    if not unified_student:
        dev_ground_truth = remap_ground_truth(dev_ground_truth, {0: new_id})
    calibration_cfg = config["calibration"]
    calibration = calibrate_threshold(
        dev_predictions,
        dev_ground_truth,
        new_id,
        float(calibration_cfg["threshold_min"]),
        float(calibration_cfg["threshold_max"]),
        float(calibration_cfg["threshold_step"]),
        float(calibration_cfg["target_precision"]),
    )
    calibration_path = report_dir / "calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _audit_event(
        config,
        protocol_id,
        "THRESHOLD_CALIBRATED",
        threshold=float(calibration["selected"]["threshold"]),
        precision=float(calibration["selected"]["precision"]),
        source="incremental_dev_only",
        artifact=rel_path(calibration_path),
        artifact_sha256=sha256_file(calibration_path),
    )

    # This is the first point where lock labels are read or transformed.
    manifest_path = dataset_dir / "manifest.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_manifest = existing_manifest if existing_manifest.get("lock_materialized_after_freeze") else materialize_lock_data(
        protocol, config["paths"]["source_splits"]["lock"], dataset_dir
    )
    _audit_event(config, protocol_id, "LOCK_UNSEALED", split_sha256=dataset_manifest["source_split_sha256"]["lock"])
    lock_images = read_split(config["paths"]["source_splits"]["lock"])
    base_test_images = read_split(dataset_dir / "base" / "splits" / "test.txt")
    incremental_test_images = read_split(dataset_dir / incremental_phase / "splits" / "test.txt")
    base_mapping = {int(key): int(value) for key, value in protocol["base_local_to_global"].items()}
    base_predictor = YOLO(str(base_weight))
    base_predictions, base_inference_ms, base_reference_map50 = validator_records(
        base_predictor,
        base_dataset,
        "test",
        base_test_images,
        base_mapping,
        config,
        device,
        "frozen_base_model",
        report_dir / "ultralytics",
        "lock_base",
        report_dir / "predictions" / "lock_base.jsonl",
    )
    incremental_predictions, incremental_inference_ms, incremental_reference_map50 = validator_records(
        incremental_predictor,
        student_dataset if unified_student else incremental_dataset,
        "test",
        incremental_test_images,
        incremental_mapping,
        config,
        device,
        "incremental_student",
        report_dir / "ultralytics",
        "lock_incremental",
        report_dir / "predictions" / "lock_incremental.jsonl",
    )
    ground_truth = yolo_ground_truth(lock_images)
    combined_predictions = (
        incremental_predictions
        if unified_student
        else class_aware_nms(base_predictions + incremental_predictions, float(config["fusion"]["nms_iou"]))
    )
    old_ids = sorted(base_mapping.values())
    retention = retention_metrics(base_predictions, combined_predictions, ground_truth, old_ids)
    old_metrics = retention["before_metrics"]
    new_metrics = evaluate_ap50(combined_predictions, ground_truth, [new_id])
    full_metrics = evaluate_ap50(combined_predictions, ground_truth, GLOBAL_CLASS_NAMES)
    old_before = float(retention["old_map50_before"])
    old_after = float(retention["old_map50_after"])
    old_prediction_equivalent = bool(retention["old_prediction_equivalent"])
    krr = float(retention["krr"])
    reference_map50 = incremental_reference_map50 if unified_student else base_reference_map50
    evaluator_error = abs(reference_map50 - (float(full_metrics["map50"]) if unified_student else old_before))
    old_channel_drift = (
        old_classification_row_drift(base_weight, incremental_weight, base_mapping)
        if unified_student
        else 0.0
    )
    shared_drift = (
        shared_parameter_relative_drift(base_weight, incremental_weight)
        if unified_student
        else 0.0
    )
    selected_threshold = float(calibration["selected"]["threshold"])
    lock_pr = precision_recall(combined_predictions, ground_truth, new_id, selected_threshold)
    false_activation = image_false_activation_rate(
        combined_predictions, ground_truth, lock_images, new_id, selected_threshold
    )
    subgroup = sensor_metrics(combined_predictions, ground_truth, lock_images, new_id)
    bootstrap_cfg = config["bootstrap"]
    bootstrap = bootstrap_metrics(
        combined_predictions,
        ground_truth,
        lock_images,
        new_id,
        int(bootstrap_cfg["iterations"]),
        int(bootstrap_cfg["seed"]),
    )

    acceptance = config["acceptance"]
    gates = {
        "data_compliance": (
            dataset_manifest["old_raw_image_count"] == 0
            and dataset_manifest.get("old_raw_label_count", 0) == 0
            and dataset_manifest.get("old_feature_cache_count", 0) == 0
            and not any(dataset_manifest["intersections"].values())
        ),
        "base_nc": dataset_manifest["base_nc"] == 3,
        "incremental_output_shape": (
            dataset_manifest.get("student_nc") == 4 if unified_student
            else dataset_manifest["specialist_nc"] == 1
        ),
        "new_map50": float(new_metrics["map50"]) >= float(acceptance["min_new_map50"]),
        "base_map50": old_before >= float(acceptance.get("min_base_map50", 0.0)),
        "krr": krr >= float(acceptance["min_krr"]),
        "old_prediction_equivalence": old_prediction_equivalent,
        "calibration_precision": bool(calibration["passed"]) and float(calibration["selected"]["precision"]) >= float(acceptance["min_calibration_precision"]),
        "evaluator_consistency": evaluator_error <= float(acceptance["max_evaluator_error"]),
        "base_weight_unchanged": base_weight_drift <= float(acceptance["max_base_weight_drift"]),
        "old_channel_isolation": old_channel_drift <= float(config.get("adaptation", {}).get("max_old_channel_drift", 1e-6)),
        "lock_after_freeze": bool(dataset_manifest["lock_materialized_after_freeze"]),
    }
    optional_gates = {
        "lock_precision": (
            float(lock_pr["precision"]) >= float(acceptance["min_lock_precision"])
            if "min_lock_precision" in acceptance else True
        ),
        "lock_recall": (
            float(lock_pr["recall"]) >= float(acceptance["min_lock_recall"])
            if "min_lock_recall" in acceptance else True
        ),
        "false_activation_rate": (
            float(false_activation["false_activation_rate"]) <= float(acceptance["max_false_activation_rate"])
            if "max_false_activation_rate" in acceptance else True
        ),
        "full_map50": (
            float(full_metrics["map50"]) >= float(acceptance["min_full_map50"])
            if "min_full_map50" in acceptance else True
        ),
    }
    gates.update(optional_gates)
    accepted = all(gates.values())
    result = {
        "schema_version": 2,
        "run_id": run_id,
        "protocol": protocol_id,
        "device": device,
        "incremental_mode": "class_incremental",
        "learning_data_scope": "incremental_dataset_only",
        "base_classes": list(protocol["base_classes"]),
        "new_class": protocol["new_class"],
        "base_local_to_global": base_mapping,
        "student_local_to_global": ({class_id: class_id for class_id in GLOBAL_CLASS_NAMES} if unified_student else None),
        "specialist_local_to_global": (None if unified_student else {0: new_id}),
        "adaptation_mode": adaptation_mode,
        "base_weight": rel_path(base_weight),
        "base_weight_sha256": base_hash_before,
        "student_weight": rel_path(incremental_weight) if unified_student else None,
        "student_weight_sha256": sha256_file(incremental_weight) if unified_student else None,
        "specialist_weight": None if unified_student else rel_path(incremental_weight),
        "specialist_weight_sha256": None if unified_student else sha256_file(incremental_weight),
        "base_weight_drift": base_weight_drift,
        "old_channel_max_abs_drift": old_channel_drift,
        "shared_parameter_relative_drift": shared_drift,
        "method_audit": method_audit,
        "old_raw_image_count": dataset_manifest["old_raw_image_count"],
        "old_raw_stems": dataset_manifest.get("old_raw_stems", []),
        "old_map50_before": old_before,
        "old_map50_after": old_after,
        "new_map50": float(new_metrics["map50"]),
        "full_map50": float(full_metrics["map50"]),
        "per_class_ap50": full_metrics["per_class_ap50"],
        "krr": krr,
        "old_prediction_equivalent": old_prediction_equivalent,
        "ultralytics_base_map50": reference_map50,
        "evaluator_error": evaluator_error,
        "calibration": calibration,
        "lock_deployment_metrics": lock_pr,
        "false_activation": false_activation,
        "sensor_metrics": subgroup,
        "bootstrap": bootstrap,
        "base_inference_ms_total": base_inference_ms,
        "specialist_inference_ms_total": 0.0 if unified_student else incremental_inference_ms,
        "student_inference_ms_total": incremental_inference_ms if unified_student else None,
        "training_seconds": time.monotonic() - started,
        "evaluation_only_recheck": recheck,
        "gates": gates,
        "accepted": accepted,
        "lock_evaluation_started_after_training_and_calibration": True,
        "reference_unified_four_class_map50": float(config["reference"]["unified_four_class_map50"]),
    }
    write_protocol_report(report_dir, result)
    _audit_event(
        config,
        protocol_id,
        "EVALUATED",
        base_map50=old_before,
        new_map50=float(new_metrics["map50"]),
        krr=krr,
        full_map50=float(full_metrics["map50"]),
    )
    if accepted:
        if unified_student:
            result["profile"] = freeze_student_profile(
                config, protocol, run_id, base_weight, incremental_weight, calibration, report_dir
            )
        else:
            result["profile"] = freeze_profile(
                config, protocol, run_id, base_weight, incremental_weight, calibration, report_dir
            )
        (report_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _audit_event(config, protocol_id, "ACCEPTED", gates=gates)
        _audit_event(config, protocol_id, "REGISTERED", profile=result.get("profile"))
    else:
        _audit_event(config, protocol_id, "REJECTED", gates=gates)
    return result


def write_summary(
    config: Mapping[str, Any],
    run_id: str,
    results: Sequence[Mapping[str, Any]],
    recheck: bool = False,
) -> Path:
    output = resolve_path(config["paths"]["report_root"]) / run_id
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "protocols": list(results),
        "passed_protocols": [row["protocol"] for row in results if row.get("accepted")],
        "failed_protocols": [row["protocol"] for row in results if not row.get("accepted")],
    }
    suffix = "_recheck" if recheck else ""
    path = output / f"summary{suffix}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 严格 3+1 类别增量双折汇总",
        "",
        "| 协议 | 新增类别 | New-mAP50 | KRR | 四类 mAP50 | 结论 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in results:
        if "error" in row:
            lines.append(f"| {row['protocol']} | - | - | - | - | 执行错误：{row['error']} |")
        else:
            lines.append(
                f"| {row['protocol']} | {row['new_class']} | {row['new_map50']:.5f} | "
                f"{row['krr']:.5f} | {row['full_map50']:.5f} | {'通过' if row['accepted'] else '未通过'} |"
            )
    (output / f"summary{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="运行严格 3+1 类别增量双折实验。")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "strict_class_incremental_3plus1.yaml")
    parser.add_argument("--run-id", help="显式指定唯一运行编号，便于多模型共享同一产物目录。")
    parser.add_argument("--check-only", action="store_true", help="只做训练环境与数据只读预检，不创建产物或启动训练。")
    parser.add_argument("--recheck-run", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_yaml(config_path)
    if args.run_id and args.recheck_run:
        raise ValueError("--run-id 与 --recheck-run 不能同时使用")
    if args.check_only and args.recheck_run:
        raise ValueError("--check-only 与 --recheck-run 不能同时使用")
    configured_run_id = config.get("experiment", {}).get("run_id")
    run_id = args.recheck_run or args.run_id or configured_run_id or datetime.now().strftime("strict-%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id):
        raise ValueError("run_id 只能包含字母、数字、点、下划线和连字符")
    if args.check_only:
        result = training_preflight(config, run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ready"] else 1
    dataset_root = resolve_path(config["paths"]["dataset_root"]) / run_id
    source_splits = config["paths"]["source_splits"]
    if not args.recheck_run:
        for protocol in config["protocols"]:
            build_protocol_dataset(protocol, source_splits, dataset_root / protocol["id"], include_lock=False)
    elif not dataset_root.exists():
        raise FileNotFoundError(f"复核数据目录不存在：{dataset_root}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("严格 3+1 实验要求 NVIDIA GPU。")
    device_count = torch.cuda.device_count()
    assignments, max_workers = resolve_device_assignments(config, device_count)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as pool:
        futures = {
            pool.submit(
                run_protocol,
                str(config_path),
                run_id,
                protocol["id"],
                assignments[protocol["id"]],
                bool(args.recheck_run),
            ): protocol["id"]
            for protocol in config["protocols"]
        }
        for future in as_completed(futures):
            protocol_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"protocol": protocol_id, "accepted": False, "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
    results.sort(key=lambda row: row["protocol"])
    summary = write_summary(config, run_id, results, recheck=bool(args.recheck_run))
    print(rel_path(summary))
    return 1 if any("error" in row or not row.get("accepted") for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
