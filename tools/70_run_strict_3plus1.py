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
from fair_agent.modules.detection_fusion import (
    apply_incremental_candidate_gates,
    learn_context_prior,
)
from fair_agent.modules.incremental_rejection import (
    apply_positive_prototype,
    calibrate_positive_prototype,
    fit_positive_prototype,
)
from fair_agent.modules.strict_incremental import (
    GLOBAL_CLASS_NAMES,
    bootstrap_metrics,
    build_protocol_dataset,
    calibrate_threshold,
    evaluate_ap50,
    fuse_old_new_predictions,
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
    specialist_model_value = config.get("adaptation", {}).get("specialist_model")
    if specialist_model_value:
        specialist_model_path = resolve_path(str(specialist_model_value))
        checks["specialist_initial_model"] = {
            "path": rel_path(specialist_model_path),
            "exists": specialist_model_path.is_file(),
        }
        if not specialist_model_path.is_file():
            errors.append(f"增量专家初始化权重不存在：{rel_path(specialist_model_path)}")
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
                # The lock split is an image-only interface until both models
                # have finished inference and their predictions are frozen.
                missing_labels = (
                    []
                    if split_name == "lock"
                    else [
                        rel_path(path)
                        for path in rows
                        if path.is_file() and not source_label(path).is_file()
                    ]
                )
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

    base_test_value = config.get("paths", {}).get("base_test_split")
    if not base_test_value:
        errors.append("paths.base_test_split 必须显式声明基础测试清单")
        checks["base_test_split"] = {"declared": False}
    else:
        base_test_path = resolve_path(base_test_value)
        if not base_test_path.is_file():
            errors.append("基础测试清单不存在")
        checks["base_test_split"] = {
            "declared": True,
            "exists": base_test_path.is_file(),
            "membership_read": False,
            "labels_read": False,
            "validation_deferred_until_after_prediction_freeze": True,
        }

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
            item["build_unified_student"] = bool(protocol.get("build_unified_student", False))
            if adaptation_mode in {"duet_yolo11s", "yolo_iod_lite"}:
                if not protocol.get("build_unified_student"):
                    raise ValueError(f"{adaptation_mode} 必须生成四类 student 数据视图")
                if adaptation_mode not in config.get("methods", {}):
                    raise ValueError(f"缺少 methods.{adaptation_mode} 配置")
            elif adaptation_mode == "frozen_base_plus_new_specialist" and protocol.get(
                "build_unified_student"
            ):
                raise ValueError("双检测器 Agent 不应生成或训练四类 student 数据视图")
            new_id = int(protocol["new_global_id"])
            counts = {
                split_name: sum(new_id in image_class_ids(image) for image in rows)
                for split_name, rows in split_rows.items()
                if split_name in {"train", "val"}
            }
            item["incremental_image_counts"] = counts
            item["base_only_image_counts"] = {
                split_name: sum(new_id not in image_class_ids(image) for image in rows)
                for split_name, rows in split_rows.items()
                if split_name in {"train", "val"}
            }
            item["lock_label_access"] = "forbidden_before_unlabeled_prediction_freeze"
            expected = protocol.get("expected_incremental_counts", {})
            for split_name in ("train", "val"):
                if split_name in expected and counts.get(split_name) != int(expected[split_name]):
                    errors.append(
                        f"{protocol_id} {split_name} 新增类样本数不符："
                        f"expected={expected[split_name]} actual={counts.get(split_name)}"
                    )
            item["valid"] = True
        except (KeyError, OSError, TypeError, ValueError) as exc:
            item["valid"] = False
            errors.append(f"{protocol_id} 协议检查失败：{exc}")
        protocol_checks.append(item)
    if not protocol_checks:
        errors.append("未声明任何增量协议")
    checks["protocols"] = protocol_checks

    agent_structure = dict(config.get("agent_structure", {}))
    checks["agent_structure"] = agent_structure
    expected_agent = {
        "architecture": "parallel_base_incremental_experts",
        "inference_scope": "every_image",
        "old_class_owner": "frozen_base",
        "new_class_owner": "incremental_specialist",
        "fusion_level": "detection_boxes",
        "scene_hard_routing": False,
        "label_aware_routing": False,
        "filename_class_routing": False,
    }
    for key, expected_value in expected_agent.items():
        if agent_structure.get(key) != expected_value:
            errors.append(f"agent_structure.{key} 必须为 {expected_value}")
    specialist_init = str(
        config.get("adaptation", {}).get("specialist_init", "base_checkpoint")
    )
    checks["specialist_initialization"] = specialist_init
    if specialist_init not in {"base_checkpoint", "generic_pretrained"}:
        errors.append(
            "adaptation.specialist_init 必须为 base_checkpoint 或 generic_pretrained"
        )

    checks["training_batch"] = int(config.get("common", {}).get("batch", 0))
    if checks["training_batch"] != 32:
        errors.append("严格 3+1 当前基准要求 batch=32")
    training_policy = dict(config.get("training_policy", {}))
    common_train = dict(config.get("common", {}))
    effective_patience = {
        phase: int({**common_train, **dict(config.get(phase, {}))}.get("patience", -1))
        for phase in ("base_train", "incremental_train")
    }
    checks["training_policy"] = {
        **training_policy,
        "effective_patience": effective_patience,
    }
    if training_policy.get("require_full_epochs") is not True:
        errors.append("training_policy.require_full_epochs 必须为 true")
    if training_policy.get("checkpoint_metric") != "map50":
        errors.append("training_policy.checkpoint_metric 必须为 map50")
    for phase, patience in effective_patience.items():
        if patience != 0:
            errors.append(f"{phase} 必须设置 patience=0 以禁用 EarlyStopping")
    predict_cfg = dict(config.get("predict", {}))
    inference_sizes = {
        "base": int(
            predict_cfg.get("base_imgsz", config.get("common", {}).get("imgsz", 0))
        ),
        "incremental": int(
            predict_cfg.get(
                "incremental_imgsz", config.get("common", {}).get("imgsz", 0)
            )
        ),
    }
    checks["inference_image_sizes"] = inference_sizes
    for owner, imgsz in inference_sizes.items():
        if imgsz <= 0 or imgsz % 32:
            errors.append(f"{owner} owner 推理分辨率必须为正整数且能被32整除")
    if bool(predict_cfg.get("augment", False)):
        errors.append("严格计分当前禁止 TTA；各 owner 仅运行一次固定尺度推理")
    acceptance = dict(config.get("acceptance", {}))
    checks["score_acceptance"] = acceptance
    required_score_keys = {"min_base_map50", "min_new_map50", "min_krr"}
    missing_score_keys = sorted(required_score_keys - set(acceptance))
    if missing_score_keys:
        errors.append(f"缺少赛题计分门槛：{missing_score_keys}")

    prototype_cfg = dict(config.get("prototype_gate", {}))
    checks["prototype_gate"] = prototype_cfg
    if prototype_cfg.get("enabled", False):
        if int(prototype_cfg.get("grid_size", 0)) < 4:
            errors.append("prototype_gate.grid_size 不能小于4")
        if not 0.0 < float(prototype_cfg.get("target_recall", 0.0)) <= 1.0:
            errors.append("prototype_gate.target_recall 必须位于(0, 1]")
        if float(prototype_cfg.get("safety_factor", 0.0)) < 1.0:
            errors.append("prototype_gate.safety_factor 不能小于1")
    cross_class = dict(config.get("fusion", {}).get("cross_class", {}))
    checks["cross_class_fusion"] = cross_class
    if cross_class.get("enabled", False):
        if not 0.0 <= float(cross_class.get("iou", -1.0)) <= 1.0:
            errors.append("fusion.cross_class.iou 必须位于[0, 1]")
        incremental_coverage = cross_class.get("incremental_coverage")
        if incremental_coverage is not None and not 0.0 <= float(
            incremental_coverage
        ) <= 1.0:
            errors.append("fusion.cross_class.incremental_coverage 必须位于[0, 1]")
        if not 0.0 <= float(cross_class.get("base_confidence", -1.0)) <= 1.0:
            errors.append("fusion.cross_class.base_confidence 必须位于[0, 1]")
        if not 0.0 <= float(cross_class.get("incremental_margin", -1.0)) <= 1.0:
            errors.append("fusion.cross_class.incremental_margin 必须位于[0, 1]")
        if not bool(cross_class.get("preserve_base_class_owners", False)):
            errors.append("双检测器 Agent 必须保留基础类别 owner 的预测")

    calibration_cfg = dict(config.get("calibration", {}))
    checks["calibration"] = calibration_cfg
    deployment_policy = str(calibration_cfg.get("deployment_policy", ""))
    if deployment_policy not in {"incremental_dev_calibrated", "fixed"}:
        errors.append(
            "calibration.deployment_policy 必须为 incremental_dev_calibrated 或 fixed"
        )
    if deployment_policy == "fixed" and calibration_cfg.get(
        "deployment_threshold"
    ) is None:
        errors.append("固定部署阈值策略必须声明 calibration.deployment_threshold")
    for key in ("threshold_min", "threshold_max", "target_precision"):
        if not 0.0 <= float(calibration_cfg.get(key, -1.0)) <= 1.0:
            errors.append(f"calibration.{key} 必须位于[0, 1]")
    if float(calibration_cfg.get("threshold_step", 0.0)) <= 0.0:
        errors.append("calibration.threshold_step 必须大于0")
    if float(calibration_cfg.get("threshold_min", 1.0)) >= float(
        calibration_cfg.get("threshold_max", 0.0)
    ):
        errors.append("calibration.threshold_min 必须小于 threshold_max")
    if calibration_cfg.get("deployment_threshold") is not None and not 0.0 <= float(
        calibration_cfg["deployment_threshold"]
    ) <= 1.0:
        errors.append("calibration.deployment_threshold 必须位于[0, 1]")

    context_gate = dict(config.get("context_gate", {}))
    checks["context_gate"] = context_gate
    if not isinstance(context_gate.get("enabled"), bool):
        errors.append("context_gate.enabled 必须为布尔值")
    if context_gate.get("enabled", False):
        context_model = resolve_path(str(context_gate.get("model") or ""))
        checks["context_gate"] = {
            **context_gate,
            "resolved_model": rel_path(context_model),
            "model_exists": context_model.is_file(),
        }
        if not context_model.is_file():
            errors.append(f"场景软门控模型不存在：{rel_path(context_model)}")
        dimensions = context_gate.get("dimensions")
        if (
            not isinstance(dimensions, list)
            or not dimensions
            or any(value not in {"scene", "sensor"} for value in dimensions)
        ):
            errors.append("context_gate.dimensions 必须是 scene/sensor 的非空列表")
        if len(set(dimensions or [])) != len(dimensions or []):
            errors.append("context_gate.dimensions 不得重复")
        if not 0.0 <= float(
            context_gate.get("max_threshold_penalty", -1.0)
        ) <= 1.0:
            errors.append("context_gate.max_threshold_penalty 必须位于[0, 1]")
        if int(context_gate.get("batch_size", 0)) < 1:
            errors.append("context_gate.batch_size 必须为正整数")

    deployment_acceptance = dict(config.get("deployment_acceptance", {}))
    checks["deployment_acceptance"] = deployment_acceptance
    required_deployment_gates = {
        "min_new_class_precision",
        "max_new_class_false_activation_rate",
    }
    missing_deployment_gates = sorted(
        required_deployment_gates - set(deployment_acceptance)
    )
    if missing_deployment_gates:
        errors.append(f"缺少 Agent 部署门槛：{missing_deployment_gates}")
    for key in required_deployment_gates & set(deployment_acceptance):
        if not 0.0 <= float(deployment_acceptance[key]) <= 1.0:
            errors.append(f"deployment_acceptance.{key} 必须位于[0, 1]")

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


def training_history(
    model: Any,
    phase: str,
    requested_epochs: int,
    *,
    require_full_epochs: bool = False,
) -> Dict[str, Any]:
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
    history = {
        "phase": phase,
        "results_csv": rel_path(results_path) if results_path.exists() else None,
        "requested_epochs": int(requested_epochs),
        "completed_epochs": completed_epochs,
        "best_epoch": int(float(best_row.get("epoch", 0))) if best_row else None,
        "best_metric": metric_key,
        "best_metric_value": float(best_row[metric_key]) if best_row and metric_key else None,
        "stopped_early": completed_epochs < int(requested_epochs),
        "stop_reason": "early_stopping_or_interruption" if completed_epochs < int(requested_epochs) else "epoch_budget_reached",
        "checkpoint_metric": getattr(trainer, "_checkpoint_metric", None),
        "epochs": rows,
    }
    if require_full_epochs and completed_epochs != int(requested_epochs):
        raise RuntimeError(
            f"{phase} 未跑满规定 epoch："
            f"completed={completed_epochs} requested={int(requested_epochs)}"
        )
    return history


def _trainer_model(trainer: Any) -> Any:
    return trainer.model.module if hasattr(trainer.model, "module") else trainer.model


def configure_map50_checkpointing(trainer: Any) -> None:
    """Make best.pt and early stopping follow the competition mAP50 metric."""
    from types import MethodType

    metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
    box = getattr(metrics, "box", None)
    if box is None:
        raise RuntimeError("Ultralytics validator 尚未初始化，无法按 mAP50 选择权重")

    def map50_fitness(metric: Any) -> float:
        return float(metric.map50)

    box.fitness = MethodType(map50_fitness, box)
    trainer._checkpoint_metric = "metrics/mAP50(B)"


def competition_score_gates(
    base_map50: float,
    new_map50: float,
    krr: float,
    acceptance: Mapping[str, Any],
) -> Dict[str, bool]:
    """Return only the three score-bearing gates stated by the competition."""
    return {
        "base_map50": float(base_map50) >= float(acceptance["min_base_map50"]),
        "new_map50": float(new_map50) >= float(acceptance["min_new_map50"]),
        "krr": float(krr) >= float(acceptance["min_krr"]),
    }


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
    if config.get("training_policy", {}).get("require_full_epochs") is True:
        if int(common.get("patience", -1)) != 0:
            raise ValueError(f"{phase} 必须设置 patience=0 以禁用 EarlyStopping")
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
    if config.get("training_policy", {}).get("require_full_epochs") is True:
        if int(arguments.get("patience", -1)) != 0:
            raise ValueError(f"methods.{method}.{stage} 必须设置 patience=0 以禁用 EarlyStopping")
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
    imgsz: int | None = None,
) -> tuple[list[Dict[str, Any]], float]:
    predict = dict(config["predict"])
    predict_args = {
        "imgsz": int(imgsz if imgsz is not None else config["common"]["imgsz"]),
        "conf": float(predict["conf"]),
        "iou": float(predict["iou"]),
        "max_det": int(predict["max_det"]),
        "rect": bool(predict.get("rect", True)),
        "augment": bool(predict.get("augment", False)),
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


def owner_inference_imgsz(config: Mapping[str, Any], owner: str) -> int:
    predict = dict(config.get("predict", {}))
    key = "base_imgsz" if owner == "base" else "incremental_imgsz"
    return int(predict.get(key, config["common"]["imgsz"]))


def write_jsonl_artifact(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Freeze one prediction-stage artifact without allowing silent overwrite."""
    if path.exists():
        raise FileExistsError(f"拒绝覆盖冻结评测记录：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "path": rel_path(path),
        "sha256": sha256_file(path),
        "row_count": len(rows),
    }


def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖冻结评测记录：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "path": rel_path(path),
        "sha256": sha256_file(path),
    }


def predict_context_records(
    model: Any,
    checkpoint: Mapping[str, Any],
    images: Sequence[Path],
    device: str,
    batch_size: int,
) -> Dict[str, Dict[str, Any]]:
    """Predict known sensor/scene probabilities without reading labels."""
    from PIL import Image

    from fair_agent.models.context import predict_context_batch

    output: Dict[str, Dict[str, Any]] = {}
    for start in range(0, len(images), int(batch_size)):
        paths = images[start : start + int(batch_size)]
        opened = []
        try:
            for path in paths:
                with Image.open(path) as source:
                    opened.append(source.convert("RGB"))
            predictions = predict_context_batch(
                model, dict(checkpoint), opened, device
            )
        finally:
            for image in opened:
                image.close()
        for path, prediction in zip(paths, predictions):
            output[path.stem] = {
                key: value
                for key, value in prediction.items()
                if key != "_inference_ms"
            }
    return output


def freeze_unlabeled_lock_predictions(
    base_predictor: Any,
    incremental_predictor: Any,
    lock_images: Sequence[Path],
    base_mapping: Mapping[int, int],
    incremental_mapping: Mapping[int, int],
    config: Mapping[str, Any],
    device: str,
    new_class_id: int,
    unified_student: bool,
    positive_prototype: Mapping[str, Any] | None,
    deployment_threshold: float,
    context_prior: Mapping[str, Any] | None,
    contexts_by_image: Mapping[str, Mapping[str, Any]] | None,
    max_context_penalty: float,
    report_dir: Path,
) -> Dict[str, Any]:
    """Run every class owner on every lock image without opening any label.

    This function is deliberately label-blind. It accepts only image paths,
    frozen models and frozen inference/fusion settings, then persists hashes of
    all prediction artifacts before the caller is allowed to unseal labels.
    """
    input_stems = [path.stem for path in lock_images]
    if not input_stems or len(input_stems) != len(set(input_stems)):
        raise ValueError("mixed_test 必须非空且图像 stem 唯一")
    agent_structure = dict(config.get("agent_structure", {}))
    if agent_structure.get("inference_scope") != "every_image":
        raise ValueError("类别增量计分必须在每张 mixed_test 图像运行全部 owner")
    for forbidden_flag in (
        "scene_hard_routing",
        "label_aware_routing",
        "filename_class_routing",
    ):
        if bool(agent_structure.get(forbidden_flag, True)):
            raise ValueError(f"严格计分禁止 agent_structure.{forbidden_flag}")

    base_predictions, base_inference_ms = predict_records(
        base_predictor,
        lock_images,
        base_mapping,
        config,
        device,
        "frozen_base_model",
        owner_inference_imgsz(config, "base"),
    )
    raw_incremental_predictions, incremental_inference_ms = predict_records(
        incremental_predictor,
        lock_images,
        incremental_mapping,
        config,
        device,
        "incremental_student" if unified_student else "incremental_specialist",
        owner_inference_imgsz(config, "incremental"),
    )

    incremental_predictions = list(raw_incremental_predictions)
    prototype_rejections: list[Dict[str, Any]] = []
    if positive_prototype is not None:
        incremental_predictions, prototype_rejections = apply_positive_prototype(
            incremental_predictions,
            lock_images,
            positive_prototype,
        )

    old_class_ids = set(int(value) for value in base_mapping.values())
    if unified_student:
        candidate_old_predictions = [
            row
            for row in incremental_predictions
            if int(row["class_id"]) in old_class_ids
        ]
        candidate_new_predictions = [
            row
            for row in incremental_predictions
            if int(row["class_id"]) == int(new_class_id)
        ]
    else:
        candidate_old_predictions = list(base_predictions)
        candidate_new_predictions = list(incremental_predictions)
    pre_activation_predictions, fusion_decisions = fuse_old_new_predictions(
        candidate_old_predictions,
        candidate_new_predictions,
        nms_iou=float(config["fusion"]["nms_iou"]),
        cross_class=config["fusion"].get("cross_class"),
    )
    combined_predictions, activation_rejections = apply_incremental_candidate_gates(
        pre_activation_predictions,
        {int(new_class_id): float(deployment_threshold)},
        contexts_by_image=contexts_by_image,
        context_prior=context_prior,
        max_context_penalty=float(max_context_penalty),
    )

    prediction_dir = report_dir / "predictions"
    input_artifact = write_json_artifact(
        prediction_dir / "lock_unlabeled_inputs.json",
        {
            "schema_version": 1,
            "input_mode": "unlabeled_images",
            "image_count": len(lock_images),
            "image_stems": input_stems,
            "base_input_stems": input_stems,
            "incremental_input_stems": input_stems,
            "base_and_incremental_inputs_identical": True,
            "owner_inference_imgsz": {
                "base": owner_inference_imgsz(config, "base"),
                "incremental": owner_inference_imgsz(config, "incremental"),
            },
            "test_time_augmentation": bool(
                config.get("predict", {}).get("augment", False)
            ),
            "scene_hard_routing": bool(agent_structure["scene_hard_routing"]),
            "label_aware_routing": bool(agent_structure["label_aware_routing"]),
            "filename_class_routing": bool(agent_structure["filename_class_routing"]),
            "context_soft_gating": bool(context_prior),
        },
    )
    artifacts = {
        "inputs": input_artifact,
        "base_raw": write_jsonl_artifact(
            prediction_dir / "lock_base_unlabeled.jsonl", base_predictions
        ),
        "incremental_raw": write_jsonl_artifact(
            prediction_dir / "lock_incremental_unlabeled.jsonl",
            raw_incremental_predictions,
        ),
        "incremental_prototype_rejected": write_jsonl_artifact(
            prediction_dir / "lock_prototype_rejected.jsonl",
            prototype_rejections,
        ),
        "fusion_decisions": write_jsonl_artifact(
            prediction_dir / "lock_fusion_decisions.jsonl", fusion_decisions
        ),
        "fused_pre_activation": write_jsonl_artifact(
            prediction_dir / "lock_fused_pre_activation.jsonl",
            pre_activation_predictions,
        ),
        "activation_rejected": write_jsonl_artifact(
            prediction_dir / "lock_activation_rejected.jsonl",
            activation_rejections,
        ),
        "contexts": write_json_artifact(
            prediction_dir / "lock_context_predictions.json",
            {
                "schema_version": 1,
                "input_mode": "unlabeled_images",
                "predictions": dict(contexts_by_image or {}),
            },
        ),
        "fused": write_jsonl_artifact(
            prediction_dir / "lock_fused_unlabeled.jsonl", combined_predictions
        ),
    }
    fusion_inputs = ["boxes", "confidence"]
    cross_class = dict(config.get("fusion", {}).get("cross_class", {}))
    if cross_class.get("enabled", False):
        fusion_inputs.append("iou")
        if cross_class.get("incremental_coverage") is not None:
            fusion_inputs.append("incremental_coverage")
    fusion_inputs.append("fixed_class_owners")
    if context_prior:
        fusion_inputs.append("soft_known_context")
    return {
        "base_predictions": base_predictions,
        "raw_incremental_predictions": raw_incremental_predictions,
        "incremental_predictions": incremental_predictions,
        "combined_predictions": combined_predictions,
        "fusion_decisions": fusion_decisions,
        "activation_rejections": activation_rejections,
        "prototype_rejections": prototype_rejections,
        "base_inference_ms": base_inference_ms,
        "incremental_inference_ms": incremental_inference_ms,
        "artifacts": artifacts,
        "audit": {
            "unlabeled_inference_completed_before_lock_labels": True,
            "input_image_count": len(lock_images),
            "base_input_stems_equal_mixed_test": True,
            "incremental_input_stems_equal_mixed_test": True,
            "base_and_incremental_input_stems_identical": True,
            "scene_hard_routing": bool(agent_structure["scene_hard_routing"]),
            "label_aware_routing": bool(agent_structure["label_aware_routing"]),
            "filename_class_routing": bool(agent_structure["filename_class_routing"]),
            "fusion_inputs": "_".join(fusion_inputs),
            "activation_threshold_source": "incremental_dev_only",
            "context_prior_source": (
                str((context_prior or {}).get("source_split"))
                if context_prior
                else None
            ),
            "context_soft_gating": bool(context_prior),
        },
    }


def declared_base_test_image_ids(
    base_test_images: Sequence[Path],
    lock_images: Sequence[Path],
    ground_truth: Sequence[Mapping[str, Any]],
    new_class_id: int,
) -> list[str]:
    """Validate and return the predeclared old-only scoring subset after freeze."""
    lock_ids = {path.stem for path in lock_images}
    base_ids = [path.stem for path in base_test_images]
    if not base_ids or len(base_ids) != len(set(base_ids)):
        raise ValueError("base_test 清单必须非空且 stem 唯一")
    if not set(base_ids) <= lock_ids:
        raise ValueError("base_test 清单必须是 mixed_test 的子集")
    classes_by_image: Dict[str, set[int]] = {path.stem: set() for path in lock_images}
    for row in ground_truth:
        classes_by_image.setdefault(str(row["image_id"]), set()).add(int(row["class_id"]))
    invalid = [
        image_id
        for image_id in base_ids
        if int(new_class_id) in classes_by_image.get(image_id, set())
    ]
    if invalid:
        raise ValueError("base_test 清单包含新增类别图像")
    return base_ids


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
    imgsz: int | None = None,
) -> tuple[list[Dict[str, Any]], float, float]:
    predict = config["predict"]
    metrics = model.val(
        validator=recording_validator_class(),
        data=str(dataset),
        split=split,
        imgsz=int(imgsz if imgsz is not None else config["common"]["imgsz"]),
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
        "base_test_map50": result["base_test_map50"],
        "base_dev_map50": result["base_dev_map50"],
        "new_map50": result["new_map50"],
        "krr": result["krr"],
        "old_map50_before": result["old_map50_before"],
        "old_map50_after": result["old_map50_after"],
        "full_map50": result["full_map50"],
        "deployment_threshold": result["calibration"]["deployment_threshold"],
        "diagnostic_calibration_threshold": result["calibration"]["selected"]["threshold"],
        "calibration_precision": result["calibration"]["selected"]["precision"],
        "calibration_recall": result["calibration"]["selected"]["recall"],
        "prototype_rejected": result.get("positive_prototype", {}).get(
            "lock_rejected_candidate_count", 0
        ),
        "fusion_rejected": result.get("fusion", {}).get(
            "rejected_incremental_count", 0
        ),
        "activation_rejected": result.get("fusion", {}).get(
            "activation_rejected_count", 0
        ),
        "lock_new_class_precision": result["lock_deployment_metrics"]["precision"],
        "lock_false_activation_rate": result["false_activation"][
            "false_activation_rate"
        ],
        "evaluator_error": result["evaluator_error"],
        "competition_accepted": result["competition_accepted"],
        "deployment_accepted": result["deployment_accepted"],
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
        f"- 赛题基础测试 mAP50：{result['base_test_map50']:.5f}",
        f"- 赛题 New-mAP50：{result['new_map50']:.5f}",
        f"- 赛题 KRR：{result['krr']:.5f}",
        f"- 基础 dev mAP50（仅选权重诊断）：{result['base_dev_map50']:.5f}",
        f"- 旧类 mAP50 before/after：{result['old_map50_before']:.5f} / {result['old_map50_after']:.5f}",
        f"- 四类总体 mAP50（仅诊断）：{result['full_map50']:.5f}",
        f"- 增量 dev 冻结的计分/Agent 共用阈值：{result['calibration']['deployment_threshold']:.2f}",
        f"- 阈值校准策略：{result['calibration']['deployment_policy']}",
        f"- dev 校准 precision/recall：{result['calibration']['selected']['precision']:.5f} / {result['calibration']['selected']['recall']:.5f}",
        f"- lock precision/recall：{result['lock_deployment_metrics']['precision']:.5f} / {result['lock_deployment_metrics']['recall']:.5f}",
        f"- lock 图像误激活率：{result['false_activation']['false_activation_rate']:.5f}",
        f"- 场景软阈值惩罚：{'启用' if result['context_gate']['enabled'] else '关闭'}（最大 +{result['context_gate']['max_threshold_penalty']:.2f}）",
        f"- 正样本原型拒绝候选数：{result.get('positive_prototype', {}).get('lock_rejected_candidate_count', 0)}",
        f"- 跨类冲突拒绝候选数：{result.get('fusion', {}).get('rejected_incremental_count', 0)}",
        f"- 阈值/场景门控拒绝候选数：{result.get('fusion', {}).get('activation_rejected_count', 0)}",
        f"- 自定义评测误差：{result['evaluator_error']:.6f}",
        f"- 共享参数相对漂移：{result.get('shared_parameter_relative_drift', 0.0):.6f}",
        f"- bootstrap New-mAP50 95% CI：[{result['bootstrap']['new_map50']['ci95_low']:.5f}, {result['bootstrap']['new_map50']['ci95_high']:.5f}]",
        f"- bootstrap 四类 mAP50 95% CI：[{result['bootstrap']['full_map50']['ci95_low']:.5f}, {result['bootstrap']['full_map50']['ci95_high']:.5f}]",
        inference_line,
        f"- 赛题三项与完整性结论：{'通过' if result['competition_accepted'] else '未通过'}",
        f"- Agent 部署质量结论：{'通过' if result['deployment_accepted'] else '未通过'}",
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
        "两份模型已先对完整 lock 执行无标签推理并哈希预测，随后才解封标签评分；本报告结果不得用于本 run 调参。",
    ]
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def freeze_context_gate(
    version_dir: Path, metrics: Mapping[str, Any]
) -> Dict[str, Any]:
    gate = dict(metrics.get("context_gate", {}))
    prior = gate.pop("prior", None)
    gate.pop("prior_artifact", None)
    if not gate.get("enabled", False):
        return {
            "context_prior": {},
            "context_gate": {**gate, "enabled": False},
            "context_prior_source": None,
            "context_prior_sha256": None,
        }
    if not isinstance(prior, Mapping) or prior.get(
        "source_split"
    ) != "incremental_train_only":
        raise ValueError("待冻结场景先验不是仅由 incremental_train 学习")
    frozen_prior = version_dir / "context_prior.json"
    frozen_prior.write_text(
        json.dumps(dict(prior), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "context_prior": dict(prior),
        "context_gate": {
            **gate,
            "enabled": True,
            "learning_data_scope": "incremental_train_only",
            "prior_source": rel_path(frozen_prior),
        },
        "context_prior_source": rel_path(frozen_prior),
        "context_prior_sha256": sha256_file(frozen_prior),
    }


def freeze_profile(
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    run_id: str,
    base_weight: Path,
    specialist_weight: Path,
    calibration: Mapping[str, Any],
    report_dir: Path,
    prototype_path: Path | None = None,
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
    frozen_prototype = version_dir / "positive_prototype.json" if prototype_path else None
    if prototype_path is not None and frozen_prototype is not None:
        shutil.copy2(prototype_path, frozen_prototype)
    shutil.copy2(report_dir / "metrics.json", version_dir / "metrics.json")
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    context_profile = freeze_context_gate(version_dir, metrics)
    local_to_global = {int(key): int(value) for key, value in protocol["base_local_to_global"].items()}
    profile = {
        "schema_version": 1,
        "profile_id": protocol["id"],
        "run_id": run_id,
        "acceptance": "passed" if metrics.get("deployment_accepted") else "rejected",
        "competition_accepted": bool(metrics.get("competition_accepted")),
        "deployment_accepted": bool(metrics.get("deployment_accepted")),
        "incremental_mode": "class_incremental",
        "deployment": "dual_detector",
        "agent_structure": dict(metrics.get("agent_structure", {})),
        "base_weight": rel_path(frozen_base),
        "base_sha256": sha256_file(frozen_base),
        "specialist_weight": rel_path(frozen_specialist),
        "specialist_sha256": sha256_file(frozen_specialist),
        "base_local_to_global": local_to_global,
        "base_local_names": {local: GLOBAL_CLASS_NAMES[global_id] for local, global_id in local_to_global.items()},
        "class_names": GLOBAL_CLASS_NAMES,
        "new_class": protocol["new_class"],
        "new_global_id": int(protocol["new_global_id"]),
        "activation_threshold": float(
            calibration.get("deployment_threshold", calibration["selected"]["threshold"])
        ),
        "calibration_source": rel_path(frozen_calibration),
        "metrics_source": rel_path(version_dir / "metrics.json"),
        "evidence_level": "verified",
        "base_test_map50": float(metrics["base_test_map50"]),
        "base_dev_map50": float(metrics["base_dev_map50"]),
        "new_map50": float(metrics["new_map50"]),
        "krr": float(metrics["krr"]),
        "full_map50": float(metrics["full_map50"]),
        "lock_precision": metrics["lock_deployment_metrics"]["precision"],
        "lock_recall": metrics["lock_deployment_metrics"]["recall"],
        "lock_false_activation_rate": metrics["false_activation"]["false_activation_rate"],
        "fusion_policy": metrics.get("fusion", {}).get("cross_class", {}),
        **context_profile,
        "base_imgsz": int(metrics["inference_image_sizes"]["base"]),
        "specialist_imgsz": int(metrics["inference_image_sizes"]["incremental"]),
        "positive_prototype": (
            json.loads(frozen_prototype.read_text(encoding="utf-8"))
            if frozen_prototype is not None else None
        ),
        "positive_prototype_source": (
            rel_path(frozen_prototype) if frozen_prototype is not None else None
        ),
        "positive_prototype_sha256": (
            sha256_file(frozen_prototype) if frozen_prototype is not None else None
        ),
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
        "base_test_map50": profile["base_test_map50"],
        "base_dev_map50": profile["base_dev_map50"],
        "new_map50": profile["new_map50"],
        "krr": profile["krr"],
        "full_map50": profile["full_map50"],
        "activation_threshold": profile["activation_threshold"],
        "lock_false_activation_rate": profile["lock_false_activation_rate"],
        "lock_precision": profile["lock_precision"],
        "competition_accepted": profile["competition_accepted"],
        "deployment_accepted": profile["deployment_accepted"],
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
    prototype_path: Path | None = None,
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
    frozen_prototype = version_dir / "positive_prototype.json" if prototype_path else None
    if prototype_path is not None and frozen_prototype is not None:
        shutil.copy2(prototype_path, frozen_prototype)
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    context_profile = freeze_context_gate(version_dir, metrics)
    profile = {
        "schema_version": 2,
        "profile_id": protocol["id"],
        "run_id": run_id,
        "acceptance": "passed" if metrics.get("deployment_accepted") else "rejected",
        "competition_accepted": bool(metrics.get("competition_accepted")),
        "deployment_accepted": bool(metrics.get("deployment_accepted")),
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
        "base_local_to_global": {
            class_id: class_id for class_id in GLOBAL_CLASS_NAMES
        },
        "base_local_names": GLOBAL_CLASS_NAMES,
        "new_class": protocol["new_class"],
        "new_global_id": int(protocol["new_global_id"]),
        "activation_threshold": float(
            calibration.get("deployment_threshold", calibration["selected"]["threshold"])
        ),
        "calibration_source": rel_path(frozen_calibration),
        "metrics_source": rel_path(version_dir / "metrics.json"),
        "evidence_level": "verified",
        "base_test_map50": float(metrics["base_test_map50"]),
        "base_dev_map50": float(metrics["base_dev_map50"]),
        "new_map50": float(metrics["new_map50"]),
        "krr": float(metrics["krr"]),
        "full_map50": float(metrics["full_map50"]),
        "old_channel_max_abs_drift": float(metrics["old_channel_max_abs_drift"]),
        "fusion_policy": metrics.get("fusion", {}).get("cross_class", {}),
        **context_profile,
        "positive_prototype": (
            json.loads(frozen_prototype.read_text(encoding="utf-8"))
            if frozen_prototype is not None else None
        ),
        "positive_prototype_source": (
            rel_path(frozen_prototype) if frozen_prototype is not None else None
        ),
        "positive_prototype_sha256": (
            sha256_file(frozen_prototype) if frozen_prototype is not None else None
        ),
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
    recheck_base_weight: str | None = None,
    recheck_incremental_weight: str | None = None,
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
        if recheck_base_weight:
            base_weight = resolve_path(recheck_base_weight)
        else:
            base_weight = resolve_path(configured_base) if configured_base else run_dir / "base" / "weights" / "best.pt"
        if recheck_incremental_weight:
            incremental_weight = resolve_path(recheck_incremental_weight)
        elif adaptation_mode == "duet_yolo11s":
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
            base_model.add_callback("on_pretrain_routine_end", configure_map50_checkpointing)
            base_train_result = base_model.train(
                **train_arguments(config, "base_train", base_dataset, run_dir, "base", device)
            )
            base_weight = best_weight(base_model, base_train_result)
            base_history = training_history(
                base_model,
                "base",
                int(config["base_train"]["epochs"]),
                require_full_epochs=bool(config["training_policy"]["require_full_epochs"]),
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
            current_model.add_callback("on_pretrain_routine_end", configure_map50_checkpointing)
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
                require_full_epochs=bool(config["training_policy"]["require_full_epochs"]),
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
            current_model.add_callback("on_pretrain_routine_end", configure_map50_checkpointing)
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
                require_full_epochs=bool(config["training_policy"]["require_full_epochs"]),
            ))
            student_model = YOLO(str(config.get("adaptation", {}).get("student_init", config["model"])))
            student_model.add_callback("on_pretrain_routine_end", configure_map50_checkpointing)
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
                require_full_epochs=bool(config["training_policy"]["require_full_epochs"]),
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
            student_model.add_callback("on_pretrain_routine_end", configure_map50_checkpointing)
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
                require_full_epochs=bool(config["training_policy"]["require_full_epochs"]),
            ))
        else:
            specialist_init = str(
                config.get("adaptation", {}).get("specialist_init", "base_checkpoint")
            )
            if specialist_init == "generic_pretrained":
                specialist_source = resolve_path(
                    str(config.get("adaptation", {}).get("specialist_model", config["model"]))
                )
            elif specialist_init == "base_checkpoint":
                specialist_source = base_weight
            else:
                raise ValueError(f"未知增量专家初始化策略：{specialist_init}")
            method_audit["specialist_initialization"] = specialist_init
            method_audit["specialist_initial_weight"] = rel_path(specialist_source)
            method_audit["specialist_initial_weight_sha256"] = sha256_file(specialist_source)
            specialist_model = YOLO(str(specialist_source))
            specialist_model.add_callback("on_pretrain_routine_end", configure_map50_checkpointing)
            specialist_train_result = specialist_model.train(
                **train_arguments(config, "incremental_train", incremental_dataset, run_dir, "specialist", device)
            )
            incremental_weight = best_weight(specialist_model, specialist_train_result)
            incremental_histories.append(training_history(
                specialist_model,
                "incremental_specialist",
                int(config["incremental_train"]["epochs"]),
                require_full_epochs=bool(config["training_policy"]["require_full_epochs"]),
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

    base_mapping = {
        int(key): int(value) for key, value in protocol["base_local_to_global"].items()
    }
    old_ids = sorted(base_mapping.values())
    base_val_images = read_split(dataset_dir / "base" / "splits" / "val.txt")
    base_predictor = YOLO(str(base_weight))
    base_dev_predictions, base_dev_inference_ms, base_dev_reference_map50 = validator_records(
        base_predictor,
        base_dataset,
        "val",
        base_val_images,
        base_mapping,
        config,
        device,
        "frozen_base_model",
        report_dir / "ultralytics",
        "base_dev",
        report_dir / "predictions" / "base_dev.jsonl",
        owner_inference_imgsz(config, "base"),
    )
    base_dev_ground_truth = remap_ground_truth(
        yolo_ground_truth(base_val_images), base_mapping
    )
    base_dev_metrics = evaluate_ap50(
        base_dev_predictions, base_dev_ground_truth, old_ids
    )
    base_dev_map50 = float(base_dev_metrics["map50"])
    base_dev_evaluator_error = abs(
        float(base_dev_reference_map50) - base_dev_map50
    )
    _audit_event(
        config,
        protocol_id,
        "BASE_EVALUATED",
        split="base_dev",
        map50=base_dev_map50,
        ultralytics_map50=float(base_dev_reference_map50),
    )

    new_id = int(protocol["new_global_id"])
    incremental_phase = "student" if unified_student else "incremental"
    incremental_train = read_split(dataset_dir / incremental_phase / "splits" / "train.txt")
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
        owner_inference_imgsz(config, "incremental"),
    )
    dev_ground_truth = yolo_ground_truth(incremental_val)
    if not unified_student:
        dev_ground_truth = remap_ground_truth(dev_ground_truth, {0: new_id})
    raw_dev_metrics = evaluate_ap50(dev_predictions, dev_ground_truth, [new_id])
    incremental_dev_evaluator_error = abs(
        float(_dev_reference_map50) - float(raw_dev_metrics["map50"])
    )
    prototype_gate_cfg = dict(config.get("prototype_gate", {}))
    positive_prototype: Dict[str, Any] | None = None
    prototype_path: Path | None = None
    dev_prototype_rejections: list[Dict[str, Any]] = []
    if prototype_gate_cfg.get("enabled", False):
        train_ground_truth = yolo_ground_truth(incremental_train)
        if not unified_student:
            train_ground_truth = remap_ground_truth(train_ground_truth, {0: new_id})
        positive_prototype = fit_positive_prototype(
            incremental_train,
            train_ground_truth,
            new_id,
            grid_size=int(prototype_gate_cfg.get("grid_size", 8)),
            minimum_scale=float(prototype_gate_cfg.get("minimum_scale", 0.10)),
        )
        positive_prototype = calibrate_positive_prototype(
            positive_prototype,
            incremental_val,
            dev_predictions,
            dev_ground_truth,
            target_recall=float(prototype_gate_cfg.get("target_recall", 0.95)),
            safety_factor=float(prototype_gate_cfg.get("safety_factor", 1.10)),
            iou_threshold=float(prototype_gate_cfg.get("iou_threshold", 0.50)),
        )
        dev_predictions, dev_prototype_rejections = apply_positive_prototype(
            dev_predictions,
            incremental_val,
            positive_prototype,
        )
        prototype_path = report_dir / "positive_prototype.json"
        prototype_path.write_text(
            json.dumps(positive_prototype, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rejected_path = report_dir / "predictions" / "dev_prototype_rejected.jsonl"
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in dev_prototype_rejections
            ),
            encoding="utf-8",
        )
        _audit_event(
            config,
            protocol_id,
            "POSITIVE_PROTOTYPE_CALIBRATED",
            source="incremental_train_and_dev_only",
            train_positive_count=positive_prototype["train_positive_count"],
            dev_positive_count=positive_prototype["dev_positive_count"],
            distance_threshold=positive_prototype["distance_threshold"],
            rejected_dev_candidate_count=len(dev_prototype_rejections),
            artifact=rel_path(prototype_path),
            artifact_sha256=sha256_file(prototype_path),
        )
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
    deployment_policy = str(
        calibration_cfg.get("deployment_policy", "incremental_dev_calibrated")
    )
    if deployment_policy == "incremental_dev_calibrated":
        deployment_threshold = float(calibration["selected"]["threshold"])
    elif deployment_policy == "fixed":
        if calibration_cfg.get("deployment_threshold") is None:
            raise ValueError("固定部署阈值策略缺少 calibration.deployment_threshold")
        deployment_threshold = float(calibration_cfg["deployment_threshold"])
    else:
        raise ValueError(f"未知部署阈值策略：{deployment_policy}")
    calibration["deployment_threshold"] = deployment_threshold
    calibration["deployment_policy"] = deployment_policy
    calibration["source_split"] = "incremental_dev_only"
    calibration["learning_data_scope"] = "incremental_dataset_only"

    context_gate_cfg = dict(config.get("context_gate", {}))
    context_model = context_checkpoint = None
    context_prior: Dict[str, Any] = {}
    context_prior_path: Path | None = None
    context_device = device if str(device).startswith("cuda:") else f"cuda:{device}"
    if context_gate_cfg.get("enabled", False):
        from fair_agent.models.context import load_context_model

        context_model_path = resolve_path(context_gate_cfg["model"])
        context_model, context_checkpoint = load_context_model(
            context_model_path, context_device
        )
        train_contexts = predict_context_records(
            context_model,
            context_checkpoint,
            incremental_train,
            context_device,
            int(context_gate_cfg.get("batch_size", 32)),
        )
        context_prior = learn_context_prior(
            list(train_contexts.values()),
            tuple(context_gate_cfg.get("dimensions", ["scene"])),
        )
        if not any(
            isinstance(context_prior.get(dimension), Mapping)
            and bool(context_prior.get(dimension))
            for dimension in context_gate_cfg.get("dimensions", ["scene"])
        ):
            raise RuntimeError("增量训练集未产生可用的已知场景先验")
        context_prior_path = report_dir / "context_prior.json"
        context_prior_path.write_text(
            json.dumps(context_prior, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _audit_event(
            config,
            protocol_id,
            "CONTEXT_PRIOR_LEARNED",
            source="incremental_train_only",
            sample_count=len(train_contexts),
            artifact=rel_path(context_prior_path),
            artifact_sha256=sha256_file(context_prior_path),
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
        deployment_threshold=float(calibration["deployment_threshold"]),
        precision=float(calibration["selected"]["precision"]),
        source="incremental_dev_only",
        artifact=rel_path(calibration_path),
        artifact_sha256=sha256_file(calibration_path),
    )

    # The lock interface remains image-only here. Every class owner receives
    # the same complete list before any label is opened or transformed.
    lock_images = read_split(config["paths"]["source_splits"]["lock"])
    lock_contexts = (
        predict_context_records(
            context_model,
            context_checkpoint,
            lock_images,
            context_device,
            int(context_gate_cfg.get("batch_size", 32)),
        )
        if context_model is not None and context_checkpoint is not None
        else {}
    )
    frozen_lock = freeze_unlabeled_lock_predictions(
        base_predictor,
        incremental_predictor,
        lock_images,
        base_mapping,
        incremental_mapping,
        config,
        device,
        new_id,
        unified_student,
        positive_prototype,
        deployment_threshold,
        context_prior,
        lock_contexts,
        float(context_gate_cfg.get("max_threshold_penalty", 0.0)),
        report_dir,
    )
    base_predictions = list(frozen_lock["base_predictions"])
    combined_predictions = list(frozen_lock["combined_predictions"])
    fusion_decisions = list(frozen_lock["fusion_decisions"])
    activation_rejections = list(frozen_lock["activation_rejections"])
    lock_prototype_rejections = list(frozen_lock["prototype_rejections"])
    base_inference_ms = float(frozen_lock["base_inference_ms"])
    incremental_inference_ms = float(frozen_lock["incremental_inference_ms"])
    lock_inference_audit = dict(frozen_lock["audit"])
    lock_prediction_artifacts = dict(frozen_lock["artifacts"])
    _audit_event(
        config,
        protocol_id,
        "LOCK_PREDICTIONS_FROZEN",
        **lock_inference_audit,
        artifacts=lock_prediction_artifacts,
    )

    # This is the first point where lock labels may be read or transformed.
    manifest_path = dataset_dir / "manifest.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_manifest = (
        existing_manifest
        if existing_manifest.get("lock_materialized_after_freeze")
        else materialize_lock_data(
            protocol, config["paths"]["source_splits"]["lock"], dataset_dir
        )
    )
    _audit_event(
        config,
        protocol_id,
        "LOCK_UNSEALED",
        split_sha256=dataset_manifest["source_split_sha256"]["lock"],
        predictions_frozen=True,
        fused_predictions_sha256=lock_prediction_artifacts["fused"]["sha256"],
    )
    ground_truth = yolo_ground_truth(lock_images)
    # base_test 成员信息也不得参与图片级模型路由；只有完整 mixed_test 的
    # 预测及其哈希冻结后，评分器才读取该预先固定的旧类子集。
    base_test_images = read_split(config["paths"]["base_test_split"])
    base_test_ids = declared_base_test_image_ids(
        base_test_images, lock_images, ground_truth, new_id
    )
    expected_base_test = protocol.get("expected_base_test_count")
    if expected_base_test is not None and len(base_test_ids) != int(expected_base_test):
        raise RuntimeError(
            f"基础测试样本数不符：expected={int(expected_base_test)} "
            f"actual={len(base_test_ids)}"
        )
    lock_inference_audit["base_test_membership_read_after_prediction_freeze"] = True
    lock_inference_audit["base_test_is_subset_of_mixed_test"] = True
    base_test_metrics = evaluate_ap50(
        subset_rows(base_predictions, base_test_ids),
        subset_rows(ground_truth, base_test_ids),
        old_ids,
    )
    base_test_map50 = float(base_test_metrics["map50"])
    retention = retention_metrics(base_predictions, combined_predictions, ground_truth, old_ids)
    new_metrics = evaluate_ap50(combined_predictions, ground_truth, [new_id])
    full_metrics = evaluate_ap50(combined_predictions, ground_truth, GLOBAL_CLASS_NAMES)
    old_before = float(retention["old_map50_before"])
    old_after = float(retention["old_map50_after"])
    old_prediction_equivalent = bool(retention["old_prediction_equivalent"])
    krr = float(retention["krr"])
    evaluator_error = max(base_dev_evaluator_error, incremental_dev_evaluator_error)
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
    deployment_threshold = float(calibration["deployment_threshold"])
    lock_pr = precision_recall(
        combined_predictions, ground_truth, new_id, deployment_threshold
    )
    false_activation = image_false_activation_rate(
        combined_predictions, ground_truth, lock_images, new_id, deployment_threshold
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
    integrity = config["integrity"]
    score_gates = competition_score_gates(
        base_test_map50,
        float(new_metrics["map50"]),
        krr,
        acceptance,
    )
    integrity_gates = {
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
        "prototype_incremental_only": (
            positive_prototype is None
            or positive_prototype.get("learning_data_scope") == "incremental_dataset_only"
        ),
        "context_prior_incremental_train_only": (
            not context_prior
            or context_prior.get("source_split") == "incremental_train_only"
        ),
        "evaluator_consistency": evaluator_error <= float(integrity["max_evaluator_error"]),
        "base_weight_unchanged": base_weight_drift <= float(integrity["max_base_weight_drift"]),
        "old_channel_isolation": old_channel_drift <= float(config.get("adaptation", {}).get("max_old_channel_drift", 1e-6)),
        "old_owner_prediction_equivalence": (
            old_prediction_equivalent if not unified_student else True
        ),
        "all_owners_every_lock_image": bool(
            lock_inference_audit["base_input_stems_equal_mixed_test"]
            and lock_inference_audit["incremental_input_stems_equal_mixed_test"]
            and lock_inference_audit["base_and_incremental_input_stems_identical"]
        ),
        "label_aware_routing_disabled": not bool(
            lock_inference_audit["label_aware_routing"]
        ),
        "scene_hard_routing_disabled": not bool(
            lock_inference_audit["scene_hard_routing"]
        ),
        "prediction_artifacts_frozen": all(
            bool(item.get("sha256"))
            for item in lock_prediction_artifacts.values()
        ),
        "lock_after_unlabeled_prediction_freeze": bool(
            dataset_manifest["lock_materialized_after_freeze"]
            and lock_inference_audit[
                "unlabeled_inference_completed_before_lock_labels"
            ]
        ),
        "declared_base_test_after_prediction_freeze": bool(
            lock_inference_audit["base_test_membership_read_after_prediction_freeze"]
            and lock_inference_audit["base_test_is_subset_of_mixed_test"]
        ),
    }
    diagnostic_gates = {
        "base_dev_map50": base_dev_map50 >= float(acceptance["min_base_map50"]),
    }
    diagnostics = dict(config.get("diagnostics", {}))
    if "min_lock_precision" in diagnostics:
        diagnostic_gates["lock_precision"] = float(lock_pr["precision"]) >= float(
            diagnostics["min_lock_precision"]
        )
    if "max_false_activation_rate" in diagnostics:
        diagnostic_gates["false_activation_rate"] = float(
            false_activation["false_activation_rate"]
        ) <= float(diagnostics["max_false_activation_rate"])
    deployment_acceptance = dict(config.get("deployment_acceptance", {}))
    deployment_gates = {
        "calibration_target_precision": bool(calibration["passed"]),
        "new_class_precision": float(lock_pr["precision"])
        >= float(deployment_acceptance["min_new_class_precision"]),
        "new_class_false_activation_rate": float(false_activation["false_activation_rate"])
        <= float(deployment_acceptance["max_new_class_false_activation_rate"]),
    }
    competition_gates = {**integrity_gates, **score_gates}
    competition_accepted = all(competition_gates.values())
    deployment_accepted = competition_accepted and all(deployment_gates.values())
    gates = {**competition_gates, **deployment_gates}
    accepted = deployment_accepted
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
        "agent_structure": dict(config.get("agent_structure", {})),
        "inference_image_sizes": {
            "base": owner_inference_imgsz(config, "base"),
            "incremental": owner_inference_imgsz(config, "incremental"),
        },
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
        "base_dev_map50": base_dev_map50,
        "base_dev_per_class_ap50": base_dev_metrics["per_class_ap50"],
        "ultralytics_base_dev_map50": float(base_dev_reference_map50),
        "base_test_map50": base_test_map50,
        "base_test_per_class_ap50": base_test_metrics["per_class_ap50"],
        "base_test_image_count": len(base_test_ids),
        "old_map50_before": old_before,
        "old_map50_after": old_after,
        "new_map50": float(new_metrics["map50"]),
        "full_map50": float(full_metrics["map50"]),
        "per_class_ap50": full_metrics["per_class_ap50"],
        "krr": krr,
        "old_prediction_equivalent": old_prediction_equivalent,
        "evaluator_error": evaluator_error,
        "base_dev_evaluator_error": base_dev_evaluator_error,
        "incremental_dev_evaluator_error": incremental_dev_evaluator_error,
        "calibration": calibration,
        "context_gate": {
            "enabled": bool(context_prior),
            "policy": "soft_threshold_penalty",
            "dimensions": list(context_gate_cfg.get("dimensions", [])),
            "max_threshold_penalty": float(
                context_gate_cfg.get("max_threshold_penalty", 0.0)
            ),
            "prior": context_prior or None,
            "prior_artifact": (
                rel_path(context_prior_path) if context_prior_path is not None else None
            ),
            "learning_data_scope": (
                "incremental_train_only" if context_prior else None
            ),
            "lock_context_count": len(lock_contexts),
            "hard_routing": False,
        },
        "positive_prototype": {
            "enabled": positive_prototype is not None,
            "artifact": rel_path(prototype_path) if prototype_path is not None else None,
            "artifact_sha256": sha256_file(prototype_path) if prototype_path is not None else None,
            "method": positive_prototype.get("method") if positive_prototype else None,
            "distance_threshold": (
                float(positive_prototype["distance_threshold"])
                if positive_prototype else None
            ),
            "train_positive_count": (
                int(positive_prototype["train_positive_count"])
                if positive_prototype else 0
            ),
            "dev_positive_count": (
                int(positive_prototype["dev_positive_count"])
                if positive_prototype else 0
            ),
            "dev_rejected_candidate_count": len(dev_prototype_rejections),
            "lock_rejected_candidate_count": len(lock_prototype_rejections),
            "learning_data_scope": (
                positive_prototype.get("learning_data_scope")
                if positive_prototype else None
            ),
        },
        "fusion": {
            "nms_iou": float(config["fusion"]["nms_iou"]),
            "cross_class": dict(config["fusion"].get("cross_class", {})),
            "decision_count": len(fusion_decisions),
            "rejected_incremental_count": sum(
                row.get("action") == "reject_specialist" for row in fusion_decisions
            ),
            "activation_rejected_count": len(activation_rejections),
            "artifact": lock_prediction_artifacts["fusion_decisions"]["path"],
            "artifact_sha256": lock_prediction_artifacts["fusion_decisions"]["sha256"],
        },
        "lock_inference_audit": lock_inference_audit,
        "lock_prediction_artifacts": lock_prediction_artifacts,
        "lock_deployment_metrics": lock_pr,
        "false_activation": false_activation,
        "sensor_metrics": subgroup,
        "bootstrap": bootstrap,
        "base_inference_ms_total": base_inference_ms,
        "base_dev_inference_ms_total": base_dev_inference_ms,
        "specialist_inference_ms_total": 0.0 if unified_student else incremental_inference_ms,
        "student_inference_ms_total": incremental_inference_ms if unified_student else None,
        "training_seconds": time.monotonic() - started,
        "evaluation_only_recheck": recheck,
        "competition_metrics": {
            "base_test_map50": {
                "value": base_test_map50,
                "threshold": float(acceptance["min_base_map50"]),
                "split": "declared_base_test_old_only_after_mixed_prediction_freeze",
                "passed": score_gates["base_map50"],
            },
            "new_map50": {
                "value": float(new_metrics["map50"]),
                "threshold": float(acceptance["min_new_map50"]),
                "split": "mixed_test",
                "passed": score_gates["new_map50"],
            },
            "krr": {
                "value": krr,
                "threshold": float(acceptance["min_krr"]),
                "split": "mixed_test",
                "passed": score_gates["krr"],
            },
        },
        "score_gates": score_gates,
        "integrity_gates": integrity_gates,
        "deployment_gates": deployment_gates,
        "diagnostic_gates": diagnostic_gates,
        "gates": gates,
        "competition_accepted": competition_accepted,
        "deployment_accepted": deployment_accepted,
        "accepted": accepted,
        "lock_evaluation_started_after_training_and_calibration": True,
        "lock_labels_opened_after_prediction_freeze": True,
    }
    write_protocol_report(report_dir, result)
    _audit_event(
        config,
        protocol_id,
        "EVALUATED",
        base_test_map50=base_test_map50,
        new_map50=float(new_metrics["map50"]),
        krr=krr,
        full_map50=float(full_metrics["map50"]),
    )
    if accepted:
        if unified_student:
            result["profile"] = freeze_student_profile(
                config, protocol, run_id, base_weight, incremental_weight, calibration,
                report_dir, prototype_path
            )
        else:
            result["profile"] = freeze_profile(
                config, protocol, run_id, base_weight, incremental_weight, calibration,
                report_dir, prototype_path
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
        "competition_passed_protocols": [
            row["protocol"] for row in results if row.get("competition_accepted")
        ],
        "deployment_passed_protocols": [
            row["protocol"] for row in results if row.get("deployment_accepted")
        ],
        "passed_protocols": [row["protocol"] for row in results if row.get("accepted")],
        "failed_protocols": [row["protocol"] for row in results if not row.get("accepted")],
    }
    suffix = "_recheck" if recheck else ""
    path = output / f"summary{suffix}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 严格 3+1 类别增量单一协议汇总",
        "",
        "| 协议 | 新增类别 | 基础测试 mAP50 | New-mAP50 | KRR | 赛题结论 | 部署结论 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in results:
        if "error" in row:
            lines.append(
                f"| {row['protocol']} | - | - | - | - | 执行错误 | {row['error']} |"
            )
        else:
            lines.append(
                f"| {row['protocol']} | {row['new_class']} | {row['base_test_map50']:.5f} | "
                f"{row['new_map50']:.5f} | {row['krr']:.5f} | "
                f"{'通过' if row['competition_accepted'] else '未通过'} | "
                f"{'通过' if row['deployment_accepted'] else '未通过'} |"
            )
    (output / f"summary{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="运行严格 3+1 类别增量单一协议实验。")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "strict_class_incremental_3plus1.yaml")
    parser.add_argument("--run-id", help="显式指定唯一运行编号，便于多模型共享同一产物目录。")
    parser.add_argument("--check-only", action="store_true", help="只做训练环境与数据只读预检，不创建产物或启动训练。")
    parser.add_argument("--recheck-run", help=argparse.SUPPRESS)
    parser.add_argument("--recheck-base-weight", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--recheck-incremental-weight", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_yaml(config_path)
    if args.run_id and args.recheck_run:
        raise ValueError("--run-id 与 --recheck-run 不能同时使用")
    if args.check_only and args.recheck_run:
        raise ValueError("--check-only 与 --recheck-run 不能同时使用")
    if (args.recheck_base_weight or args.recheck_incremental_weight) and not args.recheck_run:
        raise ValueError("复核权重覆盖参数只能与 --recheck-run 同时使用")
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
                str(args.recheck_base_weight) if args.recheck_base_weight else None,
                str(args.recheck_incremental_weight) if args.recheck_incremental_weight else None,
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
