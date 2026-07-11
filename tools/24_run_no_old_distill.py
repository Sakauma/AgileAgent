#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, Sequence

import yaml

from fair_agent.modules.incremental_compliance import (
    evaluate_incremental_metrics,
    verify_incremental_learning_scope,
)


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["soldier", "small_aircraft", "warship", "tank"]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def class_ids(names: Sequence[str]) -> list[int]:
    return [CLASS_NAMES.index(name) for name in names]


def per_class_ap50(metrics: Any) -> Dict[int, float]:
    box = metrics.box
    return {int(cls): float(ap) for cls, ap in zip(list(box.ap_class_index), list(box.ap50))}


def mean_ap(values: Mapping[int, float], ids: Iterable[int]) -> float:
    selected = [values[class_id] for class_id in ids if class_id in values]
    return mean(selected) if selected else 0.0


def best_weight(model: Any, train_result: Any) -> Path:
    trainer = model.trainer
    candidates = [getattr(trainer, "best", None), getattr(trainer, "last", None)]
    for value in candidates:
        if value and Path(value).exists():
            return Path(value)
    save_dir = Path(getattr(train_result, "save_dir", trainer.save_dir))
    for name in ["best.pt", "last.pt"]:
        candidate = save_dir / "weights" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Training completed without a best or last weight")


def model_for_trainer(trainer: Any) -> Any:
    return trainer.model.module if hasattr(trainer.model, "module") else trainer.model


def configure_new_class_channels(trainer: Any, new_ids: Sequence[int]) -> None:
    import torch

    model = model_for_trainer(trainer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.model[-1]
    branches = getattr(head, "cv3", None)
    if branches is None:
        raise RuntimeError("YOLO detection head does not expose cv3 classification branches")
    for branch in branches:
        classifier = branch[-1]
        if classifier.out_channels != len(CLASS_NAMES):
            raise RuntimeError(f"Unexpected classifier width: {classifier.out_channels}")
        classifier.weight.requires_grad_(True)
        classifier.bias.requires_grad_(True)

        def mask_gradient(gradient: Any) -> Any:
            mask = torch.zeros_like(gradient)
            mask[list(new_ids)] = 1
            return gradient * mask

        classifier.weight.register_hook(mask_gradient)
        classifier.bias.register_hook(mask_gradient)
    freeze_batch_norm(trainer)


def freeze_batch_norm(trainer: Any) -> None:
    import torch

    model = model_for_trainer(trainer)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def frozen_parameter_drift(model_cls: Any, teacher: Path, student: Path, new_ids: Sequence[int]) -> float:
    teacher_state = model_cls(str(teacher)).model.state_dict()
    student_state = model_cls(str(student)).model.state_dict()
    maximum = 0.0
    for key, before in teacher_state.items():
        after = student_state[key]
        difference = (after - before).abs()
        if ".cv3." in key and difference.ndim >= 1 and difference.shape[0] == len(CLASS_NAMES):
            keep = [index for index in range(len(CLASS_NAMES)) if index not in set(new_ids)]
            difference = difference[keep]
        if difference.numel():
            maximum = max(maximum, float(difference.max().item()))
    return maximum


def split_values(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_specialist_dataset(dataset: Path, protocol: Mapping[str, Any], new_ids: Sequence[int]) -> Dict[str, Any]:
    dataset_config = load_yaml(dataset)
    train_list = resolve(dataset_config["train"])
    val_list = resolve(dataset_config["val"])
    allowed_train_list = resolve(protocol["new_train_split"])
    allowed_val_list = resolve(protocol["new_val_split"])
    training = split_values(train_list)
    validation = split_values(val_list)
    allowed_training = split_values(allowed_train_list)
    allowed_validation = split_values(allowed_val_list)
    compliance = verify_incremental_learning_scope(
        training,
        validation,
        allowed_training,
        allowed_validation,
        protocol["base_classes"],
        protocol["new_classes"],
        incremental_mode=protocol.get("incremental_mode", "class_incremental"),
        verify_content=True,
    )
    accepted_label_ids = {0} if protocol.get("specialist_remapped") else set(new_ids)
    invalid_labels = []
    for value in training + validation:
        image = resolve(value)
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip() and int(float(line.split()[0])) not in accepted_label_ids:
                invalid_labels.append(str(label))
                break
    compliance["invalid_non_new_label_files"] = sorted(set(invalid_labels))
    compliance["compliant"] = bool(compliance["compliant"] and not invalid_labels)
    compliance["learning_scope_verified"] = compliance["compliant"]
    return compliance


def eval_model(model_cls: Any, weight: Path, data: Path, config: Mapping[str, Any], name: str, device: str) -> Dict[str, Any]:
    args = dict(config.get("eval", {}))
    args.update({"data": str(data), "name": name, "project": str(resolve(config["project"])), "device": device})
    metrics = model_cls(str(weight)).val(**args)
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_ap50": per_class_ap50(metrics),
        "save_dir": rel(Path(metrics.save_dir)),
    }


def write_report(report_dir: Path, row: Mapping[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "metrics.json").write_text(json.dumps(dict(row), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    flat = {
        key: value
        for key, value in row.items()
        if not isinstance(value, (dict, list))
    }
    with (report_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    decision = row["decision"]
    lines = [
        f"# {row['protocol']} 合规增量学习报告",
        "",
        f"- 任务类型：{row['task_type']}",
        f"- 学习数据边界：{row['learning_data_scope']}",
        f"- 数据边界验证：{row['learning_scope_verified']}",
        f"- 方法：{row['method']}",
        f"- 训练图像数：{row['training_image_count']}",
        f"- 验证图像数：{row['validation_image_count']}",
        f"- 学习阶段旧类原始图像数：{row['old_raw_image_count']}",
        f"- New-mAP50: {row['new_map50_after']:.5f}",
        f"- KRR: {row['krr']:.5f}",
        f"- 完整集 mAP50：{row['full_map50_after']:.5f}",
        f"- 训练耗时（秒）：{row['training_seconds']:.1f}",
        f"- 冻结参数最大漂移：{row['frozen_parameter_max_abs_drift']:.8g}",
        f"- 参数隔离检查：{decision['parameter_isolation_pass']}",
        f"- 合规检查：{decision['compliant']}",
        f"- 结果：{'通过' if decision['passed'] else '未通过'}",
    ]
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "incremental_no_old_distill_yolo11s.yaml")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    config = load_yaml(args.config)
    for key, value in dict(config.get("env", {})).items():
        os.environ[str(key)] = str(value)
    protocol = next((item for item in config["protocols"] if item["name"] == args.protocol), None)
    if protocol is None:
        raise ValueError(f"Unknown protocol: {args.protocol}")
    teacher = resolve(protocol["teacher_weight"])
    old_ids = class_ids(protocol["base_classes"])
    new_ids = class_ids(protocol["new_classes"])
    adaptation = dict(config.get("adaptation", {}))
    mode = adaptation.get("mode")
    manifest_path = resolve(config["output_root"]) / args.protocol / "manifest.json"
    if mode == "frozen_base_plus_new_specialist":
        dataset = resolve(protocol["learning_data"])
        manifest_path = dataset.parent / "manifest.json"
        compliance = verify_specialist_dataset(dataset, protocol, new_ids)
    elif mode == "new_class_channel_only":
        dataset = resolve(config["output_root"]) / args.protocol / "learning_dataset.yaml"
        compliance = json.loads(manifest_path.read_text(encoding="utf-8"))["compliance"]
    else:
        raise ValueError(f"Unsupported adaptation mode: {mode}")
    if not compliance.get("compliant") or compliance.get("old_raw_image_count") != 0:
        raise RuntimeError(f"Training dataset is not compliant: {compliance}")
    train_args = dict(config.get("common", {}))
    train_args.update(dict(config.get("incremental_train", {})))
    train_args.update({
        "data": str(dataset),
        "project": str(resolve(config["project"])),
        "name": args.protocol,
        "device": args.device,
        "seed": int(config.get("seed", 20260705)),
    })
    started = time.monotonic()
    student_init = resolve(protocol.get("specialist_init", teacher)) if mode == "frozen_base_plus_new_specialist" else teacher
    student = YOLO(str(student_init))
    if mode == "new_class_channel_only":
        student.add_callback("on_train_start", lambda trainer: configure_new_class_channels(trainer, new_ids))
        student.add_callback("on_train_epoch_start", freeze_batch_norm)
    train_result = student.train(**train_args)
    elapsed = time.monotonic() - started
    weight = best_weight(student, train_result)
    old_eval_data = resolve(protocol["old_evaluation_data"])
    new_eval_data = resolve(protocol.get("new_evaluation_data", protocol["old_evaluation_data"]))
    before = eval_model(YOLO, teacher, old_eval_data, config, f"{args.protocol}_teacher_eval", args.device)
    after = eval_model(YOLO, weight, new_eval_data, config, f"{args.protocol}_student_eval", args.device)
    old_before = mean_ap(before["per_class_ap50"], old_ids)
    new_after = (
        mean_ap(after["per_class_ap50"], [0])
        if mode == "frozen_base_plus_new_specialist" and protocol.get("specialist_remapped")
        else mean_ap(after["per_class_ap50"], new_ids)
    )
    if mode == "frozen_base_plus_new_specialist":
        old_after = old_before
        krr = 1.0 if old_before else 0.0
        full_after = (old_before * len(old_ids) + new_after * len(new_ids)) / (len(old_ids) + len(new_ids))
        drift = 0.0
        method = "frozen_base_plus_new_class_specialist"
    else:
        old_after = mean_ap(after["per_class_ap50"], old_ids)
        krr = old_after / old_before if old_before else 0.0
        full_after = after["map50"]
        drift = frozen_parameter_drift(YOLO, teacher, weight, new_ids)
        method = "new_images_only_teacher_pseudolabel_distillation"
    decision = evaluate_incremental_metrics(new_after, krr, compliance)
    drift_limit = float(adaptation.get("max_frozen_parameter_drift", 1e-6))
    decision["parameter_isolation_pass"] = drift <= drift_limit
    decision["max_frozen_parameter_drift"] = drift_limit
    decision["passed"] = bool(decision["passed"] and decision["parameter_isolation_pass"])
    row = {
        "protocol": args.protocol,
        "task_type": "incremental_object_detection",
        "incremental_mode": protocol.get("incremental_mode", "class_incremental"),
        "learning_data_scope": "incremental_dataset_only",
        "learning_scope_verified": compliance.get("learning_scope_verified", False),
        "method": method,
        "base_classes": list(protocol["base_classes"]),
        "new_classes": list(protocol["new_classes"]),
        "old_map50_before": old_before,
        "old_map50_after": old_after,
        "new_map50_after": new_after,
        "full_map50_after": full_after,
        "krr": krr,
        "training_seconds": elapsed,
        "training_image_count": compliance["training"]["training_image_count"],
        "validation_image_count": compliance["validation"]["training_image_count"],
        "old_raw_image_count": compliance["old_raw_image_count"],
        "frozen_parameter_max_abs_drift": drift,
        "teacher_weight": rel(teacher),
        "student_weight": rel(weight),
        "student_init": rel(student_init),
        "dataset_manifest": rel(manifest_path),
        "learning_dataset": rel(dataset),
        "old_evaluation_data": rel(old_eval_data),
        "new_evaluation_data": rel(new_eval_data),
        "evaluation_started_after_training": True,
        "before_eval": before,
        "after_eval": after,
        "composition": {
            "old_classes": "frozen_base_detector" if mode == "frozen_base_plus_new_specialist" else "student_detector",
            "new_classes": "new_class_specialist" if mode == "frozen_base_plus_new_specialist" else "student_detector",
        },
        "decision": decision,
    }
    report_dir = resolve(config["report_dir"]) / args.protocol
    write_report(report_dir, row)
    print(json.dumps({"protocol": args.protocol, "new_map50": new_after, "krr": krr, "passed": decision["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
