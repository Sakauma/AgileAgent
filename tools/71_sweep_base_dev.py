#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import load_yaml, read_split, sha256_file


PROTECTED_TRAIN_KEYS = {
    "data",
    "device",
    "epochs",
    "exist_ok",
    "name",
    "patience",
    "project",
}
ALLOWED_EVALUATION_OVERRIDE_KEYS = {
    "imgsz",
    "batch",
    "conf",
    "iou",
    "rect",
    "workers",
    "classes",
}


def resolve_local(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def candidate_train_arguments(
    sweep: Mapping[str, Any],
    strict: Mapping[str, Any],
    candidate_name: str,
    dataset: Path,
    project: Path,
    device: str,
) -> Dict[str, Any]:
    candidates = sweep.get("candidates", {})
    if candidate_name not in candidates:
        raise ValueError(f"未知基础超参候选：{candidate_name}")
    candidate = dict(candidates[candidate_name])
    overrides = dict(candidate.get("overrides", {}))
    forbidden = sorted(PROTECTED_TRAIN_KEYS.intersection(overrides))
    if forbidden:
        raise ValueError(f"候选不得覆盖完整训练与数据隔离参数：{forbidden}")

    arguments = dict(strict["common"])
    arguments.update(dict(strict["base_train"]))
    arguments.update(overrides)
    arguments.update(
        {
            "data": str(dataset),
            "project": str(project),
            "name": candidate_name,
            "device": str(device),
            "workers": int(strict["runtime"].get("workers", 8)),
            "seed": int(candidate.get("seed", strict["seed"])),
            "epochs": int(sweep["epochs"]),
            "patience": 0,
            "exist_ok": False,
        }
    )
    if int(arguments["epochs"]) != int(strict["base_train"]["epochs"]):
        raise ValueError("基础超参候选必须使用正式基础训练的完整 epoch 预算")
    if int(arguments["patience"]) != 0:
        raise ValueError("基础超参候选必须设置 patience=0")
    return arguments


def candidate_evaluation_settings(
    sweep: Mapping[str, Any], candidate_name: str
) -> Dict[str, Any]:
    candidate = dict(sweep.get("candidates", {}).get(candidate_name, {}))
    if not candidate:
        raise ValueError(f"未知基础超参候选：{candidate_name}")
    overrides = dict(candidate.get("evaluation_overrides", {}))
    unknown = sorted(set(overrides) - ALLOWED_EVALUATION_OVERRIDE_KEYS)
    if unknown:
        raise ValueError(f"候选包含未知 evaluation override：{unknown}")
    evaluation = dict(sweep["evaluation"])
    evaluation.update(overrides)
    return evaluation


def audit_base_dataset(dataset: Path) -> Dict[str, Any]:
    if not dataset.is_file():
        raise FileNotFoundError(f"基础 dataset.yaml 不存在：{dataset}")
    payload = yaml.safe_load(dataset.read_text(encoding="utf-8")) or {}
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    if "mixed_test" in serialized or "lock" in serialized:
        raise ValueError("基础 dev 调参数据路径疑似引用 mixed_test/lock")

    split_root = dataset.parent / "splits"
    train_split = split_root / "train.txt"
    val_split = split_root / "val.txt"
    train_images = read_split(train_split)
    val_images = read_split(val_split)
    train_stems = {path.stem for path in train_images}
    val_stems = {path.stem for path in val_images}
    overlap = sorted(train_stems.intersection(val_stems))
    if not train_images or not val_images or overlap:
        raise ValueError("基础 train/dev 必须非空且互斥")
    return {
        "dataset_yaml": str(dataset),
        "dataset_yaml_sha256": sha256_file(dataset),
        "train_count": len(train_images),
        "dev_count": len(val_images),
        "train_dev_overlap": overlap,
        "source_declared_test_split": "test" in payload,
        "training_declared_test_split": False,
        "lock_data_access": False,
    }


def write_train_dev_only_dataset(source: Path, destination: Path) -> Path:
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    sanitized = {
        key: value
        for key, value in payload.items()
        if key in {"path", "train", "val", "names", "nc"}
    }
    if "train" not in sanitized or "val" not in sanitized or "names" not in sanitized:
        raise ValueError("基础 dataset.yaml 缺少 train/val/names")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖基础 train/dev 清单：{destination}")
    destination.write_text(
        yaml.safe_dump(sanitized, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def metric_per_class(metrics: Any) -> Dict[str, float]:
    values = metrics.box.ap50.tolist()
    class_ids = metrics.box.ap_class_index.tolist()
    return {str(int(class_id)): float(value) for class_id, value in zip(class_ids, values)}


def validate_lineage_selection(
    path: Path, candidate_name: str, arguments: Mapping[str, Any]
) -> Dict[str, Any]:
    resolved = path.resolve()
    lowered = str(resolved).replace("\\", "/").lower()
    if any(marker in lowered for marker in ("mixed_test", "base_test", "lock")):
        raise ValueError("前向后置验证 lineage 不得引用 test/lock")
    selection = json.loads(resolved.read_text(encoding="utf-8"))
    selected = dict(selection.get("selected", {}))
    selection_scope = str(selection.get("selection_scope", ""))
    if selection_scope == "base_train_and_dev_forward_tuning_only":
        sealed = dict(selection.get("post_validation", {}))
        sealed_valid = (
            str(sealed.get("status", "")) == "sealed"
            and not bool(sealed.get("labels_opened", True))
            and bool(
                sealed.get(
                    "candidate_must_be_frozen_before_training_and_scoring",
                    False,
                )
            )
        )
    elif selection_scope == "base_train_and_dev_rolling_class_expert_selection":
        sealed = dict(selection.get("regression_window", {}))
        regression_status = str(sealed.get("status", ""))
        sealed_valid = (
            regression_status in {"sealed", "reused_not_independent"}
            and bool(sealed.get("labels_opened", False))
            == (regression_status == "reused_not_independent")
            and bool(sealed.get("must_not_participate_in_candidate_selection", False))
            and bool(selection.get("candidate_selected_without_regression_reports", False))
            and bool(selection.get("all_unknown_images_must_execute_expert", False))
            and not bool(selection.get("image_level_class_routing", True))
        )
    else:
        sealed_valid = False
    if (
        not sealed_valid
        or bool(selection.get("lock_data_access", True))
        or str(selected.get("name", "")) != candidate_name
    ):
        raise ValueError("后置验证没有使用已冻结且无图片路由的调参候选")
    selected_arguments = dict(selected.get("training_overrides", {}))
    for key, expected in selected_arguments.items():
        if expected is not None and arguments.get(key) != expected:
            raise ValueError(f"前向后置验证训练参数漂移：{key}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "selected_candidate": candidate_name,
        "selection_scope": selection_scope,
        "post_validation_was_sealed": str(sealed.get("status", "")) == "sealed",
        "regression_status": str(sealed.get("status", "")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="仅使用基础 train/dev、完整 epoch 预算运行一个基础超参候选。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "base_dev_hparam_sweep.yaml",
    )
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--lineage-selection-report",
        type=Path,
        help="后置前向验证必须引用已冻结且尚未打开后置标签的选择报告。",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", args.sweep_id):
        raise ValueError("sweep-id 只能包含字母、数字、点、下划线和连字符")
    if not re.fullmatch(r"\d+", str(args.device)):
        raise ValueError("每个基础超参候选必须绑定一个明确的单卡 device")

    sweep_path = args.config if args.config.is_absolute() else ROOT / args.config
    sweep = load_yaml(sweep_path)
    strict = load_yaml(resolve_local(sweep["strict_config"]))
    source_dataset = args.dataset.resolve()
    audit = audit_base_dataset(source_dataset)
    project = resolve_local(sweep["project_root"]) / args.sweep_id
    report_dir = resolve_local(sweep["report_root"]) / args.sweep_id
    report_path = report_dir / f"{args.candidate}.json"
    run_dir = project / args.candidate
    if report_path.exists() or run_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有基础调参候选：{args.candidate}")
    dataset = write_train_dev_only_dataset(
        source_dataset,
        project / "_train_dev_only" / f"{args.candidate}.yaml",
    )
    audit["training_dataset_yaml"] = str(dataset)
    audit["training_dataset_yaml_sha256"] = sha256_file(dataset)

    arguments = candidate_train_arguments(
        sweep,
        strict,
        args.candidate,
        dataset,
        project,
        str(args.device),
    )
    lineage_selection = (
        validate_lineage_selection(
            args.lineage_selection_report, args.candidate, arguments
        )
        if args.lineage_selection_report
        else None
    )
    for key, value in strict["runtime"].get("env", {}).items():
        os.environ[str(key)] = str(value)

    runner = runpy.run_path(str(ROOT / "tools" / "70_run_strict_3plus1.py"))
    from ultralytics import YOLO

    model_path = resolve_local(sweep.get("model", strict["model"]))
    model = YOLO(str(model_path))
    model.add_callback("on_pretrain_routine_end", runner["configure_map50_checkpointing"])
    train_result = model.train(**arguments)
    best_weight = runner["best_weight"](model, train_result)
    history = runner["training_history"](
        model,
        f"base_dev_sweep:{args.candidate}",
        int(sweep["epochs"]),
        require_full_epochs=True,
    )

    evaluation = candidate_evaluation_settings(sweep, args.candidate)
    evaluator = YOLO(str(best_weight))
    evaluation_arguments: Dict[str, Any] = {
        "data": str(dataset),
        "split": "val",
        "imgsz": int(evaluation["imgsz"]),
        "batch": int(evaluation["batch"]),
        "conf": float(evaluation["conf"]),
        "iou": float(evaluation["iou"]),
        "rect": bool(evaluation["rect"]),
        "workers": int(evaluation["workers"]),
        "device": str(args.device),
        "project": str(project / "evaluation"),
        "name": args.candidate,
        "plots": False,
        "verbose": False,
        "exist_ok": False,
    }
    class_filter = evaluation.get("classes")
    if class_filter is not None:
        evaluation_arguments["classes"] = [int(value) for value in class_filter]
    metrics = evaluator.val(**evaluation_arguments)
    result = {
        "schema_version": 1,
        "sweep_id": args.sweep_id,
        "candidate": args.candidate,
        "selection_scope": "base_train_and_dev_only",
        "lock_data_access": False,
        "lineage_selection": lineage_selection,
        "dataset_audit": audit,
        "training": {
            "requested_epochs": int(history["requested_epochs"]),
            "completed_epochs": int(history["completed_epochs"]),
            "stopped_early": bool(history["stopped_early"]),
            "checkpoint_metric": history["checkpoint_metric"],
            "best_epoch": history["best_epoch"],
            "best_metric_value": history["best_metric_value"],
            "arguments": arguments,
        },
        "evaluation": {
            "imgsz": int(evaluation["imgsz"]),
            "class_filter": evaluation_arguments.get("classes"),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "per_class_ap50": metric_per_class(metrics),
        },
        "best_weight": str(best_weight),
        "best_weight_sha256": sha256_file(best_weight),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
