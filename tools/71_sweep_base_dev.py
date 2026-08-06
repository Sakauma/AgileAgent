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

    evaluation = dict(sweep["evaluation"])
    evaluator = YOLO(str(best_weight))
    metrics = evaluator.val(
        data=str(dataset),
        split="val",
        imgsz=int(evaluation["imgsz"]),
        batch=int(evaluation["batch"]),
        conf=float(evaluation["conf"]),
        iou=float(evaluation["iou"]),
        rect=bool(evaluation["rect"]),
        workers=int(evaluation["workers"]),
        device=str(args.device),
        project=str(project / "evaluation"),
        name=args.candidate,
        plots=False,
        verbose=False,
        exist_ok=False,
    )
    result = {
        "schema_version": 1,
        "sweep_id": args.sweep_id,
        "candidate": args.candidate,
        "selection_scope": "base_train_and_dev_only",
        "lock_data_access": False,
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
