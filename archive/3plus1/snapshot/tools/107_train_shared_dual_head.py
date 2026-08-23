#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import resolve_path  # noqa: E402
from fair_agent.core.hashes import sha256_file  # noqa: E402
from fair_agent.modules.shared_dual_head import (  # noqa: E402
    ResidualAdaptedDetect,
    SharedBackboneFreezeGuard,
    build_shared_head_training_checkpoint,
    configure_map50_checkpointing,
    residual_adapter_detection_trainer,
)


def _dataset_audit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "old_raw_image_count": 0,
        "old_raw_label_count": 0,
        "old_feature_cache_count": 0,
        "original_data_modified": False,
    }
    actual = {key: payload.get(key) for key in required}
    if actual != required:
        raise RuntimeError(f"P10增量数据隔离失败：{actual} != {required}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        **actual,
    }


def _training_contract(path: Path) -> tuple[dict, dict]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != "ascend310b_full_score_method"
    ):
        raise ValueError(f"满分方法配置非法：{path}")
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError(f"满分方法配置缺少training契约：{path}")
    required = {
        "method",
        "checkpoint_metric",
        "export_checkpoints",
        "input_size",
        "epochs",
        "batch",
        "workers",
        "seed",
        "optimizer",
        "lr0",
        "lrf",
        "weight_decay",
        "deterministic",
        "patience",
        "cache",
        "plots",
        "amp",
        "cos_lr",
        "warmup_epochs",
        "warmup_bias_lr",
        "close_mosaic",
        "mosaic",
        "mixup",
        "copy_paste",
        "multi_scale",
        "degrees",
        "translate",
        "scale",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "freeze",
    }
    missing = sorted(required - set(training))
    if missing:
        raise ValueError(f"满分训练契约缺少字段：{','.join(missing)}")
    if training["checkpoint_metric"] != "map50" or training[
        "export_checkpoints"
    ] != ["best", "last"]:
        raise ValueError("满分训练契约必须按map50保存并授权best/last")
    return payload, training


def _locked_option(value, configured, option: str):
    if value is not None and value != configured:
        raise ValueError(f"{option}必须与满分方法配置一致：{value} != {configured}")
    return configured


def main() -> int:
    parser = argparse.ArgumentParser(
        description="只读取warship增量数据，冻结Base backbone/neck并训练P10新类head。"
    )
    parser.add_argument("--base-weight", required=True)
    parser.add_argument("--specialist-weight", required=True)
    parser.add_argument("--head-init-weight")
    parser.add_argument(
        "--method-config",
        default=str(ROOT / "configs/ascend310b/full_score_method.yaml"),
    )
    parser.add_argument(
        "--method",
        choices=("head_only", "residual_adapter"),
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--warmup-epochs", type=float)
    args = parser.parse_args()

    method_config = resolve_path(args.method_config)
    method_config_payload, training = _training_contract(method_config)
    new_map50_min = float(
        method_config_payload["competition"]["accuracy_gates"]["new_map50_min"]
    )
    training_method = str(_locked_option(args.method, training["method"], "--method"))
    epochs = int(_locked_option(args.epochs, training["epochs"], "--epochs"))
    batch = int(_locked_option(args.batch, training["batch"], "--batch"))
    workers = int(_locked_option(args.workers, training["workers"], "--workers"))
    seed = int(_locked_option(args.seed, training["seed"], "--seed"))
    lr0 = float(_locked_option(args.lr0, training["lr0"], "--lr0"))
    warmup_epochs = float(
        _locked_option(
            args.warmup_epochs,
            training["warmup_epochs"],
            "--warmup-epochs",
        )
    )

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("P10新类head训练要求WSL现有CUDA虚拟环境")
    artifact_dir = resolve_path(args.artifact_dir)
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise RuntimeError(f"输出目录非空，拒绝覆盖：{artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base_weight = resolve_path(args.base_weight)
    specialist_weight = resolve_path(args.specialist_weight)
    head_init_weight = (
        resolve_path(args.head_init_weight)
        if args.head_init_weight
        else specialist_weight
    )
    data = resolve_path(args.data)
    dataset_audit = _dataset_audit(resolve_path(args.dataset_manifest))
    initialization = build_shared_head_training_checkpoint(
        base_weight,
        head_init_weight,
        artifact_dir / "head-training-init.pt",
    )
    guard = SharedBackboneFreezeGuard(base_weight)
    model = YOLO(initialization["path"])

    def configure(trainer):
        configure_map50_checkpointing(trainer)
        guard.restore(trainer)

    model.add_callback("on_pretrain_routine_end", configure)
    model.add_callback("on_train_batch_start", guard.batch_start)
    model.add_callback("on_train_epoch_end", guard.restore)
    trainer_class = (
        residual_adapter_detection_trainer()
        if training_method == "residual_adapter"
        else None
    )
    train_options = dict(
        data=str(data),
        project=str(resolve_path(args.run_root)),
        name=str(args.run_name),
        exist_ok=False,
        device=str(args.device),
        workers=workers,
        seed=seed,
        deterministic=bool(training["deterministic"]),
        imgsz=int(training["input_size"]),
        batch=batch,
        epochs=epochs,
        optimizer=str(training["optimizer"]),
        patience=int(training["patience"]),
        cache=bool(training["cache"]),
        plots=bool(training["plots"]),
        amp=bool(training["amp"]),
        weight_decay=float(training["weight_decay"]),
        lr0=lr0,
        lrf=float(training["lrf"]),
        cos_lr=bool(training["cos_lr"]),
        warmup_epochs=warmup_epochs,
        warmup_bias_lr=float(training["warmup_bias_lr"]),
        close_mosaic=int(training["close_mosaic"]),
        mosaic=float(training["mosaic"]),
        mixup=float(training["mixup"]),
        copy_paste=float(training["copy_paste"]),
        multi_scale=float(training["multi_scale"]),
        degrees=float(training["degrees"]),
        translate=float(training["translate"]),
        scale=float(training["scale"]),
        fliplr=float(training["fliplr"]),
        hsv_h=float(training["hsv_h"]),
        hsv_s=float(training["hsv_s"]),
        hsv_v=float(training["hsv_v"]),
        freeze=int(training["freeze"]),
        verbose=False,
    )
    if trainer_class is not None:
        train_options["trainer"] = trainer_class
    model.train(**train_options)
    adapter_report = getattr(model.trainer, "residual_adapter_report", None)
    if training_method == "residual_adapter":
        if (
            not isinstance(model.trainer.model.model[-1], ResidualAdaptedDetect)
            or not isinstance(adapter_report, dict)
            or not adapter_report.get("zero_initialized")
        ):
            raise RuntimeError("P10 residual adapter未进入实际训练图")
    elif adapter_report is not None:
        raise RuntimeError("P10 head-only候选意外包含residual adapter")
    checkpoint_paths = {
        "best": Path(model.trainer.best).resolve(),
        "last": Path(model.trainer.last).resolve(),
    }
    checkpoint_rows = {}
    for label in training["export_checkpoints"]:
        checkpoint = checkpoint_paths[label]
        if not checkpoint.is_file():
            raise RuntimeError(f"P10缺少{label} checkpoint：{checkpoint}")
        checkpoint_model = YOLO(str(checkpoint)).model
        if (training_method == "residual_adapter") != isinstance(
            checkpoint_model.model[-1], ResidualAdaptedDetect
        ):
            raise RuntimeError(f"P10 {label} checkpoint与训练方法结构不一致")
        drift = guard.weight_drift(checkpoint)
        if drift != 0.0:
            raise RuntimeError(
                f"P10 {label}共享骨干/neck/BN/EMA发生漂移：{drift}"
            )
        metrics = YOLO(str(checkpoint)).val(
            data=str(data),
            split="test",
            imgsz=int(training["input_size"]),
            batch=max(1, batch),
            device=str(args.device),
            workers=workers,
            plots=False,
            save_json=False,
            verbose=False,
            project=str(artifact_dir),
            name=f"test-score-{label}",
            exist_ok=False,
        )
        checkpoint_rows[label] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "shared_max_drift": drift,
            "new_map50": float(metrics.box.map50),
            "new_map50_passed": float(metrics.box.map50) >= new_map50_min,
            "diagnostics": {
                "map50_95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            },
        }
    best = checkpoint_paths["best"]
    last = checkpoint_paths["last"]
    best_result = checkpoint_rows["best"]
    report = {
        "schema_version": 2,
        "kind": "shared_backbone_dual_head_training",
        "method": training_method,
        "method_config": {
            "path": str(method_config),
            "sha256": sha256_file(method_config),
        },
        "training_contract": training,
        "formal_specialist_weight": str(specialist_weight),
        "formal_specialist_weight_sha256": sha256_file(specialist_weight),
        "initialization": initialization,
        "residual_adapter": adapter_report,
        "dataset": str(data),
        "dataset_audit": dataset_audit,
        "run_root": str(resolve_path(args.run_root)),
        "run_name": str(args.run_name),
        "epochs": epochs,
        "lr0": lr0,
        "warmup_epochs": warmup_epochs,
        "imgsz": int(training["input_size"]),
        "best_weight": str(best),
        "best_weight_sha256": sha256_file(best),
        "last_weight": str(last),
        "last_weight_sha256": sha256_file(last),
        "checkpoints": checkpoint_rows,
        "shared_max_drift": max(
            float(row["shared_max_drift"]) for row in checkpoint_rows.values()
        ),
        "new_map50": float(best_result["new_map50"]),
        "new_map50_passed": bool(best_result["new_map50_passed"]),
        "any_export_checkpoint_new_map50_passed": any(
            bool(row["new_map50_passed"]) for row in checkpoint_rows.values()
        ),
        "diagnostics": best_result["diagnostics"],
    }
    report_path = artifact_dir / "training-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**report, "report_sha256": sha256_file(report_path)}, ensure_ascii=False, indent=2))
    return 0 if report["any_export_checkpoint_new_map50_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
