#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="只读取warship增量数据，冻结Base backbone/neck并训练P10新类head。"
    )
    parser.add_argument("--base-weight", required=True)
    parser.add_argument("--specialist-weight", required=True)
    parser.add_argument("--head-init-weight")
    parser.add_argument(
        "--method",
        choices=("head_only", "residual_adapter"),
        default="head_only",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--lr0", type=float, default=0.003)
    parser.add_argument("--warmup-epochs", type=float)
    args = parser.parse_args()

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
        if args.method == "residual_adapter"
        else None
    )
    warmup_epochs = (
        float(args.warmup_epochs)
        if args.warmup_epochs is not None
        else (0.0 if args.method == "residual_adapter" else 3.0)
    )
    train_options = dict(
        data=str(data),
        project=str(resolve_path(args.run_root)),
        name=str(args.run_name),
        exist_ok=False,
        device=str(args.device),
        workers=int(args.workers),
        seed=int(args.seed),
        deterministic=True,
        imgsz=896,
        batch=int(args.batch),
        epochs=int(args.epochs),
        optimizer="AdamW",
        patience=0,
        cache=False,
        plots=True,
        amp=True,
        weight_decay=0.0005,
        lr0=float(args.lr0),
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=warmup_epochs,
        warmup_bias_lr=0.0 if warmup_epochs == 0.0 else 0.1,
        close_mosaic=10,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        multi_scale=0.0,
        degrees=3.0,
        translate=0.15,
        scale=0.50,
        fliplr=0.50,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.30,
        freeze=23,
        verbose=False,
    )
    if trainer_class is not None:
        train_options["trainer"] = trainer_class
    model.train(**train_options)
    adapter_report = getattr(model.trainer, "residual_adapter_report", None)
    if args.method == "residual_adapter":
        if (
            not isinstance(model.trainer.model.model[-1], ResidualAdaptedDetect)
            or not isinstance(adapter_report, dict)
            or not adapter_report.get("zero_initialized")
        ):
            raise RuntimeError("P10 residual adapter未进入实际训练图")
    elif adapter_report is not None:
        raise RuntimeError("P10 head-only候选意外包含residual adapter")
    best = Path(model.trainer.best).resolve()
    best_model = YOLO(str(best)).model
    if (args.method == "residual_adapter") != isinstance(
        best_model.model[-1], ResidualAdaptedDetect
    ):
        raise RuntimeError("P10 best checkpoint与训练方法结构不一致")
    drift = guard.weight_drift(best)
    if drift != 0.0:
        raise RuntimeError(f"P10共享骨干/neck/BN/EMA发生漂移：{drift}")
    metrics = YOLO(str(best)).val(
        data=str(data),
        split="test",
        imgsz=896,
        batch=max(1, int(args.batch)),
        device=str(args.device),
        workers=int(args.workers),
        plots=False,
        save_json=False,
        verbose=False,
        project=str(artifact_dir),
        name="test-score",
        exist_ok=False,
    )
    report = {
        "schema_version": 1,
        "kind": "shared_backbone_dual_head_training",
        "method": str(args.method),
        "formal_specialist_weight": str(specialist_weight),
        "formal_specialist_weight_sha256": sha256_file(specialist_weight),
        "initialization": initialization,
        "residual_adapter": adapter_report,
        "dataset": str(data),
        "dataset_audit": dataset_audit,
        "run_root": str(resolve_path(args.run_root)),
        "run_name": str(args.run_name),
        "epochs": int(args.epochs),
        "lr0": float(args.lr0),
        "warmup_epochs": warmup_epochs,
        "imgsz": 896,
        "best_weight": str(best),
        "best_weight_sha256": sha256_file(best),
        "shared_max_drift": drift,
        "new_map50": float(metrics.box.map50),
        "new_map50_passed": float(metrics.box.map50) >= 0.60,
        "diagnostics": {
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
    }
    report_path = artifact_dir / "training-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**report, "report_sha256": sha256_file(report_path)}, ensure_ascii=False, indent=2))
    return 0 if report["new_map50_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
