#!/usr/bin/env python3
"""Train registry-owned confidence residual adapters on Ascend310B."""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import resource
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .core import (
    FEATURE_DIM,
    RAW_LOGIT_INDEX,
    ResidualAdapter,
    adapt_probe,
    cpu_state,
    feature_row,
    frozen_parameters,
    identity_states,
    image_sizes,
    load_calibration_module,
    load_context_prior,
    load_method,
    require_exact_probe,
    subset_probe,
    validate_calibration_contract,
)
from .protocol import EdgeProtocol, load_protocol


class NpuMonitor:
    def __init__(self) -> None:
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, float | int | None]:
        self._stop.set()
        self._thread.join(timeout=5)
        keys = ("power_w", "temperature_c", "aicore_percent", "memory_mb")
        return {
            "sample_count": len(self.samples),
            **{
                f"peak_{key}": max((row[key] for row in self.samples), default=None)
                for key in keys
            },
            **{
                f"mean_{key}": (
                    sum(row[key] for row in self.samples) / len(self.samples)
                    if self.samples
                    else None
                )
                for key in keys
            },
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    ["npu-smi", "info"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout
                board = re.search(
                    r"\|\s*0\s+310B1\s*\|[^|]*\|\s*([0-9.]+)\s+([0-9.]+)",
                    output,
                )
                device = re.search(
                    r"\|\s*0\s+0\s*\|[^|]*\|\s*([0-9.]+)\s+([0-9]+)\s*/\s*[0-9]+",
                    output,
                )
                if board and device:
                    self.samples.append(
                        {
                            "power_w": float(board.group(1)),
                            "temperature_c": float(board.group(2)),
                            "aicore_percent": float(device.group(1)),
                            "memory_mb": float(device.group(2)),
                        }
                    )
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(1.0)


def build_training_rows(
    probe: Any,
    ground_truth: Sequence[Mapping[str, Any]],
    class_id: int,
    sizes: Mapping[str, tuple[int, int]],
    box_iou: Any,
    max_negatives_per_image: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    by_image_predictions: dict[str, list[Mapping[str, Any]]] = {
        image_id: [] for image_id in probe.image_ids
    }
    by_image_targets: dict[str, list[Mapping[str, Any]]] = {
        image_id: [] for image_id in probe.image_ids
    }
    for row in probe.records:
        if (
            row.get("source") == "incremental_model"
            and int(row["class_id"]) == class_id
        ):
            by_image_predictions[str(row["image_id"])].append(row)
    for row in ground_truth:
        if int(row["class_id"]) == class_id:
            by_image_targets[str(row["image_id"])].append(row)

    features: list[list[float]] = []
    labels: list[float] = []
    positives = 0
    negatives = 0
    for image_id in sorted(probe.image_ids):
        candidates = sorted(
            by_image_predictions[image_id],
            key=lambda row: -float(row["confidence"]),
        )
        targets = by_image_targets[image_id]
        used_targets: set[int] = set()
        labeled: list[tuple[Mapping[str, Any], float]] = []
        for candidate in candidates:
            overlaps = [
                (box_iou(candidate["xyxy"], target["xyxy"]), index)
                for index, target in enumerate(targets)
                if index not in used_targets
            ]
            best_iou, best_index = max(overlaps, default=(0.0, -1))
            label = float(best_iou >= 0.5)
            if label:
                used_targets.add(best_index)
            labeled.append((candidate, label))
        positive_rows = [item for item in labeled if item[1] == 1.0]
        negative_rows = [item for item in labeled if item[1] == 0.0]
        negative_limit = max(
            32,
            min(max_negatives_per_image, 12 * max(1, len(positive_rows))),
        )
        for candidate, label in [*positive_rows, *negative_rows[:negative_limit]]:
            features.append(
                feature_row(candidate, probe.contexts[image_id], sizes[image_id])
            )
            labels.append(label)
            positives += int(label)
            negatives += int(not label)
    if not features or not positives or not negatives:
        raise RuntimeError(
            f"invalid candidate labels for class {class_id}: "
            f"samples={len(features)} positives={positives} negatives={negatives}"
        )
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        {"samples": len(features), "positives": positives, "negatives": negatives},
    )


def balanced_training_rows(
    features: np.ndarray,
    labels: np.ndarray,
    training_rows: int,
    batch_size: int,
    seed: int = 20260825,
) -> tuple[torch.Tensor, torch.Tensor]:
    if training_rows <= 0 or training_rows % batch_size or training_rows % 2:
        raise ValueError(
            "training rows must be positive, even and divisible by batch size"
        )
    positive = np.flatnonzero(labels > 0.5)
    negative = np.flatnonzero(labels <= 0.5)
    if not len(positive) or not len(negative):
        raise RuntimeError("balanced training requires positive and negative rows")
    rng = np.random.default_rng(seed)
    half = training_rows // 2
    positive_indices = rng.choice(positive, size=half, replace=len(positive) < half)
    hard_negative_order = negative[np.argsort(-features[negative, RAW_LOGIT_INDEX])]
    negative_indices = (
        hard_negative_order[:half]
        if len(hard_negative_order) >= half
        else np.resize(hard_negative_order, half)
    )
    indices = np.concatenate([positive_indices, negative_indices])
    rng.shuffle(indices)
    return torch.from_numpy(features[indices]), torch.from_numpy(labels[indices])


def train_candidate(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    device_id: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    device = torch.device(f"npu:{device_id}")
    model = ResidualAdapter().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    x = features.to(device)
    y = labels.to(device)
    losses: list[float] = []
    iteration_ms: list[float] = []
    epoch_ms: list[float] = []
    started = time.perf_counter()
    for _ in range(epochs):
        epoch_started = time.perf_counter()
        epoch_losses: list[float] = []
        for start in range(0, len(x), batch_size):
            step_started = time.perf_counter()
            batch_x = x[start : start + batch_size]
            batch_y = y[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = batch_x[:, RAW_LOGIT_INDEX] + model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            torch.npu.synchronize()
            epoch_losses.append(float(loss.detach().cpu()))
            iteration_ms.append((time.perf_counter() - step_started) * 1000.0)
        losses.append(sum(epoch_losses) / len(epoch_losses))
        epoch_ms.append((time.perf_counter() - epoch_started) * 1000.0)
    wall_seconds = time.perf_counter() - started
    steady = sorted(iteration_ms[min(32, len(iteration_ms)) :]) or iteration_ms
    state = cpu_state(model)
    with torch.no_grad():
        active_residual = model(x).detach().cpu()
    report = {
        "seed": seed,
        "learning_rate": learning_rate,
        "optimizer": "SGD",
        "epochs": epochs,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "loss_reduction_ratio": 1.0 - losses[-1] / losses[0],
        "wall_seconds": wall_seconds,
        "seconds_per_epoch": wall_seconds / epochs,
        "first_epoch_ms": epoch_ms[0],
        "steady_step_median_ms": steady[len(steady) // 2],
        "steady_step_p95_ms": steady[
            min(len(steady) - 1, int(len(steady) * 0.95))
        ],
        "batch_size": batch_size,
        "balanced_training_rows": len(features),
        "residual_abs_mean": float(active_residual.abs().mean()),
        "residual_abs_max": float(active_residual.abs().max()),
    }
    del model, optimizer, criterion, x, y
    torch.npu.empty_cache()
    return state, report


def evaluate_round(
    calibration: Any,
    protocol: EdgeProtocol,
    method: Mapping[str, Any],
    probe: Any,
    states: Mapping[int, Mapping[str, torch.Tensor]],
    sizes: Mapping[str, tuple[int, int]],
    ground_truth: Sequence[Mapping[str, Any]],
    context_prior: Mapping[str, Any],
    class_id: int,
    evaluate_ap50: Any,
    precision_recall: Any,
) -> dict[str, Any]:
    adapted = adapt_probe(calibration, probe, states, sizes)
    _, combined, counters = calibration.apply_parameters(
        adapted,
        frozen_parameters(calibration, protocol, method),
        context_prior,
        content_gate_enabled=True,
        content_gate_scene_probability=0.50,
    )
    ap = evaluate_ap50(combined, ground_truth, [class_id])
    pr = precision_recall(combined, ground_truth, class_id, 0.0)
    return {
        "map50": float(ap["map50"]),
        "precision": float(pr["precision"]),
        "recall": float(pr["recall"]),
        "f1": float(pr["f1"]),
        "tp": int(pr["tp"]),
        "fp": int(pr["fp"]),
        "targets": int(pr["targets"]),
        "final_prediction_count": len(combined),
        "policy_counters": counters,
    }


def selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = row["dev_metrics"]
    return (
        float(metrics["map50"]),
        float(metrics["f1"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        -float(row.get("residual_abs_mean", 0.0)),
    )


def parse_values(raw: str, caster: Any) -> list[Any]:
    values = [caster(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("search values must be a non-empty unique list")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="只用注册轮次 Increment train/dev 在 Ascend310B 训练轻量 Adapter。"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--context-prior", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", default="20260825,11,26,4090,310")
    parser.add_argument("--learning-rates", default="0.01,0.05,0.1")
    parser.add_argument("--training-rows", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-negatives-per-image", type=int, default=96)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    import torch_npu  # noqa: F401, PLC0415

    repo_root = args.repo_root.expanduser().resolve()
    protocol = load_protocol(args.registry, repo_root)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output_dir}")
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_negatives_per_image <= 0:
        raise ValueError("epochs, batch size and negative limit must be positive")
    from fair_agent.modules.strict_incremental import (  # noqa: PLC0415
        box_iou,
        evaluate_ap50,
        precision_recall,
        read_split,
        yolo_ground_truth,
    )

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable; refusing CPU fallback")
    torch.npu.set_device(args.device_id)
    if hasattr(torch.npu, "set_compile_mode"):
        torch.npu.set_compile_mode(jit_compile=False)
    seeds = parse_values(args.seeds, int)
    learning_rates = parse_values(args.learning_rates, float)
    if any(value <= 0 for value in learning_rates):
        raise ValueError("learning rates must be positive")
    calibration = load_calibration_module(repo_root)
    validate_calibration_contract(calibration, protocol)
    method = load_method(args.method_config.expanduser().resolve())
    context_prior = load_context_prior(args.context_prior.expanduser().resolve())
    full_probe = calibration.load_probe(args.probe.expanduser().resolve())
    training_paths = protocol.image_paths("training")
    require_exact_probe(full_probe, training_paths, "training")

    round_data: dict[int, dict[str, Any]] = {}
    all_paths: list[Path] = []
    for spec in protocol.rounds:
        train_paths = read_split(repo_root / spec.train_split)
        dev_paths = read_split(repo_root / spec.dev_split)
        all_paths.extend([*train_paths, *dev_paths])
        train_ids = {path.stem for path in train_paths}
        dev_ids = {path.stem for path in dev_paths}
        if train_ids & dev_ids:
            raise RuntimeError(f"train/dev overlap in {spec.round_id}")
        round_data[spec.class_id] = {
            "spec": spec,
            "train_paths": train_paths,
            "dev_paths": dev_paths,
            "train_probe": subset_probe(calibration, full_probe, train_ids),
            "dev_probe": subset_probe(calibration, full_probe, dev_ids),
            "train_gt": yolo_ground_truth(train_paths, [spec.class_id]),
            "dev_gt": yolo_ground_truth(dev_paths, [spec.class_id]),
        }
    sizes = image_sizes(all_paths)
    prepared = {
        class_id: build_training_rows(
            data["train_probe"],
            data["train_gt"],
            class_id,
            sizes,
            box_iou,
            args.max_negatives_per_image,
        )
        for class_id, data in round_data.items()
    }

    output_dir.mkdir(parents=True)
    monitor = NpuMonitor()
    monitor.start()
    experiment_started = time.perf_counter()
    selected_states = identity_states(protocol.new_class_ids)
    round_reports: list[dict[str, Any]] = []
    try:
        for spec in protocol.rounds:
            class_id = spec.class_id
            data = round_data[class_id]
            features_np, labels_np, counts = prepared[class_id]
            baseline = evaluate_round(
                calibration,
                protocol,
                method,
                data["dev_probe"],
                selected_states,
                sizes,
                data["dev_gt"],
                context_prior,
                class_id,
                evaluate_ap50,
                precision_recall,
            )
            candidates: list[dict[str, Any]] = [
                {
                    "kind": "identity",
                    "seed": None,
                    "learning_rate": None,
                    "residual_abs_mean": 0.0,
                    "dev_metrics": baseline,
                    "state": selected_states[class_id],
                }
            ]
            for seed in seeds:
                features, labels = balanced_training_rows(
                    features_np,
                    labels_np,
                    args.training_rows,
                    args.batch_size,
                    seed=seed,
                )
                for learning_rate in learning_rates:
                    state, train_report = train_candidate(
                        features,
                        labels,
                        seed=seed,
                        learning_rate=learning_rate,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        device_id=args.device_id,
                    )
                    trial_states = dict(selected_states)
                    trial_states[class_id] = state
                    metrics = evaluate_round(
                        calibration,
                        protocol,
                        method,
                        data["dev_probe"],
                        trial_states,
                        sizes,
                        data["dev_gt"],
                        context_prior,
                        class_id,
                        evaluate_ap50,
                        precision_recall,
                    )
                    candidates.append(
                        {
                            "kind": "trained",
                            **train_report,
                            "dev_metrics": metrics,
                            "state": state,
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "event": "candidate_complete",
                                "round_id": spec.round_id,
                                "seed": seed,
                                "learning_rate": learning_rate,
                                "wall_seconds": train_report["wall_seconds"],
                                "loss_final": train_report["loss_final"],
                                "dev_metrics": metrics,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            selected_candidate = max(candidates, key=selection_key)
            selected_states[class_id] = selected_candidate["state"]
            public_selected = {
                key: value for key, value in selected_candidate.items() if key != "state"
            }
            public_candidates = [
                {key: value for key, value in row.items() if key != "state"}
                for row in candidates
            ]
            checkpoint = output_dir / f"{spec.round_id}_adapter.pt"
            torch.save(
                {
                    "schema_version": 1,
                    "protocol_id": protocol.protocol_id,
                    "round": asdict(spec),
                    "architecture": {
                        "kind": "matmul_free_logit_residual_adapter",
                        "feature_dim": FEATURE_DIM,
                        "parameter_count": FEATURE_DIM,
                    },
                    "state_dict": selected_states[class_id],
                    "selection": public_selected,
                    "training_data_scope": "current_increment_round_train_only",
                    "selection_data_scope": "current_increment_round_dev_only",
                    "base_images_opened": False,
                    "lock_images_opened": False,
                },
                checkpoint,
            )
            round_reports.append(
                {
                    "round": asdict(spec),
                    "train_image_count": len(data["train_paths"]),
                    "dev_image_count": len(data["dev_paths"]),
                    "candidate_labels": counts,
                    "balanced_training_rows": args.training_rows,
                    "baseline_dev_metrics": baseline,
                    "selected": public_selected,
                    "candidate_count": len(public_candidates),
                    "candidates": public_candidates,
                    "checkpoint": str(checkpoint),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "round_selected",
                        "round_id": spec.round_id,
                        "selected": public_selected,
                        "checkpoint": str(checkpoint),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        torch.npu.synchronize()
        torch.npu.empty_cache()
        resource_report = monitor.stop()

    total_seconds = time.perf_counter() - experiment_started
    combined_path = output_dir / "combined_adapter_bank.pt"
    torch.save(
        {
            "schema_version": 1,
            "protocol_id": protocol.protocol_id,
            "architecture": {
                "kind": "registered_class_matmul_free_logit_residual_bank",
                "feature_dim": FEATURE_DIM,
                "parameters_per_class": FEATURE_DIM,
                "total_parameters": FEATURE_DIM * len(protocol.new_class_ids),
            },
            "class_order": list(protocol.new_class_ids),
            "class_states": selected_states,
            "class_names": {
                class_id: protocol.class_names[class_id]
                for class_id in protocol.new_class_ids
            },
            "frozen_models": [
                "base_detector.om",
                "incremental_detector.om",
                "scene_sensor_net.om",
            ],
            "base_weights_updated": False,
            "specialist_weights_updated": False,
            "scene_weights_updated": False,
        },
        combined_path,
    )
    report = {
        "schema_version": 1,
        "status": "completed",
        "phase": "incremental_learning",
        "platform": "Ascend310B1",
        "device_backend": "torch_npu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "numpy": np.__version__,
        "npu_training_required": True,
        "cpu_fallback_used": False,
        "incremental_protocol": {
            "protocol_id": protocol.protocol_id,
            "registry": str(protocol.registry_path),
            "base_images_opened_during_training": False,
            "lock_images_opened_during_training": False,
            "round_order": [spec.round_id for spec in protocol.rounds],
            "new_class_ids": list(protocol.new_class_ids),
            "historical_increment_replay": False,
        },
        "feature_schema": [
            "constant",
            "raw_logit",
            "box_area",
            "scene_air",
            "scene_forest",
            "scene_sea",
            "scene_urban",
            "sensor_ir_minus_sar",
        ],
        "epochs_per_candidate": args.epochs,
        "seeds": seeds,
        "learning_rates": learning_rates,
        "training_rows": args.training_rows,
        "batch_size": args.batch_size,
        "total_training_wall_seconds": total_seconds,
        "rounds": round_reports,
        "combined_checkpoint": str(combined_path),
        "adapter_total_parameters": FEATURE_DIM * len(protocol.new_class_ids),
        "resource_usage": {
            **resource_report,
            "process_max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
            "torch_npu_max_allocated_mb": torch.npu.max_memory_allocated() / 1024**2,
            "torch_npu_max_reserved_mb": torch.npu.max_memory_reserved() / 1024**2,
        },
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
