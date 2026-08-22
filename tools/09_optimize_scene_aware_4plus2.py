#!/usr/bin/env python3
"""Optimize and freeze a class-specific Scene-SensorNet operating point for 4+2."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CLASS_NAMES = {
    0: "soldier",
    1: "small_aircraft",
    2: "warship",
    3: "tank",
    4: "patrol_boat",
    5: "armored_vehicle",
}
BASE_CLASS_IDS = (0, 1, 2, 3)
NEW_CLASS_IDS = (4, 5)
CURRENT_THRESHOLDS = {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.18, 5: 0.08}
SYSTEM_CALIBRATION_DATA_SCOPE = {
    "scene_sensor_model_training": "base_and_incremental_train_dev",
    "scene_sensor_model_recheck": "base_and_incremental_lock_frozen_model_only",
    "base_context_prior": "base_train_only",
    "incremental_context_prior": "incremental_train_only",
    "gate_selection": "mixed_dev_only",
}


def resolve_split(
    data_root: Path,
    split_name: str,
    *,
    require_labels: bool = True,
) -> list[Path]:
    split_path = data_root / "splits" / "strict_4plus2" / split_name
    if not split_path.is_file():
        raise FileNotFoundError(f"划分不存在：{split_path}")
    images: list[Path] = []
    for line_number, raw in enumerate(
        split_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw.strip()
        if not value:
            continue
        image = Path(value)
        if not image.is_absolute():
            image = data_root / image
        image = image.resolve()
        try:
            image.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(
                f"{split_path}:{line_number} 越出数据根目录：{image}"
            ) from exc
        from fair_agent.modules.strict_incremental import source_label

        if not image.is_file() or (
            require_labels and not source_label(image).is_file()
        ):
            raise FileNotFoundError(f"划分图像或标签不存在：{image}")
        images.append(image)
    if not images or len(images) != len(set(images)):
        raise ValueError(f"划分为空或包含重复图像：{split_path}")
    return images


def ensure_mixed_contract(
    base_images: Sequence[Path],
    incremental_images: Sequence[Path],
    mixed_images: Sequence[Path],
    split_name: str,
) -> None:
    expected = [path.resolve() for path in [*base_images, *incremental_images]]
    actual = [path.resolve() for path in mixed_images]
    if Counter(expected) != Counter(actual) or len(expected) != len(actual):
        raise ValueError(f"{split_name} 不等于对应 Base + Increment 清单")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"冻结预测不存在：{path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def predict_context_cache(
    images: Sequence[Path],
    scene_weight: Path,
    cache_path: Path,
    *,
    device: str,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    expected_ids = [path.stem for path in images]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("场景概率缓存要求图像 stem 唯一")
    if cache_path.is_file():
        rows = read_jsonl(cache_path)
        if Counter(str(row["image_id"]) for row in rows) != Counter(expected_ids):
            raise ValueError(f"已有场景概率缓存与输入清单不匹配：{cache_path}")
        return {str(row["image_id"]): dict(row["context"]) for row in rows}

    from PIL import Image
    from fair_agent.models.context import load_context_model, predict_context_batch

    cuda_device = device if str(device).startswith("cuda:") else f"cuda:{device}"
    model, checkpoint = load_context_model(scene_weight, cuda_device)
    rows: list[dict[str, Any]] = []
    for offset in range(0, len(images), batch_size):
        batch_paths = images[offset : offset + batch_size]
        batch_images = []
        for path in batch_paths:
            with Image.open(path) as source:
                source.load()
                batch_images.append(source.convert("RGB"))
        contexts = predict_context_batch(
            model,
            checkpoint,
            batch_images,
            cuda_device,
        )
        for path, context in zip(batch_paths, contexts):
            clean_context = {
                key: value for key, value in context.items() if not str(key).startswith("_")
            }
            rows.append({"image_id": path.stem, "context": clean_context})
    write_jsonl(cache_path, rows)
    return {str(row["image_id"]): dict(row["context"]) for row in rows}


def learn_per_class_prior(
    images: Sequence[Path],
    contexts: Mapping[str, Mapping[str, Any]],
    class_ids: Sequence[int],
    source_split: str,
) -> dict[str, Any]:
    from fair_agent.modules.detection_fusion import learn_context_prior
    from fair_agent.modules.strict_incremental import image_class_ids

    class_contexts: dict[int, list[Mapping[str, Any]]] = {
        int(class_id): [] for class_id in class_ids
    }
    for image in images:
        present = image_class_ids(image)
        for class_id in class_ids:
            if class_id in present:
                class_contexts[int(class_id)].append(contexts[image.stem])
    per_class: dict[str, Any] = {}
    for class_id in class_ids:
        rows = class_contexts[int(class_id)]
        if not rows:
            raise ValueError(f"训练集没有类别 {class_id} 的场景先验样本")
        learned = learn_context_prior(rows, dimensions=("scene",))
        per_class[str(class_id)] = {
            "schema_version": 1,
            "class_id": int(class_id),
            "class_name": CLASS_NAMES[int(class_id)],
            "sample_count": len(rows),
            "scene": dict(learned.get("scene") or {}),
        }
    return {
        "schema_version": 2,
        "source_split": source_split,
        "dimensions": ["scene"],
        "online_inputs": ["scene_probabilities"],
        "label_aware_online_routing": False,
        "filename_aware_online_routing": False,
        "per_class": per_class,
    }


def combined_class_priors(
    base_prior: Mapping[str, Any],
    incremental_prior: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for payload in (base_prior, incremental_prior):
        for class_id, prior in dict(payload.get("per_class") or {}).items():
            output[int(class_id)] = dict(prior)
    if set(output) != set(CLASS_NAMES):
        raise ValueError("六类场景先验不完整")
    return output


def apply_scene_gates(
    rows: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    priors: Mapping[int, Mapping[str, Any]],
    thresholds: Mapping[int, float],
    penalties: Mapping[int, float],
) -> list[dict[str, Any]]:
    from fair_agent.modules.detection_fusion import context_adjusted_threshold

    kept: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        class_id = int(row["class_id"])
        context = contexts.get(str(row["image_id"]), {})
        effective, affinity = context_adjusted_threshold(
            float(thresholds[class_id]),
            context,
            priors[class_id],
            float(penalties[class_id]),
        )
        if float(row["confidence"]) >= effective:
            row["effective_threshold"] = effective
            row["context_affinity"] = affinity
            kept.append(row)
    return kept


def fuse_runtime_policy(
    base_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
    conflict: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    from fair_agent.modules.detection_fusion import arbitrate_cross_class_conflicts
    from fair_agent.modules.strict_incremental import class_aware_nms

    base_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    new_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        base_by_image[str(row["image_id"])].append(dict(row))
    for row in new_rows:
        new_by_image[str(row["image_id"])].append(dict(row))
    kept_base: list[dict[str, Any]] = []
    kept_new: list[dict[str, Any]] = []
    rejected = 0
    image_ids = sorted(set(base_by_image) | set(new_by_image))
    for image_id in image_ids:
        old = base_by_image.get(image_id, [])
        new = new_by_image.get(image_id, [])
        if conflict.get("enabled") is True:
            old, new, decisions = arbitrate_cross_class_conflicts(
                old,
                new,
                float(conflict["iou"]),
                float(conflict["base_confidence"]),
                float(conflict["specialist_margin"]),
                None,
                True,
            )
            rejected += sum(row.get("action") == "reject_specialist" for row in decisions)
        kept_base.extend(old)
        kept_new.extend(new)
    kept_new = class_aware_nms(kept_new, 0.60)
    return kept_base, kept_new, rejected


def subset(rows: Sequence[Mapping[str, Any]], image_ids: Iterable[str]) -> list[dict[str, Any]]:
    allowed = set(image_ids)
    return [dict(row) for row in rows if str(row["image_id"]) in allowed]


def score_policy(
    original_base: Sequence[Mapping[str, Any]],
    original_new: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    base_images: Sequence[Path],
    mixed_images: Sequence[Path],
    contexts: Mapping[str, Mapping[str, Any]],
    priors: Mapping[int, Mapping[str, Any]],
    thresholds: Mapping[int, float],
    penalties: Mapping[int, float],
    conflict: Mapping[str, Any],
) -> dict[str, Any]:
    from fair_agent.modules.strict_incremental import (
        evaluate_ap50,
        precision_recall,
        retention_metrics,
    )

    gated_base = apply_scene_gates(
        original_base, contexts, priors, thresholds, penalties
    )
    gated_new = apply_scene_gates(
        original_new, contexts, priors, thresholds, penalties
    )
    gated_base, gated_new, conflict_rejected = fuse_runtime_policy(
        gated_base, gated_new, conflict
    )
    fused = [*gated_base, *gated_new]
    base_ids = {path.stem for path in base_images}
    base_metrics = evaluate_ap50(
        subset(gated_base, base_ids),
        subset(ground_truth, base_ids),
        BASE_CLASS_IDS,
    )
    retention = retention_metrics(
        original_base,
        gated_base,
        ground_truth,
        BASE_CLASS_IDS,
    )
    new_metrics = evaluate_ap50(fused, ground_truth, NEW_CLASS_IDS)
    full_metrics = evaluate_ap50(fused, ground_truth, CLASS_NAMES)
    per_class: dict[str, Any] = {}
    false_image_union: set[str] = set()
    for class_id in CLASS_NAMES:
        quality = precision_recall(fused, ground_truth, class_id, 0.0)
        positive_images = {
            str(row["image_id"])
            for row in ground_truth
            if int(row["class_id"]) == class_id
        }
        negative_images = {path.stem for path in mixed_images} - positive_images
        activated = {
            str(row["image_id"])
            for row in fused
            if int(row["class_id"]) == class_id
        }
        false_images = negative_images & activated
        false_image_union.update(false_images)
        per_class[str(class_id)] = {
            "class_name": CLASS_NAMES[class_id],
            "map50": float(full_metrics["per_class_ap50"].get(class_id, 0.0)),
            **quality,
            "false_activation_image_count": len(false_images),
            "negative_image_count": len(negative_images),
            "false_activation_rate": (
                len(false_images) / len(negative_images) if negative_images else 0.0
            ),
            "threshold": float(thresholds[class_id]),
            "max_scene_penalty": float(penalties[class_id]),
        }
    total_tp = sum(int(row["tp"]) for row in per_class.values())
    total_fp = sum(int(row["fp"]) for row in per_class.values())
    new_tp = sum(int(per_class[str(class_id)]["tp"]) for class_id in NEW_CLASS_IDS)
    new_fp = sum(int(per_class[str(class_id)]["fp"]) for class_id in NEW_CLASS_IDS)
    return {
        "image_count": len(mixed_images),
        "base_map50": float(base_metrics["map50"]),
        "base_per_class_ap50": {
            str(key): value for key, value in base_metrics["per_class_ap50"].items()
        },
        "old_map50_before": float(retention["old_map50_before"]),
        "old_map50_after": float(retention["old_map50_after"]),
        "krr": float(retention["krr"]),
        "new_map50": float(new_metrics["map50"]),
        "new_per_class_ap50": {
            str(key): value for key, value in new_metrics["per_class_ap50"].items()
        },
        "full_map50": float(full_metrics["map50"]),
        "per_class": per_class,
        "overall": {
            "tp": total_tp,
            "fp": total_fp,
            "precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
            "false_activation_image_count": len(false_image_union),
        },
        "new_classes": {
            "tp": new_tp,
            "fp": new_fp,
            "precision": new_tp / (new_tp + new_fp) if new_tp + new_fp else 0.0,
        },
        "prediction_counts": {
            "base_before": len(original_base),
            "base_after": len(gated_base),
            "new_before": len(original_new),
            "new_after": len(gated_new),
            "fused": len(fused),
            "conflict_rejected": conflict_rejected,
        },
    }


def grid_values(minimum: float, maximum: float, step: float) -> list[float]:
    count = int(round((maximum - minimum) / step))
    return [round(minimum + index * step, 6) for index in range(count + 1)]


def passes_constraints(metrics: Mapping[str, Any], constraints: Mapping[str, float]) -> bool:
    return (
        float(metrics["base_map50"]) >= float(constraints["base_map50"])
        and float(metrics["new_map50"]) >= float(constraints["new_map50"])
        and float(metrics["krr"]) >= float(constraints["krr"])
    )


def objective(metrics: Mapping[str, Any], mode: str) -> tuple[float, ...]:
    overall = dict(metrics["overall"])
    new = dict(metrics["new_classes"])
    score_sum = float(metrics["base_map50"]) + float(metrics["new_map50"])
    if mode == "score_first":
        return (
            score_sum,
            float(overall["precision"]),
            float(new["precision"]),
            -float(overall["fp"]),
        )
    return (
        float(overall["precision"]),
        float(new["precision"]),
        -float(overall["false_activation_image_count"]),
        -float(overall["fp"]),
        score_sum,
    )


def coordinate_search(
    scorer: Any,
    constraints: Mapping[str, float],
    objective_mode: str,
) -> tuple[dict[int, float], dict[int, float], dict[str, Any], dict[str, Any]]:
    thresholds = dict(CURRENT_THRESHOLDS)
    penalties = {class_id: 0.0 for class_id in CLASS_NAMES}
    conflict: dict[str, Any] = {
        "enabled": False,
        "iou": 1.0,
        "base_confidence": 0.01,
        "specialist_margin": 0.0,
        "preserve_base_class_owners": True,
    }
    metrics = scorer(thresholds, penalties, conflict)
    if not passes_constraints(metrics, constraints):
        raise RuntimeError(f"当前运行点未通过 dev 搜索约束：{constraints}")

    for stage in ("coarse", "fine"):
        for class_id in CLASS_NAMES:
            if stage == "coarse":
                maximum = 0.60 if class_id in BASE_CLASS_IDS else 0.95
                threshold_values = grid_values(0.01, maximum, 0.04)
                penalty_values = grid_values(0.0, 0.80, 0.05)
            else:
                center_threshold = thresholds[class_id]
                center_penalty = penalties[class_id]
                threshold_values = grid_values(
                    max(0.01, center_threshold - 0.06),
                    min(0.99, center_threshold + 0.06),
                    0.01,
                )
                penalty_values = grid_values(
                    max(0.0, center_penalty - 0.08),
                    min(1.0, center_penalty + 0.08),
                    0.01,
                )
            threshold_values = sorted(set([*threshold_values, thresholds[class_id]]))
            penalty_values = sorted(set([*penalty_values, penalties[class_id]]))
            best = (objective(metrics, objective_mode), thresholds[class_id], penalties[class_id], metrics)
            for threshold in threshold_values:
                for penalty in penalty_values:
                    candidate_thresholds = {**thresholds, class_id: threshold}
                    candidate_penalties = {**penalties, class_id: penalty}
                    candidate_metrics = scorer(
                        candidate_thresholds, candidate_penalties, conflict
                    )
                    if not passes_constraints(candidate_metrics, constraints):
                        continue
                    candidate = (
                        objective(candidate_metrics, objective_mode),
                        threshold,
                        penalty,
                        candidate_metrics,
                    )
                    if candidate[0] > best[0]:
                        best = candidate
            _, thresholds[class_id], penalties[class_id], metrics = best

    conflict_best = (objective(metrics, objective_mode), conflict, metrics)
    for iou in grid_values(0.30, 0.90, 0.10):
        for base_confidence in grid_values(0.10, 0.80, 0.10):
            for margin in grid_values(0.0, 0.30, 0.05):
                candidate_conflict = {
                    "enabled": True,
                    "iou": iou,
                    "base_confidence": base_confidence,
                    "specialist_margin": margin,
                    "preserve_base_class_owners": True,
                }
                candidate_metrics = scorer(
                    thresholds, penalties, candidate_conflict
                )
                if not passes_constraints(candidate_metrics, constraints):
                    continue
                candidate = (
                    objective(candidate_metrics, objective_mode),
                    candidate_conflict,
                    candidate_metrics,
                )
                if candidate[0] > conflict_best[0]:
                    conflict_best = candidate
    _, conflict, metrics = conflict_best

    for class_id in CLASS_NAMES:
        center_threshold = thresholds[class_id]
        center_penalty = penalties[class_id]
        best = (objective(metrics, objective_mode), center_threshold, center_penalty, metrics)
        for threshold in grid_values(
            max(0.01, center_threshold - 0.04),
            min(0.99, center_threshold + 0.04),
            0.01,
        ):
            for penalty in grid_values(
                max(0.0, center_penalty - 0.05),
                min(1.0, center_penalty + 0.05),
                0.01,
            ):
                candidate_thresholds = {**thresholds, class_id: threshold}
                candidate_penalties = {**penalties, class_id: penalty}
                candidate_metrics = scorer(candidate_thresholds, candidate_penalties, conflict)
                if not passes_constraints(candidate_metrics, constraints):
                    continue
                candidate = (
                    objective(candidate_metrics, objective_mode),
                    threshold,
                    penalty,
                    candidate_metrics,
                )
                if candidate[0] > best[0]:
                    best = candidate
        _, thresholds[class_id], penalties[class_id], metrics = best
    return thresholds, penalties, conflict, metrics


def markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# strict 4+2 逐类场景软门控 dev 搜索",
        "",
        "本步骤属于 system_calibration，不计入 incremental_learning，也不更新 Base 或 Increment 检测器权重。",
        "线上门控只读取 Scene-SensorNet 输出的已知场景概率，不读取文件名或真值标签。",
        "Base 类先验只由 base train 正样本学习；新增类先验只由 increment train 正样本学习。",
        "",
        "| 候选 | Base mAP50 | New-mAP50 | KRR | 六类 precision | 六类 FP | 新类 precision | 新类 FP | 误激活图像 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline = payload["baseline"]
    candidates = {"baseline": {"dev_metrics": baseline}, **dict(payload["candidates"])}
    for name, candidate in candidates.items():
        metrics = candidate["dev_metrics"]
        lines.append(
            f"| {name} | {metrics['base_map50']:.6f} | {metrics['new_map50']:.6f} | "
            f"{metrics['krr']:.6f} | {metrics['overall']['precision']:.6f} | "
            f"{metrics['overall']['fp']} | {metrics['new_classes']['precision']:.6f} | "
            f"{metrics['new_classes']['fp']} | {metrics['overall']['false_activation_image_count']} |"
        )
    lines.extend(["", "## 候选参数", ""])
    for name, candidate in payload["candidates"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- 阈值：`{candidate['thresholds']}`",
                f"- 最大场景惩罚：`{candidate['max_threshold_penalties']}`",
                f"- 跨类冲突：`{candidate['conflict_policy']}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def run_dev(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser().resolve()
    evidence_dir = args.evidence_dir.expanduser().resolve()
    scene_weight = args.scene_weight.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not scene_weight.is_file():
        raise FileNotFoundError(f"Scene-SensorNet 权重不存在：{scene_weight}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_train = resolve_split(data_root, "base_train.txt")
    incremental_train = resolve_split(data_root, "increment_train.txt")
    base_dev = resolve_split(data_root, "base_dev.txt")
    incremental_dev = resolve_split(data_root, "increment_dev.txt")
    mixed_dev = resolve_split(data_root, "mixed_dev.txt")
    ensure_mixed_contract(base_dev, incremental_dev, mixed_dev, "mixed_dev")
    train_images = [*base_train, *incremental_train]
    train_contexts = predict_context_cache(
        train_images,
        scene_weight,
        output_dir / "context_cache" / "train.jsonl",
        device=args.device,
        batch_size=args.batch,
    )
    dev_contexts = predict_context_cache(
        mixed_dev,
        scene_weight,
        output_dir / "context_cache" / "mixed_dev.jsonl",
        device=args.device,
        batch_size=args.batch,
    )
    base_prior = learn_per_class_prior(
        base_train, train_contexts, BASE_CLASS_IDS, "base_train_only"
    )
    incremental_prior = learn_per_class_prior(
        incremental_train,
        train_contexts,
        NEW_CLASS_IDS,
        "incremental_train_only",
    )
    atomic_json(output_dir / "base_context_prior.json", base_prior)
    atomic_json(output_dir / "incremental_context_prior.json", incremental_prior)
    priors = combined_class_priors(base_prior, incremental_prior)

    base_predictions = read_jsonl(evidence_dir / "frozen" / "base_dev_predictions.jsonl")
    new_predictions = read_jsonl(
        evidence_dir / "frozen" / "specialist_dev_predictions.jsonl"
    )
    from fair_agent.modules.strict_incremental import yolo_ground_truth

    ground_truth = yolo_ground_truth(mixed_dev, CLASS_NAMES)

    def scorer(
        thresholds: Mapping[int, float],
        penalties: Mapping[int, float],
        conflict: Mapping[str, Any],
    ) -> dict[str, Any]:
        return score_policy(
            base_predictions,
            new_predictions,
            ground_truth,
            base_dev,
            mixed_dev,
            dev_contexts,
            priors,
            thresholds,
            penalties,
            conflict,
        )

    disabled_conflict = {
        "enabled": False,
        "iou": 1.0,
        "base_confidence": 0.01,
        "specialist_margin": 0.0,
        "preserve_base_class_owners": True,
    }
    baseline = scorer(
        CURRENT_THRESHOLDS,
        {class_id: 0.0 for class_id in CLASS_NAMES},
        disabled_conflict,
    )
    profiles = {
        "score_first": {
            "constraints": {"base_map50": 0.88, "new_map50": 0.78, "krr": 0.98},
            "objective": "score_first",
        },
        "guarded_precision": {
            "constraints": {"base_map50": 0.88, "new_map50": 0.78, "krr": 0.98},
            "objective": "precision",
        },
        "balanced_precision": {
            "constraints": {"base_map50": 0.85, "new_map50": 0.72, "krr": 0.96},
            "objective": "precision",
        },
    }
    candidates: dict[str, Any] = {}
    for name, profile in profiles.items():
        thresholds, penalties, conflict, metrics = coordinate_search(
            scorer,
            profile["constraints"],
            str(profile["objective"]),
        )
        candidate = {
            "schema_version": 2,
            "candidate_id": f"scene-aware-4plus2-{name}",
            "created_at": datetime.now().astimezone().isoformat(),
            "phase": "system_calibration",
            "counted_as_incremental_learning": False,
            "detector_weights_updated": False,
            "data_scope": dict(SYSTEM_CALIBRATION_DATA_SCOPE),
            "selection_source": "mixed_dev_only",
            "selection_constraints": profile["constraints"],
            "selection_objective": profile["objective"],
            "thresholds": {str(key): value for key, value in thresholds.items()},
            "max_threshold_penalties": {
                str(key): value for key, value in penalties.items()
            },
            "conflict_policy": conflict,
            "fusion_iou": 0.60,
            "base_context_prior": base_prior,
            "incremental_context_prior": incremental_prior,
            "online_policy": {
                "scene_sensor_probabilities_only": True,
                "hard_scene_routing": False,
                "filename_routing": False,
                "label_aware_routing": False,
            },
            "dev_metrics": metrics,
        }
        candidates[name] = candidate
        atomic_json(output_dir / "candidates" / f"{name}.json", candidate)
    result = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": "strict_4plus2_class_specific_scene_soft_gate",
        "phase": "system_calibration",
        "counted_as_incremental_learning": False,
        "detector_weights_updated": False,
        "data_scope": dict(SYSTEM_CALIBRATION_DATA_SCOPE),
        "scene_weight": scene_weight.as_posix(),
        "splits": {
            "base_train": len(base_train),
            "incremental_train": len(incremental_train),
            "mixed_dev": len(mixed_dev),
        },
        "baseline": baseline,
        "candidates": candidates,
        "lock_labels_read": False,
    }
    atomic_json(output_dir / "dev_search.json", result)
    (output_dir / "dev_search.md").write_text(
        markdown_report(result), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_lock(args: argparse.Namespace) -> int:
    if args.candidate is None:
        raise ValueError("lock 模式必须提供 --candidate")
    data_root = args.data_root.expanduser().resolve()
    evidence_dir = args.evidence_dir.expanduser().resolve()
    scene_weight = args.scene_weight.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    candidate_path = args.candidate.expanduser().resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if (
        candidate.get("selection_source") != "mixed_dev_only"
        or candidate.get("phase") not in (None, "system_calibration")
    ):
        raise ValueError("候选不是由 mixed dev 冻结")
    thresholds = {
        int(key): float(value) for key, value in candidate["thresholds"].items()
    }
    penalties = {
        int(key): float(value)
        for key, value in candidate["max_threshold_penalties"].items()
    }
    priors = combined_class_priors(
        candidate["base_context_prior"], candidate["incremental_context_prior"]
    )
    base_lock = resolve_split(data_root, "base_lock.txt", require_labels=False)
    incremental_lock = resolve_split(
        data_root, "increment_lock.txt", require_labels=False
    )
    mixed_lock = resolve_split(data_root, "mixed_lock.txt", require_labels=False)
    ensure_mixed_contract(base_lock, incremental_lock, mixed_lock, "mixed_lock")
    lock_contexts = predict_context_cache(
        mixed_lock,
        scene_weight,
        output_dir / "context_cache" / "mixed_lock.jsonl",
        device=args.device,
        batch_size=args.batch,
    )
    base_predictions = read_jsonl(evidence_dir / "frozen" / "base_lock_predictions.jsonl")
    new_predictions = read_jsonl(
        evidence_dir / "frozen" / "specialist_lock_predictions.jsonl"
    )

    # The candidate and all model outputs are fixed before lock labels are read here.
    from fair_agent.modules.strict_incremental import yolo_ground_truth

    ground_truth = yolo_ground_truth(mixed_lock, CLASS_NAMES)
    lock_metrics = score_policy(
        base_predictions,
        new_predictions,
        ground_truth,
        base_lock,
        mixed_lock,
        lock_contexts,
        priors,
        thresholds,
        penalties,
        candidate["conflict_policy"],
    )
    baseline = score_policy(
        base_predictions,
        new_predictions,
        ground_truth,
        base_lock,
        mixed_lock,
        lock_contexts,
        priors,
        CURRENT_THRESHOLDS,
        {class_id: 0.0 for class_id in CLASS_NAMES},
        {
            "enabled": False,
            "iou": 1.0,
            "base_confidence": 0.01,
            "specialist_margin": 0.0,
            "preserve_base_class_owners": True,
        },
    )
    score_gates = {
        "base_map50": float(lock_metrics["base_map50"]) >= 0.80,
        "new_map50": float(lock_metrics["new_map50"]) >= 0.60,
        "krr": float(lock_metrics["krr"]) >= 0.95,
    }
    result = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "phase": "joint_evaluation",
        "counted_as_incremental_learning": False,
        "detector_weights_updated": False,
        "model_selection_allowed": False,
        "candidate": candidate,
        "baseline": baseline,
        "lock": lock_metrics,
        "score_gates": score_gates,
        "competition_accepted": all(score_gates.values()),
        "candidate_frozen_before_lock_labels": True,
    }
    atomic_json(output_dir / "lock_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["competition_accepted"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="使用 Scene-SensorNet 实际概率优化 strict 4+2 六类逐类软门控。"
    )
    parser.add_argument("mode", choices=("dev", "lock"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=ROOT / "models" / "production" / "incremental_detection" / "evidence",
    )
    parser.add_argument(
        "--scene-weight",
        type=Path,
        default=ROOT / "models" / "context" / "scene_sensor_net.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--device", default="1")
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    return run_dev(args) if args.mode == "dev" else run_lock(args)


if __name__ == "__main__":
    raise SystemExit(main())
