#!/usr/bin/env python3
"""Evaluate the frozen production policy on lock and all labeled images."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from fair_agent.core.config import load_config  # noqa: E402
from fair_agent.modules.strict_incremental import yolo_ground_truth  # noqa: E402


CLASS_KEYS = (
    "base_map50",
    "old_map50_before",
    "old_map50_after",
    "krr",
    "new_map50",
    "full_map50",
)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def predict_records_chunked(
    weight: Path,
    images: Sequence[Path],
    local_to_global: Mapping[int, int],
    *,
    device: str,
    imgsz: int,
    batch: int,
    confidence: float,
    source_name: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run one loaded YOLO model in bounded path chunks.

    Some Ultralytics versions preprocess an in-memory path list as one tensor
    even when ``batch`` is supplied. Explicit chunks keep all-images replay
    within the same memory bound as ordinary batch inference.
    """
    from ultralytics import YOLO

    if batch < 1:
        raise ValueError("batch 必须为正整数")
    model = YOLO(str(weight))
    mapping = {int(key): int(value) for key, value in local_to_global.items()}
    expected_ids = [path.stem for path in images]
    result_ids: list[str] = []
    records: list[dict[str, Any]] = []
    speed_totals: dict[str, float] = {}
    for offset in range(0, len(images), batch):
        chunk = images[offset : offset + batch]
        results = model.predict(
            source=[str(path) for path in chunk],
            device=device,
            imgsz=imgsz,
            batch=len(chunk),
            conf=confidence,
            iou=0.70,
            max_det=300,
            rect=True,
            augment=False,
            verbose=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(
                f"预测数量不一致：expected={len(chunk)} actual={len(results)}"
            )
        # In-memory path lists are reported as image0.jpg, image1.jpg, ... by
        # some Ultralytics versions. Results preserve source order, so bind
        # them back to the explicit chunk paths instead of trusting result.path.
        for source_path, result in zip(chunk, results):
            image_id = source_path.stem
            result_ids.append(image_id)
            for key, value in (getattr(result, "speed", None) or {}).items():
                speed_totals[key] = speed_totals.get(key, 0.0) + float(value)
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for xyxy, confidence_value, local_value in zip(
                boxes.xyxy.detach().cpu().tolist(),
                boxes.conf.detach().cpu().tolist(),
                boxes.cls.detach().cpu().tolist(),
            ):
                local_id = int(local_value)
                if local_id not in mapping:
                    raise RuntimeError(
                        f"{source_name} 输出未登记的局部类别：{local_id}"
                    )
                records.append(
                    {
                        "image_id": image_id,
                        "class_id": mapping[local_id],
                        "confidence": float(confidence_value),
                        "xyxy": [float(value) for value in xyxy],
                        "source": source_name,
                    }
                )
    if Counter(expected_ids) != Counter(result_ids):
        raise RuntimeError("预测结果与输入图像 stem 不一致")
    speed = {key: value / len(images) for key, value in speed_totals.items()}
    return records, speed


def lock_reproduction_delta(
    official: Mapping[str, Any], reproduction: Mapping[str, Any]
) -> dict[str, float | int]:
    """Summarize harmless backend replay drift without replacing lock evidence."""
    return {
        "base_map50": float(reproduction["base_map50"])
        - float(official["base_map50"]),
        "new_map50": float(reproduction["new_map50"])
        - float(official["new_map50"]),
        "krr": float(reproduction["krr"]) - float(official["krr"]),
        "full_map50": float(reproduction["full_map50"])
        - float(official["full_map50"]),
        "tp": int(reproduction["overall"]["tp"])
        - int(official["overall"]["tp"]),
        "fp": int(reproduction["overall"]["fp"])
        - int(official["overall"]["fp"]),
    }


def compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Keep score and error diagnostics without embedding every NMS decision."""
    output = {
        key: metrics[key]
        for key in (
            "image_count",
            *CLASS_KEYS,
            "base_per_class_ap50",
            "new_per_class_ap50",
            "per_class",
            "overall",
            "new_classes",
            "prediction_counts",
        )
    }
    return output


def comparison(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "full_map50_delta": float(after["full_map50"])
        - float(before["full_map50"]),
        "base_map50_delta": float(after["base_map50"])
        - float(before["base_map50"]),
        "new_map50_delta": float(after["new_map50"])
        - float(before["new_map50"]),
        "krr_delta": float(after["krr"]) - float(before["krr"]),
        "precision_delta": float(after["overall"]["precision"])
        - float(before["overall"]["precision"]),
        "tp_delta": int(after["overall"]["tp"])
        - int(before["overall"]["tp"]),
        "fp_delta": int(after["overall"]["fp"])
        - int(before["overall"]["fp"]),
        "cross_class_suppressed": int(
            after["prediction_counts"]["cross_class_suppressed"]
        ),
    }


def markdown_report(payload: Mapping[str, Any]) -> str:
    primary = payload["result_1_lock"]["with_cross_class_suppression"]
    reproduction = payload["result_1_lock_reproduction"][
        "with_cross_class_suppression"
    ]
    reproduction_delta = payload["result_1_lock_reproduction"][
        "delta_from_official_lock"
    ]
    secondary_before = payload["result_2_all_images"]["before_suppression"]
    secondary = payload["result_2_all_images"]["with_cross_class_suppression"]
    split = payload["result_2_all_images"]["split_counts"]
    lines = [
        "# strict 4+2 双口径指标",
        "",
        "一号结果是独立 mixed lock；二号结果覆盖 Base/Increment 的 train、dev、lock 全部标注图像。二号结果包含训练图像，只用于描述当前冻结系统在已知数据上的拟合与错误情况，不替代一号正式结果。",
        "",
        "## 数据范围",
        "",
        "| 口径 | Base train/dev/lock | Increment train/dev/lock | 图像总数 | 独立测试口径 |",
        "| --- | ---: | ---: | ---: | --- |",
        (
            f"| 一号 mixed lock | 0 / 0 / {split['base_lock']} | "
            f"0 / 0 / {split['incremental_lock']} | "
            f"{payload['result_1_lock']['image_count']} | 是 |"
        ),
        (
            f"| 二号 all images | {split['base_train']} / {split['base_dev']} / "
            f"{split['base_lock']} | {split['incremental_train']} / "
            f"{split['incremental_dev']} / {split['incremental_lock']} | "
            f"{payload['result_2_all_images']['image_count']} | 否，诊断结果 |"
        ),
        "",
        "## 一号结果：mixed lock",
        "",
        "一号结果由版本库中冻结的 lock 预测回放得到，是正式独立测试口径。",
        "",
        "| Base mAP50 | New-mAP50 | KRR | Full-mAP50 | TP | FP | precision |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {primary['base_map50']:.6f} | {primary['new_map50']:.6f} | "
            f"{primary['krr']:.6f} | {primary['full_map50']:.6f} | "
            f"{primary['overall']['tp']} | {primary['overall']['fp']} | "
            f"{primary['overall']['precision']:.6f} |"
        ),
        "",
        "### 本机全量运行中的 lock 复现检查",
        "",
        "该行来自本次全量推理，不覆盖上方正式一号结果。",
        "",
        "| Base mAP50 | New-mAP50 | KRR | Full-mAP50 | TP | FP | Full-mAP50 差值 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {reproduction['base_map50']:.6f} | "
            f"{reproduction['new_map50']:.6f} | {reproduction['krr']:.6f} | "
            f"{reproduction['full_map50']:.6f} | "
            f"{reproduction['overall']['tp']} | "
            f"{reproduction['overall']['fp']} | "
            f"{reproduction_delta['full_map50']:+.9f} |"
        ),
        "",
        f"## 二号结果：全部 {payload['result_2_all_images']['image_count']} 张图像",
        "",
        "| 状态 | Base mAP50 | New-mAP50 | KRR | Full-mAP50 | TP | FP | precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| 抑制前 | {secondary_before['base_map50']:.6f} | "
            f"{secondary_before['new_map50']:.6f} | {secondary_before['krr']:.6f} | "
            f"{secondary_before['full_map50']:.6f} | "
            f"{secondary_before['overall']['tp']} | "
            f"{secondary_before['overall']['fp']} | "
            f"{secondary_before['overall']['precision']:.6f} |"
        ),
        (
            f"| 正式后处理 | {secondary['base_map50']:.6f} | "
            f"{secondary['new_map50']:.6f} | {secondary['krr']:.6f} | "
            f"{secondary['full_map50']:.6f} | {secondary['overall']['tp']} | "
            f"{secondary['overall']['fp']} | "
            f"{secondary['overall']['precision']:.6f} |"
        ),
        "",
        "### 二号结果逐类明细",
        "",
        "| 类别 | AP50 | precision | recall | TP | FP | 误激活图像/负图像 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for class_id, row in secondary["per_class"].items():
        lines.append(
            f"| {class_id} {row['class_name']} | {row['map50']:.6f} | "
            f"{row['precision']:.6f} | {row['recall']:.6f} | {row['tp']} | "
            f"{row['fp']} | {row['false_activation_image_count']} / "
            f"{row['negative_image_count']} |"
        )
    lines.extend(
        [
            "",
            "## 口径约束",
            "",
            "- 两个结果使用同一冻结模型、逐类阈值、场景软门控和全类别跨类别重叠抑制。",
            "- 二号结果不参与阈值选择、模型选择或正式门禁，不得表述为独立测试集成绩。",
            "- 训练标签和模型权重均未修改；这里只对固定预测进行统一后处理与评分。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在 mixed lock 与全部 890 张标注图像上复核冻结 4+2 production。"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--round-registry",
        type=Path,
        default=ROOT / "configs/incremental_round_registry_4plus2.yaml",
    )
    parser.add_argument("--round-id", default="round_02_armored_vehicle")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/agent_pipeline.yaml"
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "models/profiles/incremental-detection/active.json",
    )
    parser.add_argument(
        "--official-base-lock-cache",
        type=Path,
        default=(
            ROOT
            / "models/production/incremental_detection/evidence/frozen"
            / "base_lock_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--official-specialist-lock-cache",
        type=Path,
        default=(
            ROOT
            / "models/production/incremental_detection/evidence/frozen"
            / "specialist_lock_predictions.jsonl"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            ROOT
            / "models/production/incremental_detection/evidence"
            / "all_images_diagnostics.json"
        ),
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--context-batch", type=int, default=128)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    profile = json.loads(args.profile.expanduser().resolve().read_text(encoding="utf-8"))
    config = load_config(args.config.expanduser().resolve())

    scene_eval = importlib.import_module("tools.09_optimize_scene_aware_4plus2")
    registry, _target_round, active_rounds = scene_eval.configure_round_contract(
        args.round_registry.expanduser().resolve(), args.round_id
    )
    class_names = dict(scene_eval.CLASS_NAMES)

    base_by_role = {
        role: scene_eval.resolve_split(
            data_root, registry["base"]["splits"][role]
        )
        for role in ("train", "dev", "lock")
    }
    incremental_by_role = {
        role: scene_eval.cumulative_round_split(
            data_root, active_rounds, role
        )
        for role in ("train", "dev", "lock")
    }
    base_all = [path for role in ("train", "dev", "lock") for path in base_by_role[role]]
    incremental_all = [
        path
        for role in ("train", "dev", "lock")
        for path in incremental_by_role[role]
    ]
    mixed_all = [*base_all, *incremental_all]
    if len(mixed_all) != len(set(mixed_all)) or len(
        {path.stem for path in mixed_all}
    ) != len(mixed_all):
        raise ValueError("all-images 清单包含重复图像或重复 stem")

    output_dir.mkdir(parents=True, exist_ok=True)
    base_cache = output_dir / "base_all_predictions.jsonl"
    specialist_cache = output_dir / "specialist_all_predictions.jsonl"
    context_cache = output_dir / "all_contexts.jsonl"
    cache_files = (base_cache, specialist_cache)
    if args.reuse_cache:
        if not all(path.is_file() for path in cache_files):
            raise FileNotFoundError("--reuse-cache 要求 Base 与 Specialist 缓存均存在")
        base_predictions = read_jsonl(base_cache)
        specialist_predictions = read_jsonl(specialist_cache)
    else:
        if any(path.exists() for path in cache_files):
            raise FileExistsError("预测缓存已存在；使用新目录或显式指定 --reuse-cache")
        base_weight = (ROOT / profile["base_weight"]).resolve()
        specialist_weight = (ROOT / profile["specialist_weight"]).resolve()
        base_predictions, base_speed = predict_records_chunked(
            base_weight,
            mixed_all,
            profile["base_local_to_global"],
            device=args.device,
            imgsz=int(profile["base_imgsz"]),
            batch=args.batch,
            confidence=float(config["inference"]["confidence_min"]),
            source_name="frozen_base_model",
        )
        specialist_predictions, specialist_speed = predict_records_chunked(
            specialist_weight,
            mixed_all,
            profile["specialist_local_to_global"],
            device=args.device,
            imgsz=int(profile["specialist_imgsz"]),
            batch=args.batch,
            confidence=float(config["inference"]["confidence_min"]),
            source_name="incremental_model",
        )
        scene_eval.write_jsonl(base_cache, base_predictions)
        scene_eval.write_jsonl(specialist_cache, specialist_predictions)
        atomic_json(
            output_dir / "prediction_speed.json",
            {
                "base_ms_per_image": base_speed,
                "specialist_ms_per_image": specialist_speed,
            },
        )

    contexts = scene_eval.predict_context_cache(
        mixed_all,
        (ROOT / "models/context/scene_sensor_net.pt").resolve(),
        context_cache,
        device=args.device,
        batch_size=args.context_batch,
    )
    ground_truth = yolo_ground_truth(mixed_all, class_names)
    thresholds = {
        int(key): float(value)
        for key, value in {
            **profile["base_activation_thresholds"],
            **profile["activation_thresholds"],
        }.items()
    }
    penalties = {
        int(key): float(value)
        for key, value in {
            **profile["base_context_gate"]["max_threshold_penalties"],
            **profile["context_gate"]["max_threshold_penalties"],
        }.items()
    }
    priors = scene_eval.combined_class_priors(
        profile["base_context_prior"], profile["context_prior"]
    )
    fusion_policy = dict(profile.get("fusion_policy") or {})
    conflict_policy = {
        "enabled": bool(fusion_policy.get("enabled", False)),
        "iou": float(fusion_policy.get("iou", 1.0)),
        "base_confidence": float(fusion_policy.get("base_confidence", 0.01)),
        "specialist_margin": float(
            fusion_policy.get("specialist_margin", fusion_policy.get("incremental_margin", 0.0))
        ),
        "preserve_base_class_owners": bool(
            fusion_policy.get("preserve_base_class_owners", True)
        ),
    }
    suppression = dict(config["routing"]["cross_class_suppression"])
    if suppression.get("enabled") is not True:
        raise ValueError("正式配置未启用全类别跨类别重叠抑制")

    def score_scope(
        base_images: Sequence[Path],
        mixed_images: Sequence[Path],
        source_base_predictions: Sequence[Mapping[str, Any]],
        source_specialist_predictions: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        image_ids = {path.stem for path in mixed_images}
        scoped_base = [
            row for row in source_base_predictions if row["image_id"] in image_ids
        ]
        scoped_specialist = [
            row
            for row in source_specialist_predictions
            if row["image_id"] in image_ids
        ]
        scoped_truth = [row for row in ground_truth if row["image_id"] in image_ids]
        scoped_contexts = {
            key: value for key, value in contexts.items() if key in image_ids
        }
        before = scene_eval.score_policy(
            scoped_base,
            scoped_specialist,
            scoped_truth,
            base_images,
            mixed_images,
            scoped_contexts,
            priors,
            thresholds,
            penalties,
            conflict_policy,
            None,
        )
        after = scene_eval.score_policy(
            scoped_base,
            scoped_specialist,
            scoped_truth,
            base_images,
            mixed_images,
            scoped_contexts,
            priors,
            thresholds,
            penalties,
            conflict_policy,
            suppression,
        )
        return before, after

    lock_images = [*base_by_role["lock"], *incremental_by_role["lock"]]
    official_base_lock = read_jsonl(
        args.official_base_lock_cache.expanduser().resolve()
    )
    official_specialist_lock = read_jsonl(
        args.official_specialist_lock_cache.expanduser().resolve()
    )
    lock_before, lock_after = score_scope(
        base_by_role["lock"],
        lock_images,
        official_base_lock,
        official_specialist_lock,
    )
    lock_reproduction_before, lock_reproduction_after = score_scope(
        base_by_role["lock"],
        lock_images,
        base_predictions,
        specialist_predictions,
    )
    all_before, all_after = score_scope(
        base_all,
        mixed_all,
        base_predictions,
        specialist_predictions,
    )
    hard = config["gates"]["official_hard"]
    lock_gates = {
        "base_map50": float(lock_after["base_map50"])
        >= float(hard["base_map50_min"]),
        "new_map50": float(lock_after["new_map50"])
        >= float(hard["new_map50_min"]),
        "krr": float(lock_after["krr"]) >= float(hard["krr_min"]),
    }
    split_counts = {
        "base_train": len(base_by_role["train"]),
        "base_dev": len(base_by_role["dev"]),
        "base_lock": len(base_by_role["lock"]),
        "incremental_train": len(incremental_by_role["train"]),
        "incremental_dev": len(incremental_by_role["dev"]),
        "incremental_lock": len(incremental_by_role["lock"]),
    }
    result = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "evaluation_id": "strict_4plus2_primary_lock_and_secondary_all_images",
        "production_profile": profile["profile_id"],
        "generation_id": profile.get("generation_id", "incremental_detection_generation_4plus2"),
        "models_frozen": True,
        "thresholds_frozen": True,
        "detector_weights_updated": False,
        "model_selection_allowed": False,
        "cross_class_suppression": suppression,
        "result_1_lock": {
            "role": "primary_independent_lock",
            "prediction_source": "version_controlled_frozen_lock_predictions",
            "official_gate_scope": True,
            "includes_training_images": False,
            "image_count": len(lock_images),
            "before_suppression": compact_metrics(lock_before),
            "with_cross_class_suppression": compact_metrics(lock_after),
            "comparison": comparison(lock_before, lock_after),
            "score_gates": lock_gates,
            "competition_accepted": all(lock_gates.values()),
        },
        "result_1_lock_reproduction": {
            "role": "same_run_lock_reproduction_check",
            "prediction_source": "all_images_inference_run",
            "official_gate_scope": False,
            "includes_training_images": False,
            "may_replace_official_result": False,
            "image_count": len(lock_images),
            "before_suppression": compact_metrics(lock_reproduction_before),
            "with_cross_class_suppression": compact_metrics(
                lock_reproduction_after
            ),
            "comparison": comparison(
                lock_reproduction_before, lock_reproduction_after
            ),
            "delta_from_official_lock": lock_reproduction_delta(
                lock_after, lock_reproduction_after
            ),
        },
        "result_2_all_images": {
            "role": "secondary_all_labeled_images_diagnostic",
            "official_gate_scope": False,
            "includes_training_images": True,
            "independent_test_claim_allowed": False,
            "image_count": len(mixed_all),
            "split_counts": split_counts,
            "before_suppression": compact_metrics(all_before),
            "with_cross_class_suppression": compact_metrics(all_after),
            "comparison": comparison(all_before, all_after),
        },
    }
    atomic_json(report_path, result)
    report_path.with_suffix(".md").write_text(
        markdown_report(result), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(lock_gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
