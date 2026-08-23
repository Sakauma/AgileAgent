#!/usr/bin/env python3
"""Search an Ascend310B runtime policy on mixed dev under score constraints."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.hashes import sha256_file  # noqa: E402
from fair_agent.modules.detection_fusion import (  # noqa: E402
    arbitrate_cross_class_conflicts,
    calibrate_record_confidences,
    context_adjusted_threshold,
    context_prior_for_class,
)
from fair_agent.modules.strict_incremental import (  # noqa: E402
    evaluate_ap50,
    precision_recall,
    read_split,
    retention_metrics,
    subset_rows,
    yolo_ground_truth,
)
from fair_agent.modules.web_inference import class_aware_nms  # noqa: E402


OLD_CLASS_IDS = (0, 1, 2, 3)
NEW_CLASS_IDS = (4, 5)
ALL_CLASS_IDS = (*OLD_CLASS_IDS, *NEW_CLASS_IDS)


def parse_float_values(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("参数必须是逗号分隔数值") from exc
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("参数必须包含互异数值")
    return values


@dataclass(frozen=True)
class SearchParameters:
    thresholds: tuple[float, float, float, float, float, float]
    base_temperature: float = 1.0
    base_bias: float = 0.0
    specialist_temperature: float = 1.0
    specialist_bias: float = 0.0
    scene_penalties: tuple[float, float] = (0.0, 0.0)
    conflict_iou: float = 0.50
    conflict_base_confidence: float = 0.50
    specialist_margin: float = 0.15
    cross_class_iou: float = 0.90
    smaller_box_coverage: float | None = None
    incremental_over_base_margin: float = 0.0

    def threshold_map(self) -> Dict[int, float]:
        return {
            class_id: float(self.thresholds[index])
            for index, class_id in enumerate(ALL_CLASS_IDS)
        }

    def scene_penalty_map(self) -> Dict[int, float]:
        return {
            class_id: float(self.scene_penalties[index])
            for index, class_id in enumerate(NEW_CLASS_IDS)
        }


@dataclass(frozen=True)
class ProbeData:
    records: tuple[Dict[str, Any], ...]
    contexts: Dict[str, Dict[str, Any]]
    image_ids: frozenset[str]


_SCORE_WORKER_STATE: tuple[
    ProbeData,
    Mapping[str, Any],
    Sequence[Mapping[str, Any]],
    set[str],
    Mapping[str, float],
    bool,
    float,
] | None = None


def initialize_score_worker(
    probe: ProbeData,
    context_prior: Mapping[str, Any],
    ground_truth: Sequence[Mapping[str, Any]],
    base_image_ids: set[str],
    gates: Mapping[str, float],
    content_gate_enabled: bool,
    content_gate_scene_probability: float,
) -> None:
    """Load immutable scoring inputs once in each search worker."""

    global _SCORE_WORKER_STATE
    _SCORE_WORKER_STATE = (
        probe,
        context_prior,
        ground_truth,
        base_image_ids,
        gates,
        content_gate_enabled,
        content_gate_scene_probability,
    )


def score_parameters_worker(parameters: SearchParameters) -> Dict[str, Any]:
    """Score one parameter set using process-local immutable inputs."""

    if _SCORE_WORKER_STATE is None:
        raise RuntimeError("Ascend calibration score worker is not initialized")
    (
        probe,
        context_prior,
        ground_truth,
        base_image_ids,
        gates,
        content_gate_enabled,
        content_gate_scene_probability,
    ) = _SCORE_WORKER_STATE
    return score_parameters(
        probe,
        parameters,
        context_prior,
        ground_truth,
        base_image_ids,
        gates,
        content_gate_enabled=content_gate_enabled,
        content_gate_scene_probability=content_gate_scene_probability,
    )


def load_probe(path: Path) -> ProbeData:
    records: list[Dict[str, Any]] = []
    contexts: Dict[str, Dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        image_id = str(payload["image_id"])
        if image_id in contexts:
            raise ValueError(f"probe图像重复：{image_id}")
        context = payload.get("context")
        if not isinstance(context, Mapping):
            raise ValueError(f"probe缺少Scene概率：{image_id}")
        contexts[image_id] = dict(context)
        for detection in payload.get("detections", []):
            source = str(detection.get("source") or "")
            if source not in {"frozen_base_model", "incremental_model"}:
                raise ValueError(
                    f"probe检测缺少固定模型来源：{image_id}:{source or 'missing'}"
                )
            records.append(
                {
                    "image_id": image_id,
                    "class_id": int(detection["class_id"]),
                    "class_name": str(
                        detection.get("class_name") or detection["class_id"]
                    ),
                    "confidence": float(detection["confidence"]),
                    "xyxy": [float(value) for value in detection["xyxy"]],
                    "source": source,
                    "protocol_id": detection.get("protocol_id"),
                }
            )
    if not contexts:
        raise ValueError("probe为空")
    return ProbeData(tuple(records), contexts, frozenset(contexts))


def load_accuracy_gates(path: Path) -> Dict[str, float]:
    method = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(method, Mapping) or method.get("kind") != "ascend310b_full_score_method":
        raise ValueError(f"Ascend满分方法配置非法：{path}")
    raw = dict(method["competition"]["accuracy_gates"])
    return {
        "base_map50": float(raw["base_map50_min"]),
        "new_map50": float(raw["new_map50_min"]),
        "krr": float(raw["krr_min"]),
    }


def false_activation_rate(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    image_ids: set[str],
    class_ids: Iterable[int],
) -> tuple[float, int, int]:
    selected = {int(value) for value in class_ids}
    positive = {
        str(row["image_id"])
        for row in ground_truth
        if int(row["class_id"]) in selected
    }
    negative = image_ids - positive
    activated = {
        str(row["image_id"])
        for row in predictions
        if int(row["class_id"]) in selected
    }
    count = len(negative & activated)
    return (count / len(negative) if negative else 0.0, count, len(negative))


def apply_parameters(
    probe: ProbeData,
    parameters: SearchParameters,
    context_prior: Mapping[str, Any],
    *,
    content_gate_enabled: bool,
    content_gate_scene_probability: float,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, int]]:
    calibration = {
        "enabled": True,
        "method": "logit_affine",
        "sources": {
            "frozen_base_model": {
                "temperature": parameters.base_temperature,
                "bias": parameters.base_bias,
            },
            "incremental_model": {
                "temperature": parameters.specialist_temperature,
                "bias": parameters.specialist_bias,
            },
        },
    }
    calibrated = calibrate_record_confidences(probe.records, calibration)
    by_image: Dict[str, list[Dict[str, Any]]] = {
        image_id: [] for image_id in probe.image_ids
    }
    for row in calibrated:
        by_image[str(row["image_id"])].append(row)

    thresholds = parameters.threshold_map()
    penalties = parameters.scene_penalty_map()
    before_fusion: list[Dict[str, Any]] = []
    combined: list[Dict[str, Any]] = []
    counters = {
        "threshold_rejected": 0,
        "scene_soft_rejected": 0,
        "content_gate_images": 0,
        "conflict_suppressed": 0,
        "cross_class_suppressed": 0,
    }
    for image_id in sorted(probe.image_ids):
        context = probe.contexts[image_id]
        rows = by_image[image_id]
        base_rows: list[Dict[str, Any]] = []
        specialist_rows: list[Dict[str, Any]] = []
        for row in rows:
            class_id = int(row["class_id"])
            if row["source"] == "frozen_base_model":
                if float(row["confidence"]) >= thresholds[class_id]:
                    base_rows.append(row)
                else:
                    counters["threshold_rejected"] += 1
                continue
            class_prior = context_prior_for_class(context_prior, class_id)
            effective_threshold, _affinity = context_adjusted_threshold(
                thresholds[class_id],
                context,
                class_prior,
                penalties[class_id],
            )
            if float(row["confidence"]) >= effective_threshold:
                specialist_rows.append(row)
            else:
                counters["threshold_rejected"] += 1
                if effective_threshold > thresholds[class_id]:
                    counters["scene_soft_rejected"] += 1

        before_fusion.extend(base_rows)
        scene_probabilities = context.get("scene_probabilities") or {}
        skip_specialist = bool(
            content_gate_enabled
            and float(scene_probabilities.get("air", 0.0))
            >= content_gate_scene_probability
            and any(int(row["class_id"]) == 1 for row in base_rows)
        )
        if skip_specialist:
            specialist_rows = []
            counters["content_gate_images"] += 1

        base_rows, specialist_rows, conflicts = arbitrate_cross_class_conflicts(
            base_rows,
            specialist_rows,
            parameters.conflict_iou,
            parameters.conflict_base_confidence,
            parameters.specialist_margin,
            None,
            True,
            None,
        )
        counters["conflict_suppressed"] += sum(
            row.get("action") == "reject_specialist" for row in conflicts
        )
        fused, summary = class_aware_nms(
            [*base_rows, *specialist_rows],
            0.60,
            {
                "enabled": True,
                "strategy": "highest_confidence",
                "scope": "all_classes",
                "iou": parameters.cross_class_iou,
                "smaller_box_coverage": parameters.smaller_box_coverage,
                "incremental_over_base_margin": (
                    parameters.incremental_over_base_margin
                ),
            },
        )
        counters["cross_class_suppressed"] += int(
            summary.get("cross_class_suppressed_count", 0)
        )
        combined.extend(fused)
    return before_fusion, combined, counters


def score_parameters(
    probe: ProbeData,
    parameters: SearchParameters,
    context_prior: Mapping[str, Any],
    ground_truth: Sequence[Mapping[str, Any]],
    base_image_ids: set[str],
    gates: Mapping[str, float],
    *,
    content_gate_enabled: bool,
    content_gate_scene_probability: float,
) -> Dict[str, Any]:
    before, combined, counters = apply_parameters(
        probe,
        parameters,
        context_prior,
        content_gate_enabled=content_gate_enabled,
        content_gate_scene_probability=content_gate_scene_probability,
    )
    final_base = [
        row for row in combined if row.get("source") == "frozen_base_model"
    ]
    base = evaluate_ap50(
        subset_rows(final_base, base_image_ids),
        subset_rows(ground_truth, base_image_ids),
        OLD_CLASS_IDS,
    )
    retention = retention_metrics(before, combined, ground_truth, OLD_CLASS_IDS)
    new = evaluate_ap50(combined, ground_truth, NEW_CLASS_IDS)
    full = evaluate_ap50(combined, ground_truth, ALL_CLASS_IDS)
    per_class_pr = {
        class_id: precision_recall(combined, ground_truth, class_id, 0.0)
        for class_id in ALL_CLASS_IDS
    }
    new_precision = sum(
        float(per_class_pr[class_id]["precision"]) for class_id in NEW_CLASS_IDS
    ) / len(NEW_CLASS_IDS)
    new_recall = sum(
        float(per_class_pr[class_id]["recall"]) for class_id in NEW_CLASS_IDS
    ) / len(NEW_CLASS_IDS)
    fa_rate, fa_count, fa_negative = false_activation_rate(
        combined,
        ground_truth,
        set(probe.image_ids),
        NEW_CLASS_IDS,
    )
    fa_by_class = {
        class_id: false_activation_rate(
            combined,
            ground_truth,
            set(probe.image_ids),
            [class_id],
        )
        for class_id in NEW_CLASS_IDS
    }
    metrics = {
        "base_map50": float(base["map50"]),
        "new_map50": float(new["map50"]),
        "krr": float(retention["krr"]),
        "old_map50_before": float(retention["old_map50_before"]),
        "old_map50_after": float(retention["old_map50_after"]),
        "full_map50": float(full["map50"]),
        "new_precision": new_precision,
        "new_recall": new_recall,
        "false_activation_rate": fa_rate,
        "false_activation_image_count": fa_count,
        "negative_image_count": fa_negative,
        "false_activation_by_class": {
            str(class_id): {
                "rate": values[0],
                "count": values[1],
                "negative_images": values[2],
            }
            for class_id, values in fa_by_class.items()
        },
        "prediction_count": len(combined),
        "per_class_ap50": {
            str(key): float(value)
            for key, value in full["per_class_ap50"].items()
        },
        "per_class_precision_recall": {
            str(key): dict(value) for key, value in per_class_pr.items()
        },
        "counters": counters,
    }
    competition_gates = {
        name: float(metrics[name]) >= float(minimum)
        for name, minimum in gates.items()
    }
    violation = sum(
        max(0.0, float(minimum) - float(metrics[name])) / max(float(minimum), 1e-9)
        for name, minimum in gates.items()
    )
    passed = all(competition_gates.values())
    objective = (
        0.0 if passed else 1.0,
        float(metrics["false_activation_rate"]) if passed else violation,
        -float(metrics["new_map50"]),
        -float(metrics["full_map50"]),
        -float(metrics["new_precision"]),
        float(metrics["prediction_count"]),
    )
    return {
        "parameters": parameters,
        "metrics": metrics,
        "competition_gates": competition_gates,
        "competition_passed": passed,
        "constraint_violation": violation,
        "objective": objective,
    }


def parameter_payload(parameters: SearchParameters) -> Dict[str, Any]:
    payload = asdict(parameters)
    payload["thresholds"] = {
        str(class_id): parameters.threshold_map()[class_id]
        for class_id in ALL_CLASS_IDS
    }
    payload["scene_penalties"] = {
        str(class_id): parameters.scene_penalty_map()[class_id]
        for class_id in NEW_CLASS_IDS
    }
    return payload


def public_evaluation(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "parameters": parameter_payload(row["parameters"]),
        "metrics": row["metrics"],
        "competition_gates": row["competition_gates"],
        "competition_passed": row["competition_passed"],
        "constraint_violation": row["constraint_violation"],
    }


def select_beam(rows: Sequence[Mapping[str, Any]], size: int) -> list[SearchParameters]:
    selected: list[SearchParameters] = []
    seen: set[SearchParameters] = set()

    def add(ordered: Iterable[Mapping[str, Any]], limit: int) -> None:
        added = 0
        for row in ordered:
            parameters = row["parameters"]
            if parameters in seen:
                continue
            seen.add(parameters)
            selected.append(parameters)
            added += 1
            if added >= limit or len(selected) >= size:
                return

    objective_rows = sorted(rows, key=lambda row: row["objective"])
    add(objective_rows, max(1, size // 2))
    passed = [row for row in rows if row["competition_passed"]]
    add(
        sorted(passed, key=lambda row: -float(row["metrics"]["new_map50"])),
        max(1, size // 8),
    )
    add(
        sorted(passed, key=lambda row: -float(row["metrics"]["full_map50"])),
        max(1, size // 8),
    )
    add(
        sorted(rows, key=lambda row: float(row["constraint_violation"])),
        size,
    )
    return selected[:size]


def search(
    baseline: SearchParameters,
    evaluator: Callable[[SearchParameters], Dict[str, Any]],
    *,
    beam_size: int,
    threshold_values: Sequence[float],
    temperature_values: Sequence[float],
    specialist_bias_values: Sequence[float],
    scene_penalty_values: Sequence[float],
    source_margin_values: Sequence[float],
    overlap_iou_values: Sequence[float],
    conflict_iou_values: Sequence[float],
    specialist_margin_values: Sequence[float],
    batch_evaluator: (
        Callable[[Sequence[SearchParameters]], Sequence[Dict[str, Any]]] | None
    ) = None,
) -> tuple[Dict[SearchParameters, Dict[str, Any]], list[Dict[str, Any]]]:
    cache: Dict[SearchParameters, Dict[str, Any]] = {}
    trace: list[Dict[str, Any]] = []

    def evaluate(parameters: SearchParameters) -> Dict[str, Any]:
        if parameters not in cache:
            cache[parameters] = evaluator(parameters)
        return cache[parameters]

    def evaluate_many(
        parameters: Sequence[SearchParameters],
    ) -> list[Dict[str, Any]]:
        missing: list[SearchParameters] = []
        seen_missing: set[SearchParameters] = set()
        for item in parameters:
            if item not in cache and item not in seen_missing:
                missing.append(item)
                seen_missing.add(item)
        if missing:
            rows = (
                list(batch_evaluator(missing))
                if batch_evaluator is not None
                else [evaluator(item) for item in missing]
            )
            if len(rows) != len(missing):
                raise RuntimeError("并行评分返回数量与候选数量不一致")
            for item, row in zip(missing, rows):
                cache[item] = row
        return [cache[item] for item in parameters]

    def run_stage(
        name: str,
        beam: Sequence[SearchParameters],
        variants: Callable[[SearchParameters], Iterable[SearchParameters]],
    ) -> list[SearchParameters]:
        candidate_set = {
            variant for parent in beam for variant in variants(parent)
        }
        candidates = sorted(candidate_set, key=repr)
        rows = evaluate_many(candidates)
        selected = select_beam(rows, beam_size)
        best = min(rows, key=lambda row: row["objective"])
        stage = {
            "stage": name,
            "candidate_count": len(candidates),
            "cache_size": len(cache),
            "passing_count": sum(row["competition_passed"] for row in rows),
            "best": public_evaluation(best),
        }
        trace.append(stage)
        print(json.dumps(stage, ensure_ascii=False, separators=(",", ":")))
        return selected

    evaluate(baseline)
    beam = run_stage(
        "model_confidence_calibration",
        [baseline],
        lambda parameters: (
            replace(
                parameters,
                base_temperature=base_temperature,
                specialist_temperature=specialist_temperature,
                specialist_bias=specialist_bias,
            )
            for base_temperature in temperature_values
            for specialist_temperature in temperature_values
            for specialist_bias in specialist_bias_values
        ),
    )
    for class_id in (4, 5, 0, 1, 2, 3):
        class_index = ALL_CLASS_IDS.index(class_id)

        def threshold_variants(
            parameters: SearchParameters,
            *,
            index: int = class_index,
        ) -> Iterable[SearchParameters]:
            for value in threshold_values:
                thresholds = list(parameters.thresholds)
                thresholds[index] = value
                yield replace(parameters, thresholds=tuple(thresholds))

        beam = run_stage(
            f"per_class_threshold_{class_id}", beam, threshold_variants
        )
    for class_id in NEW_CLASS_IDS:
        penalty_index = NEW_CLASS_IDS.index(class_id)

        def penalty_variants(
            parameters: SearchParameters,
            *,
            index: int = penalty_index,
        ) -> Iterable[SearchParameters]:
            for value in scene_penalty_values:
                penalties = list(parameters.scene_penalties)
                penalties[index] = value
                yield replace(parameters, scene_penalties=tuple(penalties))

        beam = run_stage(f"known_scene_soft_gate_{class_id}", beam, penalty_variants)
    beam = run_stage(
        "overlap_source_margin",
        beam,
        lambda parameters: (
            replace(parameters, incremental_over_base_margin=value)
            for value in source_margin_values
        ),
    )
    beam = run_stage(
        "overlap_geometry",
        beam,
        lambda parameters: (
            replace(
                parameters,
                cross_class_iou=iou,
                smaller_box_coverage=coverage,
            )
            for iou in overlap_iou_values
            for coverage in (None, 0.95)
        ),
    )
    beam = run_stage(
        "old_new_conflict_iou",
        beam,
        lambda parameters: (
            replace(parameters, conflict_iou=value)
            for value in conflict_iou_values
        ),
    )
    beam = run_stage(
        "old_new_confidence_margin",
        beam,
        lambda parameters: (
            replace(parameters, specialist_margin=value)
            for value in specialist_margin_values
        ),
    )
    for class_id in (4, 5):
        class_index = ALL_CLASS_IDS.index(class_id)

        def refine_thresholds(
            parameters: SearchParameters,
            *,
            index: int = class_index,
        ) -> Iterable[SearchParameters]:
            for value in threshold_values:
                thresholds = list(parameters.thresholds)
                thresholds[index] = value
                yield replace(parameters, thresholds=tuple(thresholds))

        beam = run_stage(f"refine_new_threshold_{class_id}", beam, refine_thresholds)
    return cache, trace


def resolve_registry_path(config: Mapping[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    raw = Path(str(config["web"]["generation_registry"])).expanduser()
    return raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()


def materialize_candidate(
    source_config_path: Path,
    source_registry_path: Path | None,
    output_config: Path,
    output_registry: Path,
    output_report: Path,
    context_prior_path: Path,
    context_prior: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> None:
    config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    registry_source = resolve_registry_path(config, source_registry_path)
    registry = json.loads(registry_source.read_text(encoding="utf-8"))
    parameters: SearchParameters = selected["parameters"]
    thresholds = parameters.threshold_map()
    penalties = parameters.scene_penalty_map()
    for model in registry.get("models", []):
        owned = {int(value) for value in model.get("owns_classes", [])}
        model["per_class_thresholds"] = {
            str(class_id): thresholds[class_id] for class_id in sorted(owned)
        }
        if model.get("role") == "frozen_base":
            model["context_prior"] = {}
            model["context_gate"] = {"enabled": False}
        elif model.get("role") in {
            "class_incremental_expert",
            "target_incremental_expert",
        }:
            model["calibration_sources"] = {
                str(class_id): str(output_report) for class_id in sorted(owned)
            }
            enabled = any(penalties[class_id] > 0.0 for class_id in owned)
            model["context_prior"] = dict(context_prior) if enabled else {}
            model["context_gate"] = (
                {
                    "enabled": True,
                    "policy": "soft_threshold_penalty",
                    "hard_routing": False,
                    "learning_data_scope": "incremental_train_only",
                    "dimensions": ["scene"],
                    "max_threshold_penalty": max(
                        penalties[class_id] for class_id in owned
                    ),
                    "max_threshold_penalties": {
                        str(class_id): penalties[class_id]
                        for class_id in sorted(owned)
                    },
                    "prior_source": str(context_prior_path),
                    "prior_sha256": sha256_file(context_prior_path),
                    "online_input": "scene_sensor_net_probabilities",
                }
                if enabled
                else {"enabled": False}
            )
    for generation in registry.get("generations", []):
        if generation.get("id") == registry.get("channels", {}).get("candidate"):
            generation["candidate_system_calibration"] = {
                "phase": "system_calibration",
                "source_split": "mixed_dev_only",
                "selection": str(output_report),
                "competition_passed": True,
                "metrics": dict(selected["metrics"]),
            }

    output_registry.parent.mkdir(parents=True, exist_ok=True)
    output_registry.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config["runtime"]["server_port"] = 8502
    config["web"]["generation_registry"] = str(output_registry)
    config["generation"]["registry"] = str(output_registry)
    config["inference"]["confidence_min"] = 0.00001
    config["inference"]["confidence_default"] = min(
        0.01, min(thresholds.values())
    )
    config["routing"]["conflict_iou"] = parameters.conflict_iou
    config["routing"]["conflict_base_confidence"] = (
        parameters.conflict_base_confidence
    )
    config["routing"]["specialist_margin"] = parameters.specialist_margin
    config["routing"]["score_calibration"] = {
        "enabled": True,
        "method": "logit_affine",
        "source_split": "mixed_dev_only",
        "sources": {
            "frozen_base_model": {
                "temperature": parameters.base_temperature,
                "bias": parameters.base_bias,
            },
            "incremental_model": {
                "temperature": parameters.specialist_temperature,
                "bias": parameters.specialist_bias,
            },
        },
    }
    config["routing"]["cross_class_suppression"] = {
        "enabled": True,
        "strategy": "highest_confidence",
        "scope": "all_classes",
        "iou": parameters.cross_class_iou,
        "smaller_box_coverage": parameters.smaller_box_coverage,
        "incremental_over_base_margin": parameters.incremental_over_base_margin,
    }
    config["ascend_backend"]["validated"] = False
    config["ascend_backend"]["validation_candidate"] = True
    config["ascend_backend"]["validation_report"] = None
    config["ascend_backend"]["validation_report_sha256"] = None
    config["performance"]["report_root"] = str(output_config.parent / "reports")
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "仅在Ascend mixed dev原始候选上联合搜索逐类阈值、场景软门控、"
            "Base/Specialist置信度校准和重叠margin。"
        )
    )
    parser.add_argument("--probe-predictions", type=Path, required=True)
    parser.add_argument(
        "--mixed-dev",
        type=Path,
        default=Path("splits/strict_4plus2/mixed_dev.txt"),
    )
    parser.add_argument(
        "--base-dev",
        type=Path,
        default=Path("splits/strict_4plus2/base_dev.txt"),
    )
    parser.add_argument(
        "--context-prior",
        type=Path,
        default=Path(
            "models/production/incremental_detection/incremental_context_prior.json"
        ),
    )
    parser.add_argument(
        "--method-config",
        type=Path,
        default=Path("configs/ascend310b/full_score_method.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=89)
    parser.add_argument("--beam-size", type=int, default=48)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行评分进程数；默认1以保持板端低资源行为。",
    )
    parser.add_argument(
        "--threshold-values",
        type=parse_float_values,
        default=parse_float_values(
            "0.05,0.075,0.10,0.125,0.15,0.20,0.25,0.30,0.40,0.50,0.60"
        ),
    )
    parser.add_argument(
        "--temperature-values",
        type=parse_float_values,
        default=parse_float_values("0.75,1.0,1.25,1.5"),
    )
    parser.add_argument(
        "--specialist-bias-values",
        type=parse_float_values,
        default=parse_float_values("-0.50,-0.25,0.0,0.25"),
    )
    parser.add_argument(
        "--scene-penalty-values",
        type=parse_float_values,
        default=parse_float_values("0.0,0.05,0.10,0.15,0.20,0.30,0.40"),
    )
    parser.add_argument(
        "--source-margin-values",
        type=parse_float_values,
        default=parse_float_values("0.0,0.05,0.10,0.15,0.20"),
    )
    parser.add_argument(
        "--overlap-iou-values",
        type=parse_float_values,
        default=parse_float_values("0.50,0.60,0.70,0.80,0.90"),
    )
    parser.add_argument(
        "--conflict-iou-values",
        type=parse_float_values,
        default=parse_float_values("0.50,0.60,0.70,0.80,0.90"),
    )
    parser.add_argument(
        "--specialist-margin-values",
        type=parse_float_values,
        default=parse_float_values("0.0,0.05,0.10,0.15,0.20"),
    )
    parser.add_argument("--disable-content-execution-gate", action="store_true")
    parser.add_argument("--content-gate-scene-probability", type=float, default=0.50)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--candidate-registry", type=Path)
    parser.add_argument("--output-config", type=Path)
    parser.add_argument("--output-registry", type=Path)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    paths_to_create = [output]
    if args.output_config is not None:
        paths_to_create.append(args.output_config.expanduser().resolve())
    if args.output_registry is not None:
        paths_to_create.append(args.output_registry.expanduser().resolve())
    if any(path.exists() for path in paths_to_create):
        raise FileExistsError("搜索证据或候选输出已存在，拒绝覆盖")
    candidate_requested = any(
        value is not None
        for value in (args.candidate_config, args.output_config, args.output_registry)
    )
    if candidate_requested and not all(
        value is not None
        for value in (args.candidate_config, args.output_config, args.output_registry)
    ):
        raise ValueError(
            "物化候选必须同时提供candidate-config、output-config和output-registry"
        )
    if args.mixed_dev.name != "mixed_dev.txt" or args.base_dev.name != "base_dev.txt":
        raise ValueError("正式选参工具只接受mixed_dev.txt与base_dev.txt，禁止lock选参")
    if args.expected_images <= 0 or args.beam_size <= 0 or args.workers <= 0:
        raise ValueError("expected-images、beam-size和workers必须为正整数")

    probe_path = args.probe_predictions.expanduser().resolve()
    probe = load_probe(probe_path)
    mixed = read_split(args.mixed_dev)
    base = read_split(args.base_dev)
    mixed_ids = {path.stem for path in mixed}
    base_ids = {path.stem for path in base}
    if (
        len(mixed) != args.expected_images
        or len(mixed_ids) != len(mixed)
        or set(probe.image_ids) != mixed_ids
    ):
        raise RuntimeError("Ascend probe与mixed dev图像集合不完全一致")
    if not base_ids.issubset(mixed_ids):
        raise RuntimeError("base dev不是mixed dev子集")

    context_prior_path = args.context_prior.expanduser().resolve()
    context_prior = json.loads(context_prior_path.read_text(encoding="utf-8"))
    if context_prior.get("source_split") != "incremental_train_only":
        raise ValueError("新类场景先验必须只来自incremental train")
    gates = load_accuracy_gates(args.method_config.expanduser().resolve())

    # Labels are intentionally opened only after the unlabeled probe and image
    # identity checks above.  This process never opens a lock split.
    ground_truth = yolo_ground_truth(mixed)
    baseline = SearchParameters(thresholds=(0.10,) * len(ALL_CLASS_IDS))

    def evaluator(parameters: SearchParameters) -> Dict[str, Any]:
        return score_parameters(
            probe,
            parameters,
            context_prior,
            ground_truth,
            base_ids,
            gates,
            content_gate_enabled=not args.disable_content_execution_gate,
            content_gate_scene_probability=args.content_gate_scene_probability,
        )

    search_options = {
        "beam_size": args.beam_size,
        "threshold_values": args.threshold_values,
        "temperature_values": args.temperature_values,
        "specialist_bias_values": args.specialist_bias_values,
        "scene_penalty_values": args.scene_penalty_values,
        "source_margin_values": args.source_margin_values,
        "overlap_iou_values": args.overlap_iou_values,
        "conflict_iou_values": args.conflict_iou_values,
        "specialist_margin_values": args.specialist_margin_values,
    }
    if args.workers == 1:
        cache, trace = search(baseline, evaluator, **search_options)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_score_worker,
            initargs=(
                probe,
                context_prior,
                ground_truth,
                base_ids,
                gates,
                not args.disable_content_execution_gate,
                args.content_gate_scene_probability,
            ),
        ) as executor:
            cache, trace = search(
                baseline,
                evaluator,
                batch_evaluator=lambda parameters: list(
                    executor.map(score_parameters_worker, parameters, chunksize=1)
                ),
                **search_options,
            )
    passing = [row for row in cache.values() if row["competition_passed"]]
    selected = min(passing, key=lambda row: row["objective"]) if passing else min(
        cache.values(), key=lambda row: row["objective"]
    )
    top = sorted(passing or list(cache.values()), key=lambda row: row["objective"])[:20]
    report = {
        "schema_version": 1,
        "phase": "system_calibration",
        "platform": "Ascend310B1",
        "selection_scope": "mixed_dev_only",
        "lock_split_opened": False,
        "probe_predictions": str(probe_path),
        "image_count": len(mixed),
        "base_image_count": len(base),
        "raw_prediction_count": len(probe.records),
        "context_prior": {
            "path": str(context_prior_path),
            "source_split": context_prior["source_split"],
        },
        "constraints": gates,
        "objective": (
            "minimize_new_class_false_activation_subject_to_all_accuracy_gates;"
            "then maximize_New-mAP50_and_Full-mAP50"
        ),
        "content_execution_gate": {
            "enabled": not args.disable_content_execution_gate,
            "scene": "air",
            "scene_probability_min": args.content_gate_scene_probability,
            "base_evidence_class_ids": [1],
        },
        "evaluated_candidate_count": len(cache),
        "passing_candidate_count": len(passing),
        "baseline": public_evaluation(cache[baseline]),
        "selected": public_evaluation(selected),
        "top_candidates": [public_evaluation(row) for row in top],
        "search_trace": trace,
        "competition_passed": bool(selected["competition_passed"]),
        "candidate_materialized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if candidate_requested and selected["competition_passed"]:
        output_config = args.output_config.expanduser().resolve()
        output_registry = args.output_registry.expanduser().resolve()
        materialize_candidate(
            args.candidate_config.expanduser().resolve(),
            (
                args.candidate_registry.expanduser().resolve()
                if args.candidate_registry is not None
                else None
            ),
            output_config,
            output_registry,
            output,
            context_prior_path,
            context_prior,
            selected,
        )
        report["candidate_materialized"] = True
        report["candidate_config"] = str(output_config)
        report["candidate_registry"] = str(output_registry)
        report["required_scoring_request_confidence"] = min(
            0.01, min(selected["parameters"].threshold_map().values())
        )
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(public_evaluation(selected), ensure_ascii=False, indent=2))
    return 0 if selected["competition_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
