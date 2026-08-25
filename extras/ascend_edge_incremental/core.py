"""Shared feature extraction, replay and scoring for edge adapters."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from PIL import Image

from .protocol import EdgeProtocol


SCENES = ("air", "forest", "sea", "urban")
FEATURE_DIM = 8
RAW_LOGIT_INDEX = 1


class ResidualAdapter(torch.nn.Module):
    """Eight-parameter residual that intentionally avoids MatMul."""

    def __init__(self) -> None:
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(FEATURE_DIM))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features * self.weights).sum(dim=1)


def load_calibration_module(repo_root: Path) -> Any:
    path = repo_root / "tools/113_optimize_ascend_runtime_calibration.py"
    spec = importlib.util.spec_from_file_location("edge_ascend_calibration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import calibration module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_calibration_contract(calibration: Any, protocol: EdgeProtocol) -> None:
    old_ids = tuple(int(value) for value in calibration.OLD_CLASS_IDS)
    new_ids = tuple(int(value) for value in calibration.NEW_CLASS_IDS)
    if old_ids != protocol.base_class_ids or new_ids != protocol.new_class_ids:
        raise ValueError(
            "current Ascend calibration tool and incremental registry disagree: "
            f"calibration={old_ids}+{new_ids}, registry="
            f"{protocol.base_class_ids}+{protocol.new_class_ids}"
        )


def subset_probe(calibration: Any, probe: Any, image_ids: set[str]) -> Any:
    if not image_ids.issubset(probe.image_ids):
        missing = sorted(image_ids - set(probe.image_ids))
        raise RuntimeError(f"probe is missing registered images: {missing[:5]}")
    return calibration.ProbeData(
        tuple(row for row in probe.records if str(row["image_id"]) in image_ids),
        {image_id: probe.contexts[image_id] for image_id in image_ids},
        frozenset(image_ids),
    )


def require_exact_probe(probe: Any, paths: Sequence[Path], label: str) -> None:
    expected = {path.stem for path in paths}
    observed = set(probe.image_ids)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"{label} probe does not exactly match its registered scope: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )


def raw_logit(probability: float) -> float:
    value = min(max(float(probability), 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


def feature_row(
    row: Mapping[str, Any],
    context: Mapping[str, Any],
    size: tuple[int, int],
) -> list[float]:
    width, height = size
    x1, y1, x2, y2 = (float(value) for value in row["xyxy"])
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    scene = context.get("scene_probabilities") or {}
    sensor = context.get("sensor_probabilities") or {}
    return [
        1.0,
        raw_logit(float(row["confidence"])),
        box_width * box_height / (width * height),
        *(float(scene.get(name, 0.0)) for name in SCENES),
        float(sensor.get("ir", 0.0)) - float(sensor.get("sar", 0.0)),
    ]


def image_sizes(paths: Sequence[Path]) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    for path in paths:
        with Image.open(path) as image:
            sizes[path.stem] = tuple(int(value) for value in image.size)
    return sizes


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def identity_states(class_ids: Sequence[int]) -> dict[int, dict[str, torch.Tensor]]:
    return {int(class_id): cpu_state(ResidualAdapter().cpu()) for class_id in class_ids}


def adapter_models(
    states: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[int, ResidualAdapter]:
    models: dict[int, ResidualAdapter] = {}
    for class_id, state in states.items():
        model = ResidualAdapter().cpu().eval()
        model.load_state_dict(state)
        models[int(class_id)] = model
    return models


def adapt_probe(
    calibration: Any,
    probe: Any,
    states: Mapping[int, Mapping[str, torch.Tensor]],
    sizes: Mapping[str, tuple[int, int]],
    scales: Mapping[int, float] | None = None,
) -> Any:
    models = adapter_models(states)
    resolved_scales = {class_id: 1.0 for class_id in models}
    if scales is not None:
        resolved_scales.update({int(key): float(value) for key, value in scales.items()})
    records = [copy.deepcopy(row) for row in probe.records]
    for class_id, model in models.items():
        indices = [
            index
            for index, row in enumerate(records)
            if row.get("source") == "incremental_model"
            and int(row["class_id"]) == class_id
        ]
        if not indices:
            continue
        features = torch.tensor(
            [
                feature_row(
                    records[index],
                    probe.contexts[str(records[index]["image_id"])],
                    sizes[str(records[index]["image_id"])],
                )
                for index in indices
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            residuals = model(features).tolist()
        for index, residual in zip(indices, residuals):
            residual *= resolved_scales[class_id]
            original = float(records[index]["confidence"])
            updated = 1.0 / (1.0 + math.exp(-(raw_logit(original) + residual)))
            records[index]["adapter_raw_confidence"] = updated
            records[index]["adapter_residual_logit"] = residual
            records[index]["confidence"] = updated
    return calibration.ProbeData(
        tuple(records), dict(probe.contexts), frozenset(probe.image_ids)
    )


def load_method(method_config: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(method_config.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid Ascend method config: {method_config}")
    if payload.get("kind") != "ascend310b_full_score_method":
        raise ValueError(f"not an Ascend full-score method config: {method_config}")
    return payload


def frozen_parameters(
    calibration: Any,
    protocol: EdgeProtocol,
    method: Mapping[str, Any],
) -> Any:
    """Rebuild the current frozen production policy without selecting on lock."""

    validate_calibration_contract(calibration, protocol)
    search = method.get("threshold_search") or {}
    raw_thresholds = search.get("selected_per_class_thresholds") or {}
    thresholds = tuple(
        float(raw_thresholds[str(class_id)]) for class_id in protocol.all_class_ids
    )
    score = search.get("score_calibration") or {}
    base_score = score.get("frozen_base_model") or {}
    specialist_score = score.get("incremental_model") or {}
    penalties = search.get("known_scene_soft_penalties") or {}
    return calibration.SearchParameters(
        thresholds=thresholds,
        base_temperature=float(base_score.get("temperature", 1.0)),
        base_bias=float(base_score.get("bias", 0.0)),
        specialist_temperature=float(specialist_score.get("temperature", 1.0)),
        specialist_bias=float(specialist_score.get("bias", 0.0)),
        scene_penalties=tuple(
            float(penalties.get(str(class_id), 0.0))
            for class_id in protocol.new_class_ids
        ),
        conflict_iou=float(search.get("conflict_iou", 0.5)),
        conflict_base_confidence=float(search.get("conflict_base_confidence", 0.5)),
        specialist_margin=float(search.get("specialist_margin", 0.15)),
        cross_class_iou=float(search.get("cross_class_iou", 0.9)),
        smaller_box_coverage=(
            float(search["smaller_box_coverage"])
            if search.get("smaller_box_coverage") is not None
            else None
        ),
        incremental_over_base_margin=float(
            search.get("incremental_over_base_margin", 0.0)
        ),
    )


def accuracy_gates(method: Mapping[str, Any]) -> dict[str, float]:
    raw = (method.get("competition") or {}).get("accuracy_gates") or {}
    return {
        "base_map50": float(raw["base_map50_min"]),
        "new_map50": float(raw["new_map50_min"]),
        "krr": float(raw["krr_min"]),
    }


def load_context_prior(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_split") != "incremental_train_only":
        raise ValueError("new-class context prior is not Increment-train-only")
    return payload


def load_adapter_bank(
    path: Path, protocol: EdgeProtocol
) -> tuple[Mapping[str, Any], dict[int, Mapping[str, torch.Tensor]]]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("adapter checkpoint must contain a mapping")
    if payload.get("protocol_id") != protocol.protocol_id:
        raise ValueError("adapter checkpoint protocol does not match the registry")
    class_order = tuple(int(value) for value in payload.get("class_order") or ())
    if class_order != protocol.new_class_ids:
        raise ValueError("adapter checkpoint class order does not match the registry")
    architecture = payload.get("architecture") or {}
    if (
        architecture.get("kind")
        != "registered_class_matmul_free_logit_residual_bank"
        or int(architecture.get("feature_dim", 0)) != FEATURE_DIM
    ):
        raise ValueError("adapter checkpoint architecture is incompatible")
    raw_states = payload.get("class_states") or {}
    states = {int(key): value for key, value in raw_states.items()}
    if set(states) != set(protocol.new_class_ids):
        raise ValueError("adapter checkpoint classes do not match the registry")
    return payload, states


def false_activation(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    image_ids: set[str],
    class_ids: Sequence[int],
) -> dict[str, float | int]:
    selected = {int(value) for value in class_ids}
    positives = {
        str(row["image_id"])
        for row in ground_truth
        if int(row["class_id"]) in selected
    }
    negatives = image_ids - positives
    activated = {
        str(row["image_id"])
        for row in predictions
        if int(row["class_id"]) in selected
    }
    count = len(negatives & activated)
    return {
        "count": count,
        "negative_images": len(negatives),
        "rate": count / len(negatives) if negatives else 0.0,
    }


def score_view(
    calibration: Any,
    protocol: EdgeProtocol,
    method: Mapping[str, Any],
    probe: Any,
    ground_truth: Sequence[Mapping[str, Any]],
    base_image_ids: set[str],
    context_prior: Mapping[str, Any],
    evaluate_ap50: Any,
    precision_recall: Any,
    retention_metrics: Any,
    subset_rows: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before, combined, counters = calibration.apply_parameters(
        probe,
        frozen_parameters(calibration, protocol, method),
        context_prior,
        content_gate_enabled=True,
        content_gate_scene_probability=0.50,
    )
    final_base = [
        row for row in combined if row.get("source") == "frozen_base_model"
    ]
    base = evaluate_ap50(
        subset_rows(final_base, base_image_ids),
        subset_rows(ground_truth, base_image_ids),
        protocol.base_class_ids,
    )
    new = evaluate_ap50(combined, ground_truth, protocol.new_class_ids)
    full = evaluate_ap50(combined, ground_truth, protocol.all_class_ids)
    retention = retention_metrics(
        final_base, combined, ground_truth, protocol.base_class_ids
    )
    pre_fusion = retention_metrics(
        before, combined, ground_truth, protocol.base_class_ids
    )
    by_class: dict[str, Any] = {}
    image_ids = set(probe.image_ids)
    for class_id in protocol.all_class_ids:
        ap = evaluate_ap50(combined, ground_truth, [class_id])
        operating = precision_recall(combined, ground_truth, class_id, 0.0)
        diagnostic = precision_recall(combined, ground_truth, class_id, 0.63)
        by_class[str(class_id)] = {
            "class_name": protocol.class_names[class_id],
            "ap50": float(ap["map50"]),
            "operating_point": operating,
            "confidence_0_63": diagnostic,
            "false_activation": false_activation(
                combined, ground_truth, image_ids, [class_id]
            ),
        }
    new_pr = [by_class[str(class_id)]["confidence_0_63"] for class_id in protocol.new_class_ids]
    metrics = {
        "image_count": len(image_ids),
        "prediction_count": len(combined),
        "base_map50": float(base["map50"]),
        "new_map50": float(new["map50"]),
        "krr": float(retention["krr"]),
        "old_map50_before": float(retention["old_map50_before"]),
        "old_map50_after": float(retention["old_map50_after"]),
        "old_prediction_equivalent": bool(retention["old_prediction_equivalent"]),
        "pre_fusion_krr_diagnostic": float(pre_fusion["krr"]),
        "full_map50": float(full["map50"]),
        "new_macro_precision_at_0_63": sum(float(row["precision"]) for row in new_pr)
        / len(new_pr),
        "new_macro_recall_at_0_63": sum(float(row["recall"]) for row in new_pr)
        / len(new_pr),
        "new_false_activation": false_activation(
            combined, ground_truth, image_ids, protocol.new_class_ids
        ),
        "per_class": by_class,
        "policy_counters": counters,
    }
    return metrics, combined


def load_scales(
    path: Path | None,
    class_ids: Sequence[int],
    protocol_id: str | None = None,
) -> tuple[dict[int, float], str | None]:
    scales = {int(class_id): 1.0 for class_id in class_ids}
    if path is None:
        return scales, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if protocol_id is not None and payload.get("protocol_id") != protocol_id:
        raise ValueError("adapter scale report protocol does not match the registry")
    selected = payload.get("selected") or {}
    raw = selected.get("scales") or {}
    observed = {int(key): float(value) for key, value in raw.items()}
    if set(observed) != set(scales):
        raise ValueError("adapter scale report does not match registered new classes")
    return observed, str(path.resolve())
