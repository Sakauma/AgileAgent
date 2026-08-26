"""Runtime support for an accepted Ascend edge-incremental Adapter."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


FEATURE_DIM = 8
FEATURE_CONTRACT = "candidate_confidence_context_v1"
SCENES = ("air", "forest", "sea", "urban")


@dataclass(frozen=True)
class EdgeIncrementalAdapter:
    """A small accepted confidence Adapter that is safe to run without Torch."""

    protocol_id: str
    class_order: tuple[int, ...]
    weights: Mapping[int, tuple[float, ...]]
    manifest_path: Path
    run_id: str

    def public_status(self) -> dict[str, Any]:
        return {
            "active": True,
            "protocol_id": self.protocol_id,
            "run_id": self.run_id,
            "class_ids": list(self.class_order),
            "feature_contract": FEATURE_CONTRACT,
        }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_edge_incremental_adapter(
    options: Mapping[str, Any] | None,
    *,
    repo_root: Path,
) -> EdgeIncrementalAdapter | None:
    """Load an accepted demo-channel manifest, or return ``None`` when disabled."""

    settings = dict(options or {})
    if settings.get("enabled") is not True:
        return None
    raw_manifest = str(settings.get("manifest") or "").strip()
    if not raw_manifest:
        raise ValueError("edge incremental Adapter is enabled without a manifest")
    candidate = Path(raw_manifest).expanduser()
    manifest_path = (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported edge incremental Adapter manifest")
    if payload.get("kind") != "ascend_edge_incremental_demo_adapter":
        raise ValueError("not an Ascend edge incremental demo Adapter")
    if payload.get("feature_contract") != FEATURE_CONTRACT:
        raise ValueError("edge incremental Adapter feature contract is incompatible")
    if settings.get("require_accepted", True) is True:
        if payload.get("accepted") is not True:
            raise ValueError("edge incremental Adapter has not passed acceptance")
    if payload.get("production_modified") is not False:
        raise ValueError("edge incremental demo manifest may not claim production mutation")

    protocol_id = str(payload.get("protocol_id") or "").strip()
    required_protocol = str(settings.get("required_protocol_id") or "").strip()
    if not protocol_id or (required_protocol and protocol_id != required_protocol):
        raise ValueError("edge incremental Adapter protocol does not match runtime")
    class_order = tuple(int(value) for value in payload.get("class_order") or ())
    raw_weights = _mapping(payload.get("effective_weights"), "effective_weights")
    weights = {
        int(class_id): tuple(float(value) for value in values)
        for class_id, values in raw_weights.items()
    }
    if not class_order or set(class_order) != set(weights):
        raise ValueError("edge incremental Adapter class order is incomplete")
    if any(len(values) != FEATURE_DIM for values in weights.values()):
        raise ValueError("edge incremental Adapter must provide eight weights per class")
    if any(not math.isfinite(value) for values in weights.values() for value in values):
        raise ValueError("edge incremental Adapter contains a non-finite weight")
    return EdgeIncrementalAdapter(
        protocol_id=protocol_id,
        class_order=class_order,
        weights=weights,
        manifest_path=manifest_path,
        run_id=str(payload.get("run_id") or manifest_path.parent.name),
    )


def _raw_logit(probability: float) -> float:
    value = min(max(float(probability), 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def adapter_features(
    record: Mapping[str, Any],
    context: Mapping[str, Any],
    image_size: Sequence[int],
) -> tuple[float, ...]:
    """Build the exact eight inputs used by edge training and ONNX export."""

    width, height = (max(1, int(value)) for value in image_size)
    x1, y1, x2, y2 = (float(value) for value in record["xyxy"])
    area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)
    scenes = dict(context.get("scene_probabilities") or {})
    sensors = dict(context.get("sensor_probabilities") or {})
    return (
        1.0,
        _raw_logit(float(record["confidence"])),
        area_ratio,
        *(float(scenes.get(scene, 0.0)) for scene in SCENES),
        float(sensors.get("ir", 0.0)) - float(sensors.get("sar", 0.0)),
    )


def apply_edge_incremental_adapter(
    records: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    image_size: Sequence[int],
    adapter: EdgeIncrementalAdapter | None,
) -> list[dict[str, Any]]:
    """Apply the learned residual before frozen production score calibration."""

    if adapter is None:
        return [dict(record) for record in records]
    adapted: list[dict[str, Any]] = []
    for source in records:
        row = dict(source)
        class_id = int(row["class_id"])
        weights = adapter.weights.get(class_id)
        if row.get("source") != "incremental_model" or weights is None:
            adapted.append(row)
            continue
        features = adapter_features(row, context, image_size)
        residual = sum(value * weight for value, weight in zip(features, weights))
        original = float(row["confidence"])
        row["confidence"] = _sigmoid(_raw_logit(original) + residual)
        row["edge_adapter_raw_confidence"] = original
        row["edge_adapter_residual_logit"] = residual
        row["edge_adapter_protocol_id"] = adapter.protocol_id
        adapted.append(row)
    return adapted
