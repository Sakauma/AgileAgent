from __future__ import annotations

import io
import json
import re
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, TypeVar

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

from fair_agent.modules.detection_fusion import (
    arbitrate_cross_class_conflicts,
    box_iou,
    calibrate_record_confidences,
    context_adjusted_threshold,
    context_affinity,
    context_penalty_for_class,
    context_prior_for_class,
    pairwise_box_overlap_metrics,
    suppress_cross_class_overlaps,
)
from fair_agent.modules.edge_incremental_adapter import (
    apply_edge_incremental_adapter,
    load_edge_incremental_adapter,
)
from fair_agent.modules.incremental_rejection import apply_positive_prototype_to_image


CLASS_NAMES = {
    0: "soldier",
    1: "small_aircraft",
    2: "warship",
    3: "tank",
    4: "patrol_boat",
    5: "armored_vehicle",
}
SENSOR_LABELS = {"ir": "红外", "sar": "SAR"}
SCENE_LABELS = {"air": "空域", "forest": "林地", "sea": "海域", "urban": "城市场景"}
ASCEND_MODEL_ROLES = ("scene", "base", "specialist")
CONTENT_EXECUTION_GATE_POLICY = "skip_specialist_on_scene_and_base_evidence_v1"
T = TypeVar("T")


def fixed_neutral_context() -> Dict[str, Any]:
    """Return a score-neutral context payload without running a context model.

    This mode is restricted to explicitly configured Ascend candidates.  It
    preserves the public response schema while making no scene/sensor claim:
    both sensor classes and all four scene classes receive uniform probability.
    """

    return {
        "sensor": "ir",
        "sensor_confidence": 0.5,
        "sensor_probabilities": {"ir": 0.5, "sar": 0.5},
        "scene": "air",
        "scene_confidence": 0.25,
        "scene_probabilities": {
            "air": 0.25,
            "forest": 0.25,
            "sea": 0.25,
            "urban": 0.25,
        },
        "_inference_ms": 0.0,
    }


def _ascend_role_order(
    options: Mapping[str, Any], key: str
) -> tuple[str, str, str]:
    raw = options.get(key, ASCEND_MODEL_ROLES)
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != len(ASCEND_MODEL_ROLES)
        or set(raw) != set(ASCEND_MODEL_ROLES)
    ):
        raise RuntimeError(
            f"Ascend {key}必须是scene/base/specialist的无重复全排列"
        )
    return tuple(str(value) for value in raw)  # type: ignore[return-value]


def _active_specialist_backends(
    protocols: Mapping[str, Mapping[str, Any]],
    specialist_detectors: Mapping[str, Any],
) -> list[Any]:
    return [
        specialist_detectors[protocol_id]
        for protocol_id, protocol in protocols.items()
        if protocol.get("available")
    ]


def _ordered_group_results(
    groups: Mapping[str, Sequence[tuple[str, Callable[[], Any]]]],
    submit_order: Sequence[str],
    collect_order: Sequence[str],
    collector: Callable[[Any], Any],
) -> dict[str, Any]:
    """Submit and drain grouped work in independently configured orders."""

    pending: dict[str, Any] = {}
    submission_error: BaseException | None = None
    for role in submit_order:
        for key, submitter in groups[role]:
            try:
                pending[key] = submitter()
            except BaseException as exc:
                submission_error = exc
                break
        if submission_error is not None:
            break

    completed: dict[str, Any] = {}
    collection_error: BaseException | None = None
    for role in collect_order:
        for key, _submitter in groups[role]:
            if key not in pending:
                continue
            try:
                completed[key] = collector(pending[key])
            except BaseException as exc:
                if collection_error is None:
                    collection_error = exc
    if submission_error is not None:
        raise submission_error
    if collection_error is not None:
        raise collection_error
    return completed


class FairInferenceQueue:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._waiting = 0
        self._active = False
        self._completed = 0
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agile-agent-gpu")

    def status(self) -> Dict[str, int | bool]:
        with self._condition:
            return {
                "waiting": self._waiting,
                "active": self._active,
                "completed": self._completed,
            }

    def run(self, operation: Callable[[], T]) -> tuple[T, float]:
        if self._closed:
            raise RuntimeError("推理队列已经关闭")
        queued_at = time.perf_counter()
        with self._condition:
            self._waiting += 1

        def execute() -> tuple[T, float]:
            with self._condition:
                self._waiting -= 1
                self._active = True
            wait_ms = round((time.perf_counter() - queued_at) * 1000, 1)
            try:
                return operation(), wait_ms
            finally:
                with self._condition:
                    self._active = False
                    self._completed += 1
                    self._condition.notify_all()

        return self._executor.submit(execute).result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)


def decode_image_bytes(
    data: bytes,
    filename: str,
    decode_backend: str = "opencv",
) -> Image.Image:
    if not data:
        raise ValueError(f"无法读取图像：{filename}")
    try:
        if str(decode_backend) == "pillow":
            with Image.open(io.BytesIO(data)) as source:
                source.load()
                image = source.convert("RGB")
        else:
            import cv2
            import numpy as np

            decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"无法读取图像：{filename}")
            image = Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError(f"无法读取图像：{filename}") from exc
    return image


def decode_batch_images(
    items: Iterable[tuple[str, bytes]],
    decode_backend: str = "opencv",
    decode_workers: int = 1,
) -> List[tuple[str, bytes, Image.Image]]:
    rows = list(items)
    if not rows:
        raise ValueError("请选择至少一张图像。")
    decode_workers = min(max(1, int(decode_workers)), len(rows))
    if decode_workers > 1:
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            decoded_rows = list(
                pool.map(lambda item: decode_image_bytes(item[1], item[0], decode_backend), rows)
            )
    else:
        decoded_rows = [decode_image_bytes(data, filename, decode_backend) for filename, data in rows]
    return [
        (filename, data, image)
        for (filename, data), image in zip(rows, decoded_rows)
    ]


def result_records(result: Any, class_names: Mapping[int, str] | None = None) -> List[Dict[str, Any]]:
    names = dict(class_names or CLASS_NAMES)
    native_records = getattr(result, "records", None)
    if native_records is not None:
        source_rows = (
            (
                row["xyxy"],
                row["confidence"],
                row["class_id"],
            )
            for row in native_records
        )
    else:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().tolist()
        source_rows = zip(xyxy, confidences, class_ids)
    rows = []
    for coordinates, confidence, class_id in source_rows:
        numeric_id = int(class_id)
        rows.append(
            {
                "class_id": numeric_id,
                "class_name": names.get(numeric_id, str(numeric_id)),
                "confidence": round(float(confidence), 6),
                "xyxy": [round(float(value), 2) for value in coordinates],
            }
        )
    return rows


def summarize_records(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(item["class_name"]) for item in records).items()))


def yolo_inference_ms(result: Any) -> float:
    speed = getattr(result, "speed", None) or {}
    return max(0.0, float(speed.get("inference", 0.0)))


def yolo_timings(result: Any) -> Dict[str, float]:
    speed = getattr(result, "speed", None) or {}
    return {
        "preprocess_ms": max(0.0, float(speed.get("preprocess", 0.0))),
        "inference_ms": max(0.0, float(speed.get("inference", 0.0))),
        "postprocess_ms": max(0.0, float(speed.get("postprocess", 0.0))),
    }


def remap_specialist_records(
    records: Iterable[Dict[str, Any]],
    global_class_id: int,
    class_names: Mapping[int, str] | None = None,
    protocol_id: str | None = None,
) -> List[Dict[str, Any]]:
    class_name = dict(class_names or CLASS_NAMES).get(global_class_id, str(global_class_id))
    return [
        {
            **item,
            "class_id": global_class_id,
            "class_name": class_name,
            "source": "incremental_model",
            "protocol_id": protocol_id,
        }
        for item in records
    ]


def remap_specialist_records_dynamic(
    records: Iterable[Dict[str, Any]],
    local_to_global: Mapping[int | str, int | str],
    class_names: Mapping[int, str],
    protocol_id: str,
) -> List[Dict[str, Any]]:
    mapping = {int(key): int(value) for key, value in local_to_global.items()}
    names = {int(key): str(value) for key, value in class_names.items()}
    remapped = []
    for item in records:
        local_id = int(item["class_id"])
        if local_id not in mapping:
            continue
        global_id = mapping[local_id]
        remapped.append({
            **item,
            "class_id": global_id,
            "class_name": names.get(global_id, str(global_id)),
            "source": "incremental_model",
            "protocol_id": protocol_id,
        })
    return remapped


def protocol_class_ids(protocol: Mapping[str, Any]) -> List[int]:
    values = protocol.get("global_class_ids")
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes, Mapping)):
        return sorted({int(value) for value in values})
    if protocol.get("global_class_id") is None:
        return []
    return [int(protocol["global_class_id"])]


def protocol_thresholds(protocol: Mapping[str, Any]) -> Dict[int, float]:
    raw = protocol.get("activation_thresholds")
    if isinstance(raw, Mapping):
        return {int(key): float(value) for key, value in raw.items()}
    ids = protocol_class_ids(protocol)
    if len(ids) == 1 and protocol.get("activation_threshold") is not None:
        return {ids[0]: float(protocol["activation_threshold"])}
    return {}


def protocol_independent_class_ids(protocol: Mapping[str, Any]) -> set[int]:
    return {int(value) for value in protocol.get("independent_class_ids", [])}


def protocol_content_execution_gate(
    protocol: Mapping[str, Any],
) -> Dict[str, Any] | None:
    """Return an enabled, validated-at-registry-load execution gate.

    The gate controls whether a *physical* class-incremental expert needs to
    run. It is deliberately separate from the existing soft context gate,
    which only adjusts post-processing thresholds after the expert ran.
    """

    raw = protocol.get("content_execution_gate")
    if not isinstance(raw, Mapping) or raw.get("enabled") is not True:
        return None
    gate = dict(raw)
    if gate.get("policy") != CONTENT_EXECUTION_GATE_POLICY:
        raise RuntimeError("增量专家内容执行门控策略非法")
    return gate


def content_execution_gate_decision(
    protocol: Mapping[str, Any],
    base_records: Iterable[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate a label/filename-independent two-factor specialist gate."""

    gate = protocol_content_execution_gate(protocol)
    if gate is None:
        return {"enabled": False, "skip_specialist": False}
    scene = str(gate["scene"])
    scene_probabilities = context.get("scene_probabilities")
    if not isinstance(scene_probabilities, Mapping):
        scene_probabilities = {}
    scene_probability = float(scene_probabilities.get(scene, 0.0))
    scene_probability_min = float(gate["scene_probability_min"])
    evidence_ids = {int(value) for value in gate["base_evidence_class_ids"]}
    matched_ids = sorted(
        {
            int(row["class_id"])
            for row in base_records
            if int(row["class_id"]) in evidence_ids
        }
    )
    evidence_mode = str(gate.get("base_evidence_mode", "any"))
    base_evidence_passed = (
        evidence_ids.issubset(matched_ids)
        if evidence_mode == "all"
        else bool(matched_ids)
    )
    scene_evidence_passed = scene_probability >= scene_probability_min
    return {
        "enabled": True,
        "policy": CONTENT_EXECUTION_GATE_POLICY,
        "skip_specialist": bool(
            scene_evidence_passed and base_evidence_passed
        ),
        "scene": scene,
        "scene_probability": round(scene_probability, 6),
        "scene_probability_min": scene_probability_min,
        "scene_evidence_passed": scene_evidence_passed,
        "base_evidence_class_ids": sorted(evidence_ids),
        "base_evidence_mode": evidence_mode,
        "matched_base_class_ids": matched_ids,
        "base_evidence_passed": base_evidence_passed,
        "label_aware_online_routing": False,
        "filename_aware_online_routing": False,
    }


def apply_content_execution_gates(
    routes: Iterable[Mapping[str, Any]],
    skipped: Iterable[Mapping[str, Any]],
    base_records: Iterable[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Drop only routes whose scene and Base evidence both satisfy a gate."""

    base_rows = [dict(row) for row in base_records]
    executed: List[Dict[str, Any]] = []
    skipped_rows = [dict(row) for row in skipped]
    decisions: List[Dict[str, Any]] = []
    for raw_route in routes:
        route = dict(raw_route)
        decision = content_execution_gate_decision(
            dict(route["protocol"]), base_rows, context
        )
        if decision["enabled"]:
            audited = {"id": str(route["id"]), **decision}
            decisions.append(audited)
            if decision["skip_specialist"]:
                skipped_rows.append(
                    {
                        "id": str(route["id"]),
                        "reason": "content_execution_gate",
                        **decision,
                    }
                )
                continue
        executed.append(route)
    return executed, skipped_rows, decisions


def protocol_positive_prototypes(
    protocol: Mapping[str, Any],
) -> Dict[int, Dict[str, Any]]:
    raw = protocol.get("positive_prototypes")
    if isinstance(raw, Mapping):
        return {
            int(class_id): dict(prototype)
            for class_id, prototype in raw.items()
            if isinstance(prototype, Mapping)
        }
    single = protocol.get("positive_prototype")
    if isinstance(single, Mapping) and single.get("class_id") is not None:
        return {int(single["class_id"]): dict(single)}
    return {}


def protocol_effective_thresholds(
    protocol: Mapping[str, Any],
    context: Mapping[str, Any],
    confidence_floor: float,
) -> tuple[Dict[int, float], float]:
    """Apply a dev-frozen threshold plus optional known-context soft penalty."""
    gate = dict(protocol.get("context_gate") or {})
    prior = protocol.get("context_prior")
    effective: Dict[int, float] = {}
    affinity = 1.0
    for class_id, threshold in protocol_thresholds(protocol).items():
        class_prior = context_prior_for_class(prior, class_id)
        max_penalty = (
            context_penalty_for_class(gate, class_id)
            if gate.get("enabled") is True
            else 0.0
        )
        adjusted, affinity = context_adjusted_threshold(
            float(threshold), context, class_prior, max_penalty
        )
        effective[int(class_id)] = max(float(confidence_floor), adjusted)
    return effective, affinity


def apply_protocol_thresholds(
    records: Iterable[Mapping[str, Any]],
    thresholds: Mapping[int, float],
    affinity: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = [dict(raw_row) for raw_row in records]
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    keep_mask: np.ndarray | None = None
    if len(rows) >= 8:
        class_ids = np.fromiter(
            (int(row["class_id"]) for row in rows), dtype=np.int64, count=len(rows)
        )
        confidences = np.fromiter(
            (float(row.get("confidence", 0.0)) for row in rows),
            dtype=np.float64,
            count=len(rows),
        )
        has_threshold = np.fromiter(
            (int(class_id) in thresholds for class_id in class_ids),
            dtype=np.bool_,
            count=len(rows),
        )
        threshold_values = np.fromiter(
            (float(thresholds.get(int(class_id), 0.0)) for class_id in class_ids),
            dtype=np.float64,
            count=len(rows),
        )
        keep_mask = ~has_threshold | (confidences >= threshold_values)
    for index, row in enumerate(rows):
        threshold = thresholds.get(int(row["class_id"]))
        passes = (
            bool(keep_mask[index])
            if keep_mask is not None
            else threshold is None
            or float(row.get("confidence", 0.0)) >= float(threshold)
        )
        if passes:
            kept.append(row)
            continue
        rejected.append(
            {
                **row,
                "action": "reject_incremental_candidate",
                "reason": "below_context_adjusted_activation_threshold",
                "effective_activation_threshold": round(float(threshold), 6),
                "context_affinity": round(float(affinity), 6),
            }
        )
    return kept, rejected


def apply_unified_class_gates(
    image: Image.Image,
    records: Iterable[Mapping[str, Any]],
    gates: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    settings = dict(gates or {})
    thresholds, affinity = protocol_effective_thresholds(
        settings,
        dict(context or {}),
        0.0,
    )
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for raw_row in records:
        row = dict(raw_row)
        threshold = thresholds.get(int(row["class_id"]))
        if threshold is not None and float(row.get("confidence", 0.0)) < threshold:
            rejected.append(
                {
                    **row,
                    "action": "reject_unified_activation_threshold",
                    "activation_threshold": threshold,
                    "context_affinity": round(affinity, 6),
                    "reason": "below_context_adjusted_activation_threshold",
                }
            )
        else:
            kept.append(row)
    raw_prototypes = settings.get("positive_prototypes") or {}
    for prototype in raw_prototypes.values():
        if not isinstance(prototype, Mapping):
            continue
        kept, prototype_rejected = apply_positive_prototype_to_image(
            image, kept, prototype
        )
        rejected.extend(prototype_rejected)
    return kept, rejected


def unified_incremental_class_ids(gates: Mapping[str, Any] | None) -> set[int]:
    settings = dict(gates or {})
    if "incremental_class_ids" in settings:
        return {int(class_id) for class_id in settings.get("incremental_class_ids") or []}
    raw = settings.get("activation_thresholds") or {}
    return {int(class_id) for class_id in raw}


def apply_unified_record_ownership(
    records: Iterable[Mapping[str, Any]],
    gates: Mapping[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Preserve logical old/new owners when one detector serves all classes."""

    settings = dict(gates or {})
    new_ids = unified_incremental_class_ids(settings)
    protocol_id = settings.get("protocol_id")
    owned = []
    for raw in records:
        row = dict(raw)
        incremental = int(row["class_id"]) in new_ids
        row["source"] = "incremental_model" if incremental else "frozen_base_model"
        row["protocol_id"] = str(protocol_id) if incremental and protocol_id else None
        owned.append(row)
    return owned


def unified_logical_protocol_result(
    records: Iterable[Mapping[str, Any]],
    rejections: Iterable[Mapping[str, Any]],
    gates: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
) -> Dict[str, Any] | None:
    """Build the existing protocol audit row without running a second OM."""

    settings = dict(gates or {})
    new_ids = unified_incremental_class_ids(settings)
    if not new_ids:
        return None
    kept = [dict(row) for row in records if int(row["class_id"]) in new_ids]
    rejected = [dict(row) for row in rejections if int(row["class_id"]) in new_ids]
    prototype_rejections = [
        row for row in rejected if "prototype" in str(row.get("action", ""))
    ]
    threshold_rejections = [
        row for row in rejected if row not in prototype_rejections
    ]
    thresholds, affinity = protocol_effective_thresholds(
        settings,
        dict(context or {}),
        0.0,
    )
    activated_classes = sorted({str(row["class_name"]) for row in kept})
    protocol_id = str(settings.get("protocol_id") or "unified_incremental")
    configured_names = {
        int(key): str(value)
        for key, value in dict(settings.get("class_names") or {}).items()
    }
    class_names = [configured_names.get(class_id, str(class_id)) for class_id in sorted(new_ids)]
    return {
        "id": protocol_id,
        "class_name": class_names[0] if len(class_names) == 1 else ",".join(class_names),
        "new_class": class_names[0] if len(class_names) == 1 else class_names,
        "incremental_mode": "class_incremental",
        "new_map50": float(settings.get("new_map50", 0.0)),
        "krr": float(settings.get("krr", 0.0)),
        "status": "activated" if kept else "no_candidate",
        "activated": bool(kept),
        "activated_classes": activated_classes,
        "raw_candidate_count": len(kept) + len(rejected),
        "candidate_count": len(kept),
        "conflict_suppressed_count": 0,
        "prototype_rejected_count": len(prototype_rejections),
        "activation_rejected_count": len(threshold_rejections),
        "base_override_count": 0,
        "activation_thresholds": {
            str(key): round(float(value), 2) for key, value in thresholds.items()
        },
        "activation_threshold": (
            round(next(iter(thresholds.values())), 2)
            if len(thresholds) == 1
            else None
        ),
        "context_affinity": round(float(affinity), 6),
        "context_gate_policy": dict(settings.get("context_gate") or {}).get("policy"),
        "routing_score": 1.0,
        "activation_reason": (
            "统一检测器新类通道通过激活门限"
            if kept
            else "统一检测器未产生通过门限的新类候选框"
        ),
        "physical_model_shared": True,
    }


def remap_base_records(
    records: Iterable[Dict[str, Any]],
    local_to_global: Mapping[int, int],
    class_names: Mapping[int, str],
) -> List[Dict[str, Any]]:
    mapping = {int(key): int(value) for key, value in local_to_global.items()}
    names = {int(key): str(value) for key, value in class_names.items()}
    remapped = []
    for item in records:
        local_id = int(item["class_id"])
        if local_id not in mapping:
            raise ValueError(f"基础模型输出未注册的本地类别：{local_id}")
        global_id = mapping[local_id]
        remapped.append(
            {
                **item,
                "class_id": global_id,
                "class_name": names.get(global_id, str(global_id)),
                "source": "frozen_base_model",
                "protocol_id": None,
            }
        )
    return remapped


def consensus_specialist_records(
    base_records: Iterable[Dict[str, Any]],
    specialist_records: Iterable[Dict[str, Any]],
    global_class_id: int,
    min_iou: float,
) -> List[Dict[str, Any]]:
    references = [item for item in base_records if int(item["class_id"]) == global_class_id]
    return [
        item
        for item in specialist_records
        if any(box_iou(item["xyxy"], reference["xyxy"]) >= min_iou for reference in references)
    ]


def compose_incremental_records(
    base_records: Iterable[Dict[str, Any]],
    specialist_records: Iterable[Dict[str, Any]],
    global_class_id: int,
) -> List[Dict[str, Any]]:
    old_records = [
        {**item, "source": "frozen_base_model", "protocol_id": None}
        for item in base_records
    ]
    return old_records + remap_specialist_records(specialist_records, global_class_id)


def plan_specialist_routes(
    protocols: Mapping[str, Mapping[str, Any]],
    base_records: Iterable[Dict[str, Any]],
    context: Mapping[str, Any],
    base_class_ids: Iterable[int],
    max_specialists: int,
    detection_weight: float,
    context_weight: float,
    neutral_context_score: float,
    default_routing_prior: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_rows = list(base_records)
    base_ids = {int(value) for value in base_class_ids}
    eligible: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for protocol_id, raw_protocol in protocols.items():
        protocol = dict(raw_protocol)
        mode = str(protocol.get("incremental_mode") or "target_incremental")
        global_class_ids = protocol_class_ids(protocol)
        if not global_class_ids:
            skipped.append({"id": protocol_id, "reason": "protocol_has_no_classes"})
            continue
        if not protocol.get("available"):
            skipped.append({"id": protocol_id, "reason": "protocol_unavailable"})
            continue
        if mode == "class_incremental":
            if set(global_class_ids) & base_ids:
                skipped.append({"id": protocol_id, "reason": "new_class_id_overlaps_base"})
                continue
            thresholds = protocol_thresholds(protocol)
            sources = protocol.get("calibration_sources")
            if not isinstance(sources, Mapping) and len(global_class_ids) == 1 and protocol.get("calibration_source"):
                sources = {global_class_ids[0]: protocol["calibration_source"]}
            source_ids = {int(key) for key in (sources or {})}
            if set(global_class_ids) - set(thresholds) or set(global_class_ids) - source_ids:
                skipped.append({"id": protocol_id, "reason": "activation_threshold_not_calibrated"})
                continue
            evidence_score = float(protocol.get("routing_prior", default_routing_prior))
            mode_priority = 0
        elif mode == "target_incremental":
            references = [row for row in base_rows if int(row["class_id"]) in set(global_class_ids)]
            independent_ids = protocol_independent_class_ids(protocol) & set(global_class_ids)
            if not references and not independent_ids:
                skipped.append({"id": protocol_id, "reason": "base_class_not_detected"})
                continue
            evidence_score = (
                max(float(row.get("confidence", 0.0)) for row in references)
                if references else float(protocol.get("routing_prior", default_routing_prior))
            )
            mode_priority = 1
        else:
            skipped.append({"id": protocol_id, "reason": "unsupported_incremental_mode"})
            continue
        soft_context = context_affinity(context, protocol.get("context_prior"), neutral_context_score)
        route = {
            "id": protocol_id,
            "protocol": protocol,
            "incremental_mode": mode,
            "evidence_score": round(evidence_score, 6),
            "context_score": round(soft_context, 6),
            "routing_score": round(detection_weight * evidence_score + context_weight * soft_context, 6),
            "mode_priority": mode_priority,
        }
        eligible.append(route)
    eligible.sort(key=lambda row: (int(row["mode_priority"]), -float(row["routing_score"]), str(row["id"])))
    # Class-incremental protocols own classes that the base detector cannot
    # produce. They are therefore mandatory on every unlabeled image and must
    # never be dropped by a latency budget or context score. The budget applies
    # only to optional same-class refinement experts.
    mandatory = [
        route for route in eligible if route["incremental_mode"] == "class_incremental"
    ]
    optional = [
        route for route in eligible if route["incremental_mode"] != "class_incremental"
    ]
    limit = max(0, int(max_specialists))
    optional_limit = 0 if limit == 0 else max(0, limit - len(mandatory))
    selected_optional = optional if limit == 0 else optional[:optional_limit]
    executed = mandatory + selected_optional
    executed.sort(
        key=lambda row: (
            int(row["mode_priority"]),
            -float(row["routing_score"]),
            str(row["id"]),
        )
    )
    for route in optional[len(selected_optional) :]:
        skipped.append({"id": route["id"], "reason": "specialist_budget_exceeded", "routing_score": route["routing_score"]})
    return eligible, executed, skipped


def class_aware_nms(
    records: Iterable[Dict[str, Any]],
    iou_threshold: float = 0.60,
    cross_class_policy: Mapping[str, Any] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = list(records)
    kept: List[Dict[str, Any]] = []
    suppressed = 0
    for class_id in sorted({int(row["class_id"]) for row in rows}):
        class_rows = [row for row in rows if int(row["class_id"]) == class_id]
        if class_rows and all(
            row.get("source") == "frozen_base_model" for row in class_rows
        ):
            # The backend already performed NMS. Keeping this owner stream byte
            # equivalent prevents Agent post-processing from lowering KRR.
            kept.extend({**row, "fusion_status": "base_retained"} for row in class_rows)
            continue
        candidates = sorted(
            class_rows,
            key=lambda row: (-float(row.get("confidence", 0.0)), 0 if row.get("source") == "incremental_model" else 1),
        )
        class_kept: List[Dict[str, Any]] = []
        suppressed_mask: np.ndarray | None = None
        pairwise_iou: np.ndarray | None = None
        if len(candidates) >= 8:
            pairwise_iou, _ = pairwise_box_overlap_metrics(
                [row["xyxy"] for row in candidates],
                [row["xyxy"] for row in candidates],
            )
            suppressed_mask = np.zeros(len(candidates), dtype=np.bool_)
        for index, candidate in enumerate(candidates):
            if suppressed_mask is not None and suppressed_mask[index]:
                suppressed += 1
                continue
            if suppressed_mask is None and any(
                box_iou(candidate["xyxy"], existing["xyxy"]) >= iou_threshold
                for existing in class_kept
            ):
                suppressed += 1
                continue
            status = "specialist_kept" if candidate.get("source") == "incremental_model" else "base_retained"
            class_kept.append({**candidate, "fusion_status": status})
            if suppressed_mask is not None and pairwise_iou is not None:
                suppressed_mask[index + 1 :] |= (
                    pairwise_iou[index, index + 1 :] >= float(iou_threshold)
                )
        kept.extend(class_kept)
    policy = dict(cross_class_policy or {})
    cross_class_decisions: List[Dict[str, Any]] = []
    if policy.get("enabled") is True:
        kept, cross_class_decisions = suppress_cross_class_overlaps(
            kept,
            iou_threshold=float(policy["iou"]),
            smaller_box_coverage=(
                float(policy["smaller_box_coverage"])
                if policy.get("smaller_box_coverage") is not None
                else None
            ),
            incremental_over_base_margin=float(
                policy.get("incremental_over_base_margin", 0.0)
            ),
        )
        suppressed += len(cross_class_decisions)
    kept.sort(key=lambda row: (int(row["class_id"]), -float(row.get("confidence", 0.0))))
    summary: Dict[str, Any] = {
        "input_count": len(rows),
        "output_count": len(kept),
        "suppressed_count": suppressed,
    }
    if policy.get("enabled") is True:
        summary.update(
            {
                "same_class_suppressed_count": suppressed
                - len(cross_class_decisions),
                "cross_class_suppressed_count": len(cross_class_decisions),
                "cross_class_suppressions": cross_class_decisions,
            }
        )
    return kept, summary


def annotate_records(image: Image.Image, records: Iterable[Dict[str, Any]]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    colors = {0: "#159a91", 1: "#3978d4", 2: "#d17a35", 3: "#8b65c8"}
    line_width = max(2, round(min(canvas.size) / 320))
    for item in records:
        class_id = int(item["class_id"])
        color = colors.get(class_id, "#159a91")
        x1, y1, x2, y2 = [float(value) for value in item["xyxy"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
    return canvas


def image_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_annotated_png(source_bytes: bytes, records: Iterable[Dict[str, Any]]) -> bytes:
    with Image.open(io.BytesIO(source_bytes)) as source:
        source.load()
        return image_png_bytes(annotate_records(source.convert("RGB"), records))


def result_json_bytes(payload: Dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def build_batch_zip(results: Iterable[Dict[str, Any]]) -> bytes:
    rows = list(results)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        metadata = []
        for index, item in enumerate(rows, 1):
            source_stem = Path(str(item["filename"])).stem
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_stem).strip("._") or "image"
            annotated_png = item.get("annotated_png")
            if annotated_png is None:
                annotated_png = render_annotated_png(item["source_bytes"], item.get("detections", []))
            archive.writestr(f"annotated/{index:03d}_{safe_stem}.png", annotated_png)
            metadata.append(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"annotated_png", "annotated_image", "source_bytes"}
                }
            )
        archive.writestr("results.json", result_json_bytes({"image_count": len(rows), "results": metadata}))
    return buffer.getvalue()


class WebInferenceEngine:
    def __init__(
        self,
        detector_path: Path,
        context_path: Path,
        generation_id: str,
        base_model_id: str,
        device_index: str = "0",
        predict_options: Mapping[str, Any] | None = None,
        incremental_protocols: Mapping[str, Mapping[str, Any]] | None = None,
        class_names: Mapping[int, str] | None = None,
        base_class_ids: Iterable[int] | None = None,
        base_local_to_global: Mapping[int, int] | None = None,
        routing_options: Mapping[str, Any] | None = None,
        class_owners: Mapping[int, str] | None = None,
        backend_name: str = "ultralytics_cuda",
        native_options: Mapping[str, Any] | None = None,
        unified_class_gates: Mapping[str, Any] | None = None,
    ) -> None:
        self._closed = False
        from fair_agent.backends.inference import create_backend
        from fair_agent.models.context import (
            load_context_model,
            load_tensorrt_context_model,
            predict_context,
            predict_context_batch,
        )

        self.device_index = str(device_index)
        self.device = f"cuda:{self.device_index}"
        self.backend_name = str(backend_name)
        self.native_options = dict(native_options or {})
        self._create_backend = create_backend
        self.detector = create_backend(
            self.backend_name,
            detector_path,
            self.device_index,
            self._backend_options_for_role("base"),
        )
        self.context_mode = str(self.native_options.get("context_mode", "model"))
        self.fixed_neutral_context = (
            self.backend_name == "ascend_acl"
            and self.context_mode == "fixed_neutral_v1"
        )
        self.context_model_id = (
            "fixed_neutral_context_v1"
            if self.fixed_neutral_context
            else "scene_sensor_net_v1"
        )
        if self.backend_name == "ascend_acl":
            from fair_agent.backends.ascend_acl import load_ascend_context_model

            self.context_model, self.context_checkpoint = load_ascend_context_model(
                self.native_options
            )
            self.ascend_stream_priority_status = dict(
                self.detector.model.runtime.stream_priority_status
                or {
                    "requested": False,
                    "supported": False,
                    "reason": "not_requested",
                    "values": {},
                }
            )
            self.device = f"ascend:{self.device_index}"
        elif self.backend_name == "tensorrt_engine":
            self.context_model, self.context_checkpoint = load_tensorrt_context_model(
                context_path,
                dict(self.native_options["context_engine"]),
                self.device,
            )
        else:
            self.context_model, self.context_checkpoint = load_context_model(context_path, self.device)
        self.class_names = {int(key): str(value) for key, value in (class_names or CLASS_NAMES).items()}
        configured_base_ids = {int(value) for value in (base_class_ids or self.class_names)}
        self.base_local_to_global = {
            int(key): int(value)
            for key, value in (base_local_to_global or {class_id: class_id for class_id in configured_base_ids}).items()
        }
        self.base_class_ids = set(self.base_local_to_global.values())
        self.generation_id = str(generation_id)
        self.base_model_id = str(base_model_id)
        self.class_owners = {
            int(key): str(value)
            for key, value in (class_owners or {class_id: self.base_model_id for class_id in self.base_class_ids}).items()
        }
        self.base_local_names = {
            local_id: self.class_names[global_id] for local_id, global_id in self.base_local_to_global.items()
        }
        self.unified_class_gates = dict(unified_class_gates or {})
        options = dict(predict_options or {})
        routing = dict(routing_options or {})
        self.imgsz = int(options["imgsz"])
        self.specialist_imgsz = int(options["specialist_imgsz"])
        self.iou = float(options["iou"])
        self.max_det = int(options["max_det"])
        self.batch_size = int(options["batch_size"])
        self.default_confidence = float(options["confidence_default"])
        self.warmup_iterations = int(options["warmup_iterations"])
        self.warmup_batch_size = int(options["warmup_batch_size"])
        self.warmup_width = int(options["warmup_width"])
        self.warmup_height = int(options["warmup_height"])
        self.preload_specialists = bool(options["preload_specialists"])
        self.quantize = options["quantize"]
        self.compile = bool(options["compile"])
        self.fusion_iou = float(routing["fusion_iou"])
        self.cross_class_suppression = dict(
            routing.get("cross_class_suppression") or {"enabled": False}
        )
        self.score_calibration = dict(
            routing.get("score_calibration") or {"enabled": False}
        )
        self.edge_incremental_adapter = load_edge_incremental_adapter(
            routing.get("edge_incremental_adapter"),
            repo_root=Path(__file__).resolve().parents[2],
        )
        self.edge_incremental_adapter_status = (
            self.edge_incremental_adapter.public_status()
            if self.edge_incremental_adapter is not None
            else {"active": False}
        )
        self.max_specialists = int(routing["max_specialists_per_image"])
        self.conflict_iou = float(routing["conflict_iou"])
        self.conflict_incremental_coverage = (
            float(routing["conflict_incremental_coverage"])
            if routing.get("conflict_incremental_coverage") is not None
            else None
        )
        self.conflict_base_confidence = float(routing["conflict_base_confidence"])
        self.specialist_margin = float(routing["specialist_margin"])
        self.preserve_base_class_owners = bool(routing["preserve_base_class_owners"])
        self.detection_evidence_weight = float(routing["detection_evidence_weight"])
        self.context_evidence_weight = float(routing["context_evidence_weight"])
        self.neutral_context_score = float(routing["neutral_context_score"])
        self.default_routing_prior = float(routing["default_routing_prior"])
        self.parallel_model_execution = bool(routing["parallel_model_execution"])
        self.parallel_context_execution = bool(routing["parallel_context_execution"])
        self.parallel_context_batch_execution = bool(routing["parallel_context_batch_execution"])
        self.max_model_workers = int(routing["max_model_workers"])
        if self.backend_name == "ascend_acl":
            self.context_stream = None
            self.parallel_context_batch_execution = False
        else:
            import torch

            torch.backends.cudnn.benchmark = bool(options["cudnn_benchmark"])
            self.context_stream = torch.cuda.Stream(device=int(self.device_index))
        self._model_executor = ThreadPoolExecutor(max_workers=self.max_model_workers)
        self._encoded_hardware_lock = threading.Lock()
        self._encoded_batch_executor = ThreadPoolExecutor(max_workers=2)
        context_size = int(self.context_checkpoint["preprocessing"]["image_size"])
        warmup_image = Image.new("RGB", (self.warmup_width, self.warmup_height))
        context_warmup = Image.new("RGB", (context_size, context_size))
        for _ in range(self.warmup_iterations):
            if getattr(self, "fixed_neutral_context", False):
                pass
            elif self.backend_name == "ascend_acl":
                self.context_model.predict(context_warmup)
            else:
                predict_context(
                    self.context_model,
                    self.context_checkpoint,
                    context_warmup,
                    self.device,
                    self.context_stream,
                )
            self.detector.predict(
                warmup_image,
                imgsz=self.imgsz,
                conf=self.default_confidence,
                iou=self.iou,
                max_det=self.max_det,
                quantize=self.quantize,
                compile=self.compile,
            )
        warmup_batch = [warmup_image] * self.warmup_batch_size
        if getattr(self, "fixed_neutral_context", False):
            pass
        elif self.backend_name == "ascend_acl":
            self.context_model.predict_batch([context_warmup] * self.warmup_batch_size)
        else:
            predict_context_batch(
                self.context_model,
                self.context_checkpoint,
                [context_warmup] * self.warmup_batch_size,
                self.device,
                self.context_stream,
            )
        self.detector.predict_batch(
            warmup_batch,
            imgsz=self.imgsz,
            conf=self.default_confidence,
            iou=self.iou,
            max_det=self.max_det,
            quantize=self.quantize,
            compile=self.compile,
        )
        self.incremental_protocols = {name: dict(value) for name, value in (incremental_protocols or {}).items()}
        self.specialist_detectors: Dict[str, Any] = {}
        if self.preload_specialists:
            for protocol_id, protocol in self.incremental_protocols.items():
                if not protocol.get("available"):
                    continue
                backend = self._create_backend(
                    self.backend_name,
                    protocol["weights"],
                    self.device_index,
                    self._backend_options_for_role("specialist"),
                )
                backend.predict(
                    warmup_image,
                    imgsz=self.specialist_imgsz,
                    conf=self.default_confidence,
                    iou=self.iou,
                    max_det=self.max_det,
                    quantize=self.quantize,
                    compile=self.compile,
                )
                backend.predict_batch(
                    warmup_batch,
                    imgsz=self.specialist_imgsz,
                    conf=self.default_confidence,
                    iou=self.iou,
                    max_det=self.max_det,
                    quantize=self.quantize,
                    compile=self.compile,
                )
                self.specialist_detectors[protocol_id] = backend
        self.encoded_preprocessor = None
        if (
            self.backend_name == "ascend_acl"
            and self.native_options.get("encoded_preprocessing", "cpu") == "dvpp"
        ):
            active_specialists = _active_specialist_backends(
                self.incremental_protocols,
                self.specialist_detectors,
            )
            if len(active_specialists) != 1:
                raise RuntimeError("Ascend DVPP设备预处理当前要求恰好一个活动增量模型")
            if protocol_positive_prototypes(self.unified_class_gates) or any(
                protocol_positive_prototypes(protocol)
                for protocol in self.incremental_protocols.values()
                if protocol.get("available")
            ):
                raise RuntimeError("Ascend DVPP编码输入不支持启用像素原型门控")
            from fair_agent.backends.ascend_acl import AscendEncodedPreprocessor

            self.encoded_preprocessor = AscendEncodedPreprocessor(
                self.detector.model.runtime,
                self.detector.model,
                active_specialists[0].model,
                self.context_model.model,
                scene_resize_stages=self.native_options.get(
                    "dvpp_scene_resize_stages", []
                ),
                prepare_context=not self.fixed_neutral_context,
            )
            self._encoded_image_stub = Image.new(
                "RGB",
                (
                    self.encoded_preprocessor.source_width,
                    self.encoded_preprocessor.source_height,
                ),
            )
        self.queue = FairInferenceQueue()

    def _backend_options_for_role(self, role: str) -> Mapping[str, Any]:
        if self.backend_name != "ascend_acl":
            return self.native_options
        if role not in {"base", "specialist"}:
            raise RuntimeError(f"Ascend检测后端角色非法：{role}")
        return {**self.native_options, "_stream_role": role}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        queue = getattr(self, "queue", None)
        if queue is not None:
            queue.close()
        batch_executor = getattr(self, "_encoded_batch_executor", None)
        if batch_executor is not None:
            batch_executor.shutdown(wait=True, cancel_futures=True)
        executor = getattr(self, "_model_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        resources = [
            getattr(self, "encoded_preprocessor", None),
            *getattr(self, "specialist_detectors", {}).values(),
            getattr(self, "detector", None),
            getattr(self, "context_model", None),
        ]
        closed: set[int] = set()
        for resource in resources:
            if resource is None or id(resource) in closed:
                continue
            closed.add(id(resource))
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def queue_status(self) -> Dict[str, int | bool]:
        return self.queue.status()

    def predict(
        self,
        image: Image.Image,
        filename: str,
        confidence: float | None = None,
        incremental_protocol: str | None = None,
    ) -> Dict[str, Any]:
        result, queue_wait_ms = self.queue.run(
            lambda: self._predict_unlocked(
                image, filename, self.default_confidence if confidence is None else confidence,
                incremental_protocol,
            )
        )
        result["queue_wait_ms"] = queue_wait_ms
        return result

    def accepts_encoded(self, data: bytes) -> bool:
        preprocessor = getattr(self, "encoded_preprocessor", None)
        return preprocessor is not None and preprocessor.accepts(data)

    def predict_encoded(
        self,
        data: bytes,
        filename: str,
        confidence: float | None = None,
        incremental_protocol: str | None = None,
        speculative_specialists: bool = False,
    ) -> Dict[str, Any]:
        if not self.accepts_encoded(data):
            raise ValueError("当前Ascend编码输入不符合DVPP固定生产契约")
        resolved_confidence = (
            self.default_confidence if confidence is None else confidence
        )
        result, queue_wait_ms = self.queue.run(
            lambda: (
                self._predict_encoded_unlocked(
                    data,
                    filename,
                    resolved_confidence,
                    incremental_protocol,
                    True,
                )
                if speculative_specialists
                else self._predict_encoded_unlocked(
                    data,
                    filename,
                    resolved_confidence,
                    incremental_protocol,
                )
            )
        )
        result["queue_wait_ms"] = queue_wait_ms
        return result

    def predict_encoded_batch(
        self,
        items: Iterable[tuple[bytes, str]],
        confidence: float | None = None,
        incremental_protocol: str | None = "auto",
        speculative_specialists: bool = False,
    ) -> List[Dict[str, Any]]:
        rows = list(items)
        if not rows:
            raise ValueError("请选择至少一张图像。")
        if not all(self.accepts_encoded(data) for data, _filename in rows):
            raise ValueError("批量Ascend编码输入不符合DVPP固定生产契约")
        resolved_confidence = (
            self.default_confidence if confidence is None else confidence
        )
        pipeline_hardware = (
            speculative_specialists
            and len(rows) > 1
            and self.backend_name == "ascend_acl"
            and self.native_options.get("execution_mode", "synchronous")
            == "async_stream"
            and self.native_options.get("schedule_mode", "threaded_execute")
            == "unified_enqueue"
            and not bool(self.native_options.get("detailed_event_timing", False))
        )

        def predict_rows() -> List[Dict[str, Any]]:
            if pipeline_hardware:
                futures = [
                    self._encoded_batch_executor.submit(
                        self._predict_encoded_unlocked,
                        data,
                        filename,
                        resolved_confidence,
                        incremental_protocol,
                        True,
                        True,
                    )
                    for data, filename in rows
                ]
                return [future.result() for future in futures]
            return [
                (
                    self._predict_encoded_unlocked(
                        data,
                        filename,
                        resolved_confidence,
                        incremental_protocol,
                        True,
                    )
                    if speculative_specialists
                    else self._predict_encoded_unlocked(
                        data,
                        filename,
                        resolved_confidence,
                        incremental_protocol,
                    )
                )
                for data, filename in rows
            ]

        results, queue_wait_ms = self.queue.run(predict_rows)
        for result in results:
            result["queue_wait_ms"] = queue_wait_ms
        return results

    def _predict_encoded_unlocked(
        self,
        data: bytes,
        filename: str,
        confidence: float,
        incremental_protocol: str | None,
        speculative_specialists: bool = False,
        pipeline_hardware: bool = False,
    ) -> Dict[str, Any]:
        hardware_release: Callable[[], None] | None = None
        hardware_locked = False
        hardware_wait_started = time.perf_counter()
        hardware_started = hardware_wait_started
        hardware_elapsed_ms = 0.0
        if pipeline_hardware:
            self._encoded_hardware_lock.acquire()
            hardware_locked = True
            hardware_started = time.perf_counter()

            def release_hardware() -> None:
                nonlocal hardware_elapsed_ms, hardware_locked
                if hardware_locked:
                    hardware_elapsed_ms = (
                        time.perf_counter() - hardware_started
                    ) * 1000
                    hardware_locked = False
                    self._encoded_hardware_lock.release()

            hardware_release = release_hardware
        try:
            result = self._predict_unlocked(
                self._encoded_image_stub,
                filename,
                confidence,
                incremental_protocol,
                encoded_data=data,
                speculative_specialists=speculative_specialists,
                hardware_release=hardware_release,
            )
            if pipeline_hardware:
                result.setdefault("timings", {}).update(
                    {
                        "pipeline_hardware_wait_ms": round(
                            (hardware_started - hardware_wait_started) * 1000,
                            3,
                        ),
                        "pipeline_hardware_ms": round(hardware_elapsed_ms, 3),
                    }
                )
            return result
        finally:
            if hardware_locked:
                self._encoded_hardware_lock.release()

    def predict_batch(
        self,
        items: Iterable[tuple[Image.Image, str]],
        confidence: float | None = None,
        incremental_protocol: str | None = "auto",
    ) -> List[Dict[str, Any]]:
        rows = list(items)
        if not rows:
            return []
        results, queue_wait_ms = self.queue.run(
            lambda: self._predict_batch_unlocked(
                rows, self.default_confidence if confidence is None else confidence, incremental_protocol
            )
        )
        for result in results:
            result["queue_wait_ms"] = queue_wait_ms
        return results

    def _predict_batch_unlocked(
        self,
        items: List[tuple[Image.Image, str]],
        confidence: float,
        incremental_protocol: str | None,
    ) -> List[Dict[str, Any]]:
        from fair_agent.models.context import predict_context_batch

        batch_started = time.perf_counter()
        images = [
            image if image.mode == "RGB" else image.convert("RGB")
            for image, _filename in items
        ]
        automatic = incremental_protocol == "auto"
        if automatic:
            protocol_pool = self.incremental_protocols
        elif incremental_protocol:
            protocol = self.incremental_protocols.get(incremental_protocol)
            if protocol is None or not protocol.get("available"):
                raise ValueError("所选增量能力当前不可用。")
            protocol_pool = {incremental_protocol: protocol}
        else:
            protocol_pool = {}

        def context_batch_task() -> list[Dict[str, Any]]:
            if getattr(self, "fixed_neutral_context", False):
                return [fixed_neutral_context() for _image in images]
            if self.backend_name == "ascend_acl":
                return self.context_model.predict_batch(images)
            return predict_context_batch(
                self.context_model,
                self.context_checkpoint,
                images,
                self.device,
                self.context_stream,
            )

        def detector_batch_task(
            backend: Any, imgsz: int, selected_images: Sequence[Image.Image] | None = None
        ) -> Sequence[Any]:
            source_images = list(selected_images or images)
            predictions: List[Any] = []
            for start in range(0, len(source_images), self.batch_size):
                predictions.extend(
                    backend.predict_batch(
                        source_images[start:start + self.batch_size],
                        imgsz=imgsz,
                        conf=float(confidence),
                        iou=self.iou,
                        max_det=self.max_det,
                        quantize=self.quantize,
                        compile=self.compile,
                    )
                )
            return predictions

        route_candidate_ids = [
            protocol_id for protocol_id, protocol in protocol_pool.items()
            if protocol.get("available") and (
                protocol.get("incremental_mode") == "class_incremental"
                or protocol_independent_class_ids(protocol)
            )
        ][: self.max_specialists]
        prefetch_ids = [
            protocol_id
            for protocol_id in route_candidate_ids
            if protocol_content_execution_gate(protocol_pool[protocol_id]) is None
        ]
        for protocol_id in prefetch_ids:
            protocol = protocol_pool[protocol_id]
            if protocol_id not in self.specialist_detectors:
                self.specialist_detectors[protocol_id] = self._create_backend(
                    self.backend_name,
                    protocol["weights"],
                    self.device_index,
                    self._backend_options_for_role("specialist"),
                )
        prefetched_batches: Dict[str, Sequence[Any]] = {}
        if self.parallel_model_execution:
            context_future = self._model_executor.submit(context_batch_task)
            detector_future = self._model_executor.submit(detector_batch_task, self.detector, self.imgsz)
            specialist_futures = {
                protocol_id: self._model_executor.submit(
                    detector_batch_task, self.specialist_detectors[protocol_id], self.specialist_imgsz
                )
                for protocol_id in prefetch_ids
            }
            contexts = context_future.result()
            base_predictions = detector_future.result()
            prefetched_batches = {
                protocol_id: future.result() for protocol_id, future in specialist_futures.items()
            }
        elif self.parallel_context_batch_execution:
            context_future = self._model_executor.submit(context_batch_task)
            base_predictions = detector_batch_task(self.detector, self.imgsz)
            contexts = context_future.result()
        else:
            contexts = context_batch_task()
            base_predictions = detector_batch_task(self.detector, self.imgsz)
        raw_base_records_by_image = [
            calibrate_record_confidences(
                remap_base_records(
                    result_records(prediction, self.base_local_names),
                    self.base_local_to_global,
                    self.class_names,
                ),
                getattr(self, "score_calibration", None),
            )
            for prediction in base_predictions
        ]
        base_records_by_image: List[List[Dict[str, Any]]] = []
        unified_gate_decisions_by_image: List[List[Dict[str, Any]]] = []
        for image, records, context in zip(
            images, raw_base_records_by_image, contexts
        ):
            gated, decisions = apply_unified_class_gates(
                image,
                records,
                getattr(self, "unified_class_gates", {}),
                context,
            )
            base_records_by_image.append(gated)
            unified_gate_decisions_by_image.append(decisions)

        raw_route_rows = [
            plan_specialist_routes(
                protocol_pool,
                base_records,
                context,
                self.base_class_ids,
                self.max_specialists if automatic else 1,
                self.detection_evidence_weight,
                self.context_evidence_weight,
                self.neutral_context_score,
                self.default_routing_prior,
            )
            for base_records, context in zip(base_records_by_image, contexts)
        ]
        route_rows = []
        content_gate_decisions_by_image: List[List[Dict[str, Any]]] = []
        for base_records, context, (eligible, executed, skipped) in zip(
            base_records_by_image, contexts, raw_route_rows
        ):
            filtered, skipped_rows, gate_decisions = apply_content_execution_gates(
                executed, skipped, base_records, context
            )
            route_rows.append((eligible, filtered, skipped_rows))
            content_gate_decisions_by_image.append(gate_decisions)
        protocol_outputs: List[List[Dict[str, Any]]] = []
        for records, rejections, context in zip(
            base_records_by_image,
            unified_gate_decisions_by_image,
            contexts,
        ):
            logical = unified_logical_protocol_result(
                records,
                rejections,
                getattr(self, "unified_class_gates", {}),
                context,
            )
            protocol_outputs.append([logical] if logical is not None else [])
        specialists_by_image: List[List[Dict[str, Any]]] = [[] for _ in images]
        conflicts_by_image: List[List[Dict[str, Any]]] = [
            list(rows) for rows in unified_gate_decisions_by_image
        ]
        specialist_timing_by_image = [
            {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0}
            for _ in images
        ]

        for protocol_id, protocol_raw in protocol_pool.items():
            selected = [
                index
                for index, (_eligible, executed, _skipped) in enumerate(route_rows)
                if any(str(route["id"]) == str(protocol_id) for route in executed)
            ]
            if not selected:
                continue
            protocol = dict(protocol_raw)
            if protocol_id not in self.specialist_detectors:
                self.specialist_detectors[protocol_id] = self._create_backend(
                    self.backend_name,
                    protocol["weights"],
                    self.device_index,
                    self._backend_options_for_role("specialist"),
                )
            if protocol_id in prefetched_batches:
                predictions = [prefetched_batches[protocol_id][index] for index in selected]
            else:
                predictions = detector_batch_task(
                    self.specialist_detectors[protocol_id],
                    self.specialist_imgsz,
                    [images[index] for index in selected],
                )
            for image_index, prediction in zip(selected, predictions):
                timing = yolo_timings(prediction)
                for key in specialist_timing_by_image[image_index]:
                    specialist_timing_by_image[image_index][key] += timing[key]
                route = next(
                    row for row in route_rows[image_index][1] if str(row["id"]) == str(protocol_id)
                )
                class_ids = protocol_class_ids(protocol)
                local_to_global = protocol.get("local_to_global")
                if not isinstance(local_to_global, Mapping):
                    local_to_global = {0: class_ids[0]}
                effective_thresholds, context_score = protocol_effective_thresholds(
                    protocol, contexts[image_index], float(confidence)
                )
                remapped_candidates = remap_specialist_records_dynamic(
                    result_records(prediction),
                    local_to_global,
                    self.class_names,
                    str(protocol_id),
                )
                adapted_candidates = apply_edge_incremental_adapter(
                    remapped_candidates,
                    contexts[image_index],
                    images[image_index].size,
                    getattr(self, "edge_incremental_adapter", None),
                )
                raw_candidates = calibrate_record_confidences(
                    adapted_candidates,
                    getattr(self, "score_calibration", None),
                )
                threshold_candidates, threshold_rejections = apply_protocol_thresholds(
                    raw_candidates, effective_thresholds, context_score
                )
                prototype_rejections: List[Dict[str, Any]] = []
                for prototype in protocol_positive_prototypes(protocol).values():
                    threshold_candidates, rejected_by_prototype = apply_positive_prototype_to_image(
                        images[image_index], threshold_candidates, prototype
                    )
                    prototype_rejections.extend(rejected_by_prototype)
                independent_ids = protocol_independent_class_ids(protocol)
                if protocol.get("incremental_mode") == "class_incremental":
                    candidates = threshold_candidates
                    activation_reason = (
                        "通过新类置信度与正样本原型门控"
                        if protocol_positive_prototypes(protocol)
                        else "通过独立新类置信度门限"
                    )
                else:
                    candidates = []
                    for class_id in class_ids:
                        class_candidates = [
                            item for item in threshold_candidates if int(item["class_id"]) == class_id
                        ]
                        if class_id in independent_ids:
                            candidates.extend(class_candidates)
                        else:
                            candidates.extend(consensus_specialist_records(
                                base_records_by_image[image_index], class_candidates, class_id,
                                float(protocol.get("consensus_iou", 0.30)),
                            ))
                    activation_reason = (
                        "通过冻结专家链独立置信度门限"
                        if independent_ids else "通过基础同类目标与空间一致性检查"
                    )
                base_records_by_image[image_index], candidates, conflict_decisions = arbitrate_cross_class_conflicts(
                    base_records_by_image[image_index],
                    candidates,
                    self.conflict_iou,
                    self.conflict_base_confidence,
                    self.specialist_margin,
                    protocol.get("confusion_graph"),
                    self.preserve_base_class_owners,
                    self.conflict_incremental_coverage,
                )
                rejected = [
                    row for row in conflict_decisions if row["action"] == "reject_specialist"
                ]
                activated = bool(candidates)
                activated_class_names = sorted({str(item["class_name"]) for item in candidates})
                specialists_by_image[image_index].extend(candidates)
                conflicts_by_image[image_index].extend(
                    threshold_rejections + prototype_rejections + conflict_decisions
                )
                protocol_outputs[image_index].append({
                    "id": str(protocol_id),
                    "class_name": protocol["class_name"],
                    "new_class": protocol["class_name"],
                    "incremental_mode": protocol["incremental_mode"],
                    "new_map50": protocol["new_map50"],
                    "krr": protocol["krr"],
                    "status": "activated" if activated else "no_candidate",
                    "activated": activated,
                    "activated_classes": activated_class_names,
                    "raw_candidate_count": len(raw_candidates),
                    "candidate_count": len(candidates),
                    "conflict_suppressed_count": len(rejected),
                    "prototype_rejected_count": len(prototype_rejections),
                    "activation_rejected_count": len(threshold_rejections),
                    "base_override_count": sum(
                        row["action"] == "suppress_base" for row in conflict_decisions
                    ),
                    "activation_thresholds": {
                        str(key): round(value, 2) for key, value in effective_thresholds.items()
                    },
                    "activation_threshold": (
                        round(next(iter(effective_thresholds.values())), 2)
                        if len(effective_thresholds) == 1 else None
                    ),
                    "context_affinity": round(context_score, 6),
                    "context_gate_policy": dict(protocol.get("context_gate") or {}).get(
                        "policy"
                    ),
                    "routing_score": route["routing_score"],
                    "activation_reason": activation_reason if activated else "未产生通过门限的候选框",
                })

        results: List[Dict[str, Any]] = []
        batch_total_ms = (time.perf_counter() - batch_started) * 1000
        per_image_batch_total = batch_total_ms / len(images)
        for index, ((_, filename), image, context, base_prediction, base_records) in enumerate(
            zip(items, images, contexts, base_predictions, base_records_by_image)
        ):
            context_inference_ms = float(context.pop("_inference_ms", 0.0))
            detector_timing = yolo_timings(base_prediction)
            specialist_timing = specialist_timing_by_image[index]
            inference_ms = (
                context_inference_ms
                + detector_timing["inference_ms"]
                + specialist_timing["inference_ms"]
            )
            base_with_source = apply_unified_record_ownership(
                base_records,
                getattr(self, "unified_class_gates", {}),
            )
            records, fusion_summary = class_aware_nms(
                base_with_source + specialists_by_image[index],
                self.fusion_iou,
                getattr(self, "cross_class_suppression", None),
            )
            fusion_summary["conflict_suppressed_count"] = len(conflicts_by_image[index])
            eligible, executed, skipped = route_rows[index]
            activated_classes = sorted({
                class_name
                for item in protocol_outputs[index]
                for class_name in item.get("activated_classes", [])
            })
            models_used = [
                getattr(self, "context_model_id", "scene_sensor_net_v1"),
                self.base_model_id,
            ]
            models_used.extend(str(route["id"]) for route in executed)
            gate_decisions = content_gate_decisions_by_image[index]
            decision = {
                "mode": "automatic" if automatic else ("manual" if protocol_pool else "unified_only"),
                "input_mode": "unlabeled_image",
                "inference_scope": "production",
                "routing_basis": "all_class_owners_plus_optional_content_routing",
                "class_incremental_execution_policy": (
                    "scene_and_base_evidence_gate"
                    if gate_decisions
                    else "every_image"
                ),
                "label_aware_routing": False,
                "scene_hard_routing": bool(gate_decisions),
                "content_execution_gates": gate_decisions,
                "evaluated_specialists": len(executed),
                "base_detection_count": sum(
                    row["source"] == "frozen_base_model" for row in base_with_source
                ),
                "final_detection_count": len(records),
                "activated_classes": activated_classes,
                "eligible_protocols": [
                    {"id": row["id"], "routing_score": row["routing_score"], "incremental_mode": row["incremental_mode"]}
                    for row in eligible
                ],
                "executed_protocols": [str(row["id"]) for row in executed],
                "skipped_protocols": skipped,
                "fusion_summary": fusion_summary,
                "conflict_suppressions": conflicts_by_image[index],
                "reason": "已融合通过模式感知门控的专项候选" if activated_classes else "未发现通过模式感知门控的专项候选，保持冻结基础检测结果",
                "generation_id": self.generation_id,
                "base_model_id": self.base_model_id,
                "class_owners": {str(key): value for key, value in self.class_owners.items()},
                "edge_incremental_adapter": getattr(
                    self, "edge_incremental_adapter_status", {"active": False}
                ),
            }
            results.append({
                "filename": filename,
                "image_width": int(image.width),
                "image_height": int(image.height),
                "context": context,
                "detections": records,
                "class_counts": summarize_records(records),
                "detection_count": len(records),
                "confidence_threshold": round(float(confidence), 2),
                "inference_ms": round(inference_ms, 1),
                "timings": {
                    "context_inference_ms": round(context_inference_ms, 3),
                    "detector_preprocess_ms": round(detector_timing["preprocess_ms"], 3),
                    "detector_inference_ms": round(detector_timing["inference_ms"], 3),
                    "detector_postprocess_ms": round(detector_timing["postprocess_ms"], 3),
                    "specialist_preprocess_ms": round(specialist_timing["preprocess_ms"], 3),
                    "specialist_inference_ms": round(specialist_timing["inference_ms"], 3),
                    "specialist_postprocess_ms": round(specialist_timing["postprocess_ms"], 3),
                    "batch_engine_total_per_image_ms": round(per_image_batch_total, 3),
                },
                "agent": {
                    "mode": "automatic_orchestration" if automatic else "standard_detection",
                    "models_used": models_used,
                    "protocol": protocol_outputs[index][0] if not automatic and protocol_outputs[index] else None,
                    "protocols": protocol_outputs[index],
                    "decision": decision,
                },
            })
        return results

    def _predict_unlocked(
        self,
        image: Image.Image,
        filename: str,
        confidence: float,
        incremental_protocol: str | None,
        encoded_data: bytes | None = None,
        speculative_specialists: bool = False,
        hardware_release: Callable[[], None] | None = None,
    ) -> Dict[str, Any]:
        from fair_agent.models.context import predict_context

        pipeline_started = time.perf_counter()
        rgb_image = image if image.mode == "RGB" else image.convert("RGB")
        ascend_rgb_array = None
        dvpp_preprocess_ms = 0.0
        dvpp_device_ms = 0.0
        ascend_submit_ms = 0.0
        ascend_wait_ms = 0.0
        ascend_input_copy_max_ms = 0.0
        ascend_output_copy_max_ms = 0.0
        encoded_preloaded = encoded_data is not None
        if encoded_preloaded:
            if self.encoded_preprocessor is None:
                raise RuntimeError("Ascend DVPP编码预处理器尚未初始化")
            dvpp_preprocess_ms = self.encoded_preprocessor.prepare(encoded_data)
        elif self.backend_name == "ascend_acl":
            import numpy as np

            ascend_rgb_array = np.ascontiguousarray(np.asarray(rgb_image))
        automatic = incremental_protocol == "auto"
        if automatic:
            protocol_pool = self.incremental_protocols
        elif incremental_protocol:
            protocol = self.incremental_protocols.get(incremental_protocol)
            if protocol is None or not protocol.get("available"):
                raise ValueError("所选增量能力当前不可用。")
            protocol_pool = {incremental_protocol: protocol}
        else:
            protocol_pool = {}

        def context_task() -> tuple[Dict[str, Any], float]:
            started = time.perf_counter()
            if getattr(self, "fixed_neutral_context", False):
                value = fixed_neutral_context()
            elif self.backend_name == "ascend_acl":
                value = (
                    self.context_model.predict_preloaded(
                        self.encoded_preprocessor.context_ready_event
                    )
                    if encoded_preloaded
                    else self.context_model.predict(rgb_image)
                )
            else:
                value = predict_context(
                    self.context_model,
                    self.context_checkpoint,
                    rgb_image,
                    self.device,
                    self.context_stream,
                )
            return value, (time.perf_counter() - started) * 1000

        def encoded_detector_contract(backend: Any) -> tuple[Any, Dict[str, Any]]:
            if self.encoded_preprocessor is None:
                raise RuntimeError("Ascend DVPP编码预处理器尚未初始化")
            if backend is self.detector:
                ready_event = self.encoded_preprocessor.base_ready_event
            else:
                ready_event = self.encoded_preprocessor.specialist_ready_event
            if ready_event is None:
                raise RuntimeError("Ascend DVPP检测模型缺少对应的输入就绪事件")
            source_width = int(self.encoded_preprocessor.source_width)
            source_height = int(self.encoded_preprocessor.source_height)
            target_width = int(backend.expected_width)
            target_height = int(backend.expected_height)
            scale = min(target_width / source_width, target_height / source_height)
            resized_width = int(round(source_width * scale))
            resized_height = int(round(source_height * scale))
            if resized_width > target_width or resized_height > target_height:
                raise RuntimeError("Ascend DVPP letterbox尺寸计算越界")
            return ready_event, {
                "original_height": source_height,
                "original_width": source_width,
                "scale": scale,
                "pad_left": (target_width - resized_width) // 2,
                "pad_top": (target_height - resized_height) // 2,
            }

        def detector_task(backend: Any, imgsz: int) -> tuple[Any, float]:
            started = time.perf_counter()
            predict_options = {
                "imgsz": imgsz,
                "conf": float(confidence),
                "iou": self.iou,
                "max_det": self.max_det,
                "quantize": self.quantize,
                "compile": self.compile,
            }
            if encoded_preloaded:
                ready_event, info = encoded_detector_contract(backend)
                value = backend.predict_preloaded(
                    info, ready_event=ready_event, **predict_options
                )
                return value, (time.perf_counter() - started) * 1000
            if ascend_rgb_array is not None:
                predict_options["_ascend_rgb_array"] = ascend_rgb_array
            value = backend.predict(rgb_image, **predict_options)
            return value, (time.perf_counter() - started) * 1000

        def context_submit_task() -> tuple[Any, float]:
            started = time.perf_counter()
            if getattr(self, "fixed_neutral_context", False):
                return self._model_executor.submit(fixed_neutral_context), started
            handle = (
                self.context_model.submit_preloaded(
                    self.encoded_preprocessor.context_ready_event
                )
                if encoded_preloaded
                else self.context_model.submit(rgb_image)
            )
            return handle, started

        def detector_submit_task(backend: Any, imgsz: int) -> tuple[Any, float]:
            started = time.perf_counter()
            predict_options = {
                "imgsz": imgsz,
                "conf": float(confidence),
                "iou": self.iou,
                "max_det": self.max_det,
                "quantize": self.quantize,
                "compile": self.compile,
            }
            if encoded_preloaded:
                ready_event, info = encoded_detector_contract(backend)
                handle = backend.submit_preloaded(
                    info, ready_event, **predict_options
                )
                return handle, started
            if ascend_rgb_array is not None:
                predict_options["_ascend_rgb_array"] = ascend_rgb_array
            return backend.submit(rgb_image, **predict_options), started

        route_candidate_ids = [
            protocol_id for protocol_id, protocol in protocol_pool.items()
            if protocol.get("available") and (
                protocol.get("incremental_mode") == "class_incremental"
                or protocol_independent_class_ids(protocol)
            )
        ][: self.max_specialists]
        prefetch_ids = (
            list(route_candidate_ids)
            if speculative_specialists
            else [
                protocol_id
                for protocol_id in route_candidate_ids
                if protocol_content_execution_gate(protocol_pool[protocol_id]) is None
            ]
        )
        deferred_specialist_ids = set(route_candidate_ids) - set(prefetch_ids)
        physical_specialist_ids = tuple(prefetch_ids)
        for protocol_id in physical_specialist_ids:
            protocol = protocol_pool[protocol_id]
            if protocol_id not in self.specialist_detectors:
                self.specialist_detectors[protocol_id] = self._create_backend(
                    self.backend_name,
                    protocol["weights"],
                    self.device_index,
                    self._backend_options_for_role("specialist"),
                )
        prefetched_predictions: Dict[str, tuple[Any, float]] = {}
        ascend_submit_order = _ascend_role_order(
            self.native_options, "submit_order"
        )
        ascend_collect_order = _ascend_role_order(
            self.native_options, "collect_order"
        )
        fixed_context = getattr(self, "fixed_neutral_context", False)
        context = fixed_neutral_context() if fixed_context else None
        context_total_ms = 0.0
        unified_ascend_submit = (
            self.backend_name == "ascend_acl"
            and self.parallel_model_execution
            and bool(
                physical_specialist_ids
                or deferred_specialist_ids
            )
            and self.native_options.get("execution_mode", "synchronous")
            == "async_stream"
            and self.native_options.get("schedule_mode", "threaded_execute")
            == "unified_enqueue"
        )
        if unified_ascend_submit:
            groups = {
                "scene": (
                    () if fixed_context else (("context", context_submit_task),)
                ),
                "base": ((
                    "detector",
                    lambda: detector_submit_task(self.detector, self.imgsz),
                ),),
                "specialist": tuple(
                    (
                        protocol_id,
                        lambda protocol_id=protocol_id: detector_submit_task(
                            self.specialist_detectors[protocol_id],
                            self.specialist_imgsz,
                        ),
                    )
                    for protocol_id in physical_specialist_ids
                ),
            }

            def collect_ascend_handle(pending: tuple[Any, float]) -> tuple[Any, float]:
                handle, started = pending
                return handle.result(), (time.perf_counter() - started) * 1000

            completed = _ordered_group_results(
                groups,
                ascend_submit_order,
                ascend_collect_order,
                collect_ascend_handle,
            )
            if not fixed_context:
                context, context_total_ms = completed["context"]
            prediction, detector_total_ms = completed["detector"]
            prefetched_predictions = {
                protocol_id: completed[protocol_id]
                for protocol_id in physical_specialist_ids
            }
        elif (
            self.backend_name == "ascend_acl"
            and self.parallel_model_execution
            and bool(
                physical_specialist_ids
                or deferred_specialist_ids
            )
            and self.native_options.get("execution_mode", "synchronous")
            == "async_stream"
            and self.native_options.get("schedule_mode", "threaded_execute")
            == "threaded_execute"
        ):
            groups = {
                "scene": (
                    ()
                    if fixed_context
                    else ((
                        "context",
                        lambda: self._model_executor.submit(context_task),
                    ),)
                ),
                "base": ((
                    "detector",
                    lambda: self._model_executor.submit(
                        detector_task, self.detector, self.imgsz
                    ),
                ),),
                "specialist": tuple(
                    (
                        protocol_id,
                        lambda protocol_id=protocol_id: self._model_executor.submit(
                            detector_task,
                            self.specialist_detectors[protocol_id],
                            self.specialist_imgsz,
                        ),
                    )
                    for protocol_id in physical_specialist_ids
                ),
            }
            completed = _ordered_group_results(
                groups,
                ascend_submit_order,
                ascend_collect_order,
                lambda future: future.result(),
            )
            if not fixed_context:
                context, context_total_ms = completed["context"]
            prediction, detector_total_ms = completed["detector"]
            prefetched_predictions = {
                protocol_id: completed[protocol_id]
                for protocol_id in physical_specialist_ids
            }
        elif self.parallel_model_execution and (
            physical_specialist_ids
            or deferred_specialist_ids
        ):
            context_future = (
                None
                if fixed_context
                else self._model_executor.submit(context_task)
            )
            detector_future = self._model_executor.submit(detector_task, self.detector, self.imgsz)
            specialist_futures = {
                protocol_id: self._model_executor.submit(
                    detector_task, self.specialist_detectors[protocol_id], self.specialist_imgsz
                )
                for protocol_id in physical_specialist_ids
            }
            if context_future is not None:
                context, context_total_ms = context_future.result()
            prediction, detector_total_ms = detector_future.result()
            prefetched_predictions = {
                protocol_id: future.result() for protocol_id, future in specialist_futures.items()
            }
        elif self.parallel_context_execution:
            context_future = (
                None
                if fixed_context
                else self._model_executor.submit(context_task)
            )
            prediction, detector_total_ms = detector_task(self.detector, self.imgsz)
            if context_future is not None:
                context, context_total_ms = context_future.result()
        else:
            if not fixed_context:
                context, context_total_ms = context_task()
            prediction, detector_total_ms = detector_task(self.detector, self.imgsz)

        # Encoded batch workers serialize only the resident DVPP/model buffers.
        # Once every OM output has been copied to its per-result host arrays,
        # the next frame may enter the NPU while this frame completes routing,
        # calibration and NMS on the CPU.
        if hardware_release is not None:
            hardware_release()
            # Hand the interpreter to the next batch worker immediately.  The
            # routing stage below is Python-heavy enough to otherwise retain
            # the GIL until it has finished, which would serialize the very
            # NPU/CPU overlap this pipeline is intended to create.
            time.sleep(0)

        if context is None:
            raise RuntimeError("上下文执行未返回结果")

        context_inference_ms = float(context.pop("_inference_ms", 0.0))
        context_ascend_timings = dict(context.pop("_ascend_timings", {}) or {})

        def accumulate_ascend_timings(timings: Mapping[str, Any]) -> None:
            nonlocal ascend_submit_ms, ascend_wait_ms
            nonlocal ascend_input_copy_max_ms, ascend_output_copy_max_ms
            ascend_submit_ms += float(timings.get("ascend_submit", 0.0))
            ascend_wait_ms += float(timings.get("ascend_wait", 0.0))
            ascend_input_copy_max_ms = max(
                ascend_input_copy_max_ms,
                float(timings.get("ascend_input_copy", 0.0)),
            )
            ascend_output_copy_max_ms = max(
                ascend_output_copy_max_ms,
                float(timings.get("ascend_output_copy", 0.0)),
            )

        accumulate_ascend_timings(context_ascend_timings)
        detector_timings = yolo_timings(prediction)
        accumulate_ascend_timings(dict(getattr(prediction, "speed", {}) or {}))
        for prefetched_prediction, _total_ms in prefetched_predictions.values():
            accumulate_ascend_timings(
                dict(getattr(prefetched_prediction, "speed", {}) or {})
            )
        inference_ms = context_inference_ms + detector_timings["inference_ms"]
        conversion_started = time.perf_counter()
        raw_base_records = calibrate_record_confidences(
            remap_base_records(
                result_records(prediction, self.base_local_names),
                self.base_local_to_global,
                self.class_names,
            ),
            getattr(self, "score_calibration", None),
        )
        routing_conversion_ms = (time.perf_counter() - conversion_started) * 1000
        gate_started = time.perf_counter()
        base_records, unified_gate_rejections = apply_unified_class_gates(
            rgb_image,
            raw_base_records,
            getattr(self, "unified_class_gates", {}),
            context,
        )
        routing_gate_ms = (time.perf_counter() - gate_started) * 1000
        eligible_routes, executed_routes, skipped_protocols = plan_specialist_routes(
            protocol_pool,
            base_records,
            context,
            self.base_class_ids,
            self.max_specialists if automatic else 1,
            self.detection_evidence_weight,
            self.context_evidence_weight,
            self.neutral_context_score,
            self.default_routing_prior,
        )
        executed_routes, skipped_protocols, content_gate_decisions = (
            apply_content_execution_gates(
                executed_routes,
                skipped_protocols,
                base_records,
                context,
            )
        )

        routing_started = time.perf_counter()
        logical_protocol = unified_logical_protocol_result(
            base_records,
            unified_gate_rejections,
            getattr(self, "unified_class_gates", {}),
            context,
        )
        protocol_results = [logical_protocol] if logical_protocol is not None else []
        specialist_records: List[Dict[str, Any]] = []
        specialist_preprocess_ms = 0.0
        specialist_inference_ms = 0.0
        specialist_postprocess_ms = 0.0
        routing_conflict_ms = 0.0
        routing_nms_ms = 0.0
        routing_decision_ms = 0.0
        conflict_rejections: List[Dict[str, Any]] = list(unified_gate_rejections)
        for route in executed_routes:
            protocol_id = str(route["id"])
            protocol = dict(route["protocol"])
            if (
                protocol_id not in prefetched_predictions
                and protocol_id not in self.specialist_detectors
            ):
                self.specialist_detectors[protocol_id] = self._create_backend(
                    self.backend_name,
                    protocol["weights"],
                    self.device_index,
                    self._backend_options_for_role("specialist"),
                )
            if protocol_id in prefetched_predictions:
                specialist_prediction, _specialist_total_ms = prefetched_predictions[protocol_id]
            else:
                specialist_prediction, _specialist_total_ms = detector_task(
                    self.specialist_detectors[protocol_id], self.specialist_imgsz
                )
            specialist_timing = yolo_timings(specialist_prediction)
            if protocol_id not in prefetched_predictions:
                accumulate_ascend_timings(
                    dict(getattr(specialist_prediction, "speed", {}) or {})
                )
            specialist_preprocess_ms += specialist_timing["preprocess_ms"]
            specialist_inference_ms += specialist_timing["inference_ms"]
            specialist_postprocess_ms += specialist_timing["postprocess_ms"]
            inference_ms += specialist_timing["inference_ms"]
            class_ids = protocol_class_ids(protocol)
            local_to_global = protocol.get("local_to_global")
            if not isinstance(local_to_global, Mapping):
                local_to_global = {0: class_ids[0]}
            conversion_started = time.perf_counter()
            remapped_candidates = remap_specialist_records_dynamic(
                result_records(specialist_prediction),
                local_to_global,
                self.class_names,
                protocol_id,
            )
            adapted_candidates = apply_edge_incremental_adapter(
                remapped_candidates,
                context,
                rgb_image.size,
                getattr(self, "edge_incremental_adapter", None),
            )
            raw_candidates = calibrate_record_confidences(
                adapted_candidates,
                getattr(self, "score_calibration", None),
            )
            routing_conversion_ms += (time.perf_counter() - conversion_started) * 1000
            gate_started = time.perf_counter()
            effective_thresholds, context_score = protocol_effective_thresholds(
                protocol, context, float(confidence)
            )
            threshold_candidates, threshold_rejections = apply_protocol_thresholds(
                raw_candidates, effective_thresholds, context_score
            )
            prototype_rejections: List[Dict[str, Any]] = []
            for prototype in protocol_positive_prototypes(protocol).values():
                threshold_candidates, rejected_by_prototype = apply_positive_prototype_to_image(
                    rgb_image, threshold_candidates, prototype
                )
                prototype_rejections.extend(rejected_by_prototype)
            independent_ids = protocol_independent_class_ids(protocol)
            if protocol.get("incremental_mode") == "class_incremental":
                candidates = threshold_candidates
                activation_reason = (
                    "通过新类置信度与正样本原型门控"
                    if protocol_positive_prototypes(protocol)
                    else "通过独立新类置信度门限"
                )
            else:
                candidates = []
                for class_id in class_ids:
                    class_candidates = [
                        item for item in threshold_candidates if int(item["class_id"]) == class_id
                    ]
                    if class_id in independent_ids:
                        candidates.extend(class_candidates)
                    else:
                        candidates.extend(consensus_specialist_records(
                            base_records, class_candidates, class_id,
                            float(protocol.get("consensus_iou", 0.30)),
                        ))
                activation_reason = (
                    "通过冻结专家链独立置信度门限"
                    if independent_ids else "通过基础同类目标与空间一致性检查"
                )
            routing_gate_ms += (time.perf_counter() - gate_started) * 1000
            conflict_started = time.perf_counter()
            base_records, candidates, conflict_decisions = arbitrate_cross_class_conflicts(
                base_records,
                candidates,
                self.conflict_iou,
                self.conflict_base_confidence,
                self.specialist_margin,
                protocol.get("confusion_graph"),
                self.preserve_base_class_owners,
                self.conflict_incremental_coverage,
            )
            routing_conflict_ms += (time.perf_counter() - conflict_started) * 1000
            rejected = [
                row for row in conflict_decisions if row["action"] == "reject_specialist"
            ]
            conflict_rejections.extend(
                threshold_rejections + prototype_rejections + conflict_decisions
            )
            activated = bool(candidates)
            activated_class_names = sorted({str(item["class_name"]) for item in candidates})
            if activated:
                specialist_records.extend(candidates)
            decision_started = time.perf_counter()
            protocol_results.append({
                "id": protocol_id,
                "class_name": protocol["class_name"],
                "new_class": protocol["class_name"],
                "incremental_mode": protocol["incremental_mode"],
                "new_map50": protocol["new_map50"],
                "krr": protocol["krr"],
                "status": "activated" if activated else "no_candidate",
                "activated": activated,
                "activated_classes": activated_class_names,
                "raw_candidate_count": len(raw_candidates),
                "candidate_count": len(candidates),
                "conflict_suppressed_count": len(rejected),
                "prototype_rejected_count": len(prototype_rejections),
                "activation_rejected_count": len(threshold_rejections),
                "base_override_count": sum(
                    row["action"] == "suppress_base" for row in conflict_decisions
                ),
                "activation_thresholds": {
                    str(key): round(value, 2) for key, value in effective_thresholds.items()
                },
                "activation_threshold": (
                    round(next(iter(effective_thresholds.values())), 2)
                    if len(effective_thresholds) == 1 else None
                ),
                "context_affinity": round(context_score, 6),
                "context_gate_policy": dict(protocol.get("context_gate") or {}).get(
                    "policy"
                ),
                "routing_score": route["routing_score"],
                "activation_reason": activation_reason if activated else "未产生通过门限的候选框",
            })
            routing_decision_ms += (time.perf_counter() - decision_started) * 1000

        base_with_source = apply_unified_record_ownership(
            base_records,
            getattr(self, "unified_class_gates", {}),
        )
        nms_started = time.perf_counter()
        records, fusion_summary = class_aware_nms(
            base_with_source + specialist_records,
            self.fusion_iou,
            getattr(self, "cross_class_suppression", None),
        )
        routing_nms_ms = (time.perf_counter() - nms_started) * 1000
        fusion_summary["conflict_suppressed_count"] = len(conflict_rejections)
        routing_fusion_ms = (time.perf_counter() - routing_started) * 1000
        if encoded_preloaded:
            dvpp_device_ms = self.encoded_preprocessor.device_ms()
        decision_started = time.perf_counter()
        base_model_id = self.base_model_id
        generation_id = self.generation_id
        models_used = [
            getattr(self, "context_model_id", "scene_sensor_net_v1"),
            base_model_id,
        ]
        models_used.extend(str(route["id"]) for route in executed_routes)
        activated_classes = sorted({
            class_name
            for item in protocol_results
            for class_name in item.get("activated_classes", [])
        })
        decision = {
            "mode": "automatic" if automatic else ("manual" if protocol_pool else "unified_only"),
            "input_mode": "unlabeled_image",
            "inference_scope": "production",
            "execution_profile": (
                "speculative_low_latency"
                if speculative_specialists
                else "gated_standard"
            ),
            "prefetched_specialists": list(physical_specialist_ids),
            "routing_basis": "all_class_owners_plus_optional_content_routing",
            "class_incremental_execution_policy": (
                "scene_and_base_evidence_gate"
                if content_gate_decisions
                else "every_image"
            ),
            "label_aware_routing": False,
            "scene_hard_routing": bool(content_gate_decisions),
            "content_execution_gates": content_gate_decisions,
            "evaluated_specialists": len(executed_routes),
            "base_detection_count": sum(
                row["source"] == "frozen_base_model" for row in base_with_source
            ),
            "final_detection_count": len(records),
            "activated_classes": activated_classes,
            "eligible_protocols": [
                {"id": row["id"], "routing_score": row["routing_score"], "incremental_mode": row["incremental_mode"]}
                for row in eligible_routes
            ],
            "executed_protocols": [str(row["id"]) for row in executed_routes],
            "skipped_protocols": skipped_protocols,
            "fusion_summary": fusion_summary,
            "conflict_suppressions": conflict_rejections,
            "reason": (
                "已融合通过模式感知门控的专项候选"
                if activated_classes
                else "未发现通过模式感知门控的专项候选，保持冻结基础检测结果"
            ),
            "generation_id": generation_id,
            "base_model_id": base_model_id,
            "class_owners": {
                str(key): value for key, value in getattr(self, "class_owners", {}).items()
            },
            "edge_incremental_adapter": getattr(
                self, "edge_incremental_adapter_status", {"active": False}
            ),
        }
        routing_decision_ms += (time.perf_counter() - decision_started) * 1000
        result = {
            "filename": filename,
            "image_width": int(rgb_image.width),
            "image_height": int(rgb_image.height),
            "context": context,
            "detections": records,
            "class_counts": summarize_records(records),
            "detection_count": len(records),
            "confidence_threshold": round(float(confidence), 2),
            "inference_ms": round(inference_ms, 1),
            "timings": {
                "dvpp_enqueue_ms": round(dvpp_preprocess_ms, 3),
                "dvpp_device_ms": round(dvpp_device_ms, 3),
                "ascend_submit_ms": round(ascend_submit_ms, 3),
                "ascend_wait_ms": round(ascend_wait_ms, 3),
                "ascend_input_copy_max_ms": round(ascend_input_copy_max_ms, 3),
                "ascend_output_copy_max_ms": round(ascend_output_copy_max_ms, 3),
                "context_total_ms": round(context_total_ms, 3),
                "context_inference_ms": round(context_inference_ms, 3),
                "detector_total_ms": round(detector_total_ms, 3),
                "detector_preprocess_ms": round(detector_timings["preprocess_ms"], 3),
                "detector_inference_ms": round(detector_timings["inference_ms"], 3),
                "detector_postprocess_ms": round(detector_timings["postprocess_ms"], 3),
                "specialist_preprocess_ms": round(specialist_preprocess_ms, 3),
                "specialist_inference_ms": round(specialist_inference_ms, 3),
                "specialist_postprocess_ms": round(specialist_postprocess_ms, 3),
                "routing_fusion_ms": round(routing_fusion_ms, 3),
                "routing_conversion_ms": round(routing_conversion_ms, 3),
                "routing_gate_ms": round(routing_gate_ms, 3),
                "routing_conflict_ms": round(routing_conflict_ms, 3),
                "routing_nms_ms": round(routing_nms_ms, 3),
                "routing_decision_ms": round(routing_decision_ms, 3),
            },
            "agent": {
                "mode": "automatic_orchestration" if automatic else "standard_detection",
                "models_used": models_used,
                "protocol": protocol_results[0] if not automatic and protocol_results else None,
                "protocols": protocol_results,
                "decision": decision,
            },
        }
        result["timings"]["engine_total_ms"] = round((time.perf_counter() - pipeline_started) * 1000, 3)
        return result
