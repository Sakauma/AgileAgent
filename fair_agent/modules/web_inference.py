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

from PIL import Image, ImageDraw, UnidentifiedImageError

from fair_agent.modules.detection_fusion import (
    arbitrate_cross_class_conflicts,
    box_iou,
    context_adjusted_threshold,
    context_affinity,
)
from fair_agent.modules.incremental_rejection import apply_positive_prototype_to_image


CLASS_NAMES = {0: "soldier", 1: "small_aircraft", 2: "warship", 3: "tank"}
SENSOR_LABELS = {"ir": "红外", "sar": "SAR"}
SCENE_LABELS = {"air": "空域", "forest": "林地", "sea": "海域", "urban": "城市场景"}
T = TypeVar("T")


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
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().tolist()
    names = dict(class_names or CLASS_NAMES)
    rows = []
    for coordinates, confidence, class_id in zip(xyxy, confidences, class_ids):
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
    max_penalty = (
        float(gate.get("max_threshold_penalty", 0.0))
        if gate.get("enabled") is True
        else 0.0
    )
    effective: Dict[int, float] = {}
    affinity = 1.0
    for class_id, threshold in protocol_thresholds(protocol).items():
        adjusted, affinity = context_adjusted_threshold(
            float(threshold), context, prior, max_penalty
        )
        effective[int(class_id)] = max(float(confidence_floor), adjusted)
    return effective, affinity


def apply_protocol_thresholds(
    records: Iterable[Mapping[str, Any]],
    thresholds: Mapping[int, float],
    affinity: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for raw_row in records:
        row = dict(raw_row)
        threshold = thresholds.get(int(row["class_id"]))
        if threshold is None or float(row.get("confidence", 0.0)) >= float(threshold):
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
        remapped.append({**item, "class_id": global_id, "class_name": names.get(global_id, str(global_id))})
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
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
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
        for candidate in candidates:
            if any(box_iou(candidate["xyxy"], existing["xyxy"]) >= iou_threshold for existing in class_kept):
                suppressed += 1
                continue
            status = "specialist_kept" if candidate.get("source") == "incremental_model" else "base_retained"
            class_kept.append({**candidate, "fusion_status": status})
        kept.extend(class_kept)
    kept.sort(key=lambda row: (int(row["class_id"]), -float(row.get("confidence", 0.0))))
    return kept, {"input_count": len(rows), "output_count": len(kept), "suppressed_count": suppressed}


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
            self.backend_name, detector_path, self.device_index, self.native_options
        )
        if self.backend_name == "ascend_acl":
            from fair_agent.backends.ascend_acl import load_ascend_context_model

            self.context_model, self.context_checkpoint = load_ascend_context_model(
                self.native_options
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
        context_size = int(self.context_checkpoint["preprocessing"]["image_size"])
        warmup_image = Image.new("RGB", (self.warmup_width, self.warmup_height))
        context_warmup = Image.new("RGB", (context_size, context_size))
        for _ in range(self.warmup_iterations):
            if self.backend_name == "ascend_acl":
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
        if self.backend_name == "ascend_acl":
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
                    self.backend_name, protocol["weights"], self.device_index, self.native_options
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
            active_specialists = [
                self.specialist_detectors[protocol_id]
                for protocol_id, protocol in self.incremental_protocols.items()
                if protocol.get("available")
            ]
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
            )
            self._encoded_image_stub = Image.new(
                "RGB",
                (
                    self.encoded_preprocessor.source_width,
                    self.encoded_preprocessor.source_height,
                ),
            )
        self.queue = FairInferenceQueue()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        queue = getattr(self, "queue", None)
        if queue is not None:
            queue.close()
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
    ) -> Dict[str, Any]:
        if not self.accepts_encoded(data):
            raise ValueError("当前Ascend编码输入不符合DVPP固定生产契约")
        result, queue_wait_ms = self.queue.run(
            lambda: self._predict_encoded_unlocked(
                data,
                filename,
                self.default_confidence if confidence is None else confidence,
                incremental_protocol,
            )
        )
        result["queue_wait_ms"] = queue_wait_ms
        return result

    def _predict_encoded_unlocked(
        self,
        data: bytes,
        filename: str,
        confidence: float,
        incremental_protocol: str | None,
    ) -> Dict[str, Any]:
        return self._predict_unlocked(
            self._encoded_image_stub,
            filename,
            confidence,
            incremental_protocol,
            encoded_data=data,
        )

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

        prefetch_ids = [
            protocol_id for protocol_id, protocol in protocol_pool.items()
            if protocol.get("available") and (
                protocol.get("incremental_mode") == "class_incremental"
                or protocol_independent_class_ids(protocol)
            )
        ][: self.max_specialists]
        for protocol_id in prefetch_ids:
            protocol = protocol_pool[protocol_id]
            if protocol_id not in self.specialist_detectors:
                self.specialist_detectors[protocol_id] = self._create_backend(
                    self.backend_name, protocol["weights"], self.device_index, self.native_options
                )
        prefetched_batches: Dict[str, Sequence[Any]] = {}
        if self.parallel_model_execution and prefetch_ids:
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
            remap_base_records(
                result_records(prediction, self.base_local_names),
                self.base_local_to_global,
                self.class_names,
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

        route_rows = [
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
        protocol_outputs: List[List[Dict[str, Any]]] = [[] for _ in images]
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
                    self.native_options,
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
                raw_candidates = remap_specialist_records_dynamic(
                    result_records(prediction), local_to_global, self.class_names, str(protocol_id)
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
            base_with_source = [
                {**item, "source": "frozen_base_model", "protocol_id": None}
                for item in base_records
            ]
            records, fusion_summary = class_aware_nms(
                base_with_source + specialists_by_image[index], self.fusion_iou
            )
            fusion_summary["conflict_suppressed_count"] = len(conflicts_by_image[index])
            eligible, executed, skipped = route_rows[index]
            activated_classes = sorted({
                class_name
                for item in protocol_outputs[index]
                for class_name in item.get("activated_classes", [])
            })
            models_used = ["scene_sensor_net_v1", self.base_model_id]
            models_used.extend(str(route["id"]) for route in executed)
            decision = {
                "mode": "automatic" if automatic else ("manual" if protocol_pool else "unified_only"),
                "input_mode": "unlabeled_image",
                "inference_scope": "production",
                "routing_basis": "all_class_owners_plus_optional_content_routing",
                "class_incremental_execution_policy": "every_image",
                "label_aware_routing": False,
                "scene_hard_routing": False,
                "evaluated_specialists": len(executed),
                "base_detection_count": len(base_records),
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
    ) -> Dict[str, Any]:
        from fair_agent.models.context import predict_context

        pipeline_started = time.perf_counter()
        rgb_image = image if image.mode == "RGB" else image.convert("RGB")
        ascend_rgb_array = None
        dvpp_preprocess_ms = 0.0
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
            if self.backend_name == "ascend_acl":
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
                if (backend.expected_height, backend.expected_width) == (736, 896):
                    ready_event = self.encoded_preprocessor.base_ready_event
                    info = {
                        "original_height": 512,
                        "original_width": 640,
                        "scale": 1.4,
                        "pad_left": 0,
                        "pad_top": 9,
                    }
                elif (backend.expected_height, backend.expected_width) == (512, 640):
                    ready_event = self.encoded_preprocessor.specialist_ready_event
                    info = {
                        "original_height": 512,
                        "original_width": 640,
                        "scale": 1.0,
                        "pad_left": 0,
                        "pad_top": 0,
                    }
                else:
                    raise RuntimeError(
                        "Ascend DVPP检测输入契约不受支持："
                        f"{backend.expected_height}x{backend.expected_width}"
                    )
                value = backend.predict_preloaded(
                    info, ready_event=ready_event, **predict_options
                )
                return value, (time.perf_counter() - started) * 1000
            if ascend_rgb_array is not None:
                predict_options["_ascend_rgb_array"] = ascend_rgb_array
            value = backend.predict(rgb_image, **predict_options)
            return value, (time.perf_counter() - started) * 1000

        prefetch_ids = [
            protocol_id for protocol_id, protocol in protocol_pool.items()
            if protocol.get("available") and (
                protocol.get("incremental_mode") == "class_incremental"
                or protocol_independent_class_ids(protocol)
            )
        ][: self.max_specialists]
        for protocol_id in prefetch_ids:
            protocol = protocol_pool[protocol_id]
            if protocol_id not in self.specialist_detectors:
                self.specialist_detectors[protocol_id] = self._create_backend(
                    self.backend_name, protocol["weights"], self.device_index, self.native_options
                )
        prefetched_predictions: Dict[str, tuple[Any, float]] = {}
        if self.parallel_model_execution and prefetch_ids:
            context_future = self._model_executor.submit(context_task)
            detector_future = self._model_executor.submit(detector_task, self.detector, self.imgsz)
            specialist_futures = {
                protocol_id: self._model_executor.submit(
                    detector_task, self.specialist_detectors[protocol_id], self.specialist_imgsz
                )
                for protocol_id in prefetch_ids
            }
            context, context_total_ms = context_future.result()
            prediction, detector_total_ms = detector_future.result()
            prefetched_predictions = {
                protocol_id: future.result() for protocol_id, future in specialist_futures.items()
            }
        elif self.parallel_context_execution:
            context_future = self._model_executor.submit(context_task)
            prediction, detector_total_ms = detector_task(self.detector, self.imgsz)
            context, context_total_ms = context_future.result()
        else:
            context, context_total_ms = context_task()
            prediction, detector_total_ms = detector_task(self.detector, self.imgsz)

        context_inference_ms = float(context.pop("_inference_ms", 0.0))
        detector_timings = yolo_timings(prediction)
        inference_ms = context_inference_ms + detector_timings["inference_ms"]
        raw_base_records = remap_base_records(
            result_records(prediction, self.base_local_names),
            self.base_local_to_global,
            self.class_names,
        )
        base_records, unified_gate_rejections = apply_unified_class_gates(
            rgb_image,
            raw_base_records,
            getattr(self, "unified_class_gates", {}),
            context,
        )
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

        routing_started = time.perf_counter()
        protocol_results = []
        specialist_records: List[Dict[str, Any]] = []
        specialist_preprocess_ms = 0.0
        specialist_inference_ms = 0.0
        specialist_postprocess_ms = 0.0
        conflict_rejections: List[Dict[str, Any]] = list(unified_gate_rejections)
        for route in executed_routes:
            protocol_id = str(route["id"])
            protocol = dict(route["protocol"])
            if protocol_id not in self.specialist_detectors:
                self.specialist_detectors[protocol_id] = self._create_backend(
                    self.backend_name,
                    protocol["weights"],
                    self.device_index,
                    self.native_options,
                )
            if protocol_id in prefetched_predictions:
                specialist_prediction, _specialist_total_ms = prefetched_predictions[protocol_id]
            else:
                specialist_prediction, _specialist_total_ms = detector_task(
                    self.specialist_detectors[protocol_id], self.specialist_imgsz
                )
            specialist_timing = yolo_timings(specialist_prediction)
            specialist_preprocess_ms += specialist_timing["preprocess_ms"]
            specialist_inference_ms += specialist_timing["inference_ms"]
            specialist_postprocess_ms += specialist_timing["postprocess_ms"]
            inference_ms += specialist_timing["inference_ms"]
            class_ids = protocol_class_ids(protocol)
            local_to_global = protocol.get("local_to_global")
            if not isinstance(local_to_global, Mapping):
                local_to_global = {0: class_ids[0]}
            effective_thresholds, context_score = protocol_effective_thresholds(
                protocol, context, float(confidence)
            )
            raw_candidates = remap_specialist_records_dynamic(
                result_records(specialist_prediction), local_to_global, self.class_names, protocol_id
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

        base_with_source = [
            {**item, "source": "frozen_base_model", "protocol_id": None}
            for item in base_records
        ]
        records, fusion_summary = class_aware_nms(
            base_with_source + specialist_records,
            self.fusion_iou,
        )
        fusion_summary["conflict_suppressed_count"] = len(conflict_rejections)
        routing_fusion_ms = (time.perf_counter() - routing_started) * 1000
        base_model_id = self.base_model_id
        generation_id = self.generation_id
        models_used = ["scene_sensor_net_v1", base_model_id]
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
            "routing_basis": "all_class_owners_plus_optional_content_routing",
            "class_incremental_execution_policy": "every_image",
            "label_aware_routing": False,
            "scene_hard_routing": False,
            "evaluated_specialists": len(executed_routes),
            "base_detection_count": len(base_records),
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
        }
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
