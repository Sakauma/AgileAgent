from __future__ import annotations

import io
import hashlib
import json
import re
import threading
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, TypeVar

from PIL import Image, ImageDraw, UnidentifiedImageError


CLASS_NAMES = {0: "soldier", 1: "small_aircraft", 2: "warship", 3: "tank"}
SENSOR_LABELS = {"ir": "红外", "sar": "SAR"}
SCENE_LABELS = {"air": "空域", "forest": "林地", "sea": "海域", "urban": "城市场景"}
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_BATCH_FILES = 20
MAX_BATCH_BYTES = 200 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "BMP", "TIFF"}
T = TypeVar("T")


class FairInferenceQueue:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0
        self._waiting = 0
        self._active = False
        self._completed = 0

    def status(self) -> Dict[str, int | bool]:
        with self._condition:
            return {
                "waiting": self._waiting,
                "active": self._active,
                "completed": self._completed,
            }

    def run(self, operation: Callable[[], T]) -> tuple[T, float]:
        queued_at = time.perf_counter()
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            self._waiting += 1
            while ticket != self._serving_ticket:
                self._condition.wait()
            self._waiting -= 1
            self._active = True
        wait_ms = round((time.perf_counter() - queued_at) * 1000, 1)
        try:
            return operation(), wait_ms
        finally:
            with self._condition:
                self._active = False
                self._completed += 1
                self._serving_ticket += 1
                self._condition.notify_all()


def content_task_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_image_bytes(
    data: bytes,
    filename: str,
    max_bytes: int = MAX_FILE_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> tuple[Image.Image, str]:
    if not data:
        raise ValueError(f"图像为空：{filename}")
    if len(data) > max_bytes:
        raise ValueError(f"图像超过 {max_bytes // (1024 * 1024)}MB 限制：{filename}")
    try:
        with Image.open(io.BytesIO(data)) as source:
            image_format = str(source.format or "").upper()
            width, height = source.size
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise ValueError(f"不支持的图像格式：{image_format or 'unknown'}")
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ValueError(f"图像像素超过限制：{width}x{height}")
            source.load()
            image = source.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError(f"无法读取图像：{filename}") from exc
    return image, content_task_id(data)


def validate_batch_uploads(items: Iterable[tuple[str, bytes]]) -> List[tuple[str, bytes, Image.Image, str]]:
    rows = list(items)
    if not rows:
        raise ValueError("请选择至少一张图像。")
    if len(rows) > MAX_BATCH_FILES:
        raise ValueError(f"单批最多处理 {MAX_BATCH_FILES} 张图像。")
    total_bytes = sum(len(data) for _, data in rows)
    if total_bytes > MAX_BATCH_BYTES:
        raise ValueError(f"单批总大小不能超过 {MAX_BATCH_BYTES // (1024 * 1024)}MB。")
    validated = []
    seen_ids = set()
    for filename, data in rows:
        image, task_id = validate_image_bytes(data, filename)
        if task_id in seen_ids:
            raise ValueError(f"批次中存在重复图像：{filename}")
        seen_ids.add(task_id)
        validated.append((filename, data, image, task_id))
    return validated


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


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


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


def context_affinity(context: Mapping[str, Any], prior: Mapping[str, Any] | None) -> float:
    """Return a soft context score. Missing priors stay neutral and never reject a route."""
    prior = dict(prior or {})
    components: List[float] = []
    for dimension in ("sensor", "scene"):
        weights = prior.get(dimension)
        probabilities = context.get(f"{dimension}_probabilities")
        if isinstance(weights, Mapping) and isinstance(probabilities, Mapping) and weights:
            score = sum(float(probabilities.get(name, 0.0)) * float(weight) for name, weight in weights.items())
            components.append(max(0.0, min(1.0, score)))
    return sum(components) / len(components) if components else 0.5


def plan_specialist_routes(
    protocols: Mapping[str, Mapping[str, Any]],
    base_records: Iterable[Dict[str, Any]],
    context: Mapping[str, Any],
    base_class_ids: Iterable[int],
    max_specialists: int = 4,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_rows = list(base_records)
    base_ids = {int(value) for value in base_class_ids}
    eligible: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for protocol_id, raw_protocol in protocols.items():
        protocol = dict(raw_protocol)
        mode = str(protocol.get("incremental_mode") or "target_incremental")
        global_class_id = int(protocol["global_class_id"])
        if not protocol.get("available"):
            skipped.append({"id": protocol_id, "reason": "protocol_unavailable"})
            continue
        if mode == "class_incremental":
            if global_class_id in base_ids:
                skipped.append({"id": protocol_id, "reason": "new_class_id_overlaps_base"})
                continue
            if protocol.get("activation_threshold") is None or not protocol.get("calibration_source"):
                skipped.append({"id": protocol_id, "reason": "activation_threshold_not_calibrated"})
                continue
            evidence_score = float(protocol.get("routing_prior", 0.5))
            mode_priority = 0
        elif mode == "target_incremental":
            references = [row for row in base_rows if int(row["class_id"]) == global_class_id]
            if not references:
                skipped.append({"id": protocol_id, "reason": "base_class_not_detected"})
                continue
            evidence_score = max(float(row.get("confidence", 0.0)) for row in references)
            mode_priority = 1
        else:
            skipped.append({"id": protocol_id, "reason": "unsupported_incremental_mode"})
            continue
        soft_context = context_affinity(context, protocol.get("context_prior"))
        route = {
            "id": protocol_id,
            "protocol": protocol,
            "incremental_mode": mode,
            "evidence_score": round(evidence_score, 6),
            "context_score": round(soft_context, 6),
            "routing_score": round(0.7 * evidence_score + 0.3 * soft_context, 6),
            "mode_priority": mode_priority,
        }
        eligible.append(route)
    eligible.sort(key=lambda row: (int(row["mode_priority"]), -float(row["routing_score"]), str(row["id"])))
    limit = max(0, int(max_specialists))
    executed = eligible if limit == 0 else eligible[:limit]
    for route in eligible[len(executed) :]:
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
        candidates = sorted(
            (row for row in rows if int(row["class_id"]) == class_id),
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
            archive.writestr(f"annotated/{index:03d}_{safe_stem}.png", item["annotated_png"])
            metadata.append(
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"annotated_png", "annotated_image", "task_id"}
                }
            )
        archive.writestr("results.json", result_json_bytes({"image_count": len(rows), "results": metadata}))
    return buffer.getvalue()


class WebInferenceEngine:
    def __init__(
        self,
        detector_path: Path,
        context_path: Path,
        device_index: str = "0",
        predict_options: Mapping[str, Any] | None = None,
        incremental_protocols: Mapping[str, Mapping[str, Any]] | None = None,
        class_names: Mapping[int, str] | None = None,
        base_class_ids: Iterable[int] | None = None,
        base_local_to_global: Mapping[int, int] | None = None,
        routing_options: Mapping[str, Any] | None = None,
    ) -> None:
        from ultralytics import YOLO

        from fair_agent.models.context import load_context_model, predict_context

        self.device_index = str(device_index)
        self.device = f"cuda:{self.device_index}"
        self._yolo_class = YOLO
        self.detector = YOLO(str(detector_path))
        self.context_model, self.context_checkpoint = load_context_model(context_path, self.device)
        self.class_names = {int(key): str(value) for key, value in (class_names or CLASS_NAMES).items()}
        configured_base_ids = {int(value) for value in (base_class_ids or self.class_names)}
        self.base_local_to_global = {
            int(key): int(value)
            for key, value in (base_local_to_global or {class_id: class_id for class_id in configured_base_ids}).items()
        }
        self.base_class_ids = set(self.base_local_to_global.values())
        self.base_local_names = {
            local_id: self.class_names[global_id] for local_id, global_id in self.base_local_to_global.items()
        }
        options = dict(predict_options or {})
        routing = dict(routing_options or {})
        self.imgsz = int(options.get("imgsz", 640))
        self.iou = float(options.get("iou", 0.7))
        self.max_det = int(options.get("max_det", 300))
        self.fusion_iou = float(routing.get("fusion_iou", 0.60))
        self.max_specialists = int(routing.get("max_specialists_per_image", 4))
        context_size = int(self.context_checkpoint["preprocessing"]["image_size"])
        warmup_image = Image.new("RGB", (self.imgsz, self.imgsz))
        predict_context(
            self.context_model,
            self.context_checkpoint,
            Image.new("RGB", (context_size, context_size)),
            self.device,
        )
        self.detector.predict(
            source=warmup_image,
            imgsz=self.imgsz,
            conf=0.50,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device_index,
            verbose=False,
        )
        self.incremental_protocols = {name: dict(value) for name, value in (incremental_protocols or {}).items()}
        self.specialist_detectors: Dict[str, Any] = {}
        self.queue = FairInferenceQueue()

    def queue_status(self) -> Dict[str, int | bool]:
        return self.queue.status()

    def predict(
        self,
        image: Image.Image,
        filename: str,
        confidence: float = 0.50,
        task_id: str | None = None,
        incremental_protocol: str | None = None,
    ) -> Dict[str, Any]:
        result, queue_wait_ms = self.queue.run(
            lambda: self._predict_unlocked(image, filename, confidence, task_id, incremental_protocol)
        )
        result["queue_wait_ms"] = queue_wait_ms
        return result

    def _predict_unlocked(
        self,
        image: Image.Image,
        filename: str,
        confidence: float,
        task_id: str | None,
        incremental_protocol: str | None,
    ) -> Dict[str, Any]:
        from fair_agent.models.context import predict_context

        rgb_image = image.convert("RGB")
        context = predict_context(self.context_model, self.context_checkpoint, rgb_image, self.device)
        inference_ms = float(context.pop("_inference_ms", 0.0))
        prediction = self.detector.predict(
            source=rgb_image,
            imgsz=self.imgsz,
            conf=float(confidence),
            iou=self.iou,
            max_det=self.max_det,
            device=self.device_index,
            verbose=False,
        )[0]
        inference_ms += yolo_inference_ms(prediction)
        base_records = remap_base_records(
            result_records(prediction, self.base_local_names),
            self.base_local_to_global,
            self.class_names,
        )
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

        eligible_routes, executed_routes, skipped_protocols = plan_specialist_routes(
            protocol_pool,
            base_records,
            context,
            self.base_class_ids,
            self.max_specialists if automatic else 1,
        )

        protocol_results = []
        specialist_records: List[Dict[str, Any]] = []
        for route in executed_routes:
            protocol_id = str(route["id"])
            protocol = dict(route["protocol"])
            if protocol_id not in self.specialist_detectors:
                self.specialist_detectors[protocol_id] = self._yolo_class(str(protocol["weights"]))
            specialist_prediction = self.specialist_detectors[protocol_id].predict(
                source=rgb_image,
                imgsz=self.imgsz,
                conf=float(confidence),
                iou=self.iou,
                max_det=self.max_det,
                device=self.device_index,
                verbose=False,
            )[0]
            inference_ms += yolo_inference_ms(specialist_prediction)
            global_class_id = int(protocol["global_class_id"])
            activation_threshold = max(float(confidence), float(protocol.get("activation_threshold", confidence)))
            raw_candidates = remap_specialist_records(
                result_records(specialist_prediction),
                global_class_id,
                self.class_names,
                protocol_id,
            )
            threshold_candidates = [
                item for item in raw_candidates if float(item.get("confidence", 0.0)) >= activation_threshold
            ]
            if protocol.get("incremental_mode") == "class_incremental":
                candidates = threshold_candidates
                activation_reason = "通过独立新类置信度门限"
            else:
                candidates = consensus_specialist_records(
                    base_records,
                    threshold_candidates,
                    global_class_id,
                    float(protocol.get("consensus_iou", 0.30)),
                )
                activation_reason = "通过基础同类目标与空间一致性检查"
            activated = bool(candidates)
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
                "raw_candidate_count": len(raw_candidates),
                "candidate_count": len(candidates),
                "activation_threshold": round(activation_threshold, 2),
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
        annotated = annotate_records(rgb_image, records)
        models_used = ["scene_sensor_net_v1", "unified_yolo11s_v1"]
        if executed_routes:
            models_used.append("incremental_model_bank_v1")
        activated_classes = [item["class_name"] for item in protocol_results if item["activated"]]
        decision = {
            "mode": "automatic" if automatic else ("manual" if protocol_pool else "unified_only"),
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
            "reason": (
                "已融合通过模式感知门控的专项候选"
                if activated_classes
                else "未发现通过模式感知门控的专项候选，保持统一检测结果"
            ),
        }
        return {
            "filename": filename,
            "task_id": task_id,
            "context": context,
            "detections": records,
            "class_counts": summarize_records(records),
            "detection_count": len(records),
            "confidence_threshold": round(float(confidence), 2),
            "inference_ms": round(inference_ms, 1),
            "agent": {
                "mode": "automatic_orchestration" if automatic else "standard_detection",
                "models_used": models_used,
                "protocol": protocol_results[0] if not automatic and protocol_results else None,
                "protocols": protocol_results,
                "decision": decision,
            },
            "annotated_png": image_png_bytes(annotated),
        }
