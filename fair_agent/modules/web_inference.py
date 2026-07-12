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


def result_records(result: Any) -> List[Dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().tolist()
    rows = []
    for coordinates, confidence, class_id in zip(xyxy, confidences, class_ids):
        numeric_id = int(class_id)
        rows.append(
            {
                "class_id": numeric_id,
                "class_name": CLASS_NAMES.get(numeric_id, str(numeric_id)),
                "confidence": round(float(confidence), 6),
                "xyxy": [round(float(value), 2) for value in coordinates],
            }
        )
    return rows


def summarize_records(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(item["class_name"]) for item in records).items()))


def remap_specialist_records(records: Iterable[Dict[str, Any]], global_class_id: int) -> List[Dict[str, Any]]:
    class_name = CLASS_NAMES[global_class_id]
    return [
        {**item, "class_id": global_class_id, "class_name": class_name, "source": "incremental_model"}
        for item in records
    ]


def compose_incremental_records(
    base_records: Iterable[Dict[str, Any]],
    specialist_records: Iterable[Dict[str, Any]],
    global_class_id: int,
) -> List[Dict[str, Any]]:
    old_records = [
        {**item, "source": "frozen_base_model"}
        for item in base_records
        if int(item["class_id"]) != global_class_id
    ]
    return old_records + remap_specialist_records(specialist_records, global_class_id)


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
        label = f'{item["class_name"]} {float(item["confidence"]):.2f}'
        text_box = draw.textbbox((0, 0), label)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_y = max(0.0, y1 - text_height - 7)
        draw.rounded_rectangle(
            (x1, text_y, x1 + text_width + 8, text_y + text_height + 6),
            radius=3,
            fill=color,
        )
        draw.text((x1 + 4, text_y + 2), label, fill="white")
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
            task_suffix = str(item.get("task_id") or "unknown")[:8]
            archive.writestr(f"annotated/{index:03d}_{safe_stem}_{task_suffix}.png", item["annotated_png"])
            metadata.append(
                {key: value for key, value in item.items() if key not in {"annotated_png", "annotated_image"}}
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
    ) -> None:
        from ultralytics import YOLO

        from fair_agent.models.context import load_context_model

        self.device_index = str(device_index)
        self.device = f"cuda:{self.device_index}"
        self._yolo_class = YOLO
        self.detector = YOLO(str(detector_path))
        self.context_model, self.context_checkpoint = load_context_model(context_path, self.device)
        options = dict(predict_options or {})
        self.imgsz = int(options.get("imgsz", 640))
        self.iou = float(options.get("iou", 0.7))
        self.max_det = int(options.get("max_det", 300))
        self.incremental_protocols = {name: dict(value) for name, value in (incremental_protocols or {}).items()}
        self.specialist_protocol: str | None = None
        self.specialist_detector: Any = None
        self.queue = FairInferenceQueue()

    def queue_status(self) -> Dict[str, int | bool]:
        return self.queue.status()

    def predict(
        self,
        image: Image.Image,
        filename: str,
        confidence: float = 0.15,
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

        started = time.perf_counter()
        rgb_image = image.convert("RGB")
        context = predict_context(self.context_model, self.context_checkpoint, rgb_image, self.device)
        prediction = self.detector.predict(
            source=rgb_image,
            imgsz=self.imgsz,
            conf=float(confidence),
            iou=self.iou,
            max_det=self.max_det,
            device=self.device_index,
            verbose=False,
        )[0]
        base_records = result_records(prediction)
        protocol_result = None
        if incremental_protocol:
            protocol = self.incremental_protocols.get(incremental_protocol)
            if protocol is None or not protocol.get("available"):
                raise ValueError("所选增量协议当前不可用于演示。")
            global_class_id = int(protocol["global_class_id"])
            if self.specialist_protocol != incremental_protocol:
                self.specialist_detector = self._yolo_class(str(protocol["weights"]))
                self.specialist_protocol = incremental_protocol
            specialist_prediction = self.specialist_detector.predict(
                source=rgb_image,
                imgsz=self.imgsz,
                conf=float(confidence),
                iou=self.iou,
                max_det=self.max_det,
                device=self.device_index,
                verbose=False,
            )[0]
            records = compose_incremental_records(
                base_records,
                result_records(specialist_prediction),
                global_class_id,
            )
            annotated = annotate_records(rgb_image, records)
            protocol_result = {
                "id": incremental_protocol,
                "new_class": protocol["new_class"],
                "new_map50": protocol["new_map50"],
                "krr": protocol["krr"],
                "status": "activated",
                "activated": True,
                "reason": None,
            }
        else:
            records = base_records
            plotted = prediction.plot(labels=True, conf=True, line_width=2)
            annotated = Image.fromarray(plotted[:, :, ::-1])
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        models_used = ["scene_sensor_net_v1", "unified_yolo11s_v1"]
        if protocol_result:
            models_used.append("incremental_model_bank_v1")
        return {
            "filename": filename,
            "task_id": task_id,
            "context": context,
            "detections": records,
            "class_counts": summarize_records(records),
            "detection_count": len(records),
            "elapsed_ms": elapsed_ms,
            "agent": {
                "mode": "incremental_demo" if protocol_result else "standard_detection",
                "models_used": models_used,
                "protocol": protocol_result,
            },
            "annotated_png": image_png_bytes(annotated),
        }
