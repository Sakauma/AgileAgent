from __future__ import annotations

import io
import json
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image


CLASS_NAMES = {0: "soldier", 1: "small_aircraft", 2: "warship", 3: "tank"}
SENSOR_LABELS = {"ir": "红外", "sar": "SAR"}
SCENE_LABELS = {"air": "空域", "forest": "林地", "sea": "海域", "urban": "城市场景"}


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
        for item in rows:
            safe_stem = Path(str(item["filename"])).stem
            archive.writestr(f"annotated/{safe_stem}.png", item["annotated_png"])
            metadata.append(
                {key: value for key, value in item.items() if key not in {"annotated_png", "annotated_image"}}
            )
        archive.writestr("results.json", result_json_bytes({"image_count": len(rows), "results": metadata}))
    return buffer.getvalue()


class WebInferenceEngine:
    def __init__(self, detector_path: Path, context_path: Path, device_index: str = "0") -> None:
        from ultralytics import YOLO

        from fair_agent.models.context import load_context_model

        self.device_index = str(device_index)
        self.device = f"cuda:{self.device_index}"
        self.detector = YOLO(str(detector_path))
        self.context_model, self.context_checkpoint = load_context_model(context_path, self.device)

    def predict(self, image: Image.Image, filename: str, confidence: float = 0.15) -> Dict[str, Any]:
        from fair_agent.models.context import predict_context

        started = time.perf_counter()
        rgb_image = image.convert("RGB")
        context = predict_context(self.context_model, self.context_checkpoint, rgb_image, self.device)
        prediction = self.detector.predict(
            source=rgb_image,
            imgsz=640,
            conf=float(confidence),
            iou=0.7,
            max_det=300,
            device=self.device_index,
            verbose=False,
        )[0]
        records = result_records(prediction)
        plotted = prediction.plot(labels=True, conf=True, line_width=2)
        annotated = Image.fromarray(plotted[:, :, ::-1])
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "filename": filename,
            "context": context,
            "detections": records,
            "class_counts": summarize_records(records),
            "detection_count": len(records),
            "elapsed_ms": elapsed_ms,
            "annotated_image": annotated,
            "annotated_png": image_png_bytes(annotated),
        }
