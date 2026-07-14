from __future__ import annotations

import io
import json
import threading
import time
import zipfile

from PIL import Image

from fair_agent.modules.web_inference import (
    FairInferenceQueue,
    MAX_BATCH_FILES,
    annotate_records,
    box_iou,
    build_batch_zip,
    content_task_id,
    image_png_bytes,
    result_records,
    summarize_records,
    compose_incremental_records,
    class_aware_nms,
    context_affinity,
    plan_specialist_routes,
    remap_specialist_records,
    remap_base_records,
    consensus_specialist_records,
    yolo_inference_ms,
    validate_batch_uploads,
    validate_image_bytes,
    WebInferenceEngine,
)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[1, 2, 30, 40], [5, 6, 50, 60]])
        self.conf = FakeTensor([0.91, 0.72])
        self.cls = FakeTensor([0, 3])

    def __len__(self):
        return 2


class FakeResult:
    boxes = FakeBoxes()
    speed = {"preprocess": 1.2, "inference": 7.35, "postprocess": 2.1}


class DynamicBoxes:
    def __init__(self, rows):
        self.xyxy = FakeTensor([row[0] for row in rows])
        self.conf = FakeTensor([row[1] for row in rows])
        self.cls = FakeTensor([row[2] for row in rows])

    def __len__(self):
        return len(self.conf.values)


class DynamicResult:
    def __init__(self, rows, inference=1.0):
        self.boxes = DynamicBoxes(rows)
        self.speed = {"inference": inference}


class FakeDetector:
    def __init__(self, result):
        self.result = result

    def predict(self, **_kwargs):
        return [self.result]


def test_detection_records_are_public_and_serializable() -> None:
    records = result_records(FakeResult())
    assert records[0] == {
        "class_id": 0,
        "class_name": "soldier",
        "confidence": 0.91,
        "xyxy": [1.0, 2.0, 30.0, 40.0],
    }
    assert summarize_records(records) == {"soldier": 1, "tank": 1}
    assert yolo_inference_ms(FakeResult()) == 7.35
    json.dumps(records)


def test_annotation_draws_boxes_without_label_background() -> None:
    source = Image.new("RGB", (64, 64), "black")
    annotated = annotate_records(
        source,
        [{"class_id": 2, "class_name": "warship", "confidence": 0.91, "xyxy": [10, 20, 50, 50]}],
    )
    assert annotated.crop((0, 0, 64, 20)).tobytes() == source.crop((0, 0, 64, 20)).tobytes()
    assert annotated.getpixel((10, 20)) != source.getpixel((10, 20))


def test_batch_zip_contains_images_and_json() -> None:
    annotated = image_png_bytes(Image.new("RGB", (16, 16), "white"))
    payload = {
        "filename": "sample.png",
        "context": {"sensor": "sar", "scene": "sea"},
        "detections": [],
        "class_counts": {},
        "detection_count": 0,
        "inference_ms": 12.3,
        "annotated_image": Image.new("RGB", (16, 16), "white"),
        "annotated_png": annotated,
    }
    archive_bytes = build_batch_zip([payload])
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert sorted(archive.namelist()) == ["annotated/001_sample.png", "results.json"]
        metadata = json.loads(archive.read("results.json"))
    assert metadata["image_count"] == 1
    assert metadata["results"][0]["filename"] == "sample.png"
    assert "annotated_png" not in metadata["results"][0]
    assert "annotated_image" not in metadata["results"][0]


def test_upload_validation_uses_content_hash_not_filename() -> None:
    first = image_png_bytes(Image.new("RGB", (16, 16), "white"))
    second = image_png_bytes(Image.new("RGB", (16, 16), "black"))
    image, task_id = validate_image_bytes(first, "same.png")
    assert image.size == (16, 16)
    assert task_id == content_task_id(first)
    assert task_id != content_task_id(second)


def test_upload_validation_rejects_size_pixels_and_duplicate_batch() -> None:
    data = image_png_bytes(Image.new("RGB", (16, 16), "white"))
    try:
        validate_image_bytes(data, "large.png", max_bytes=10)
    except ValueError as exc:
        assert "超过" in str(exc)
    else:
        raise AssertionError("oversized upload was accepted")
    try:
        validate_image_bytes(data, "pixels.png", max_pixels=100)
    except ValueError as exc:
        assert "像素超过限制" in str(exc)
    else:
        raise AssertionError("oversized image was accepted")
    try:
        validate_batch_uploads([("a.png", data), ("copy.png", data)])
    except ValueError as exc:
        assert "重复图像" in str(exc)
    else:
        raise AssertionError("duplicate image was accepted")
    try:
        validate_batch_uploads([(f"{index}.png", data + bytes([index])) for index in range(MAX_BATCH_FILES + 1)])
    except ValueError as exc:
        assert "最多处理" in str(exc)
    else:
        raise AssertionError("oversized batch was accepted")


def test_fair_inference_queue_serializes_concurrent_work() -> None:
    queue = FairInferenceQueue()
    guard = threading.Lock()
    active = 0
    max_active = 0
    results = []

    def worker(value: int) -> None:
        def operation() -> int:
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            return value

        result, wait_ms = queue.run(operation)
        results.append((result, wait_ms))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(value for value, _ in results) == [0, 1, 2, 3]
    assert max_active == 1
    assert queue.status() == {"waiting": 0, "active": False, "completed": 4}


def test_batch_zip_uses_index_and_task_hash_for_duplicate_stems() -> None:
    annotated = image_png_bytes(Image.new("RGB", (8, 8), "white"))
    items = [
        {"filename": "same.jpg", "task_id": "a" * 64, "annotated_png": annotated},
        {"filename": "same.png", "task_id": "b" * 64, "annotated_png": annotated},
    ]
    with zipfile.ZipFile(io.BytesIO(build_batch_zip(items))) as archive:
        names = archive.namelist()
    assert "annotated/001_same.png" in names
    assert "annotated/002_same.png" in names


def test_incremental_composition_keeps_old_classes_and_remaps_specialist() -> None:
    base = [
        {"class_id": 0, "class_name": "soldier", "confidence": 0.8, "xyxy": [1, 1, 2, 2]},
        {"class_id": 2, "class_name": "warship", "confidence": 0.7, "xyxy": [3, 3, 4, 4]},
    ]
    specialist = [
        {"class_id": 0, "class_name": "specialist", "confidence": 0.9, "xyxy": [5, 5, 6, 6]},
    ]
    remapped = remap_specialist_records(specialist, 2)
    assert remapped[0]["class_name"] == "warship"
    assert remapped[0]["source"] == "incremental_model"
    composed = compose_incremental_records(base, specialist, 2)
    assert [item["class_id"] for item in composed] == [0, 2, 2]
    assert [item["source"] for item in composed] == [
        "frozen_base_model", "frozen_base_model", "incremental_model"
    ]


def test_specialist_candidates_require_spatial_consensus() -> None:
    base = [{"class_id": 2, "xyxy": [10, 10, 30, 30]}]
    candidates = [
        {"class_id": 2, "xyxy": [11, 11, 29, 29]},
        {"class_id": 2, "xyxy": [40, 40, 55, 55]},
    ]
    assert box_iou(base[0]["xyxy"], candidates[0]["xyxy"]) > 0.30
    assert consensus_specialist_records(base, candidates, 2, 0.30) == [candidates[0]]
    assert consensus_specialist_records([], candidates, 2, 0.30) == []


def test_class_incremental_route_does_not_require_base_same_class() -> None:
    protocols = {
        "p05_new_vehicle": {
            "available": True,
            "incremental_mode": "class_incremental",
            "global_class_id": 4,
            "activation_threshold": 0.65,
            "calibration_source": "incremental_val/calibration.json",
            "routing_prior": 0.8,
            "context_prior": {},
        }
    }
    eligible, executed, skipped = plan_specialist_routes(
        protocols,
        [{"class_id": 0, "confidence": 0.8, "xyxy": [1, 1, 2, 2]}],
        {"sensor": "sar", "scene": "urban"},
        {0, 1, 2, 3},
    )
    assert [item["id"] for item in eligible] == ["p05_new_vehicle"]
    assert [item["id"] for item in executed] == ["p05_new_vehicle"]
    assert skipped == []


def test_target_incremental_route_requires_base_class_but_not_scene_match() -> None:
    protocol = {
        "available": True,
        "incremental_mode": "target_incremental",
        "global_class_id": 2,
        "activation_threshold": 0.5,
        "context_prior": {"scene": {"sea": 1.0, "urban": 0.0}},
    }
    eligible, executed, skipped = plan_specialist_routes(
        {"p02": protocol},
        [{"class_id": 2, "confidence": 0.9, "xyxy": [1, 1, 2, 2]}],
        {"scene_probabilities": {"sea": 0.0, "urban": 1.0}},
        {0, 1, 2, 3},
    )
    assert len(eligible) == len(executed) == 1
    assert eligible[0]["context_score"] == 0.0
    assert skipped == []

    _eligible, _executed, skipped = plan_specialist_routes(
        {"p02": protocol}, [], {}, {0, 1, 2, 3}
    )
    assert skipped == [{"id": "p02", "reason": "base_class_not_detected"}]


def test_fusion_preserves_unmatched_base_and_removes_same_class_duplicate() -> None:
    rows = [
        {"class_id": 2, "class_name": "warship", "confidence": 0.80, "xyxy": [0, 0, 20, 20], "source": "frozen_base_model"},
        {"class_id": 2, "class_name": "warship", "confidence": 0.70, "xyxy": [40, 40, 60, 60], "source": "frozen_base_model"},
        {"class_id": 2, "class_name": "warship", "confidence": 0.90, "xyxy": [1, 1, 19, 19], "source": "incremental_model", "protocol_id": "p02"},
    ]
    fused, summary = class_aware_nms(rows, 0.60)
    assert len(fused) == 2
    assert any(item["xyxy"] == [40, 40, 60, 60] for item in fused)
    assert any(item["source"] == "incremental_model" for item in fused)
    assert summary == {"input_count": 3, "output_count": 2, "suppressed_count": 1}


def test_dynamic_new_class_mapping_and_neutral_context() -> None:
    remapped = remap_specialist_records(
        [{"class_id": 0, "confidence": 0.9, "xyxy": [1, 1, 2, 2]}],
        7,
        {7: "new_vehicle"},
        "p07",
    )
    assert remapped[0]["class_id"] == 7
    assert remapped[0]["class_name"] == "new_vehicle"
    assert remapped[0]["protocol_id"] == "p07"
    assert context_affinity({}, {}) == 0.5


def test_strict_base_local_ids_remap_to_global_space() -> None:
    records = [
        {"class_id": 0, "class_name": "soldier", "confidence": 0.9, "xyxy": [1, 1, 2, 2]},
        {"class_id": 1, "class_name": "warship", "confidence": 0.8, "xyxy": [3, 3, 4, 4]},
        {"class_id": 2, "class_name": "tank", "confidence": 0.7, "xyxy": [5, 5, 6, 6]},
    ]
    remapped = remap_base_records(records, {0: 0, 1: 2, 2: 3}, {0: "soldier", 2: "warship", 3: "tank"})
    assert [item["class_id"] for item in remapped] == [0, 2, 3]
    assert [item["class_name"] for item in remapped] == ["soldier", "warship", "tank"]


def test_full_engine_auto_route_activates_true_new_class_and_preserves_base(monkeypatch) -> None:
    monkeypatch.setattr(
        "fair_agent.models.context.predict_context",
        lambda *_args, **_kwargs: {
            "sensor": "sar",
            "sensor_confidence": 0.9,
            "sensor_probabilities": {"ir": 0.1, "sar": 0.9},
            "scene": "urban",
            "scene_confidence": 0.8,
            "scene_probabilities": {"air": 0.0, "forest": 0.1, "sea": 0.1, "urban": 0.8},
            "_inference_ms": 1.0,
        },
    )
    engine = WebInferenceEngine.__new__(WebInferenceEngine)
    engine.context_model = object()
    engine.context_checkpoint = {}
    engine.device = "cuda:0"
    engine.device_index = "0"
    engine.imgsz = 640
    engine.iou = 0.7
    engine.max_det = 300
    engine.fusion_iou = 0.60
    engine.max_specialists = 4
    engine.class_names = {0: "soldier", 1: "small_aircraft", 2: "warship", 3: "tank", 4: "new_vehicle"}
    engine.base_class_ids = {0, 1, 2, 3}
    engine.base_local_to_global = {0: 0, 1: 1, 2: 2, 3: 3}
    engine.base_local_names = {0: "soldier", 1: "small_aircraft", 2: "warship", 3: "tank"}
    engine.detector = FakeDetector(
        DynamicResult([
            ([0, 0, 20, 20], 0.80, 2),
            ([40, 40, 60, 60], 0.70, 2),
        ])
    )
    specialist_results = {
        "target.pt": DynamicResult([([1, 1, 19, 19], 0.95, 0)]),
        "new.pt": DynamicResult([([70, 70, 90, 90], 0.88, 0)]),
    }
    engine._yolo_class = lambda path: FakeDetector(specialist_results[path])
    engine.specialist_detectors = {}
    engine.incremental_protocols = {
        "p05_new_vehicle": {
            "available": True,
            "incremental_mode": "class_incremental",
            "global_class_id": 4,
            "class_name": "new_vehicle",
            "activation_threshold": 0.5,
            "calibration_source": "incremental_val/calibration.json",
            "routing_prior": 0.8,
            "context_prior": {},
            "weights": "new.pt",
            "new_map50": 0.8,
            "krr": 0.95,
        },
        "p02_warship": {
            "available": True,
            "incremental_mode": "target_incremental",
            "global_class_id": 2,
            "class_name": "warship",
            "activation_threshold": 0.5,
            "context_prior": {},
            "consensus_iou": 0.3,
            "weights": "target.pt",
            "new_map50": 0.83,
            "krr": 1.0,
        },
    }
    engine.queue = FairInferenceQueue()

    result = engine.predict(Image.new("RGB", (100, 100)), "sample.png", 0.5, "task", "auto")
    assert result["class_counts"] == {"new_vehicle": 1, "warship": 2}
    assert result["agent"]["decision"]["executed_protocols"] == ["p05_new_vehicle", "p02_warship"]
    assert result["agent"]["decision"]["fusion_summary"]["suppressed_count"] == 1
    assert all("fusion_status" in item for item in result["detections"])
