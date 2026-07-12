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
    build_batch_zip,
    content_task_id,
    image_png_bytes,
    result_records,
    summarize_records,
    compose_incremental_records,
    remap_specialist_records,
    yolo_inference_ms,
    validate_batch_uploads,
    validate_image_bytes,
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
    assert [item["class_id"] for item in composed] == [0, 2]
    assert [item["source"] for item in composed] == ["frozen_base_model", "incremental_model"]
