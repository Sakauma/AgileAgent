from __future__ import annotations

import io
import json
import zipfile

from PIL import Image

from fair_agent.modules.web_inference import (
    build_batch_zip,
    image_png_bytes,
    result_records,
    summarize_records,
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


def test_detection_records_are_public_and_serializable() -> None:
    records = result_records(FakeResult())
    assert records[0] == {
        "class_id": 0,
        "class_name": "soldier",
        "confidence": 0.91,
        "xyxy": [1.0, 2.0, 30.0, 40.0],
    }
    assert summarize_records(records) == {"soldier": 1, "tank": 1}
    json.dumps(records)


def test_batch_zip_contains_images_and_json() -> None:
    annotated = image_png_bytes(Image.new("RGB", (16, 16), "white"))
    payload = {
        "filename": "sample.png",
        "context": {"sensor": "sar", "scene": "sea"},
        "detections": [],
        "class_counts": {},
        "detection_count": 0,
        "elapsed_ms": 12.3,
        "annotated_image": Image.new("RGB", (16, 16), "white"),
        "annotated_png": annotated,
    }
    archive_bytes = build_batch_zip([payload])
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert sorted(archive.namelist()) == ["annotated/sample.png", "results.json"]
        metadata = json.loads(archive.read("results.json"))
    assert metadata["image_count"] == 1
    assert metadata["results"][0]["filename"] == "sample.png"
    assert "annotated_png" not in metadata["results"][0]
    assert "annotated_image" not in metadata["results"][0]
