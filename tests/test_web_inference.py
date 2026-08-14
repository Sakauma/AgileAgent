from __future__ import annotations

import io
import json
import threading
import time
import zipfile

from PIL import Image

from fair_agent.modules.web_inference import (
    FairInferenceQueue,
    annotate_records,
    box_iou,
    build_batch_zip,
    image_png_bytes,
    result_records,
    summarize_records,
    compose_incremental_records,
    class_aware_nms,
    context_affinity,
    protocol_effective_thresholds,
    plan_specialist_routes,
    remap_specialist_records,
    remap_base_records,
    consensus_specialist_records,
    yolo_inference_ms,
    decode_batch_images,
    decode_image_bytes,
    WebInferenceEngine,
)


ROUTING_ARGS = (4, 0.70, 0.30, 0.50, 0.50)


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
        self.calls = []

    def predict(self, _image, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def predict_batch(self, images, **_kwargs):
        return [self.result for _ in images]


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


def test_detection_records_prefer_native_mappings_without_box_conversion() -> None:
    class NativeResult:
        records = (
            {"class_id": 2, "confidence": 0.91234567, "xyxy": [1.234, 2.345, 30.456, 40.567]},
        )

        @property
        def boxes(self):
            raise AssertionError("native records must bypass the boxes adapter")

    assert result_records(NativeResult()) == [
        {
            "class_id": 2,
            "class_name": "warship",
            "confidence": 0.912346,
            "xyxy": [1.23, 2.35, 30.46, 40.57],
        }
    ]


def test_native_result_box_adapter_supports_iteration_contract() -> None:
    from fair_agent.backends.inference import _NativeResult

    result = _NativeResult(
        {
            "detections": [
                {"class_id": 2, "confidence": 0.91, "xyxy": [1, 2, 30, 40]}
            ]
        }
    )
    box = result.boxes[0]
    assert box.cls.item() == 2
    assert box.conf.item() == 0.91
    assert box.xyxy[0].tolist() == [1, 2, 30, 40]


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


def test_upload_validation_decodes_without_content_identity() -> None:
    first = image_png_bytes(Image.new("RGB", (16, 16), "white"))
    second = image_png_bytes(Image.new("RGB", (16, 16), "black"))
    assert decode_image_bytes(first, "same.png", "pillow").size == (16, 16)
    assert decode_image_bytes(second, "same.png", "pillow").size == (16, 16)


def test_image_decode_only_rejects_unreadable_content() -> None:
    data = image_png_bytes(Image.new("RGB", (16, 16), "white"))
    try:
        decode_image_bytes(b"not-an-image", "invalid.png", "pillow")
    except ValueError as exc:
        assert "无法读取图像" in str(exc)
    else:
        raise AssertionError("unreadable image was accepted")
    assert len(decode_batch_images([("a.png", data), ("copy.png", data)], "pillow")) == 2


def test_parallel_opencv_decode_preserves_order_without_identity_routing() -> None:
    colors = ["red", "green", "blue", "white"]
    rows = [
        (f"{index}.png", image_png_bytes(Image.new("RGB", (16, 16), color)))
        for index, color in enumerate(colors)
    ]
    validated = decode_batch_images(rows, "opencv", 4)
    assert [item[0] for item in validated] == [row[0] for row in rows]
    assert [item[2].getpixel((0, 0)) for item in validated] == [
        Image.new("RGB", (1, 1), color).getpixel((0, 0)) for color in colors
    ]
    assert len(decode_batch_images([rows[0], ("duplicate.png", rows[0][1])], "opencv", 4)) == 2


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


def test_batch_zip_uses_index_for_duplicate_stems() -> None:
    annotated = image_png_bytes(Image.new("RGB", (8, 8), "white"))
    items = [
        {"filename": "same.jpg", "annotated_png": annotated},
        {"filename": "same.png", "annotated_png": annotated},
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
        *ROUTING_ARGS,
    )
    assert [item["id"] for item in eligible] == ["p05_new_vehicle"]
    assert [item["id"] for item in executed] == ["p05_new_vehicle"]
    assert skipped == []


def test_class_incremental_owners_are_not_dropped_by_specialist_budget() -> None:
    protocols = {
        f"p{class_id}": {
            "available": True,
            "incremental_mode": "class_incremental",
            "global_class_id": class_id,
            "activation_threshold": 0.5,
            "calibration_source": f"p{class_id}/calibration.json",
            "routing_prior": 0.5,
        }
        for class_id in (4, 5, 6)
    }

    eligible, executed, skipped = plan_specialist_routes(
        protocols,
        [],
        {},
        {0, 1, 2, 3},
        1,
        *ROUTING_ARGS[1:],
    )

    assert {row["id"] for row in eligible} == {"p4", "p5", "p6"}
    assert {row["id"] for row in executed} == {"p4", "p5", "p6"}
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
        *ROUTING_ARGS,
    )
    assert len(eligible) == len(executed) == 1
    assert eligible[0]["context_score"] == 0.0
    assert skipped == []

    _eligible, _executed, skipped = plan_specialist_routes(
        {"p02": protocol}, [], {}, {0, 1, 2, 3}, *ROUTING_ARGS
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


def test_fusion_does_not_renms_a_frozen_base_only_owner_stream() -> None:
    rows = [
        {"class_id": 0, "confidence": 0.90, "xyxy": [0, 0, 20, 20], "source": "frozen_base_model"},
        {"class_id": 0, "confidence": 0.80, "xyxy": [1, 1, 19, 19], "source": "frozen_base_model"},
    ]

    fused, summary = class_aware_nms(rows, 0.60)

    assert len(fused) == 2
    assert summary == {"input_count": 2, "output_count": 2, "suppressed_count": 0}


def test_vectorized_fusion_preserves_stable_ties_and_input_audit() -> None:
    rows = [
        {
            "class_id": 4,
            "confidence": 0.9 if index < 2 else 0.8 - index / 100,
            "xyxy": [index * 30, 0, index * 30 + 20, 20],
            "source": "incremental_model" if index != 1 else "frozen_base_model",
            "protocol_id": f"p{index}",
        }
        for index in range(8)
    ]
    rows[1]["xyxy"] = [1, 1, 19, 19]

    fused, summary = class_aware_nms(rows, 0.60)

    assert [row["protocol_id"] for row in fused] == ["p0", "p2", "p3", "p4", "p5", "p6", "p7"]
    assert summary == {"input_count": 8, "output_count": 7, "suppressed_count": 1}


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
    assert context_affinity({}, {}, 0.5) == 0.5


def test_class_incremental_context_is_soft_and_does_not_disable_execution() -> None:
    protocol = {
        "available": True,
        "incremental_mode": "class_incremental",
        "global_class_id": 4,
        "activation_threshold": 0.69,
        "calibration_source": "incremental_dev/calibration.json",
        "context_prior": {
            "source_split": "incremental_train_only",
            "scene": {"sea": 1.0, "forest": 0.0},
        },
        "context_gate": {
            "enabled": True,
            "policy": "soft_threshold_penalty",
            "max_threshold_penalty": 0.05,
        },
    }
    context = {"scene_probabilities": {"sea": 0.0, "forest": 1.0}}

    thresholds, affinity = protocol_effective_thresholds(protocol, context, 0.01)
    _eligible, executed, skipped = plan_specialist_routes(
        {"new": protocol}, [], context, {0, 1, 2, 3}, *ROUTING_ARGS
    )

    assert thresholds == {4: 0.74}
    assert affinity == 0.0
    assert [row["id"] for row in executed] == ["new"]
    assert skipped == []


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
    engine.generation_id = "test_generation"
    engine.base_model_id = "test_base_model"
    engine.imgsz = 640
    engine.specialist_imgsz = 512
    engine.iou = 0.7
    engine.max_det = 300
    engine.quantize = 16
    engine.compile = False
    engine.fusion_iou = 0.60
    engine.max_specialists = 4
    engine.conflict_iou = 0.50
    engine.conflict_incremental_coverage = None
    engine.conflict_base_confidence = 0.50
    engine.specialist_margin = 0.15
    engine.preserve_base_class_owners = True
    engine.detection_evidence_weight = 0.70
    engine.context_evidence_weight = 0.30
    engine.neutral_context_score = 0.50
    engine.default_routing_prior = 0.50
    engine.parallel_model_execution = False
    engine.parallel_context_execution = False
    engine.context_stream = None
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
    created_specialists = {}

    def create_specialist(_backend, path, _device, _native):
        detector = FakeDetector(specialist_results[str(path)])
        created_specialists[str(path)] = detector
        return detector

    engine._create_backend = create_specialist
    engine.backend_name = "ultralytics_cuda"
    engine.native_options = {}
    engine.specialist_detectors = {}
    engine.incremental_protocols = {
        "p05_new_vehicle": {
            "available": True,
            "incremental_mode": "class_incremental",
            "global_class_id": 4,
            "class_name": "new_vehicle",
            "activation_threshold": 0.63,
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

    result = engine.predict(Image.new("RGB", (100, 100)), "sample.png", 0.5, "auto")
    assert result["class_counts"] == {"new_vehicle": 1, "warship": 2}
    assert result["agent"]["decision"]["executed_protocols"] == ["p05_new_vehicle", "p02_warship"]
    assert result["agent"]["decision"]["fusion_summary"]["suppressed_count"] == 1
    assert all("fusion_status" in item for item in result["detections"])
    assert created_specialists["new.pt"].calls[0]["conf"] == 0.5


def test_ascend_async_path_submits_three_models_before_collecting_results() -> None:
    events = []

    class Handle:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def result(self):
            events.append(f"result:{self.name}")
            return self.value

    class AsyncContext:
        def submit(self, _image):
            events.append("submit:scene")
            return Handle(
                "scene",
                {
                    "sensor": "sar",
                    "sensor_confidence": 0.9,
                    "sensor_probabilities": {"ir": 0.1, "sar": 0.9},
                    "scene": "urban",
                    "scene_confidence": 0.8,
                    "scene_probabilities": {
                        "air": 0.0,
                        "forest": 0.1,
                        "sea": 0.1,
                        "urban": 0.8,
                    },
                    "_inference_ms": 1.0,
                    "_ascend_timings": {
                        "ascend_submit": 0.1,
                        "ascend_wait": 0.2,
                        "ascend_input_copy": 0.3,
                        "ascend_output_copy": 0.4,
                    },
                },
            )

    class AsyncDetector:
        def __init__(self, name, result, timing):
            self.name = name
            self.result_value = result
            self.result_value.speed.update(timing)

        def submit(self, _image, **_options):
            events.append(f"submit:{self.name}")
            return Handle(self.name, self.result_value)

    base_result = DynamicResult([([0, 0, 20, 20], 0.80, 0)], inference=2.0)
    specialist_result = DynamicResult(
        [([30, 30, 50, 50], 0.88, 0)], inference=3.0
    )
    engine = WebInferenceEngine.__new__(WebInferenceEngine)
    engine.context_model = AsyncContext()
    engine.context_checkpoint = {}
    engine.device = "ascend:0"
    engine.device_index = "0"
    engine.generation_id = "test_generation"
    engine.base_model_id = "test_base_model"
    engine.imgsz = 640
    engine.specialist_imgsz = 512
    engine.iou = 0.7
    engine.max_det = 300
    engine.quantize = 16
    engine.compile = False
    engine.fusion_iou = 0.60
    engine.max_specialists = 1
    engine.conflict_iou = 0.50
    engine.conflict_incremental_coverage = None
    engine.conflict_base_confidence = 0.50
    engine.specialist_margin = 0.15
    engine.preserve_base_class_owners = True
    engine.detection_evidence_weight = 0.70
    engine.context_evidence_weight = 0.30
    engine.neutral_context_score = 0.50
    engine.default_routing_prior = 0.50
    engine.parallel_model_execution = True
    engine.parallel_context_execution = True
    engine.context_stream = None
    engine.class_names = {0: "soldier", 4: "new_vehicle"}
    engine.base_class_ids = {0}
    engine.base_local_to_global = {0: 0}
    engine.base_local_names = {0: "soldier"}
    engine.class_owners = {0: "test_base_model", 4: "p05_new_vehicle"}
    engine.unified_class_gates = {}
    engine.backend_name = "ascend_acl"
    engine.native_options = {"execution_mode": "async_stream"}
    engine.encoded_preprocessor = None
    engine.detector = AsyncDetector(
        "base",
        base_result,
        {
            "ascend_submit": 0.2,
            "ascend_wait": 0.3,
            "ascend_input_copy": 0.4,
            "ascend_output_copy": 0.5,
        },
    )
    engine.specialist_detectors = {
        "p05_new_vehicle": AsyncDetector(
            "specialist",
            specialist_result,
            {
                "ascend_submit": 0.25,
                "ascend_wait": 0.35,
                "ascend_input_copy": 0.45,
                "ascend_output_copy": 0.55,
            },
        )
    }
    engine.incremental_protocols = {
        "p05_new_vehicle": {
            "available": True,
            "incremental_mode": "class_incremental",
            "global_class_id": 4,
            "local_to_global": {0: 4},
            "class_name": "new_vehicle",
            "activation_threshold": 0.63,
            "calibration_source": "incremental_val/calibration.json",
            "routing_prior": 0.8,
            "context_prior": {},
            "weights": "new.pt",
            "new_map50": 0.8,
            "krr": 0.95,
        }
    }

    result = engine._predict_unlocked(
        Image.new("RGB", (100, 100)), "sample.png", 0.5, "auto"
    )

    assert events == [
        "submit:scene",
        "submit:base",
        "submit:specialist",
        "result:scene",
        "result:base",
        "result:specialist",
    ]
    assert result["timings"]["dvpp_device_ms"] == 0.0
    assert result["timings"]["ascend_submit_ms"] == 0.55
    assert result["timings"]["ascend_wait_ms"] == 0.85
    assert result["timings"]["ascend_input_copy_max_ms"] == 0.45
    assert result["timings"]["ascend_output_copy_max_ms"] == 0.55
