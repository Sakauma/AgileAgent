from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import yaml
from PIL import Image
from starlette.testclient import TestClient

from fair_agent.core.runtime_log import StructuredEventLog, mirror_state_event
from fair_agent.modules.incremental_workbench import IncrementalBatchStore, TrainingJobManager
from fair_agent.web.app import create_app


def png_bytes(color: str = "white") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(output, format="PNG")
    return output.getvalue()


def dataset_zip(class_name: str = "unknown_vehicle", count: int = 5, class_id: int = 0) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("data.yaml", f"names:\n  {class_id}: {class_name}\n")
        for index in range(count):
            split = "val" if index == count - 1 else "train"
            archive.writestr(f"images/{split}/sample_{index}.png", png_bytes("white" if index % 2 else "black"))
            archive.writestr(f"labels/{split}/sample_{index}.txt", f"{class_id} 0.5 0.5 0.2 0.2\n")
    return output.getvalue()


def unnamed_dataset_zip(label_line: str = "0 0.5 0.5 0.2 0.2") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for index in range(2):
            archive.writestr(f"images/sample_{index}.png", png_bytes("white" if index else "black"))
            archive.writestr(f"labels/sample_{index}.txt", label_line + "\n")
    return output.getvalue()


def settings(tmp_path: Path) -> dict:
    return {
        "root": str(tmp_path / "batches"),
        "max_archive_bytes": 10 * 1024 * 1024,
        "max_extracted_bytes": 20 * 1024 * 1024,
        "max_extracted_files": 100,
        "max_image_pixels": 1_000_000,
        "allowed_image_extensions": [".png", ".jpg"],
        "allowed_label_formats": ["class_id_bbox", "bbox_only"],
        "require_labels": True,
        "validation_fraction": 0.20,
        "minimum_images": 2,
        "preview_limit": 12,
        "job_log_tail_lines": 300,
        "poll_interval_ms": 2000,
        "training": {
            "python": None,
            "initial_weights": "models/production/incremental_detection/three_class_base_detector.pt",
            "device": "0", "imgsz": 640, "batch": 32, "epochs": 1, "patience": 1,
            "workers": 0, "optimizer": "AdamW", "lr0": 0.001, "seed": 20260705,
            "deterministic": True, "amp": True,
        },
    }


def make_store(tmp_path: Path) -> tuple[IncrementalBatchStore, StructuredEventLog]:
    event_log = StructuredEventLog(tmp_path / "logs", 1024 * 1024, 3)
    store = IncrementalBatchStore(settings(tmp_path), event_log, ["soldier", "warship"])
    return store, event_log


def test_upload_audit_and_injection_are_persistent_and_incremental_only(tmp_path: Path) -> None:
    store, event_log = make_store(tmp_path)
    manifest = store.create("new-targets.zip", dataset_zip(), "未知车辆批次")
    assert manifest["status"] == "AUDITED"
    assert manifest["audit"]["image_count"] == 5
    assert manifest["audit"]["object_count"] == 5
    assert manifest["audit"]["incremental_mode"] == "class_incremental"
    assert manifest["audit"]["old_raw_image_count"] == 0
    assert manifest["class_registry"]["revision"] == 1
    assert (store.root / manifest["batch_id"] / "class_registry.yaml").is_file()
    injected = store.inject(manifest["batch_id"])
    assert injected["status"] == "INJECTED"
    assert injected["injection"]["counts"] == {"train": 4, "val": 1}
    batch_dir = store.root / manifest["batch_id"]
    assert (batch_dir / "batch.yaml").is_file()
    assert (batch_dir / "prepared" / "dataset.yaml").is_file()
    assert len(store.list()) == 1
    events = event_log.query(batch_id=manifest["batch_id"])
    assert {row["event"] for row in events} >= {
        "incremental.upload.saved", "incremental.audit.completed", "incremental.inject.completed"
    }


def test_existing_class_is_detected_as_target_incremental(tmp_path: Path) -> None:
    store, _event_log = make_store(tmp_path)
    manifest = store.create("warship.zip", dataset_zip("warship"))
    assert manifest["audit"]["incremental_mode"] == "target_incremental"


def test_archive_path_traversal_is_rejected_without_writing_outside_root(tmp_path: Path) -> None:
    store, _event_log = make_store(tmp_path)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("../escape.txt", "blocked")
    manifest = store.create("unsafe.zip", output.getvalue())
    assert manifest["status"] == "REJECTED"
    assert "不安全路径" in manifest["error"]
    assert not (tmp_path / "escape.txt").exists()


def test_generated_names_are_stable_and_do_not_block_injection(tmp_path: Path) -> None:
    store, _event_log = make_store(tmp_path)
    manifest = store.create("unnamed.zip", unnamed_dataset_zip())
    assert manifest["audit"]["requires_class_confirmation"] is True
    assert manifest["audit"]["class_map"] == {"0": "类别3"}
    assert manifest["audit"]["local_to_global"] == {"0": 2}
    assert manifest["audit"]["class_bindings"][0]["semantic_status"] == "provisional"
    injected = store.inject(manifest["batch_id"])
    assert injected["status"] == "INJECTED"


def test_multiple_unnamed_batches_reserve_sequential_names_and_global_ids(tmp_path: Path) -> None:
    store, _event_log = make_store(tmp_path)
    first = store.create("first.zip", unnamed_dataset_zip())
    second = store.create("second.zip", unnamed_dataset_zip())
    assert first["audit"]["class_map"] == {"0": "类别3"}
    assert first["audit"]["local_to_global"] == {"0": 2}
    assert second["audit"]["class_map"] == {"0": "类别4"}
    assert second["audit"]["local_to_global"] == {"0": 3}


def test_competition_profile_accepts_bbox_only_labels_with_backend_warning(tmp_path: Path) -> None:
    store, event_log = make_store(tmp_path)
    manifest = store.create("bbox-only.zip", unnamed_dataset_zip("0.5 0.5 0.2 0.2"))
    assert manifest["status"] == "AUDITED"
    assert manifest["audit"]["label_format"] == "bbox_only"
    assert manifest["audit"]["warnings"][0]["code"] == "bbox_only_labels_without_class_id"
    assert manifest["audit"]["requires_class_confirmation"] is True
    warning_events = event_log.query(batch_id=manifest["batch_id"], level="warning")
    assert any(row["event"] == "incremental.audit.warning" for row in warning_events)
    injected = store.inject(manifest["batch_id"])
    prepared_label = next((store.root / injected["batch_id"] / "prepared" / "labels").rglob("*.txt"))
    assert prepared_label.read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"


def test_dataset_names_and_unused_source_id_are_preserved(tmp_path: Path) -> None:
    event_log = StructuredEventLog(tmp_path / "logs", 1024 * 1024, 3)
    store = IncrementalBatchStore(settings(tmp_path), event_log, {0: "soldier", 1: "small_aircraft", 2: "tank"})
    manifest = store.create("official-new-class.zip", dataset_zip("warship", class_id=3))
    binding = manifest["audit"]["class_bindings"][0]
    assert binding == {
        "source_class_id": 3,
        "training_class_id": 0,
        "global_class_id": 3,
        "display_name": "warship",
        "semantic_status": "confirmed",
        "semantic_source": "dataset_names",
        "is_existing_class": False,
    }


def test_class_rename_keeps_ids_and_updates_prepared_yaml(tmp_path: Path) -> None:
    store, event_log = make_store(tmp_path)
    manifest = store.create("unnamed.zip", unnamed_dataset_zip())
    injected = store.inject(manifest["batch_id"])
    before = injected["audit"]["class_bindings"][0]
    renamed = store.rename_classes(manifest["batch_id"], {0: "新型车辆"})
    after = renamed["audit"]["class_bindings"][0]
    assert after["display_name"] == "新型车辆"
    assert after["semantic_status"] == "confirmed"
    assert after["source_class_id"] == before["source_class_id"]
    assert after["training_class_id"] == before["training_class_id"]
    assert after["global_class_id"] == before["global_class_id"]
    assert renamed["class_registry"]["revision"] == 2
    assert (store.root / manifest["batch_id"] / "class_registry_history" / "revision-0001.yaml").is_file()
    assert (store.root / manifest["batch_id"] / "class_registry_history" / "revision-0002.yaml").is_file()
    dataset = yaml.safe_load(
        (store.root / manifest["batch_id"] / "prepared" / "dataset.yaml").read_text(encoding="utf-8")
    )
    assert dataset["names"] == {0: "新型车辆"}
    assert event_log.query(batch_id=manifest["batch_id"], component="incremental")[0]["event"] == "incremental.classes.renamed"


def test_training_snapshot_is_immutable_after_late_class_rename(tmp_path: Path) -> None:
    store, event_log = make_store(tmp_path)
    manifest = store.create("unnamed.zip", unnamed_dataset_zip())
    store.inject(manifest["batch_id"])
    manager = TrainingJobManager(store, settings(tmp_path), event_log)
    snapshot = manager._create_training_snapshot(manifest["batch_id"], "train-snapshot-test")
    snapshot_registry_path = Path(snapshot["class_registry"])
    if not snapshot_registry_path.is_absolute():
        snapshot_registry_path = Path.cwd() / snapshot_registry_path
    frozen = yaml.safe_load(snapshot_registry_path.read_text(encoding="utf-8"))
    assert frozen["bindings"][0]["display_name"] == "类别3"

    store.rename_classes(manifest["batch_id"], {0: "新型车辆"})
    current = yaml.safe_load(
        (store.root / manifest["batch_id"] / "class_registry.yaml").read_text(encoding="utf-8")
    )
    frozen_after = yaml.safe_load(snapshot_registry_path.read_text(encoding="utf-8"))
    assert current["bindings"][0]["display_name"] == "新型车辆"
    assert current["revision"] == 2
    assert frozen_after == frozen
    assert snapshot["class_registry_revision"] == 1


class NoopEngine:
    def queue_status(self):
        return {"waiting": 0, "active": False, "completed": 0}


class NoopTrainingManager:
    def list(self, batch_id=None):
        return []


def test_web_incremental_upload_browse_and_inject_flow(tmp_path: Path) -> None:
    store, event_log = make_store(tmp_path)
    client = TestClient(
        create_app(
            engine_provider=lambda: NoopEngine(),
            incremental_store=store,
            training_manager=NoopTrainingManager(),
            event_log=event_log,
        )
    )
    upload = client.post(
        "/api/incremental/batches",
        files={"file": ("new-targets.zip", dataset_zip(), "application/zip")},
        data={"name": "网页批次"},
    )
    assert upload.status_code == 201
    batch_id = upload.json()["batch_id"]
    listing = client.get("/api/incremental/batches").json()["batches"]
    assert listing[0]["batch_id"] == batch_id
    detail = client.get(f"/api/incremental/batches/{batch_id}")
    assert detail.status_code == 200
    image = client.get(f"/api/incremental/batches/{batch_id}/images/0")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    renamed = client.patch(
        f"/api/incremental/batches/{batch_id}/classes",
        json={"names": {"0": "海上新目标"}},
    )
    assert renamed.status_code == 200
    assert renamed.json()["audit"]["class_bindings"][0]["display_name"] == "海上新目标"
    injected = client.post(f"/api/incremental/batches/{batch_id}/inject")
    assert injected.status_code == 200
    assert injected.json()["status"] == "INJECTED"
    logs = client.get(f"/api/logs?component=incremental&batch_id={batch_id}")
    assert logs.status_code == 200
    assert len(logs.json()["events"]) >= 3


def test_structured_log_redacts_secrets_and_filters(tmp_path: Path) -> None:
    event_log = StructuredEventLog(tmp_path / "logs", 1024, 2)
    event_log.append(
        "unit.event", component="test", trace_id="trace-1", batch_id="batch-1",
        details={"token": "do-not-log", "nested": {"password": "hidden", "value": 3}},
    )
    rows = event_log.query(component="test", batch_id="batch-1")
    assert len(rows) == 1
    assert rows[0]["details"]["token"] == "***"
    assert rows[0]["details"]["nested"]["password"] == "***"
    assert "do-not-log" not in json.dumps(rows, ensure_ascii=False)


def test_state_events_are_queryable_across_experiment_identifiers(tmp_path: Path) -> None:
    event_log = StructuredEventLog(tmp_path / "logs", 1024 * 1024, 2)
    mirror_state_event(
        event_log,
        "THRESHOLD_CALIBRATED",
        status="completed",
        experiment_id="warship_3plus1",
        run_id="run-01",
        protocol_id="round_01",
        details={"threshold": 0.63, "artifact_sha256": "a" * 64},
    )
    rows = event_log.query(
        experiment_id="warship_3plus1", run_id="run-01", protocol_id="round_01"
    )
    assert len(rows) == 1
    assert rows[0]["event"] == "incremental.dev_calibration.completed"
    assert rows[0]["details"]["threshold"] == 0.63
