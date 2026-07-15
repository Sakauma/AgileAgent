from __future__ import annotations

import io
import json
import sys
import threading
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from fair_agent.backends.inference import TensorRTEngineBackend
from fair_agent.cli import cmd_incremental
from fair_agent.core.config import rel_path
from fair_agent.core.hashes import sha256_file
from fair_agent.core.runtime_log import StructuredEventLog
from fair_agent.modules.incremental_lifecycle import IncrementalLifecycle
from fair_agent.modules.incremental_lineage import _canonical_sha256, audit_incremental_records
from fair_agent.modules.incremental_workbench import IncrementalBatchStore
from fair_agent.modules.model_generations import generation_web_settings, load_generation_registry
from fair_agent.modules.web_inference import plan_specialist_routes
from fair_agent.web.app import AtomicEngineProvider


def _catalog(path: Path, files: list[dict], cache_files: list[dict] | None = None) -> None:
    payload = {
        "schema_version": 1, "catalog_id": "base", "kind": "test", "files": files,
        "cache_files": cache_files or [],
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_lineage_detects_renamed_old_image_label_and_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "old.npy").write_bytes(b"feature")
    base = tmp_path / "lineage" / "base.json"
    _catalog(
        base,
        [{"stem": "original", "image_sha256": "a" * 64, "label_sha256": "b" * 64}],
        [{"path": str(cache / "old.npy"), "sha256": sha256_file(cache / "old.npy")}],
    )
    settings = {
        "lineage": {
            "required": True, "root": str(tmp_path / "lineage"), "base_manifest": str(base),
            "base_experiment_config": str(tmp_path / "unused.yaml"), "auto_initialize_base": False,
            "cache_roots": [str(cache)],
        }
    }
    result = audit_incremental_records([{
        "image": "renamed.png", "label": "renamed.txt",
        "image_sha256": "a" * 64, "label_sha256": "b" * 64,
    }], settings)
    assert result["compliance"] == "blocked"
    assert result["old_raw_image_count"] == 1
    assert result["old_raw_label_count"] == 1
    assert result["old_cache_count"] == 1


def _png(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(output, format="PNG")
    return output.getvalue()


def _multi_class_zip() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("data.yaml", "names:\n  4: class_four\n  7: class_seven\n")
        for index in range(15):
            class_id = 4 if index % 2 == 0 else 7
            archive.writestr(f"images/train/sample_{index}.png", _png((index, 0, 0)))
            archive.writestr(f"labels/train/sample_{index}.txt", f"{class_id} 0.5 0.5 0.2 0.2\n")
    return output.getvalue()


def _store_settings(tmp_path: Path) -> dict:
    base = tmp_path / "lineage" / "base.json"
    _catalog(base, [{"stem": "old", "image_sha256": "f" * 64, "label_sha256": "e" * 64}])
    return {
        "root": str(tmp_path / "batches"), "max_archive_bytes": 10_000_000,
        "max_extracted_bytes": 20_000_000, "max_extracted_files": 100,
        "max_image_pixels": 1_000_000, "allowed_image_extensions": [".png"],
        "allowed_label_formats": ["class_id_bbox"], "require_labels": True,
        "validation_fraction": 0.2, "lock_fraction": 0.2, "split_seed": 20260705,
        "minimum_images": 2, "preview_limit": 5, "job_log_tail_lines": 30,
        "poll_interval_ms": 1000,
        "lineage": {
            "required": True, "root": str(tmp_path / "lineage"), "base_manifest": str(base),
            "base_experiment_config": str(tmp_path / "unused.yaml"), "auto_initialize_base": False,
            "cache_roots": [],
        },
        "training": {"seed": 20260705},
    }


def test_auto_lock_is_deterministic_stratified_and_not_training_reachable(tmp_path: Path) -> None:
    event_log = StructuredEventLog(tmp_path / "logs", 1_000_000, 2)
    store = IncrementalBatchStore(_store_settings(tmp_path), event_log, {0: "old"})
    manifest = store.create("multi.zip", _multi_class_zip())
    injected = store.inject(manifest["batch_id"])
    assert injected["injection"]["counts"] == {"train": 9, "val": 3, "lock": 3}
    batch_dir = store.root / manifest["batch_id"]
    assert not (batch_dir / "prepared" / "images" / "lock").exists()
    assert len(list((batch_dir / "sealed_lock" / "images" / "lock").glob("*.png"))) == 3
    lock_manifest = json.loads((batch_dir / "sealed_lock" / "lock_manifest.json").read_text(encoding="utf-8"))
    assert lock_manifest["auto_sealed"] is True
    assert {class_id for row in lock_manifest["files"] for class_id in next(
        item["classes"] for item in manifest["files"] if Path(item["image"]).stem == row["stem"]
    )} == {4, 7}
    second = store.create("multi-second.zip", _multi_class_zip())
    store.inject(second["batch_id"])
    second_lock = json.loads(
        (store.root / second["batch_id"] / "sealed_lock" / "lock_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["stem"] for row in second_lock["files"]] == [
        row["stem"] for row in lock_manifest["files"]
    ]


def test_generation_schema_supports_one_multi_class_expert(tmp_path: Path) -> None:
    payload = json.loads(Path("models/generations.json").read_text(encoding="utf-8"))
    payload = deepcopy(payload)
    payload["class_map"]["4"] = "new_vehicle"
    expert = next(item for item in payload["models"] if item["id"] == "incremental_detector")
    expert["owns_classes"] = [2, 4]
    expert["local_to_global"]["1"] = 4
    expert["per_class_thresholds"]["4"] = 0.71
    expert["calibration_sources"]["4"] = expert["calibration_source"]
    expert["metrics"]["per_class"] = {"2": {"map50": 0.79}, "4": {"map50": 0.81}}
    generation = next(item for item in payload["generations"] if item["id"] == "incremental_detection_generation")
    generation["classes"].append(4)
    generation["class_owners"]["4"] = "incremental_detector"
    generation["new_class_ids"] = [2, 4]
    path = tmp_path / "generations.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    registry = load_generation_registry(path)
    settings = generation_web_settings(registry)
    protocol = settings["protocols"]["incremental_detector"]
    assert protocol["global_class_ids"] == [2, 4]
    assert protocol["local_to_global"] == {0: 2, 1: 4}
    eligible, executed, skipped = plan_specialist_routes(
        settings["protocols"], [], {}, settings["base_class_ids"], 4, 0.7, 0.3, 0.5, 0.5
    )
    assert [row["id"] for row in eligible] == ["incremental_detector"]
    assert len(executed) == 1
    assert skipped == []


class _LifecycleStore:
    def __init__(self, root: Path, manifest: dict) -> None:
        self.root = root
        self.manifest = manifest
        self.states = []

    def _batch_dir(self, _batch_id: str) -> Path:
        return self.root

    def get(self, _batch_id: str) -> dict:
        return self.manifest

    def update_training(self, _batch_id: str, _job_id: str, status: str, **details):
        self.states.append(status)
        self.manifest.setdefault("training", {}).update(details)
        return self.manifest


def test_lifecycle_runs_calibration_recheck_shadow_and_promotion(monkeypatch, tmp_path: Path) -> None:
    job_id = "train-1"
    candidate_dir = tmp_path / "training" / job_id
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "candidate_manifest.json").write_text(json.dumps({"job_id": job_id}), encoding="utf-8")
    manifest = {
        "batch_id": "batch-1", "name": "multi", "files": [],
        "audit": {"local_to_global": {"0": 4}, "class_bindings": [{"global_class_id": 4}]},
        "injection": {"dataset_fingerprint": "a" * 64},
    }
    store = _LifecycleStore(tmp_path, manifest)
    logger = StructuredEventLog(tmp_path / "logs", 1_000_000, 2)
    calibration = {
        "calibrated": True, "per_class_thresholds": {"4": 0.7},
        "calibration_sources": {"4": "calibration.json"}, "new_map50": 0.8,
    }
    monkeypatch.setattr("fair_agent.modules.incremental_lifecycle.calibrate_candidate", lambda *_args: calibration)
    monkeypatch.setattr(
        "fair_agent.modules.incremental_lifecycle.register_trained_candidate",
        lambda *_args: {"generation_id": "g1", "parent_generation_id": "g0"},
    )
    monkeypatch.setattr("fair_agent.modules.incremental_lifecycle.recheck_generation", lambda *_args: {"accepted": True, "manifest": "recheck.json"})
    monkeypatch.setattr("fair_agent.modules.incremental_lifecycle.shadow_load_generation", lambda *_args: (object(), {"ok": True}))
    accepted = tmp_path / "accepted.json"
    accepted.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("fair_agent.modules.incremental_lifecycle.freeze_accepted_batch", lambda *_args: accepted)
    promoted = []
    lifecycle = IncrementalLifecycle(
        store,
        {"generation": {"auto_promote": True}, "incremental_workbench": {}},
        logger,
        lambda generation_id, manifest_path, _engine, _smoke: promoted.append((generation_id, manifest_path)) or {"production": generation_id},
    )
    result = lifecycle.run("batch-1", job_id)
    assert result["status"] == "PROMOTED"
    assert store.states == ["CALIBRATING", "CALIBRATED", "REGISTERED_CANDIDATE", "LOCK_RECHECKING", "ACCEPTED", "SHADOW_LOADING", "PROMOTED"]
    assert promoted == [("g1", "recheck.json")]


def test_int8_lifecycle_quantizes_before_lock_recheck(monkeypatch, tmp_path: Path) -> None:
    job_id = "train-int8"
    candidate_dir = tmp_path / "training" / job_id
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "candidate_manifest.json").write_text(
        json.dumps({"job_id": job_id}), encoding="utf-8"
    )
    store = _LifecycleStore(tmp_path, {
        "batch_id": "batch-int8", "name": "multi", "files": [],
        "audit": {"local_to_global": {"0": 6}, "class_bindings": [{"global_class_id": 6}]},
        "injection": {"dataset_fingerprint": "d" * 64},
    })
    logger = StructuredEventLog(tmp_path / "logs", 1_000_000, 2)
    monkeypatch.setattr(
        "fair_agent.modules.incremental_lifecycle.calibrate_candidate",
        lambda *_args: {
            "calibrated": True, "per_class_thresholds": {"6": 0.7},
            "calibration_sources": {"6": "calibration.json"}, "new_map50": 0.8,
        },
    )
    monkeypatch.setattr(
        "fair_agent.modules.incremental_lifecycle.register_trained_candidate",
        lambda *_args: {"generation_id": "g1", "parent_generation_id": "g0"},
    )
    calls = []
    monkeypatch.setattr(
        "fair_agent.modules.incremental_lifecycle.quantize_incremental_candidate",
        lambda *_args: calls.append("quantize") or {"deployment": {"precision": "int8"}},
    )
    monkeypatch.setattr(
        "fair_agent.modules.incremental_lifecycle.recheck_generation",
        lambda *_args: calls.append("recheck") or {"accepted": True, "manifest": "recheck.json"},
    )
    lifecycle = IncrementalLifecycle(
        store,
        {
            "generation": {"auto_promote": False},
            "incremental_workbench": {},
            "inference": {"backend": "tensorrt_engine"},
            "tensorrt_backend": {
                "precision": "int8",
                "int8_calibration": {"auto_quantize_incremental": True},
            },
        },
        logger,
    )
    result = lifecycle.run("batch-int8", job_id)
    assert result["status"] == "ACCEPTED"
    assert calls == ["quantize", "recheck"]
    assert "QUANTIZING" in store.states and "QUANTIZED" in store.states


@pytest.mark.parametrize("failure_stage", ["shadow_load", "lineage_freeze"])
def test_lifecycle_failure_rolls_back_to_parent(
    monkeypatch, tmp_path: Path, failure_stage: str
) -> None:
    job_id = "train-rollback"
    candidate_dir = tmp_path / "training" / job_id
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "candidate_manifest.json").write_text(
        json.dumps({"job_id": job_id}), encoding="utf-8"
    )
    manifest = {
        "batch_id": "batch-rollback", "name": "new-class", "files": [],
        "audit": {"local_to_global": {"0": 8}, "class_bindings": [{"global_class_id": 8}]},
        "injection": {"dataset_fingerprint": "b" * 64},
    }
    store = _LifecycleStore(tmp_path, manifest)
    logger = StructuredEventLog(tmp_path / "logs", 1_000_000, 2)
    monkeypatch.setattr(
        "fair_agent.modules.incremental_lifecycle.calibrate_candidate",
        lambda *_args: {
            "calibrated": True, "per_class_thresholds": {"8": 0.72},
            "calibration_sources": {"8": "calibration.json"}, "new_map50": 0.82,
        },
    )
    monkeypatch.setattr(
        "fair_agent.modules.incremental_lifecycle.register_trained_candidate",
        lambda *_args: {"generation_id": "g1", "parent_generation_id": "g0"},
    )
    monkeypatch.setattr(
        "fair_agent.modules.incremental_lifecycle.recheck_generation",
        lambda *_args: {"accepted": True, "manifest": "recheck.json"},
    )
    if failure_stage == "shadow_load":
        monkeypatch.setattr(
            "fair_agent.modules.incremental_lifecycle.shadow_load_generation",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("warmup failed")),
        )
    else:
        monkeypatch.setattr(
            "fair_agent.modules.incremental_lifecycle.shadow_load_generation",
            lambda *_args: (object(), {"ok": True}),
        )
        monkeypatch.setattr(
            "fair_agent.modules.incremental_lifecycle.freeze_accepted_batch",
            lambda *_args: (_ for _ in ()).throw(OSError("lineage write failed")),
        )
    promoted = []
    rolled_back = []
    lifecycle = IncrementalLifecycle(
        store,
        {"generation": {"auto_promote": True}, "incremental_workbench": {}},
        logger,
        lambda generation_id, *_args: promoted.append(generation_id) or {"production": generation_id},
        lambda target_id: rolled_back.append(target_id) or {"production": target_id},
    )
    result = lifecycle.run("batch-rollback", job_id)
    assert result["status"] == "ROLLED_BACK"
    assert result["failed_stage"] == failure_stage
    assert store.states[-1] == "ROLLED_BACK"
    assert rolled_back == ["g0"]
    assert promoted == ([] if failure_stage == "shadow_load" else ["g1"])


def test_lifecycle_reports_rollback_failure_explicitly(monkeypatch, tmp_path: Path) -> None:
    job_id = "train-rollback-failed"
    candidate_dir = tmp_path / "training" / job_id
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "candidate_manifest.json").write_text(
        json.dumps({"job_id": job_id}), encoding="utf-8"
    )
    store = _LifecycleStore(tmp_path, {
        "batch_id": "batch-rollback-failed", "name": "new-class", "files": [],
        "audit": {"local_to_global": {"0": 9}, "class_bindings": [{"global_class_id": 9}]},
        "injection": {"dataset_fingerprint": "c" * 64},
    })
    logger = StructuredEventLog(tmp_path / "logs", 1_000_000, 2)
    monkeypatch.setattr(
        "fair_agent.modules.incremental_lifecycle.calibrate_candidate",
        lambda *_args: {
            "calibrated": True, "per_class_thresholds": {"9": 0.8},
            "calibration_sources": {"9": "calibration.json"}, "new_map50": 0.85,
        },
    )
    monkeypatch.setattr(
        "fair_agent.modules.incremental_lifecycle.register_trained_candidate",
        lambda *_args: {"generation_id": "g1", "parent_generation_id": "g0"},
    )
    monkeypatch.setattr(
        "fair_agent.modules.incremental_lifecycle.recheck_generation",
        lambda *_args: {"accepted": True, "manifest": "recheck.json"},
    )
    monkeypatch.setattr(
        "fair_agent.modules.incremental_lifecycle.shadow_load_generation",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("warmup failed")),
    )
    lifecycle = IncrementalLifecycle(
        store,
        {"generation": {"auto_promote": True}, "incremental_workbench": {}},
        logger,
        rollback_callback=lambda _target: (_ for _ in ()).throw(RuntimeError("registry unavailable")),
    )
    with pytest.raises(RuntimeError, match="无法恢复父代际"):
        lifecycle.run("batch-rollback-failed", job_id)
    assert store.states[-1] == "ROLLBACK_FAILED"


def test_atomic_runtime_promotion_failure_restores_registry_and_engine(
    monkeypatch,
) -> None:
    manager = AtomicEngineProvider.__new__(AtomicEngineProvider)
    manager.config = {}
    old_engine = object()
    manager._engine = old_engine
    manager._settings = {"generation_id": "g0"}
    manager._lock = threading.RLock()
    manager._fallbacks = {}
    calls = []
    monkeypatch.setattr(
        "fair_agent.modules.generation_management.promote_generation",
        lambda *_args: calls.append("promote") or {"production": "g1"},
    )
    monkeypatch.setattr(
        "fair_agent.modules.generation_management.rollback_generation",
        lambda _config, target: calls.append(f"rollback:{target}") or {"production": target},
    )
    monkeypatch.setattr(
        "fair_agent.web.app.build_web_settings",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("settings failed")),
    )
    with pytest.raises(RuntimeError, match="settings failed"):
        manager.promote("g1", "manifest.json", object(), {"ok": True})
    assert calls == ["promote", "rollback:g0"]
    assert manager._engine is old_engine
    assert manager._settings == {"generation_id": "g0"}


def test_cli_incremental_run_waits_for_complete_lifecycle(monkeypatch) -> None:
    calls = []

    class Store:
        def get(self, batch_id, include_files=True):
            assert batch_id == "batch-1"
            return {"batch_id": batch_id, "status": "INJECTED"}

    class Manager:
        def start(self, batch_id, wait=False):
            calls.append((batch_id, wait))
            return {"job_id": "job-1", "status": "PROMOTED"}

    monkeypatch.setattr("fair_agent.cli.load_args_config", lambda _args: {})
    monkeypatch.setattr(
        "fair_agent.cli._incremental_services", lambda _config: (Store(), Manager(), object())
    )
    args = SimpleNamespace(
        incremental_action="run", batch="batch-1", name=None, class_names=None
    )
    assert cmd_incremental(args) == 0
    assert calls == [("batch-1", True)]


def test_tensorrt_engine_accepts_dynamic_batches(monkeypatch, tmp_path: Path) -> None:
    weights = tmp_path / "model.pt"
    engine = tmp_path / "model.engine"
    weights.write_bytes(b"weights")
    engine.write_bytes(b"engine")
    fake_result = SimpleNamespace(speed={"inference": 1.0})
    calls = []

    class FakeModel:
        def predict(self, source, **options):
            calls.append((source, options))
            count = len(source) if isinstance(source, list) else 1
            return [fake_result for _ in range(count)]

    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="10.8.0.43"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_capability=lambda _index: (8, 9),
    )))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda *_args, **_kwargs: FakeModel()))
    backend = TensorRTEngineBackend(weights, "0", {
        "validated": True, "expected_version": "10.8.0.43", "require_exact_gpu": True,
        "expected_compute_capability": "8.9", "dynamic": True,
        "engines": {rel_path(weights): {
            "path": str(engine), "sha256": sha256_file(engine), "imgsz": 640,
            "min_batch_size": 1, "opt_batch_size": 8, "batch_size": 20,
        }},
    })
    images = [Image.new("RGB", (32, 24)) for _ in range(8)]
    backend.predict(images[0])
    assert len(backend.predict_batch(images, imgsz=640)) == 8
    assert [options["imgsz"] for _, options in calls] == [640, 640]
    assert [options["rect"] for _, options in calls] == [True, True]
    with pytest.raises(RuntimeError, match="最大batch"):
        backend.predict_batch(images * 3, imgsz=640)
