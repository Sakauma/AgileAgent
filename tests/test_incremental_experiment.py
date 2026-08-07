from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from PIL import Image

from fair_agent.modules.incremental_experiment import (
    ExperimentLedger,
    dataset_snapshot,
    load_experiment_config,
    reproduce_experiment,
    validate_experiment,
)
from fair_agent.core.runtime_log import StructuredEventLog
from fair_agent.modules.model_generations import load_generation_registry
from fair_agent.modules.strict_incremental import retention_metrics


def _image(path: Path, color: str) -> None:
    Image.new("RGB", (16, 12), color).save(path)


def _split(path: Path, images: list[Path]) -> None:
    path.write_text("\n".join(str(image) for image in images) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> Path:
    base = tmp_path / "ir_r1_base_forest_000001.png"
    new = tmp_path / "sar_r1_base_sea_000002.png"
    lock = tmp_path / "sar_r1_base_sea_000003.png"
    _image(base, "red")
    _image(new, "blue")
    _image(lock, "green")
    base.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    new.with_suffix(".txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    # A sealed lock split must not parse labels during validate/audit.
    lock.with_suffix(".txt").write_text("not a yolo label\n", encoding="utf-8")
    train = tmp_path / "train.txt"
    dev = tmp_path / "dev.txt"
    lock_split = tmp_path / "lock.txt"
    _split(train, [base, new])
    _split(dev, [])
    _split(lock_split, [lock])
    config = {
        "schema_version": 1,
        "experiment": {"id": "synthetic", "seed": 20260705, "output_root": str(tmp_path / "runs")},
        "dataset": {
            "source_splits": {"train": str(train), "dev": str(dev), "lock": str(lock_split)},
            "base_test_split": str(lock_split),
            "class_map": {0: "old", 2: "new"},
        },
        "audit": {"cache_roots": []},
        "partition": {
            "base_class_ids": [0],
            "cooccurrence_policy": "reject",
            "rounds": [{
                "id": "round_01",
                "new_class_ids": [2],
                "train_selector": "contains_only_new_classes",
                "dev_selector": "contains_only_new_classes",
            }],
        },
        "training": {"adapter_config": "configs/strict_class_incremental_3plus1.yaml"},
        "acceptance": {
            "min_base_map50": 0.8,
            "min_new_map50": 0.6,
            "min_krr": 0.95,
        },
        "diagnostics": {
            "calibration_target_precision": 0.99,
            "min_lock_precision": 0.7,
            "max_false_activation_rate": 0.15,
        },
        "integrity": {
            "max_base_weight_drift": 0.0,
        },
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_snapshot_hash_is_stable_and_lock_remains_sealed(tmp_path: Path) -> None:
    config = load_experiment_config(_fixture(tmp_path))
    first = dataset_snapshot(config)
    second = dataset_snapshot(config)
    assert first["snapshot_sha256"] == second["snapshot_sha256"]
    assert first["lock"]["sealed"] is True
    assert first["lock"]["files"] == []
    assert first["old_raw_image_count"] == 0
    assert first["old_raw_label_count"] == 0


def test_content_hash_detects_renamed_old_image(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    config = load_experiment_config(config_path)
    train = Path(config["dataset"]["source_splits"]["train"])
    rows = [Path(line) for line in train.read_text(encoding="utf-8").splitlines() if line]
    shutil.copyfile(rows[0], rows[1])
    snapshot = dataset_snapshot(config)
    assert snapshot["old_raw_stems"] == []
    assert len(snapshot["old_raw_content_hashes"]) == 1
    assert snapshot["old_raw_image_count"] == 1
    assert validate_experiment(config_path)["valid"] is False


def test_cooccurring_old_and_new_class_is_rejected(tmp_path: Path) -> None:
    config = load_experiment_config(_fixture(tmp_path))
    new_image = Path(config["dataset"]["source_splits"]["train"])
    rows = [Path(line) for line in new_image.read_text(encoding="utf-8").splitlines() if line]
    rows[1].with_suffix(".txt").write_text(
        "0 0.5 0.5 0.2 0.2\n2 0.4 0.4 0.1 0.1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="共现"):
        dataset_snapshot(config)


def test_nonempty_declared_feature_cache_blocks_validation(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache = tmp_path / "feature_cache"
    cache.mkdir()
    (cache / "old_features.npy").write_bytes(b"cached-old-features")
    payload["audit"]["cache_roots"] = [str(cache)]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    result = validate_experiment(config_path)
    assert result["valid"] is False
    assert result["old_feature_cache_count"] == 1


def test_reproduce_rejects_changed_source_data_before_training(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    config = load_experiment_config(config_path)
    snapshot = dataset_snapshot(config)
    manifest_path = tmp_path / "parent_manifest.json"
    manifest_path.write_text(json.dumps({
        "config_snapshot": str(config_path),
        "dataset_snapshot_sha256": snapshot["snapshot_sha256"],
    }), encoding="utf-8")
    base_image = Path(config["dataset"]["source_splits"]["train"])
    base_path = Path(base_image.read_text(encoding="utf-8").splitlines()[0])
    _image(base_path, "black")
    with pytest.raises(ValueError, match="指纹"):
        reproduce_experiment(manifest_path, run_id="must-not-start")
    assert not (tmp_path / "runs" / "must-not-start").exists()


def test_generation_zero_excludes_warship_and_benchmark_cannot_own_production(tmp_path: Path) -> None:
    registry = load_generation_registry("models/generations.json")
    generation_zero = registry["generations_by_id"]["base_detection_generation"]
    assert set(generation_zero["classes"]) == {0, 1, 3}
    assert 2 not in generation_zero["class_owners"]
    production = registry["generations_by_id"][registry["channels"]["production"]]
    assert set(production["classes"]) == {0, 1, 2, 3}
    assert production["class_owners"][2] == "incremental_detector"

    payload = json.loads(Path("models/generations.json").read_text(encoding="utf-8"))
    broken = deepcopy(payload)
    generation = next(item for item in broken["generations"] if item["id"] == "base_detection_generation")
    generation["classes"].append(2)
    generation["class_owners"]["2"] = "four_class_unified_benchmark"
    broken_path = tmp_path / "generations.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark_only"):
        load_generation_registry(broken_path)

    premature = deepcopy(payload)
    expert = next(item for item in premature["models"] if item["id"] == "incremental_detector")
    expert["acceptance"]["passed"] = False
    premature_path = tmp_path / "premature.json"
    premature_path.write_text(json.dumps(premature), encoding="utf-8")
    with pytest.raises(ValueError, match="未通过部署门禁"):
        load_generation_registry(premature_path)


def test_krr_is_recomputed_from_actual_combined_predictions() -> None:
    target = {"image_id": "a", "class_id": 0, "xyxy": [0, 0, 10, 10]}
    prediction = {**target, "confidence": 0.9}
    retained = retention_metrics([prediction], [], [target], [0])
    assert retained["old_map50_before"] > 0.99
    assert retained["old_map50_after"] == 0.0
    assert retained["krr"] == 0.0
    assert retained["old_prediction_equivalent"] is False


def test_experiment_ledger_mirrors_state_to_global_agent_log(tmp_path: Path) -> None:
    event_log = StructuredEventLog(tmp_path / "global-logs", 1024 * 1024, 2)
    ledger = ExperimentLedger(
        tmp_path / "runs", "run-01", experiment_id="synthetic", event_log=event_log
    )
    ledger.event(
        "LOCK_UNSEALED", protocol_id="round_01", split_sha256="b" * 64
    )
    local = [json.loads(line) for line in ledger.events_path.read_text(encoding="utf-8").splitlines()]
    global_rows = event_log.query(
        experiment_id="synthetic", run_id="run-01", protocol_id="round_01"
    )
    assert local[0]["state"] == "LOCK_UNSEALED"
    assert global_rows[0]["event"] == "incremental.lock.unsealed"
    assert global_rows[0]["details"]["split_sha256"] == "b" * 64
