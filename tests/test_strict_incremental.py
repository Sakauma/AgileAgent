from __future__ import annotations

import hashlib
import json
import copy
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from PIL import Image

import fair_agent.modules.strict_incremental as strict
from fair_agent.modules.strict_incremental import (
    bootstrap_metrics,
    build_protocol_dataset,
    calibrate_threshold,
    class_aware_nms,
    evaluate_ap50,
    load_experiment_profile,
    materialize_lock_data,
    load_yaml,
)


def make_image(root: Path, stem: str, labels: list[int]) -> Path:
    image = root / f"{stem}.png"
    color = tuple(hashlib.sha256(stem.encode("utf-8")).digest()[:3])
    Image.new("RGB", (100, 100), color).save(image)
    image.with_suffix(".txt").write_text(
        "".join(f"{class_id} 0.5 0.5 0.2 0.2\n" for class_id in labels),
        encoding="utf-8",
    )
    return image


def write_split(path: Path, images: list[Path]) -> None:
    path.write_text("\n".join(str(image) for image in images) + "\n", encoding="utf-8")


def protocol() -> dict:
    return {
        "id": "strict-p01",
        "base_classes": ["soldier", "warship", "tank"],
        "new_class": "small_aircraft",
        "new_global_id": 1,
        "base_local_to_global": {0: 0, 1: 2, 2: 3},
    }


def test_repository_strict_config_has_only_disjoint_aircraft_and_warship_folds() -> None:
    config = load_yaml("configs/strict_class_incremental_3plus1.yaml")
    assert [item["id"] for item in config["protocols"]] == ["strict-p01", "warship-incremental"]
    assert [item["new_class"] for item in config["protocols"]] == ["small_aircraft", "warship"]
    assert config["acceptance"]["min_new_map50"] == 0.60
    assert config["acceptance"]["min_krr"] == 0.95
    assert config["bootstrap"]["iterations"] == 1000
    assert config["predict"]["evaluation_batch"] == 1
    assert config["predict"]["rect"] is True
    runner = Path("tools/70_run_strict_3plus1.py").read_text(encoding="utf-8")
    assert 'mp_context=get_context("spawn")' in runner
    assert 'getattr(result, "path", "")' in runner
    assert '"image_id": image_id' in runner
    assert "multi_label=True" in runner
    smoke = Path("scripts/smoke_models.py").read_text(encoding="utf-8")
    assert 'base_local_to_global=settings.get("base_local_to_global")' in smoke
    assert 'generation_id=settings["generation_id"]' in smoke


def test_strict_dataset_is_disjoint_and_lock_is_deferred(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    old_train = make_image(source, "ir_old_train", [0, 2, 3])
    new_train = make_image(source, "ir_new_train", [1])
    old_val = make_image(source, "ir_old_val", [0])
    new_val = make_image(source, "ir_new_val", [1])
    lock_old = make_image(source, "sar_lock_old", [2])
    lock_new = make_image(source, "ir_lock_new", [1])
    splits = {}
    for name, images in {
        "train": [old_train, new_train],
        "val": [old_val, new_val],
        "lock": [lock_old, lock_new],
    }.items():
        split = tmp_path / f"{name}.txt"
        write_split(split, images)
        splits[name] = split

    output = tmp_path / "strict"
    original_label = old_train.with_suffix(".txt").read_text(encoding="utf-8")
    manifest = build_protocol_dataset(protocol(), splits, output)
    assert manifest["counts"]["base"] == {"train": 1, "val": 1, "test": 0}
    assert manifest["counts"]["incremental"] == {"train": 1, "val": 1, "test": 0}
    assert manifest["lock_materialized_after_freeze"] is False
    assert manifest["source_split_intersections"] == {
        "train_val": [], "train_lock": [], "val_lock": []
    }
    assert not (output / "base" / "test" / "labels").exists()
    assert old_train.with_suffix(".txt").read_text(encoding="utf-8") == original_label
    base_label = (output / "base" / "train" / "labels" / "ir_old_train.txt").read_text(encoding="utf-8")
    assert [line.split()[0] for line in base_label.splitlines()] == ["0", "1", "2"]
    incremental_label = (output / "incremental" / "train" / "labels" / "ir_new_train.txt").read_text(encoding="utf-8")
    assert incremental_label.startswith("0 ")
    base_split_text = (output / "base" / "splits" / "train.txt").read_text(encoding="utf-8")
    assert str(output.absolute()) in base_split_text
    assert str(source.absolute()) not in base_split_text

    frozen_manifest = materialize_lock_data(protocol(), splits["lock"], output)
    assert frozen_manifest["lock_materialized_after_freeze"] is True
    assert frozen_manifest["counts"]["base"]["test"] == 2
    assert frozen_manifest["counts"]["incremental"]["test"] == 2


def test_strict_dataset_rejects_duplicate_stems_across_source_splits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    shared = make_image(source, "ir_shared", [0])
    new = make_image(source, "ir_new", [1])
    lock = make_image(source, "sar_lock", [2])
    splits = {}
    for name, images in {"train": [shared, new], "val": [shared], "lock": [lock]}.items():
        split = tmp_path / f"{name}.txt"
        write_split(split, images)
        splits[name] = split
    with pytest.raises(RuntimeError, match="重复 stem"):
        build_protocol_dataset(protocol(), splits, tmp_path / "strict")


def test_strict_dataset_rejects_new_old_class_cooccurrence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    mixed = make_image(source, "ir_mixed", [0, 1])
    old = make_image(source, "ir_old", [2])
    lock = make_image(source, "sar_lock", [1])
    splits = {}
    for name, images in {"train": [mixed], "val": [old], "lock": [lock]}.items():
        split = tmp_path / f"{name}.txt"
        write_split(split, images)
        splits[name] = split
    with pytest.raises(ValueError, match="旧类共现"):
        build_protocol_dataset(protocol(), splits, tmp_path / "strict")


def test_strict_dataset_rejects_renamed_old_image_content(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    old = make_image(source, "ir_old", [0])
    renamed_old = make_image(source, "ir_new_name", [1])
    renamed_old.write_bytes(old.read_bytes())
    lock = make_image(source, "sar_lock", [1])
    splits = {}
    for name, images in {
        "train": [old, renamed_old],
        "val": [],
        "lock": [lock],
    }.items():
        split = tmp_path / f"{name}.txt"
        write_split(split, images)
        splits[name] = split
    with pytest.raises(RuntimeError, match="包含旧图"):
        build_protocol_dataset(protocol(), splits, tmp_path / "strict")


def test_unified_student_dataset_keeps_global_new_class_id(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    old_train = make_image(source, "ir_old_train", [0, 2, 3])
    new_train = make_image(source, "ir_new_train", [1])
    old_val = make_image(source, "ir_old_val", [2])
    new_val = make_image(source, "ir_new_val", [1])
    lock = make_image(source, "sar_lock", [0, 1, 2, 3])
    splits = {}
    for name, images in {
        "train": [old_train, new_train], "val": [old_val, new_val], "lock": [lock]
    }.items():
        split = tmp_path / f"{name}.txt"
        write_split(split, images)
        splits[name] = split
    spec = protocol()
    spec["build_unified_student"] = True
    output = tmp_path / "strict"
    manifest = build_protocol_dataset(spec, splits, output)
    assert manifest["student_nc"] == 4
    label = output / "student" / "train" / "labels" / "ir_new_train.txt"
    assert label.read_text(encoding="utf-8").startswith("1 ")
    materialize_lock_data(spec, splits["lock"], output)
    lock_label = output / "student" / "test" / "labels" / "sar_lock.txt"
    assert [int(line.split()[0]) for line in lock_label.read_text().splitlines()] == [0, 1, 2, 3]


def test_ap_calibration_nms_and_bootstrap_are_deterministic() -> None:
    ground_truth = [
        {"image_id": "ir_positive", "class_id": 1, "xyxy": [10, 10, 30, 30]},
        {"image_id": "sar_old", "class_id": 0, "xyxy": [40, 40, 60, 60]},
    ]
    predictions = [
        {"image_id": "ir_positive", "class_id": 1, "confidence": 0.95, "xyxy": [10, 10, 30, 30]},
        {"image_id": "ir_positive", "class_id": 1, "confidence": 0.80, "xyxy": [11, 11, 29, 29]},
        {"image_id": "sar_old", "class_id": 0, "confidence": 0.90, "xyxy": [40, 40, 60, 60]},
    ]
    fused = class_aware_nms(predictions, 0.60)
    assert len(fused) == 2
    metrics = evaluate_ap50(fused, ground_truth, [0, 1])
    assert metrics["map50"] > 0.99
    calibration = calibrate_threshold(predictions, ground_truth, 1, target_precision=0.90)
    assert calibration["passed"] is True
    assert calibration["selected"]["precision"] == 1.0
    images = [Path("ir_positive.png"), Path("sar_old.png")]
    first = bootstrap_metrics(fused, ground_truth, images, 1, iterations=10, seed=7)
    second = bootstrap_metrics(fused, ground_truth, images, 1, iterations=10, seed=7)
    assert first == second


def test_ap50_matches_ultralytics_partial_recall_sentinel() -> None:
    ground_truth = [
        {"image_id": "one", "class_id": 1, "xyxy": [0, 0, 10, 10]},
        {"image_id": "two", "class_id": 1, "xyxy": [0, 0, 10, 10]},
    ]
    predictions = [
        {"image_id": "one", "class_id": 1, "confidence": 0.9, "xyxy": [0, 0, 10, 10]},
    ]
    result = evaluate_ap50(predictions, ground_truth, [1])
    assert 0.49 <= result["map50"] <= 0.51


def test_ap50_matches_targets_in_confidence_order() -> None:
    ground_truth = [{"image_id": "one", "class_id": 1, "xyxy": [0, 0, 10, 10]}]
    predictions = [
        {"image_id": "one", "class_id": 1, "confidence": 0.9, "xyxy": [0, 0, 8, 10]},
        {"image_id": "one", "class_id": 1, "confidence": 0.1, "xyxy": [0, 0, 10, 10]},
    ]
    result = evaluate_ap50(predictions, ground_truth, [1])
    assert result["map50"] > 0.99


def test_predict_records_uses_result_path_when_ultralytics_reorders_batch(tmp_path: Path) -> None:
    script = runpy.run_path("tools/70_run_strict_3plus1.py")

    class TensorValue:
        def __init__(self, values) -> None:
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class Boxes:
        def __init__(self, class_id: int) -> None:
            self.xyxy = TensorValue([[0.0, 0.0, 10.0, 10.0]])
            self.conf = TensorValue([0.9])
            self.cls = TensorValue([class_id])

        def __len__(self) -> int:
            return 1

    class Result:
        def __init__(self, path: str, class_id: int) -> None:
            self.path = path
            self.boxes = Boxes(class_id)
            self.speed = {"inference": 3.0}

    class Model:
        def predict(self, **_kwargs):
            return [Result(str(tmp_path / "second.png"), 1), Result(str(tmp_path / "first.png"), 0)]

    (tmp_path / "first.png").write_bytes(b"")
    (tmp_path / "second.png").write_bytes(b"")

    config = {
        "common": {"imgsz": 640},
        "predict": {"conf": 0.001, "iou": 0.7, "max_det": 300, "batch": 32},
    }
    rows, inference_ms = script["predict_records"](
        Model(),
        [tmp_path / "first.png", tmp_path / "second.png"],
        {0: 0, 1: 2},
        config,
        "1",
        "student",
    )
    assert [(row["image_id"], row["class_id"]) for row in rows] == [
        ("second", 2),
        ("first", 0),
    ]
    assert inference_ms == 6.0


def test_predict_records_rejects_mismatched_result_paths(tmp_path: Path) -> None:
    script = runpy.run_path("tools/70_run_strict_3plus1.py")

    class Result:
        path = str(tmp_path / "unexpected.png")
        boxes = None
        speed = {"inference": 0.0}

    class Model:
        def predict(self, **_kwargs):
            return [Result()]

    (tmp_path / "expected.png").write_bytes(b"")

    config = {
        "common": {"imgsz": 640},
        "predict": {"conf": 0.001, "iou": 0.7, "max_det": 300, "batch": 32},
    }
    try:
        script["predict_records"](
            Model(), [tmp_path / "expected.png"], {0: 0}, config, "1", "student"
        )
    except RuntimeError as exc:
        assert "预测图像标识不一致" in str(exc)
    else:
        raise AssertionError("错误的 result.path 未被拒绝")


def test_expanded_student_initializes_ema_and_freezes_batchnorm() -> None:
    script = runpy.run_path("tools/70_run_strict_3plus1.py")

    class Head(nn.Module):
        def __init__(self, class_count: int) -> None:
            super().__init__()
            self.cv3 = nn.ModuleList(
                [nn.Sequential(nn.Identity(), nn.Identity(), nn.Conv2d(2, class_count, 1))]
            )

    class Detector(nn.Module):
        def __init__(self, class_count: int) -> None:
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
            self.model = nn.ModuleList([Head(class_count)])

    teacher = Detector(3)
    student = Detector(4)
    trainer = SimpleNamespace(
        model=student,
        ema=SimpleNamespace(ema=copy.deepcopy(student), updates=17),
    )
    script["configure_expanded_student"](
        trainer,
        Path("unused.pt"),
        {0: 0, 1: 1, 2: 3},
        2,
        teacher_model=teacher,
    )
    for name, value in student.state_dict().items():
        assert torch.equal(value, trainer.ema.ema.state_dict()[name])
    assert trainer.ema.updates == 0
    student.train()
    script["restore_expanded_student"](trainer)
    assert student.stem[1].training is False


def test_training_history_records_epochs_best_epoch_and_early_stop(tmp_path: Path) -> None:
    script = runpy.run_path("tools/70_run_strict_3plus1.py")
    (tmp_path / "results.csv").write_text(
        "epoch,metrics/mAP50(B),train/box_loss\n0,0.40,1.2\n1,0.75,0.8\n",
        encoding="utf-8",
    )
    model = SimpleNamespace(trainer=SimpleNamespace(save_dir=tmp_path))
    history = script["training_history"](model, "base", 5)
    assert history["completed_epochs"] == 2
    assert history["best_epoch"] == 1
    assert history["best_metric_value"] == 0.75
    assert history["stopped_early"] is True
    assert len(history["epochs"]) == 2


def test_context_class_weights_allow_a_missing_incremental_scene() -> None:
    script = runpy.run_path("tools/60_train_scene_sensor.py")
    images = [
        Path("ir_r1_base_air_000001.png"),
        Path("ir_r1_base_forest_000002.png"),
        Path("sar_r1_base_urban_000003.png"),
    ]
    weights = script["class_weights"](images, 1, 4, torch.device("cpu"))
    assert torch.isfinite(weights).all()
    assert float(weights[2]) == 0.0


def test_experiment_profile_requires_passed_hash_verified_assets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(strict, "ROOT", tmp_path)
    profile_root = tmp_path / "models" / "profiles" / "strict-p01"
    profile_root.mkdir(parents=True)
    base = tmp_path / "base.pt"
    specialist = tmp_path / "specialist.pt"
    base.write_bytes(b"base")
    specialist.write_bytes(b"specialist")
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"selected":{"threshold":0.5}}', encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({
        "accepted": True,
        "incremental_mode": "class_incremental",
        "learning_data_scope": "incremental_dataset_only",
        "old_raw_image_count": 0,
        "gates": {"data": True},
        "lock_deployment_metrics": {"precision": 0.9, "recall": 0.8},
        "false_activation": {"false_activation_rate": 0.0},
    }), encoding="utf-8")
    payload = {
        "profile_id": "strict-p01",
        "acceptance": "passed",
        "incremental_mode": "class_incremental",
        "evidence_level": "verified",
        "activation_threshold": 0.5,
        "new_global_id": 1,
        "base_local_to_global": {"0": 0, "1": 2, "2": 3},
        "calibration_source": str(calibration),
        "metrics_source": str(metrics),
        "base_weight": str(base),
        "base_sha256": hashlib.sha256(b"base").hexdigest(),
        "specialist_weight": str(specialist),
        "specialist_sha256": hashlib.sha256(b"specialist").hexdigest(),
    }
    (profile_root / "active.json").write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_experiment_profile("strict-p01")
    assert loaded["acceptance"] == "passed"
    assert loaded["deployment_accepted"] is True
    discovered = strict.discover_experiment_profiles(profile_root.parent)
    assert discovered["true_class_incremental_verified"] is True
    assert discovered["verified_count"] == 1
    assert discovered["core_verified_count"] == 1
    specialist.write_bytes(b"changed")
    try:
        load_experiment_profile("strict-p01")
    except ValueError as exc:
        assert "权重校验失败" in str(exc)
    else:
        raise AssertionError("篡改后的实验档被接受")
