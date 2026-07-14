from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
from PIL import Image

from fair_agent.modules.full_yolo_iod import (
    GLOBAL_TO_OFFICIAL,
    OFFICIAL_ALL_CLASSES,
    _coco_document,
    _ensure_runtime_directories,
    _inject_gradient_accumulation,
    _rewrite_base_dependency,
    cpr_is_enabled,
    count_cpr_pseudo_labels,
    load_full_yolo_iod_config,
    summarize_disabled_cpr,
    training_batch_plan,
    validate_full_yolo_iod_config,
)


def _image(tmp_path: Path, name: str, labels: str) -> Path:
    path = tmp_path / f"{name}.png"
    Image.new("RGB", (100, 80)).save(path)
    path.with_suffix(".txt").write_text(labels, encoding="utf-8")
    return path


def test_coco_conversion_uses_old_first_official_order(tmp_path: Path) -> None:
    image = _image(
        tmp_path,
        "sample",
        "0 0.5 0.5 0.2 0.2\n2 0.4 0.4 0.1 0.1\n3 0.6 0.6 0.1 0.1\n",
    )
    document = _coco_document([image], {0, 2, 3})
    assert [row["name"] for row in document["categories"]] == list(OFFICIAL_ALL_CLASSES)
    assert [row["category_id"] for row in document["annotations"]] == [1, 4, 3]
    assert document["annotations"][0]["segmentation"][0] == pytest.approx(
        [40.0, 32.0, 60.0, 32.0, 60.0, 48.0, 40.0, 48.0]
    )
    assert GLOBAL_TO_OFFICIAL == {0: 0, 1: 1, 3: 2, 2: 3}


def test_config_rejects_non_official_mapping() -> None:
    config = {
        "experiment": {
            "protocol": "strict-p02",
            "official_commit": "3d9d05a6e88561c88916657367412f9adb7341a7",
        },
        "classes": {
            "old": ["soldier", "small aircraft", "tank"],
            "new": "warship",
            "global_to_official": {0: 0, 1: 1, 2: 2, 3: 3},
        },
        "runtime": {"devices": ["1", "2", "3"]},
    }
    with pytest.raises(ValueError, match="映射错误"):
        validate_full_yolo_iod_config(config)


def test_cpr_count_separates_added_old_labels(tmp_path: Path) -> None:
    source = {
        "annotations": [{"id": 1, "category_id": 4}],
    }
    refined = {
        "annotations": [
            {"id": 1, "category_id": 4},
            {"id": 2, "category_id": 1},
            {"id": 3, "category_id": 4},
        ]
    }
    source_path = tmp_path / "source.json"
    refined_path = tmp_path / "refined.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    refined_path.write_text(json.dumps(refined), encoding="utf-8")
    result = count_cpr_pseudo_labels(source_path, refined_path)
    assert result == {
        "source_annotations": 1,
        "refined_annotations": 3,
        "pseudo_annotations": 2,
        "old_class_pseudo_annotations": 1,
    }


def test_generated_config_uses_stable_base_dependency() -> None:
    template = (
        "_base_ = (\n"
        "    '../../third_party/mmyolo/configs/yolov8/'\n"
        "    'yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py')\n"
    )
    dependency = "/opt/YOLO-IOD/third_party/mmyolo/configs/yolov8/base.py"
    result = _rewrite_base_dependency(template, dependency)
    assert dependency in result
    assert "../../third_party" not in result


def test_prepare_creates_gps_importance_directory(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _ensure_runtime_directories(data_root)
    assert (data_root / "importance").is_dir()


def test_resume_command_passes_explicit_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch_2.pth"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "last_checkpoint").write_text(str(checkpoint), encoding="utf-8")
    runner = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/72_run_full_yolo_iod.py"))
    command = runner["_resume_command"](["python", "train.py"], tmp_path)
    assert command == ["python", "train.py", "--resume", str(checkpoint)]


def test_official_patch_restores_accumulation_count_after_gps_scan() -> None:
    root = Path(__file__).resolve().parents[1]
    patch = (root / "patches/yolo_iod_single_class_grad_mask.patch").read_text(
        encoding="utf-8"
    )
    assert "train_iter = self._train_loop.iter" in patch
    assert "self._train_loop._iter = train_iter" in patch
    assert "self.optim_wrapper.initialize_count_status(" in patch


def test_r04_disables_cpr_and_reaches_effective_batch_16() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_full_yolo_iod_config(root / "configs/full_yolo_iod_p02_r04.yaml")
    assert cpr_is_enabled(config) is False
    assert config["dataset"]["incremental_validation_only"] is True
    plan = training_batch_plan(config)
    assert {row["effective_batch_size"] for row in plan.values()} == {16}
    assert plan["final"]["micro_batch_per_gpu"] == 2
    assert plan["final"]["gradient_accumulation_steps"] == 8


def test_r05_uses_gpu3_without_changing_training_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_full_yolo_iod_config(root / "configs/full_yolo_iod_p02_r05_gpu3.yaml")
    assert config["runtime"]["devices"] == ["3"]
    assert config["training"]["final"]["devices"] == ["3"]
    assert config["evaluation"]["device"] == "3"
    assert cpr_is_enabled(config) is False
    assert {row["effective_batch_size"] for row in training_batch_plan(config).values()} == {16}


def test_schema2_rejects_multiple_or_mismatched_devices() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_full_yolo_iod_config(root / "configs/full_yolo_iod_p02_r05_gpu3.yaml")
    config["runtime"]["devices"] = ["2", "3"]
    with pytest.raises(ValueError, match="一张有效 GPU"):
        validate_full_yolo_iod_config(config)

    config["runtime"]["devices"] = ["3"]
    config["training"]["final"]["devices"] = ["2"]
    with pytest.raises(ValueError, match="最终训练设备"):
        validate_full_yolo_iod_config(config)


def test_gradient_accumulation_is_written_into_official_config() -> None:
    source = "optim_wrapper = dict(\n    constructor='YOLOWv5OptimizerConstructor')\n"
    result = _inject_gradient_accumulation(source, 8)
    assert "accumulative_counts=8" in result
    assert result.count("YOLOWv5OptimizerConstructor") == 1


def test_disabled_cpr_summary_contains_no_pseudo_labels(tmp_path: Path) -> None:
    source = tmp_path / "current_train.json"
    source.write_text(json.dumps({"annotations": [{"id": 1}, {"id": 2}]}), encoding="utf-8")
    result = summarize_disabled_cpr(source, "disabled_no_old_class_cooccurrence")
    assert result["status"] == "skipped_by_data_gate"
    assert result["source_annotations"] == 2
    assert result["pseudo_annotations"] == 0
    assert result["old_class_pseudo_annotations"] == 0
