from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from PIL import Image

from fair_agent.modules.strict_incremental import source_label


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "75_build_small_object_crops.py"))


def write_source(root: Path, stem: str, class_id: int) -> Path:
    image = root / f"{stem}.png"
    Image.new("L", (640, 512), color=0).save(image)
    image.with_suffix(".txt").write_text(
        f"{class_id} 0.5 0.5 0.02 0.02\n",
        encoding="utf-8",
    )
    return image


def test_temporal_holdout_uses_sequence_tail() -> None:
    images = [Path(f"ir_r1_base_air_{index:06d}.png") for index in range(1, 11)]
    train, val, audit = MODULE["temporal_holdout"](images, 0.20)

    assert [path.stem for path in train] == [
        f"ir_r1_base_air_{index:06d}" for index in range(1, 9)
    ]
    assert [path.stem for path in val] == [
        "ir_r1_base_air_000009",
        "ir_r1_base_air_000010",
    ]
    assert audit["ir_r1_base_air"]["train_last_frame"] == 8
    assert audit["ir_r1_base_air"]["val_first_frame"] == 9


def test_crop_labels_clips_only_visible_centered_boxes() -> None:
    labels = [
        (0, 0.50, 0.50, 0.10, 0.10),
        (2, 0.80, 0.50, 0.50, 0.20),
        (1, 0.05, 0.05, 0.02, 0.02),
    ]
    cropped = MODULE["crop_labels"](
        labels,
        (640, 512),
        (160, 128, 480, 384),
        0.50,
    )

    assert [row[0] for row in cropped] == [0]
    assert cropped[0][1:] == pytest.approx((0.5, 0.5, 0.2, 0.2))


def test_build_dataset_excludes_incremental_and_never_crops_val(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    rows = []
    for prefix, class_id in (("ir_r1_base_air", 1), ("ir_r1_base_forest", 3)):
        rows.extend(
            write_source(source_root, f"{prefix}_{index:06d}", class_id)
            for index in range(1, 6)
        )
    rows.append(write_source(source_root, "ir_r1_base_sea_000001", 2))
    split = tmp_path / "pool_train.txt"
    split.write_text("\n".join(str(path) for path in rows) + "\n", encoding="utf-8")

    output = tmp_path / "generated"
    manifest = MODULE["build_dataset"](
        split,
        output,
        (320, 256),
        "temporal",
        holdout_fraction=0.20,
        jitter_fraction=0.0,
    )

    assert manifest["source_audit"]["base_selected_count"] == 10
    assert manifest["source_audit"]["incremental_excluded_count"] == 1
    assert manifest["train_source_count"] == 8
    assert manifest["train_materialized_count"] == 16
    assert manifest["train_crop_count"] == 8
    assert manifest["val_source_count"] == 2
    assert manifest["val_crop_count"] == 0
    assert manifest["source_train_val_overlap"] == []
    assert len((output / "splits" / "train.txt").read_text().splitlines()) == 16
    assert len((output / "splits" / "val.txt").read_text().splitlines()) == 2
    assert not list((output / "images" / "val").glob("*__small_*.png"))
    for label in (output / "labels").rglob("*.txt"):
        assert {int(line.split()[0]) for line in label.read_text().splitlines()} <= {0, 1, 2}
    generated_val = Path((output / "splits" / "val.txt").read_text().splitlines()[0])
    assert source_label(generated_val).parent == output / "labels" / "val"


def test_build_dataset_can_materialize_no_crop_control(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    rows = [
        write_source(source_root, f"ir_r1_base_air_{index:06d}", 1)
        for index in range(1, 6)
    ]
    split = tmp_path / "pool_train.txt"
    split.write_text("\n".join(str(path) for path in rows) + "\n", encoding="utf-8")

    output = tmp_path / "control"
    manifest = MODULE["build_dataset"](
        split,
        output,
        (320, 256),
        "temporal",
        holdout_fraction=0.20,
        include_crops=False,
    )

    assert manifest["crop_enabled"] is False
    assert manifest["train_source_count"] == 4
    assert manifest["train_materialized_count"] == 4
    assert manifest["train_crop_count"] == 0
    assert not list((output / "images" / "train").glob("*__small_*.png"))


def test_build_dataset_can_materialize_sliding_grid_with_empty_tiles(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    rows = [
        write_source(source_root, f"ir_r1_base_air_{index:06d}", 1)
        for index in range(1, 6)
    ]
    split = tmp_path / "pool_train.txt"
    split.write_text("\n".join(str(path) for path in rows) + "\n", encoding="utf-8")

    output = tmp_path / "sliding"
    manifest = MODULE["build_dataset"](
        split,
        output,
        (384, 320),
        "temporal",
        holdout_fraction=0.20,
        crop_strategy="sliding",
        crop_overlap=0.20,
    )

    assert manifest["crop_strategy"] == "sliding"
    assert manifest["train_source_count"] == 4
    assert manifest["train_crop_count"] == 16
    assert manifest["train_materialized_count"] == 20
    assert len(list((output / "images" / "train").glob("*__grid_*.png"))) == 16


def test_build_dataset_can_weight_recent_training_frames(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    rows = [
        write_source(source_root, f"ir_r1_base_air_{index:06d}", 1)
        for index in range(1, 11)
    ]
    split = tmp_path / "pool_train.txt"
    split.write_text("\n".join(str(path) for path in rows) + "\n", encoding="utf-8")

    output = tmp_path / "recent"
    manifest = MODULE["build_dataset"](
        split,
        output,
        (320, 256),
        "temporal",
        holdout_fraction=0.20,
        include_crops=False,
        recent_fraction=0.25,
        recent_full_repeats=1,
    )

    assert manifest["train_source_count"] == 8
    assert manifest["recent_source_count"] == 2
    assert manifest["recent_source_stems"] == [
        "ir_r1_base_air_000007",
        "ir_r1_base_air_000008",
    ]
    assert manifest["train_materialized_count"] == 10
    assert len(list((output / "images" / "train").glob("*__recent_r1.png"))) == 2
