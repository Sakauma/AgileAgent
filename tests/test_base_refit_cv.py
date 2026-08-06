from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "78_build_base_refit_cv.py"))


def write_image(root: Path, sequence: str, frame: int, class_id: int) -> Path:
    image = root / f"{sequence}_{frame:06d}.png"
    image.write_bytes(b"image")
    image.with_suffix(".txt").write_text(
        f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    return image


def write_split(path: Path, images: list[Path]) -> None:
    path.write_text("\n".join(str(image) for image in images) + "\n", encoding="utf-8")


def test_refit_folds_cover_each_non_test_image_once(tmp_path: Path) -> None:
    images = []
    for sequence, class_id in (("ir_r1_base_air", 1), ("sar_r1_base_forest", 3)):
        images.extend(write_image(tmp_path, sequence, frame, class_id) for frame in range(1, 11))
    train = [image for image in images if int(image.stem.rsplit("_", 1)[1]) <= 8]
    dev = [image for image in images if int(image.stem.rsplit("_", 1)[1]) > 8]
    train_split = tmp_path / "base_train.txt"
    dev_split = tmp_path / "base_dev.txt"
    write_split(train_split, train)
    write_split(dev_split, dev)

    output = tmp_path / "refit_cv"
    manifest = MODULE["build_refit_folds"](train_split, dev_split, output, 5)

    assert manifest["lock_data_access"] is False
    assert manifest["combined_non_test_count"] == 20
    assert manifest["validation_coverage_count"] == 20
    assert len(manifest["folds"]) == 5
    validation_stems = []
    for fold in manifest["folds"]:
        train_rows = Path(fold["train_split"]).read_text(encoding="utf-8").splitlines()
        val_rows = Path(fold["val_split"]).read_text(encoding="utf-8").splitlines()
        assert len(train_rows) == 16
        assert len(val_rows) == 4
        assert {Path(row).stem for row in train_rows}.isdisjoint(
            {Path(row).stem for row in val_rows}
        )
        validation_stems.extend(Path(row).stem for row in val_rows)
    assert len(validation_stems) == len(set(validation_stems)) == 20
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_refit_folds_reject_incremental_class(tmp_path: Path) -> None:
    images = [write_image(tmp_path, "ir_r1_base_forest", frame, 0) for frame in range(1, 7)]
    images[-1].with_suffix(".txt").write_text(
        "2 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    train_split = tmp_path / "base_train.txt"
    dev_split = tmp_path / "base_dev.txt"
    write_split(train_split, images[:4])
    write_split(dev_split, images[4:])

    with pytest.raises(ValueError, match="非基础类别"):
        MODULE["build_refit_folds"](
            train_split, dev_split, tmp_path / "refit_cv", 2
        )


def test_sparse_refit_keeps_sequence_endpoints_for_training(tmp_path: Path) -> None:
    images = []
    for sequence, class_id in (("ir_r1_base_air", 1), ("sar_r1_base_forest", 3)):
        images.extend(write_image(tmp_path, sequence, frame, class_id) for frame in range(1, 21))
    train_split = tmp_path / "base_train.txt"
    dev_split = tmp_path / "base_dev.txt"
    write_split(train_split, [image for image in images if int(image.stem.rsplit("_", 1)[1]) <= 15])
    write_split(dev_split, [image for image in images if int(image.stem.rsplit("_", 1)[1]) > 15])

    output = tmp_path / "sparse_refit"
    manifest = MODULE["build_sparse_refit"](
        train_split, dev_split, output, 5, 2
    )

    assert manifest["performance_evidence"] is False
    assert manifest["train_count"] == 32
    assert manifest["val_count"] == 8
    train_stems = {
        Path(row).stem
        for row in Path(manifest["train_split"]).read_text(encoding="utf-8").splitlines()
    }
    for sequence in ("ir_r1_base_air", "sar_r1_base_forest"):
        assert f"{sequence}_000001" in train_stems
        assert f"{sequence}_000020" in train_stems


@pytest.mark.parametrize(
    "config_name,expected_candidates",
    [
        ("base_refit_s896_cv.yaml", {f"fold_{index}" for index in range(5)}),
        ("base_final_refit_s896.yaml", {"seed_a", "seed_b"}),
    ],
)
def test_s896_candidates_keep_full_training_budget(
    config_name: str, expected_candidates: set[str]
) -> None:
    config = yaml.safe_load((ROOT / "configs" / config_name).read_text(encoding="utf-8"))

    assert config["model"] == "yolo11s.pt"
    assert config["epochs"] == 160
    assert set(config["candidates"]) == expected_candidates
    for candidate in config["candidates"].values():
        assert candidate["overrides"]["imgsz"] == 896
        assert candidate["overrides"]["multi_scale"] == 0.0
