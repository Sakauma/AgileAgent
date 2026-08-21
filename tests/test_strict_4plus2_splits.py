from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = ROOT / "splits" / "strict_4plus2"


def read_split(name: str) -> list[str]:
    return [
        line.strip()
        for line in (SPLIT_ROOT / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sequence(path: str) -> str:
    parts = Path(path).stem.split("_")
    return f"{parts[0]}|{parts[3]}"


def frame(path: str) -> int:
    return int(Path(path).stem.split("_")[-1])


def test_active_pointer_and_all_list_counts_are_frozen() -> None:
    active = json.loads((ROOT / "splits" / "active.json").read_text(encoding="utf-8"))
    assert active["active_protocol"] == "strict_4plus2"
    assert active["manifest"] == "splits/strict_4plus2/manifest.json"

    manifest = json.loads((SPLIT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "competition_score_priority_strict_4plus2_partition"
    assert manifest["training_started"] is False
    for item in manifest["lists"].values():
        path = ROOT / item["path"]
        assert path.is_file()
        assert len([line for line in path.read_text(encoding="utf-8").splitlines() if line]) == item["count"]


def test_base_and_increment_splits_are_complete_and_disjoint() -> None:
    expected = {
        "base": {"train": 600, "dev": 75, "lock": 75, "all": 750},
        "increment": {"train": 112, "dev": 14, "lock": 14, "all": 140},
    }
    for prefix in ("base", "increment"):
        split = {name: set(read_split(f"{prefix}_{name}")) for name in ("train", "dev", "lock")}
        for name in split:
            assert len(split[name]) == expected[prefix][name]
        assert not split["train"] & split["dev"]
        assert not split["train"] & split["lock"]
        assert not split["dev"] & split["lock"]
        assert split["train"] | split["dev"] | split["lock"] == set(read_split(f"{prefix}_all"))
        assert len(read_split(f"{prefix}_all")) == expected[prefix]["all"]
        assert set(read_split(f"{prefix}_train_plus_dev")) == split["train"] | split["dev"]


def test_every_sensor_scene_sequence_appears_in_each_split() -> None:
    expected_base = {
        "ir|air",
        "ir|forest",
        "ir|sea",
        "ir|urban",
        "sar|forest",
        "sar|sea",
        "sar|urban",
    }
    expected_increment = {
        "ir|forest",
        "ir|sea",
        "ir|urban",
        "sar|forest",
        "sar|sea",
        "sar|urban",
    }
    for name in ("train", "dev", "lock"):
        assert {sequence(path) for path in read_split(f"base_{name}")} == expected_base
        assert {sequence(path) for path in read_split(f"increment_{name}")} == expected_increment


def test_score_priority_split_intentionally_has_adjacent_train_frames() -> None:
    manifest = json.loads((SPLIT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["allocation_policy"]["temporal_isolation"] is False
    assert manifest["allocation_policy"]["adjacent_frame_leakage_intentional"] is True

    expected_minimum_ratio = {"base": 0.98, "increment": 1.0}
    for prefix in ("base", "increment"):
        train_frames: dict[str, set[int]] = {}
        for path in read_split(f"{prefix}_train"):
            train_frames.setdefault(sequence(path), set()).add(frame(path))
        for split_name in ("dev", "lock"):
            values = read_split(f"{prefix}_{split_name}")
            adjacent = sum(
                min(abs(frame(path) - train_frame) for train_frame in train_frames[sequence(path)]) <= 1
                for path in values
            )
            assert adjacent / len(values) >= expected_minimum_ratio[prefix]


def test_mixed_lists_and_legacy_archive_are_complete() -> None:
    assert set(read_split("mixed_dev")) == set(read_split("base_dev")) | set(read_split("increment_dev"))
    assert set(read_split("mixed_lock")) == set(read_split("base_lock")) | set(read_split("increment_lock"))
    assert len(read_split("mixed_dev")) == 89
    assert len(read_split("mixed_lock")) == 89

    archive_root = ROOT / "splits" / "archive" / "2026-08-21_strict_3plus1"
    archive = json.loads((archive_root / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
    assert archive["source_protocol"] == "full_coverage_strict_3plus1_dataset_partition"
    assert archive["file_count"] == 21
    assert len(archive["files"]) == 21
    for relative in archive["files"]:
        archived = archive_root / "snapshot" / relative
        assert archived.is_file()


def test_increment_sequence_quotas_match_fixed_80_10_10_design() -> None:
    manifest = json.loads((SPLIT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    quotas = manifest["increment_dataset"]["sequence_quotas"]
    assert quotas == {
        "ir|forest": {"train": 20, "dev": 3, "lock": 2},
        "ir|sea": {"train": 40, "dev": 5, "lock": 5},
        "ir|urban": {"train": 20, "dev": 2, "lock": 3},
        "sar|forest": {"train": 8, "dev": 1, "lock": 1},
        "sar|sea": {"train": 16, "dev": 2, "lock": 2},
        "sar|urban": {"train": 8, "dev": 1, "lock": 1},
    }

    distributions = manifest["increment_dataset"]["distribution"]
    assert all(distributions[name]["class_images"].get("4", 0) > 0 for name in ("train", "dev", "lock"))
    assert all(distributions[name]["class_images"].get("5", 0) > 0 for name in ("train", "dev", "lock"))

    armored_vehicle_only = {
        f"datasets_r2_inc_train/ir_r2_inc_urban_{frame_id:06d}.png"
        for frame_id in (18, 19, 20, 21, 23)
    }
    assert set(read_split("increment_dev")) & armored_vehicle_only
    assert set(read_split("increment_lock")) & armored_vehicle_only
