from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = ROOT / "splits"
EXPECTED_POOL_COUNTS = {
    "pool_train": 522,
    "pool_dev": 88,
    "mixed_test": 89,
    "embargo": 51,
}


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sequence_and_frame(path: str) -> tuple[str, int]:
    parts = Path(path).stem.split("_")
    assert len(parts) == 5
    return "|".join(parts[:4]), int(parts[4])


def test_active_source_pools_are_complete_disjoint_and_temporally_isolated() -> None:
    assert not (ROOT / "splits_v2").exists()
    assert not any(
        (SPLIT_ROOT / name).exists()
        for name in ("train.txt", "dev_val.txt", "lock_val.txt")
    )
    pools = {
        name: read_split(SPLIT_ROOT / f"{name}.txt")
        for name in EXPECTED_POOL_COUNTS
    }
    for name, expected in EXPECTED_POOL_COUNTS.items():
        assert len(pools[name]) == expected
        assert len(pools[name]) == len(set(pools[name]))
        assert all(path.startswith("datasets_r1_base_train/") for path in pools[name])
        assert all(path.endswith(".png") for path in pools[name])

    all_paths = [path for rows in pools.values() for path in rows]
    assert len(all_paths) == 750
    assert len(set(all_paths)) == 750

    for first_name, second_name in (
        ("pool_train", "pool_dev"),
        ("pool_train", "mixed_test"),
        ("pool_dev", "mixed_test"),
    ):
        first = [sequence_and_frame(path) for path in pools[first_name]]
        second = [sequence_and_frame(path) for path in pools[second_name]]
        for first_sequence, first_frame in first:
            distances = [
                abs(first_frame - second_frame)
                for second_sequence, second_frame in second
                if second_sequence == first_sequence
            ]
            assert distances
            assert min(distances) > 4


def test_active_sensor_subsets_match_source_pools() -> None:
    for name in ("pool_train", "pool_dev", "mixed_test"):
        complete = set(read_split(SPLIT_ROOT / f"{name}.txt"))
        ir = set(read_split(SPLIT_ROOT / f"{name}_ir.txt"))
        sar = set(read_split(SPLIT_ROOT / f"{name}_sar.txt"))
        assert not ir & sar
        assert ir | sar == complete
        assert all(Path(path).name.startswith("ir_") for path in ir)
        assert all(Path(path).name.startswith("sar_") for path in sar)


def test_fixed_protocol_is_exactly_three_base_classes_plus_warship() -> None:
    top = json.loads((SPLIT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    protocol_path = ROOT / top["strict_3plus1_manifest"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_root = protocol_path.parent

    assert top["protocol"] == "temporal_strict_3plus1_dataset_partition"
    assert top["counts"] == EXPECTED_POOL_COUNTS
    assert top["simulated_increment_class"] == "warship"
    assert protocol["protocol"] == "strict_3plus1_class_incremental_simulation"
    assert protocol["base_class_ids"] == [0, 1, 3]
    assert protocol["base_class_names"] == ["soldier", "small_aircraft", "tank"]
    assert protocol["increment_class_id"] == 2
    assert protocol["increment_class_name"] == "warship"

    base_train = set(read_split(protocol_root / "base_train.txt"))
    base_dev = set(read_split(protocol_root / "base_dev.txt"))
    increment_train = set(read_split(protocol_root / "increment_train.txt"))
    increment_dev = set(read_split(protocol_root / "increment_dev.txt"))
    assert len(base_train) == 405
    assert len(base_dev) == 70
    assert len(increment_train) == 117
    assert len(increment_dev) == 18
    assert not base_train & increment_train
    assert not base_dev & increment_dev
    assert base_train | increment_train == set(read_split(SPLIT_ROOT / "pool_train.txt"))
    assert base_dev | increment_dev == set(read_split(SPLIT_ROOT / "pool_dev.txt"))
    assert all("_sea_" not in path for path in base_train | base_dev)
    assert all("_sea_" in path for path in increment_train | increment_dev)
    assert not (SPLIT_ROOT / "pseudo_incremental").exists()


def test_mixed_detection_test_and_known_scene_lists_use_correct_scopes() -> None:
    protocol_root = SPLIT_ROOT / "strict_3plus1"
    mixed = set(read_split(protocol_root / "mixed_test.txt"))
    old_positive = set(read_split(protocol_root / "mixed_test_old_positive.txt"))
    new_positive = set(read_split(protocol_root / "mixed_test_new_positive.txt"))
    assert len(mixed) == 89
    assert len(old_positive) == 70
    assert len(new_positive) == 19
    assert old_positive | new_positive == mixed
    assert not old_positive & new_positive
    assert mixed == set(read_split(SPLIT_ROOT / "mixed_test.txt"))

    assert set(read_split(protocol_root / "scene_train.txt")) == set(
        read_split(SPLIT_ROOT / "pool_train.txt")
    )
    assert set(read_split(protocol_root / "scene_dev.txt")) == set(
        read_split(SPLIT_ROOT / "pool_dev.txt")
    )
    assert set(read_split(protocol_root / "scene_test.txt")) == mixed

    protocol = json.loads((protocol_root / "manifest.json").read_text(encoding="utf-8"))
    assert protocol["detection_contract"] == {
        "base_training_classes": [0, 1, 3],
        "base_training_contains_increment_class": False,
        "increment_training_classes": [2],
        "increment_training_contains_base_classes": False,
        "test_inference_scope": "complete_mixed_test_for_parent_and_candidate",
        "same_image_old_new_required": False,
        "same_image_old_new_count": 0,
    }
    assert protocol["scene_contract"]["known_scenes"] == ["air", "forest", "sea", "urban"]
    assert protocol["scene_contract"]["target_class_labels_access"] is False
    assert protocol["scene_contract"]["scene_to_target_class_hard_binding_allowed"] is False
