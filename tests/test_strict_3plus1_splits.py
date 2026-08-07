from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = ROOT / "splits"
EXPECTED_POOL_COUNTS = {
    "pool_train": 573,
    "pool_dev": 88,
    "mixed_test": 89,
    "embargo": 0,
}


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def logical_sha256(rows: list[str]) -> str:
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def test_active_source_pools_are_complete_disjoint_and_use_all_750_images() -> None:
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
    assert pools["embargo"] == []
    assert logical_sha256(pools["pool_dev"]) == (
        "aaf49d10ffe77157ed1f32c46af13a7bb7c24156dc06c9479f14348a04c29eb7"
    )
    assert logical_sha256(pools["mixed_test"]) == (
        "c3e28ffdfedc4e52149c18b1ec05119ef4ff8d9f17974a096a4fd6d5df813602"
    )


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

    assert top["protocol"] == "full_coverage_strict_3plus1_dataset_partition"
    assert top["counts"] == EXPECTED_POOL_COUNTS
    assert top["allocation_policy"] == {
        "all_source_images_used": True,
        "reclaimed_previous_embargo_to_pool_train": True,
        "reclaimed_image_count": 51,
        "dev_and_test_membership_preserved": True,
        "temporal_gap_constraint": None,
    }
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
    base_test = set(read_split(protocol_root / "base_test.txt"))
    assert len(base_train) == 441
    assert len(base_dev) == 70
    assert len(increment_train) == 132
    assert len(increment_dev) == 18
    assert len(base_test) == 70
    assert not base_train & increment_train
    assert not base_dev & increment_dev
    assert base_train | increment_train == set(read_split(SPLIT_ROOT / "pool_train.txt"))
    assert base_dev | increment_dev == set(read_split(SPLIT_ROOT / "pool_dev.txt"))
    assert all("_sea_" not in path for path in base_train | base_dev)
    assert all("_sea_" in path for path in increment_train | increment_dev)
    assert all("_sea_" not in path for path in base_test)
    assert not (SPLIT_ROOT / "pseudo_incremental").exists()


def test_mixed_detection_test_and_known_scene_lists_use_correct_scopes() -> None:
    protocol_root = SPLIT_ROOT / "strict_3plus1"
    mixed = set(read_split(protocol_root / "mixed_test.txt"))
    base_test = set(read_split(protocol_root / "base_test.txt"))
    assert len(mixed) == 89
    assert len(base_test) == 70
    assert base_test < mixed
    assert mixed == set(read_split(SPLIT_ROOT / "mixed_test.txt"))

    assert set(read_split(protocol_root / "scene_train.txt")) == set(
        read_split(SPLIT_ROOT / "pool_train.txt")
    )
    assert set(read_split(protocol_root / "scene_dev.txt")) == set(
        read_split(SPLIT_ROOT / "pool_dev.txt")
    )
    assert set(read_split(protocol_root / "scene_test.txt")) == mixed

    protocol = json.loads((protocol_root / "manifest.json").read_text(encoding="utf-8"))
    assert protocol["mixed_test_composition"] == {
        "old_class_images": 70,
        "new_class_images": 19,
        "base_test_list_published_for_scoring_only": True,
    }
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
    assert protocol["evaluation"]["score_gates"] == {
        "base_test_map50": 0.8,
        "new_map50": 0.6,
        "krr": 0.95,
    }
    assert protocol["evaluation"]["full_map50_role"] == "diagnostic_only"
    assert protocol["evaluation"]["unlabeled_inference_before_scoring"] is True
    assert protocol["evaluation"]["label_aware_routing_allowed"] is False
