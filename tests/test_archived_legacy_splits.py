from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = ROOT / "archive" / "splits_legacy_random_560_95_95"
EXPECTED_COUNTS = {"train": 560, "dev_val": 95, "lock_val": 95}


def read_split(name: str) -> list[str]:
    return [
        line.strip()
        for line in (SPLIT_ROOT / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_archived_legacy_splits_are_complete_disjoint_and_relative() -> None:
    splits = {name: read_split(name) for name in EXPECTED_COUNTS}
    for name, expected in EXPECTED_COUNTS.items():
        assert len(splits[name]) == expected
        assert len(splits[name]) == len(set(splits[name]))
        assert all(path.startswith("datasets_r1_base_train/") for path in splits[name])
        assert all(path.endswith(".png") for path in splits[name])

    all_paths = [path for rows in splits.values() for path in rows]
    assert len(all_paths) == 750
    assert len(set(all_paths)) == 750


def test_sensor_subsets_match_archived_legacy_splits() -> None:
    for split_name in EXPECTED_COUNTS:
        complete = set(read_split(split_name))
        ir = set(read_split(f"{split_name}_ir"))
        sar = set(read_split(f"{split_name}_sar"))
        assert not ir & sar
        assert ir | sar == complete
        assert all(Path(path).name.startswith("ir_") for path in ir)
        assert all(Path(path).name.startswith("sar_") for path in sar)
