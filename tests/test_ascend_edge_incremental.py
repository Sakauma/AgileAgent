from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from extras.ascend_edge_incremental.protocol import (
    ensure_isolated_output,
    load_protocol,
)
from extras.ascend_edge_incremental.workflow import build_parser, build_stages


def build_registry(repo_root: Path, *, old_raw_image_count: int = 0) -> Path:
    split_values = {
        "base_train": "data/base_train.png",
        "base_dev": "data/base_dev.png",
        "base_lock": "data/base_lock.png",
        "round_01_train": "data/new4_train.png",
        "round_01_dev": "data/new4_dev.png",
        "round_01_lock": "data/new4_lock.png",
        "round_02_train": "data/new5_train.png",
        "round_02_dev": "data/new5_dev.png",
        "round_02_lock": "data/new5_lock.png",
    }
    (repo_root / "data").mkdir(parents=True)
    (repo_root / "splits").mkdir()
    for name, value in split_values.items():
        (repo_root / value).write_bytes(b"png")
        (repo_root / "splits" / f"{name}.txt").write_text(
            value + "\n", encoding="utf-8"
        )
    payload = {
        "schema_version": 1,
        "protocol_id": "synthetic_4plus2",
        "classes": {
            0: {"name": "old0", "introduced_in": "base"},
            1: {"name": "old1", "introduced_in": "base"},
            2: {"name": "old2", "introduced_in": "base"},
            3: {"name": "old3", "introduced_in": "base"},
            4: {"name": "new4", "introduced_in": "round_01"},
            5: {"name": "new5", "introduced_in": "round_02"},
        },
        "base": {
            "generation_id": "base_generation",
            "class_ids": [0, 1, 2, 3],
            "splits": {
                key: f"splits/base_{key}.txt" for key in ("train", "dev", "lock")
            },
        },
        "rounds": [
            {
                "round_id": "round_01",
                "round_index": 1,
                "generation_id": "generation_01",
                "parent_generation_id": "base_generation",
                "new_class_ids": [4],
                "learned_class_ids": [0, 1, 2, 3, 4],
                "learning_data_scope": "incremental_dataset_only",
                "validation_data_scope": "incremental_dataset_only",
                "base_detector_weights_frozen": True,
                "old_expert_weights_frozen": True,
                "old_raw_image_count": old_raw_image_count,
                "label_projection": "current_round_classes_only",
                "splits": {
                    key: f"splits/round_01_{key}.txt"
                    for key in ("train", "dev", "lock")
                },
                "specialist": {"local_to_global": {0: 4}},
            },
            {
                "round_id": "round_02",
                "round_index": 2,
                "generation_id": "generation_02",
                "parent_generation_id": "generation_01",
                "new_class_ids": [5],
                "learned_class_ids": [0, 1, 2, 3, 4, 5],
                "learning_data_scope": "incremental_dataset_only",
                "validation_data_scope": "incremental_dataset_only",
                "base_detector_weights_frozen": True,
                "old_expert_weights_frozen": True,
                "old_raw_image_count": 0,
                "label_projection": "current_round_classes_only",
                "splits": {
                    key: f"splits/round_02_{key}.txt"
                    for key in ("train", "dev", "lock")
                },
                "specialist": {"local_to_global": {0: 5}},
            },
        ],
    }
    registry = repo_root / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return registry


def test_registry_drives_rounds_and_data_scopes(tmp_path: Path) -> None:
    registry = build_registry(tmp_path)
    protocol = load_protocol(registry, tmp_path)

    assert protocol.new_class_ids == (4, 5)
    assert [item.round_id for item in protocol.rounds] == ["round_01", "round_02"]
    assert len(protocol.image_paths("training")) == 4
    assert len(protocol.image_paths("selection")) == 3
    assert len(protocol.image_paths("lock")) == 3
    assert len(protocol.image_paths("all")) == 9
    assert protocol.base_image_ids("lock") == {"base_lock"}


def test_registry_rejects_historical_raw_replay(tmp_path: Path) -> None:
    registry = build_registry(tmp_path, old_raw_image_count=1)

    with pytest.raises(ValueError, match="replays historical raw images"):
        load_protocol(registry, tmp_path)


def test_training_scope_rejects_cross_round_image_replay(tmp_path: Path) -> None:
    registry = build_registry(tmp_path)
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    payload["rounds"][1]["splits"]["train"] = payload["rounds"][0]["splits"][
        "train"
    ]
    registry.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    protocol = load_protocol(registry, tmp_path)

    with pytest.raises(ValueError, match="duplicate image stem"):
        protocol.image_paths("training")


def test_output_guard_blocks_production_and_overwrite(tmp_path: Path) -> None:
    production = tmp_path / "models" / "production"
    production.mkdir(parents=True)

    with pytest.raises(ValueError, match="protected project assets"):
        ensure_isolated_output(production / "edge", tmp_path)

    existing = tmp_path / "edge-output"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ensure_isolated_output(existing, tmp_path)


def test_workflow_plan_separates_runtimes_and_orders_freeze_before_lock(
    tmp_path: Path,
) -> None:
    registry = build_registry(tmp_path)
    training_python = tmp_path / "train-python"
    production_python = tmp_path / "production-python"
    parser = build_parser()
    args = parser.parse_args(
        [
            "plan",
            "--repo-root",
            str(tmp_path),
            "--registry",
            str(registry),
            "--output-root",
            str(tmp_path / "edge-output"),
            "--training-python",
            str(training_python),
            "--production-python",
            str(production_python),
            "--baseline-fps",
            "38.2175",
            "--include-all-diagnostics",
        ]
    )
    stages = build_stages(args)
    by_name = {stage.name: stage for stage in stages}
    names = [stage.name for stage in stages]

    assert by_name["freeze_training_probe"].command[0] == str(production_python)
    assert by_name["compile_om"].use_production_python_path is True
    assert by_name["train_registered_rounds"].command[0] == str(training_python)
    assert "--registry" in by_name["train_registered_rounds"].command
    assert names.index("train_registered_rounds") < names.index("evaluate_lock")
    assert names.index("select_adapter_scales") < names.index("freeze_lock_probe")
    assert "evaluate_all" in names


def test_training_source_explicitly_refuses_cpu_fallback() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "extras/ascend_edge_incremental/train.py"
    ).read_text(encoding="utf-8")

    assert "refusing CPU fallback" in source
    assert "ROUNDS =" not in source
    assert "CLASS_NAMES =" not in source
