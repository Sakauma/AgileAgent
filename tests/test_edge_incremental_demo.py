from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import yaml
from PIL import Image

from extras.ascend_edge_incremental.demo_contract import materialize_demo_contract
from extras.ascend_edge_incremental.promote_demo import build_demo_manifest
from extras.ascend_edge_incremental.protocol import load_protocol
from fair_agent.core.config import load_config, validate_config
from fair_agent.modules.edge_incremental_adapter import (
    apply_edge_incremental_adapter,
    load_edge_incremental_adapter,
)


def _image(path: Path, class_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16)).save(path)
    path.with_suffix(".txt").write_text(
        f"{class_id} 0.5 0.5 0.25 0.25\n", encoding="utf-8"
    )


def _write_split(repo: Path, name: str, values: list[Path]) -> str:
    path = repo / "splits" / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{value.relative_to(repo).as_posix()}\n" for value in values),
        encoding="utf-8",
    )
    return path.relative_to(repo).as_posix()


def _demo_registry(repo: Path, increment_reference: Path) -> Path:
    base_paths = {}
    for split in ("train", "dev", "lock"):
        image = repo / "base" / f"base_{split}.png"
        _image(image, 0)
        base_paths[split] = _write_split(repo, f"base_{split}", [image])
    round_splits: dict[int, dict[str, str]] = {4: {}, 5: {}}
    for class_id in (4, 5):
        for split in ("train", "dev", "lock"):
            image = increment_reference / f"new{class_id}_{split}.png"
            _image(image, class_id)
            round_splits[class_id][split] = _write_split(
                repo, f"new{class_id}_{split}", [image]
            )
    registry = {
        "schema_version": 1,
        "protocol_id": "strict_4plus2_sequential_class_incremental",
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
            "splits": base_paths,
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
                "old_raw_image_count": 0,
                "label_projection": "current_round_classes_only",
                "splits": round_splits[4],
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
                "splits": round_splits[5],
                "specialist": {"local_to_global": {0: 5}},
            },
        ],
    }
    path = repo / "reference_registry.yaml"
    path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    return path


def test_demo_contract_rebinds_only_incremental_round_images(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    reference_increment = repo / "reference_increment"
    registry = _demo_registry(repo, reference_increment)
    supplied = repo / "supplied_increment"
    for source in reference_increment.glob("*.png"):
        _image(supplied / source.name, int(source.stem[3]))

    audit = materialize_demo_contract(
        repo_root=repo,
        reference_registry=registry,
        incremental_data=supplied,
        contract_root=repo / "runs/demo/contract",
    )
    protocol = load_protocol(Path(audit["registry"]), repo)

    assert audit["base_images_used_for_training"] == 0
    assert audit["old_raw_image_count"] == 0
    assert audit["incremental_images_registered"] == 6
    assert all(
        supplied in path.parents for path in protocol.image_paths("training")
    )
    assert not ({path.stem for path in protocol.image_paths("training")} & {"base_train"})


def test_runtime_adapter_matches_eight_feature_residual(tmp_path: Path) -> None:
    manifest = tmp_path / "adapter.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "ascend_edge_incremental_demo_adapter",
                "feature_contract": "candidate_confidence_context_v1",
                "protocol_id": "strict_4plus2_sequential_class_incremental",
                "run_id": "demo-1",
                "class_order": [4, 5],
                "effective_weights": {
                    "4": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "5": [0.0] * 8,
                },
                "accepted": True,
                "production_modified": False,
            }
        ),
        encoding="utf-8",
    )
    adapter = load_edge_incremental_adapter(
        {
            "enabled": True,
            "manifest": str(manifest),
            "required_protocol_id": "strict_4plus2_sequential_class_incremental",
        },
        repo_root=tmp_path,
    )
    context = {
        "scene_probabilities": {
            "air": 0.1,
            "forest": 0.2,
            "sea": 0.3,
            "urban": 0.4,
        },
        "sensor_probabilities": {"ir": 0.75, "sar": 0.25},
    }
    records = [
        {
            "class_id": 4,
            "source": "incremental_model",
            "confidence": 0.5,
            "xyxy": [0.0, 0.0, 8.0, 8.0],
        },
        {
            "class_id": 0,
            "source": "frozen_base_model",
            "confidence": 0.5,
            "xyxy": [0.0, 0.0, 8.0, 8.0],
        },
    ]

    adapted = apply_edge_incremental_adapter(records, context, (16, 16), adapter)

    assert adapted[0]["confidence"] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert adapted[0]["edge_adapter_residual_logit"] == pytest.approx(1.0)
    assert adapted[1]["confidence"] == 0.5


def test_runtime_adapter_rejects_unaccepted_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "adapter.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "ascend_edge_incremental_demo_adapter",
                "feature_contract": "candidate_confidence_context_v1",
                "protocol_id": "demo",
                "class_order": [4],
                "effective_weights": {"4": [0.0] * 8},
                "accepted": False,
                "production_modified": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has not passed acceptance"):
        load_edge_incremental_adapter(
            {
                "enabled": True,
                "manifest": str(manifest),
                "required_protocol_id": "demo",
            },
            repo_root=tmp_path,
        )


def test_demo_promotion_requires_accuracy_numerical_and_fps_gates(
    tmp_path: Path,
) -> None:
    export = {
        "protocol_id": "demo",
        "feature_contract": "candidate_confidence_context_v1",
        "class_order": [4, 5],
        "effective_weights": {"4": [0.0] * 8, "5": [0.1] * 8},
    }
    evaluation = {
        "competition_passed": True,
        "edge_adapter": {
            "base_map50": 0.82,
            "new_map50": 0.65,
            "krr": 1.0,
            "full_map50": 0.74,
        },
    }
    benchmark = {
        "numerical_equivalence": {"passed": True},
        "projected_integrated_pipeline": {
            "fps_gate_30_passed": True,
            "projected_fps": 37.9,
        },
    }

    manifest = build_demo_manifest(
        run_id="run-1",
        protocol_id="demo",
        export_report=export,
        evaluation_report=evaluation,
        benchmark_report=benchmark,
        om_path=tmp_path / "adapter.om",
    )

    assert manifest["accepted"] is True
    assert manifest["production_modified"] is False
    assert manifest["effective_weights"]["5"] == [0.1] * 8
    rejected = copy.deepcopy(benchmark)
    rejected["projected_integrated_pipeline"]["fps_gate_30_passed"] = False
    with pytest.raises(ValueError, match="30 FPS"):
        build_demo_manifest(
            run_id="run-2",
            protocol_id="demo",
            export_report=export,
            evaluation_report=evaluation,
            benchmark_report=rejected,
            om_path=tmp_path / "adapter.om",
        )


def test_config_accepts_isolated_edge_adapter_contract() -> None:
    config = load_config("configs/agent_pipeline.yaml")
    config["routing"]["edge_incremental_adapter"] = {
        "enabled": True,
        "manifest": "/tmp/adapter_manifest.json",
        "require_accepted": True,
        "required_protocol_id": "strict_4plus2_sequential_class_incremental",
    }

    validate_config(config)


def test_one_command_wrapper_forces_offline_mode() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_ascend310b_incremental_demo.sh"
    ).read_text(encoding="utf-8")

    assert "PIP_NO_INDEX=1" in source
    assert "HF_HUB_OFFLINE=1" in source
    assert "stop_agent_ascend310b.sh" in source
    assert "extras.ascend_edge_incremental.demo" in source


def test_runtime_benchmark_reuses_the_formal_low_latency_pool() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "extras/ascend_edge_incremental/benchmark_demo_runtime.py"
    ).read_text(encoding="utf-8")

    assert "AtomicEngineProvider(config)" in source
    assert "predict_encoded_low_latency_batch" in source
    assert 'parser.add_argument("--confidence", type=float, default=0.5)' in source
