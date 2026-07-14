from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from fair_agent.modules.incremental_methods import (
    copy_incremental_head_rows,
    is_class_output_key,
    is_task_shared_key,
    merge_task_vectors,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_comparison_tool():
    path = ROOT / "tools" / "71_compare_incremental_methods.py"
    spec = importlib.util.spec_from_file_location("incremental_method_comparison_tool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_vector_merge_uses_reference_and_two_disjoint_deltas() -> None:
    reference = {"model.0.weight": torch.tensor([1.0, 2.0])}
    old = {"model.0.weight": torch.tensor([3.0, 4.0])}
    new = {"model.0.weight": torch.tensor([5.0, 8.0])}
    template = {"model.0.weight": torch.zeros(2)}

    merged, report = merge_task_vectors(
        reference,
        old,
        new,
        template,
        alpha_old=0.5,
        alpha_new=0.25,
        shared_key_exclude=("model.23",),
    )

    expected = reference["model.0.weight"]
    expected = expected + 0.5 * (old["model.0.weight"] - reference["model.0.weight"])
    expected = expected + 0.25 * (new["model.0.weight"] - reference["model.0.weight"])
    assert torch.allclose(merged["model.0.weight"], expected)
    assert report["merged_shared_tensor_count"] == 1


def test_task_vector_merge_preserves_excluded_old_detector_head() -> None:
    key = "model.23.cv2.0.2.weight"
    reference = {key: torch.tensor([1.0])}
    old = {key: torch.tensor([2.0])}
    new = {key: torch.tensor([9.0])}
    template = {key: torch.tensor([0.0])}

    merged, report = merge_task_vectors(
        reference,
        old,
        new,
        template,
        alpha_old=1.0,
        alpha_new=1.0,
        shared_key_exclude=("model.23",),
    )

    assert torch.equal(merged[key], old[key])
    assert report["preserved_old_tensor_count"] == 1


def test_incremental_head_rows_map_local_old_and_new_to_global_ids() -> None:
    weight_key = "model.23.cv3.0.2.weight"
    bias_key = "model.23.cv3.0.2.bias"
    state = {
        weight_key: torch.zeros(4, 2),
        bias_key: torch.zeros(4),
    }
    old = {
        weight_key: torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        bias_key: torch.tensor([1.0, 2.0, 3.0]),
    }
    new = {
        weight_key: torch.tensor([[9.0, 9.0]]),
        bias_key: torch.tensor([9.0]),
    }

    report = copy_incremental_head_rows(state, old, new, {0: 0, 1: 1, 2: 3}, 2)

    assert torch.equal(state[weight_key][:, 0], torch.tensor([1.0, 2.0, 9.0, 3.0]))
    assert torch.equal(state[bias_key], torch.tensor([1.0, 2.0, 9.0, 3.0]))
    assert report == {"copied_old_rows": 6, "copied_new_rows": 2}


def test_yolo_class_output_and_shared_key_detection() -> None:
    assert is_class_output_key("model.23.cv3.2.2.weight")
    assert not is_class_output_key("model.23.cv3.2.1.weight")
    assert is_task_shared_key("model.10.cv1.conv.weight", ("model.23",))
    assert not is_task_shared_key("model.23.cv2.0.0.weight", ("model.23",))


def test_comparison_contract_rejects_different_incremental_protocols() -> None:
    tool = _load_comparison_tool()
    config = {
        "paths": {"shared_base_checkpoint": "base.pt"},
        "methods": {"duet_yolo11s": {}, "yolo_iod_lite": {}},
        "protocols": [
            {
                "adaptation_mode": "duet_yolo11s",
                "base_classes": ["a", "b", "c"],
                "new_class": "d",
                "new_global_id": 3,
                "base_local_to_global": {0: 0, 1: 1, 2: 2},
                "expected_incremental_counts": {"train": 10},
                "preferred_device": "1",
            },
            {
                "adaptation_mode": "yolo_iod_lite",
                "base_classes": ["a", "b", "c"],
                "new_class": "different",
                "new_global_id": 4,
                "base_local_to_global": {0: 0, 1: 1, 2: 2},
                "expected_incremental_counts": {"train": 10},
                "preferred_device": "2",
            },
        ],
    }

    result = tool.validate_comparison_config(config)

    assert result["valid"] is False
    assert any("数据协议" in error for error in result["errors"])


def test_repository_comparison_config_uses_same_protocol_and_distinct_gpus() -> None:
    tool = _load_comparison_tool()
    config = tool.load_config(ROOT / "configs" / "incremental_method_comparison.yaml")

    result = tool.validate_comparison_config(config)

    assert result["valid"] is True
    assert result["errors"] == []
