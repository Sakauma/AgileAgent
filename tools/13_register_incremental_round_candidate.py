#!/usr/bin/env python3
"""Register one frozen strict 4+2 round candidate in the source registry."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.hashes import sha256_file  # noqa: E402
from fair_agent.modules.incremental_round_registry import (  # noqa: E402
    DEFAULT_ROUND_REGISTRY,
    load_incremental_round_registry,
    rounds_through,
    select_round,
)
from fair_agent.modules.model_generations import load_generation_registry  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{resolved}")
    return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def relative_to_root(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"正式候选资产必须复制到工程目录内：{resolved}") from exc


def count_split(path: str | Path) -> int:
    resolved = ROOT / Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"轮次清单不存在：{resolved}")
    rows = [
        line.strip()
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"轮次清单为空或包含重复项：{resolved}")
    return len(rows)


def copy_immutable(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"候选资产不存在：{source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            raise FileExistsError(f"拒绝覆盖内容不同的候选资产：{destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def expected_lineage(round_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "round_id": round_spec["round_id"],
        "round_index": round_spec["round_index"],
        "parent_generation_id": round_spec["parent_generation_id"],
        "generation_id": round_spec["generation_id"],
        "old_class_ids": round_spec["old_class_ids"],
        "new_class_ids": round_spec["new_class_ids"],
        "learned_class_ids": round_spec["learned_class_ids"],
    }


def validate_selection(
    payload: Mapping[str, Any], round_spec: Mapping[str, Any]
) -> Path:
    expected_specialist = round_spec["specialist"]
    registered_specialist = dict(payload.get("specialist") or {})
    registered_mapping = {
        int(key): int(value)
        for key, value in dict(registered_specialist.get("local_to_global") or {}).items()
    }
    if (
        payload.get("phase") != "incremental_learning"
        or payload.get("counted_as_incremental_learning") is not True
        or payload.get("detector_weights_updated") is not False
        or payload.get("round_id") != round_spec["round_id"]
        or payload.get("round_index") != round_spec["round_index"]
        or payload.get("parent_generation_id")
        != round_spec["parent_generation_id"]
        or payload.get("generation_id") != round_spec["generation_id"]
        or payload.get("new_class_ids") != round_spec["new_class_ids"]
        or payload.get("old_class_ids") != round_spec["old_class_ids"]
        or payload.get("training_data_scope") != "incremental_dataset_only"
        or payload.get("validation_data_scope") != "incremental_dataset_only"
        or int(payload.get("old_raw_image_count", -1)) != 0
        or int(payload.get("old_raw_label_count", -1)) != 0
        or payload.get("base_detector_weights_frozen") is not True
        or payload.get("old_expert_weights_frozen") is not True
        or payload.get("lock_used") is not False
        or registered_specialist.get("model_id") != expected_specialist["model_id"]
        or registered_mapping != expected_specialist["local_to_global"]
    ):
        raise ValueError("选模记录不满足当轮 Increment dev 与冻结父代契约")
    selected = payload.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("选模记录缺少 selected")
    weight = Path(str(selected.get("promoted_weight") or selected.get("weight") or ""))
    resolved = weight.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"选中权重不存在：{resolved}")
    return resolved


def validate_calibration(
    payload: Mapping[str, Any],
    round_spec: Mapping[str, Any],
    expected_incremental_ids: set[int],
) -> dict[int, float]:
    thresholds = {
        int(key): float(value)
        for key, value in dict(payload.get("per_class_thresholds") or {}).items()
    }
    if (
        payload.get("phase") != "system_calibration"
        or payload.get("counted_as_incremental_learning") is not False
        or payload.get("detector_weights_updated") is not False
        or payload.get("round_id") != round_spec["round_id"]
        or payload.get("round_index") != round_spec["round_index"]
        or payload.get("parent_generation_id")
        != round_spec["parent_generation_id"]
        or payload.get("generation_id") != round_spec["generation_id"]
        or payload.get("source_split") != "cumulative_dev_only"
        or payload.get("old_incremental_thresholds_frozen") is not True
        or set(thresholds) != expected_incremental_ids
        or (
            int(round_spec["round_index"]) == 1
            and payload.get("parent_calibration") is not None
        )
        or (
            int(round_spec["round_index"]) > 1
            and not str(payload.get("parent_calibration") or "").strip()
        )
    ):
        raise ValueError("校准文件不满足 system_calibration 冻结契约")
    if any(not 0.01 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError("校准阈值越界")
    return thresholds


def cumulative_incremental_ids(
    active_rounds: Sequence[Mapping[str, Any]],
) -> set[int]:
    return {
        int(class_id)
        for row in active_rounds
        for class_id in row["new_class_ids"]
    }


def validate_evaluation(
    payload: Mapping[str, Any],
    round_spec: Mapping[str, Any],
    active_rounds: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    lineage = dict(payload.get("lineage") or {})
    if any(
        lineage.get(key) != value
        for key, value in expected_lineage(round_spec).items()
    ):
        raise ValueError("累计评测的父子代际或类别集合不一致")
    score_gates = dict(payload.get("score_gates") or {})
    if (
        payload.get("phase") != "joint_evaluation"
        or payload.get("counted_as_incremental_learning") is not False
        or payload.get("detector_weights_updated") is not False
        or payload.get("model_selection_allowed") is not False
        or payload.get("incremental_learning_data_scope")
        != "incremental_dataset_only"
        or int(payload.get("old_raw_image_count", -1)) != 0
        or int(payload.get("old_raw_label_count", -1)) != 0
        or payload.get("predictions_frozen_before_lock_labels") is not True
        or payload.get("accepted") is not True
        or payload.get("competition_accepted") is not True
        or set(score_gates) != {"base_map50", "new_map50", "krr"}
        or not all(value is True for value in score_gates.values())
    ):
        raise ValueError("累计评测尚未通过正式 mAP50/KRR 门禁")
    specialists = payload.get("specialists")
    if not isinstance(specialists, list):
        raise ValueError("累计评测缺少专家列表")
    by_round = {str(item.get("round_id")): item for item in specialists}
    expected_ids = {str(row["round_id"]) for row in active_rounds}
    if set(by_round) != expected_ids or len(by_round) != len(specialists):
        raise ValueError("累计评测必须且只能包含截至当前轮的专家")
    for row in active_rounds:
        item = by_round[str(row["round_id"])]
        mapping = {
            int(key): int(value)
            for key, value in dict(item.get("local_to_global") or {}).items()
        }
        if (
            item.get("model_id") != row["specialist"]["model_id"]
            or mapping != row["specialist"]["local_to_global"]
        ):
            raise ValueError(f"累计评测专家映射不一致：{row['round_id']}")
    return by_round


def validate_parent_chain(
    raw_registry: Mapping[str, Any],
    round_spec: Mapping[str, Any],
    active_rounds: Sequence[Mapping[str, Any]],
    evaluated: Mapping[str, Mapping[str, Any]],
    current_weight_hash: str,
    base_model_id: str,
    evaluated_base_weight: Path,
) -> tuple[dict[str, Any], list[str]]:
    generations = {
        str(item["id"]): item for item in raw_registry.get("generations", [])
    }
    models = {str(item["id"]): item for item in raw_registry.get("models", [])}
    parent_id = str(round_spec["parent_generation_id"])
    if parent_id not in generations:
        raise ValueError(f"父代尚未登记：{parent_id}")
    parent = generations[parent_id]
    if set(int(value) for value in parent["classes"]) != set(
        round_spec["old_class_ids"]
    ):
        raise ValueError("父代累计类别与轮次注册表不一致")
    if base_model_id not in models:
        raise ValueError(f"Base 模型尚未登记：{base_model_id}")
    if (
        not evaluated_base_weight.is_file()
        or sha256_file(evaluated_base_weight) != models[base_model_id]["sha256"]
    ):
        raise ValueError("累计评测的 Base 权重与冻结 Base 登记不一致")
    expected_parent_members: list[str] = [base_model_id]
    for previous in active_rounds[:-1]:
        model_id = str(previous["specialist"]["model_id"])
        generation_id = str(previous["generation_id"])
        if model_id not in models or generation_id not in generations:
            raise ValueError(f"历史轮次尚未登记：{previous['round_id']}")
        model = models[model_id]
        generation = generations[generation_id]
        mapping = {
            int(key): int(value)
            for key, value in dict(model.get("local_to_global") or {}).items()
        }
        if (
            generation.get("parent") != previous["parent_generation_id"]
            or set(int(value) for value in generation.get("classes", []))
            != set(previous["learned_class_ids"])
            or set(int(value) for value in model.get("owns_classes", []))
            != set(previous["new_class_ids"])
            or mapping != previous["specialist"]["local_to_global"]
        ):
            raise ValueError(f"历史轮次登记不完整：{previous['round_id']}")
        evaluated_weight = Path(str(evaluated[str(previous["round_id"])]["weight"]))
        if (
            not evaluated_weight.expanduser().resolve().is_file()
            or sha256_file(evaluated_weight.expanduser().resolve()) != model["sha256"]
        ):
            raise ValueError(f"历史专家评测权重与登记权重不一致：{previous['round_id']}")
        expected_parent_members.append(model_id)
    current_eval = Path(str(evaluated[str(round_spec["round_id"])]["weight"]))
    if (
        not current_eval.expanduser().resolve().is_file()
        or sha256_file(current_eval.expanduser().resolve()) != current_weight_hash
    ):
        raise ValueError("当前轮选中权重与累计评测权重不一致")
    parent_members = list(
        dict.fromkeys(
            str(value)
            for value in (parent.get("model_members") or parent["class_owners"].values())
        )
    )
    if parent_members != expected_parent_members:
        raise ValueError("父代模型成员未严格继承 Base 与历史轮次专家")
    return parent, expected_parent_members


def registry_identity_matches(
    model: Mapping[str, Any],
    generation: Mapping[str, Any],
    round_spec: Mapping[str, Any],
    weight_hash: str,
) -> bool:
    mapping = {
        int(key): int(value)
        for key, value in dict(model.get("local_to_global") or {}).items()
    }
    return (
        str(model.get("id")) == str(round_spec["specialist"]["model_id"])
        and str(model.get("sha256")) == weight_hash
        and set(int(value) for value in model.get("owns_classes", []))
        == set(round_spec["new_class_ids"])
        and mapping == round_spec["specialist"]["local_to_global"]
        and str(generation.get("id")) == str(round_spec["generation_id"])
        and generation.get("parent") == round_spec["parent_generation_id"]
        and set(int(value) for value in generation.get("classes", []))
        == set(round_spec["learned_class_ids"])
    )


def write_validated_registry(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        load_generation_registry(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="登记一轮已选定、已冻结并通过累计评测的严格类别增量候选。"
    )
    parser.add_argument(
        "--round-registry", type=Path, default=ROOT / DEFAULT_ROUND_REGISTRY
    )
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="默认使用 evaluation 同目录下的 calibration.json",
    )
    parser.add_argument(
        "--generation-registry",
        type=Path,
        default=ROOT / "models" / "generations.json",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT / "models" / "candidates" / "incremental_detection",
    )
    args = parser.parse_args()

    registry = load_incremental_round_registry(args.round_registry)
    round_spec = select_round(registry, args.round_id)
    active_rounds = rounds_through(registry, args.round_id)
    selection_path = args.selection.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    calibration_path = (
        args.calibration.expanduser().resolve()
        if args.calibration is not None
        else evaluation_path.parent / "calibration.json"
    )
    selection = read_json(selection_path)
    evaluation = read_json(evaluation_path)
    calibration = read_json(calibration_path)
    selected_weight = validate_selection(selection, round_spec)
    selected_hash = sha256_file(selected_weight)
    evaluated = validate_evaluation(evaluation, round_spec, active_rounds)
    expected_incremental = cumulative_incremental_ids(active_rounds)
    thresholds = validate_calibration(
        calibration, round_spec, expected_incremental
    )

    artifact_root = args.artifact_root.expanduser().resolve()
    relative_to_root(artifact_root)
    round_root = artifact_root / str(round_spec["generation_id"])
    evidence_root = round_root / "evidence"
    model_id = str(round_spec["specialist"]["model_id"])
    generation_id = str(round_spec["generation_id"])
    stored_weight = round_root / f"{model_id}.pt"
    stored_selection = evidence_root / "incremental_selection.json"
    stored_evaluation = evidence_root / "metrics.json"
    stored_calibration = evidence_root / "calibration.json"
    registration_path = round_root / "registration.json"

    generation_registry_path = args.generation_registry.expanduser().resolve()
    raw_registry = read_json(generation_registry_path)
    existing_models = {
        str(item["id"]): item for item in raw_registry.get("models", [])
    }
    existing_generations = {
        str(item["id"]): item for item in raw_registry.get("generations", [])
    }
    if model_id in existing_models or generation_id in existing_generations:
        if model_id not in existing_models or generation_id not in existing_generations:
            raise ValueError("同名模型或代际只存在一项，注册表已不完整")
        existing_model = existing_models[model_id]
        existing_generation = existing_generations[generation_id]
        existing_evidence = dict(existing_model.get("training_evidence") or {})
        stored_inputs = (
            (selected_weight, stored_weight),
            (selection_path, stored_selection),
            (evaluation_path, stored_evaluation),
            (calibration_path, stored_calibration),
        )
        registration = (
            read_json(registration_path) if registration_path.is_file() else {}
        )
        if (
            not registry_identity_matches(
                existing_model,
                existing_generation,
                round_spec,
                selected_hash,
            )
            or existing_generation.get("status") != "registered_candidate"
            or raw_registry.get("channels", {}).get("candidate") != generation_id
            or existing_model.get("path") != relative_to_root(stored_weight)
            or existing_evidence.get("selection_source")
            != relative_to_root(stored_selection)
            or existing_evidence.get("evaluation_source")
            != relative_to_root(stored_evaluation)
            or existing_evidence.get("calibration_source")
            != relative_to_root(stored_calibration)
            or existing_evidence.get("registration_source")
            != relative_to_root(registration_path)
            or any(
                not destination.is_file()
                or sha256_file(source) != sha256_file(destination)
                for source, destination in stored_inputs
            )
            or registration.get("status") != "registered_candidate"
            or registration.get("round_id") != round_spec["round_id"]
            or registration.get("generation_id") != generation_id
            or registration.get("model_id") != model_id
            or registration.get("model_path") != relative_to_root(stored_weight)
            or registration.get("model_sha256") != selected_hash
            or registration.get("production_unchanged") is not True
        ):
            raise ValueError("同名候选已存在，但登记资产或证据身份不完整")
        print(
            json.dumps(
                {**registration, "status": "already_registered"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    parent, parent_members = validate_parent_chain(
        raw_registry,
        round_spec,
        active_rounds,
        evaluated,
        selected_hash,
        str(registry["base"]["model_id"]),
        Path(str(dict(evaluation.get("weights") or {}).get("base") or ""))
        .expanduser()
        .resolve(),
    )
    copy_immutable(selected_weight, stored_weight)
    copy_immutable(selection_path, stored_selection)
    copy_immutable(evaluation_path, stored_evaluation)
    copy_immutable(calibration_path, stored_calibration)
    selection_report = selection_path.with_suffix(".md")
    if selection_report.is_file():
        copy_immutable(selection_report, evidence_root / "incremental_selection.md")

    stored_weight_hash = sha256_file(stored_weight)
    if stored_weight_hash != selected_hash:
        raise RuntimeError("候选权重复制后身份发生变化")
    owned_ids = [int(value) for value in round_spec["new_class_ids"]]
    local_to_global = {
        str(key): value
        for key, value in round_spec["specialist"]["local_to_global"].items()
    }
    lock = dict(evaluation["lock"])
    quality = {
        str(class_id): dict(lock["new_class_quality"][str(class_id)])
        for class_id in owned_ids
    }
    false_activation = {
        str(class_id): dict(lock["false_activation"][str(class_id)])
        for class_id in owned_ids
    }
    split_counts = {
        role: count_split(round_spec["splits"][role])
        for role in ("train", "dev", "lock")
    }
    training_evidence = {
        "phase": "incremental_learning",
        "counted_as_incremental_learning": True,
        "round_id": round_spec["round_id"],
        "round_index": round_spec["round_index"],
        "parent_generation_id": round_spec["parent_generation_id"],
        "generation_id": generation_id,
        "training_data_scope": "incremental_dataset_only",
        "validation_data_scope": "incremental_dataset_only",
        "old_raw_image_count": 0,
        "old_raw_label_count": 0,
        "base_detector_weights_frozen": True,
        "historical_incremental_expert_weights_frozen": True,
        "split_counts": split_counts,
        "selection_source": relative_to_root(stored_selection),
        "evaluation_source": relative_to_root(stored_evaluation),
        "calibration_source": relative_to_root(stored_calibration),
        "registration_source": relative_to_root(registration_path),
    }
    model = {
        "id": model_id,
        "display_name": "{} 单类增量专家".format(round_spec["round_id"]),
        "role": "class_incremental_expert",
        "incremental_mode": "class_incremental",
        "backend": "ultralytics",
        "architecture": str(
            dict(selection.get("selected") or {}).get("model_tag")
            or "registry_selected"
        ),
        "path": relative_to_root(stored_weight),
        "sha256": stored_weight_hash,
        "imgsz": int(selection.get("imgsz") or 1280),
        "owns_classes": owned_ids,
        "local_to_global": local_to_global,
        "per_class_thresholds": {
            str(class_id): thresholds[class_id] for class_id in owned_ids
        },
        "calibration_sources": {
            str(class_id): relative_to_root(stored_calibration)
            for class_id in owned_ids
        },
        "context_prior": {},
        "context_gate": {"enabled": False},
        "metrics": {
            "new_map50": float(lock["new_map50"]),
            "per_class": quality,
        },
        "deployment_metrics": {
            "false_activation_rate_by_class": {
                class_id: row["false_activation_rate"]
                for class_id, row in false_activation.items()
            },
            "diagnostic_only": True,
        },
        "training_evidence": training_evidence,
        "acceptance": {
            "official_score_gates_passed": True,
            "competition_gates_passed": True,
            "deployment_quality_gates_passed": False,
            "integrity_gates_passed": True,
            "passed": True,
        },
        "status": "registered_candidate",
    }
    if len(owned_ids) == 1:
        only = owned_ids[0]
        model["activation_threshold"] = thresholds[only]
        model["calibration_source"] = relative_to_root(stored_calibration)

    owners = {str(key): str(value) for key, value in parent["class_owners"].items()}
    for class_id in owned_ids:
        if str(class_id) in owners:
            raise ValueError(f"本轮新类已被父代拥有：{class_id}")
        owners[str(class_id)] = model_id
    members = [*parent_members, model_id]
    round_metrics = {
        key: float(evaluation["round_metrics"][key])
        for key in ("new_map50", "krr", "full_map50")
    }
    generation = {
        "id": generation_id,
        "display_name": "{} 严格类别增量代际".format(round_spec["round_id"]),
        "parent": round_spec["parent_generation_id"],
        "classes": round_spec["learned_class_ids"],
        "class_owners": owners,
        "model_members": members,
        "status": "registered_candidate",
        "old_class_ids": round_spec["old_class_ids"],
        "new_class_ids": owned_ids,
        "updated_class_ids": [],
        "incremental_mode": "class_incremental",
        "dataset_fingerprint": "splits/strict_4plus2/manifest.json",
        "data_compliance": {
            "compliance": "passed",
            "lineage_evidence": "strict_sequential_round_registry",
            "round_id": round_spec["round_id"],
            "training_data_scope": "incremental_dataset_only",
            "validation_data_scope": "incremental_dataset_only",
            "old_raw_image_count": 0,
            "old_raw_label_count": 0,
            "old_cache_count": 0,
            "unverified_cache_count": 0,
            **{f"incremental_{key}_count": value for key, value in split_counts.items()},
        },
        "evaluation_lock": {
            "split_sources": [
                registry["base"]["splits"]["lock"],
                *[row["splits"]["lock"] for row in active_rounds],
            ],
            "prediction_scope": "all_registered_owners_before_cumulative_lock_labels",
            "predictions_frozen_before_lock_labels": True,
            "metrics_source": relative_to_root(stored_evaluation),
        },
        "metrics": {
            **round_metrics,
            "base_test_map50": float(lock["base_map50"]),
            "old_map50_before": float(lock["old_map50_before"]),
            "old_map50_after": float(lock["old_map50_after"]),
            "per_class": quality,
            "cumulative_per_class": dict(lock["full_per_class_ap50"]),
            "specialist_count": len(active_rounds),
        },
        "acceptance": {
            "core_metrics_passed": True,
            "competition_gates_passed": True,
            "deployment_recheck_passed": False,
            "status": "registered_candidate",
        },
        "sequential_round_evidence": len(active_rounds) >= 2,
        "phase_contract": {
            "incremental_learning": {
                "counted_as_incremental_learning": True,
                "training_data_scope": "incremental_dataset_only",
                "validation_data_scope": "incremental_dataset_only",
                "detector_weights_updated": [model_id],
                "base_detector_weights_frozen": True,
                "historical_incremental_expert_weights_frozen": True,
            },
            "system_calibration": {
                "counted_as_incremental_learning": False,
                "detector_weights_updated": False,
            },
            "joint_evaluation": {
                "counted_as_incremental_learning": False,
                "detector_weights_updated": False,
                "model_selection_allowed": False,
            },
        },
    }
    registration = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "registered_candidate",
        "round_registry": relative_to_root(Path(registry["path"])),
        "round_id": round_spec["round_id"],
        "round_index": round_spec["round_index"],
        "parent_generation_id": round_spec["parent_generation_id"],
        "generation_id": generation_id,
        "model_id": model_id,
        "new_class_ids": owned_ids,
        "local_to_global": local_to_global,
        "model_path": relative_to_root(stored_weight),
        "model_sha256": stored_weight_hash,
        "selection_source": relative_to_root(stored_selection),
        "evaluation_source": relative_to_root(stored_evaluation),
        "calibration_source": relative_to_root(stored_calibration),
        "generation_registry": relative_to_root(generation_registry_path),
        "production_unchanged": True,
        "candidate_channel": generation_id,
    }
    atomic_json(registration_path, registration)
    raw_registry["models"].append(model)
    raw_registry["generations"].append(generation)
    raw_registry["channels"]["candidate"] = generation_id
    write_validated_registry(generation_registry_path, raw_registry)
    print(json.dumps(registration, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
