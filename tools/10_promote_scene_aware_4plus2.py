#!/usr/bin/env python3
"""Promote a frozen scene-aware 4+2 candidate into the x86/CUDA release metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from fair_agent.modules.incremental_round_registry import (  # noqa: E402
    DEFAULT_ROUND_REGISTRY,
    load_incremental_round_registry,
    rounds_through,
    select_round,
)


CLASS_NAMES: dict[int, str] = {}
BASE_IDS: tuple[int, ...] = ()
NEW_IDS: tuple[int, ...] = ()
SYSTEM_CALIBRATION_DATA_SCOPE = {
    "scene_sensor_model_training": "base_and_incremental_train_dev",
    "scene_sensor_model_recheck": "base_and_incremental_lock_frozen_model_only",
    "base_context_prior": "base_train_only",
    "incremental_context_prior": "incremental_train_only",
    "gate_selection": "mixed_dev_only",
}
PHASE_CONTRACT = {
    "base_learning": {
        "counted_as_incremental_learning": False,
        "detector_weights_updated": ["base_detector"],
        "training_data_scope": "base_dataset_only",
    },
    "incremental_learning": {
        "counted_as_incremental_learning": True,
        "detector_weights_updated": ["current_round_incremental_expert"],
        "training_data_scope": "incremental_dataset_only",
        "validation_data_scope": "incremental_dataset_only",
        "base_detector_weights_frozen": True,
        "historical_incremental_expert_weights_frozen": True,
    },
    "system_calibration": {
        "counted_as_incremental_learning": False,
        "detector_weights_updated": False,
        "base_detector_weights_frozen": True,
        "incremental_detector_weights_frozen": True,
        "data_scope": SYSTEM_CALIBRATION_DATA_SCOPE,
    },
    "joint_evaluation": {
        "counted_as_incremental_learning": False,
        "detector_weights_updated": False,
        "model_selection_allowed": False,
    },
}


def configure_round_contract(
    registry_path: Path, round_id: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    global CLASS_NAMES, BASE_IDS, NEW_IDS

    registry = load_incremental_round_registry(registry_path)
    target_round = select_round(registry, round_id)
    active_rounds = rounds_through(registry, round_id)
    CLASS_NAMES = {
        class_id: registry["class_names"][class_id]
        for class_id in target_round["learned_class_ids"]
    }
    BASE_IDS = tuple(registry["base"]["class_ids"])
    NEW_IDS = tuple(
        class_id
        for round_spec in active_rounds
        for class_id in round_spec["new_class_ids"]
    )
    return registry, target_round, active_rounds


def registered_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def copy_immutable_weight(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            raise FileExistsError(
                f"拒绝覆盖内容不同的 production 专家权重：{destination}"
            )
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def validate_round_evidence(
    payload: Mapping[str, Any],
    registry: Mapping[str, Any],
    target_round: Mapping[str, Any],
    active_rounds: list[dict[str, Any]],
) -> None:
    expected_rounds = list(registry["rounds"])
    rows = list(payload.get("rounds") or [])
    registrations = list(payload.get("registrations") or [])
    if (
        target_round["round_index"] != len(expected_rounds)
        or active_rounds != expected_rounds
        or payload.get("protocol") != registry["protocol_id"]
        or payload.get("evidence_status") != "complete"
        or payload.get("sequential_class_incremental_verified") is not True
        or payload.get("registered_lineage_verified") is not True
        or payload.get("all_rounds_competition_accepted") is not True
        or int(payload.get("distinct_round_count", 0)) < 2
        or int(payload.get("distinct_new_class_count", 0)) < 2
        or payload.get("scene_sensor_is_incremental_learner") is not False
        or len(rows) != len(expected_rounds)
        or len(registrations) != len(expected_rounds)
    ):
        raise ValueError("晋级必须提供完整的两轮顺序类别增量证据")
    for expected, row, registration in zip(
        expected_rounds, rows, registrations
    ):
        metrics = dict(row.get("metrics") or {})
        if (
            row.get("round_id") != expected["round_id"]
            or row.get("round_index") != expected["round_index"]
            or row.get("parent_generation_id")
            != expected["parent_generation_id"]
            or row.get("generation_id") != expected["generation_id"]
            or row.get("old_class_ids") != expected["old_class_ids"]
            or row.get("new_class_ids") != expected["new_class_ids"]
            or row.get("learned_class_ids") != expected["learned_class_ids"]
            or row.get("competition_accepted") is not True
            or set(metrics) != {"new_map50", "krr", "full_map50"}
            or registration.get("model_id")
            != expected["specialist"]["model_id"]
            or registration.get("generation_id") != expected["generation_id"]
        ):
            raise ValueError(
                f"顺序增量汇总与注册表不一致：{expected['round_id']}"
            )


def validate_registered_lineage(
    raw: Mapping[str, Any],
    registry: Mapping[str, Any],
    target_round: Mapping[str, Any],
    active_rounds: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    models = {str(row["id"]): row for row in raw.get("models", [])}
    generations = {
        str(row["id"]): row for row in raw.get("generations", [])
    }
    base_model_id = str(registry["base"]["model_id"])
    base_generation_id = str(registry["base"]["generation_id"])
    if base_model_id not in models or base_generation_id not in generations:
        raise ValueError("严格轮次缺少已冻结的 Base 模型或 Base 代际")
    expected_owners = {
        str(class_id): base_model_id for class_id in registry["base"]["class_ids"]
    }
    expected_members = [base_model_id]
    previous_generation_id = base_generation_id
    for round_spec in active_rounds:
        model_id = str(round_spec["specialist"]["model_id"])
        generation_id = str(round_spec["generation_id"])
        if model_id not in models or generation_id not in generations:
            raise ValueError(
                f"严格轮次尚未登记：{round_spec['round_id']}"
            )
        model = models[model_id]
        generation = generations[generation_id]
        local_to_global = {
            int(key): int(value)
            for key, value in dict(model.get("local_to_global") or {}).items()
        }
        evidence = dict(model.get("training_evidence") or {})
        model_path = registered_path(str(model.get("path") or ""))
        expected_members.append(model_id)
        for class_id in round_spec["new_class_ids"]:
            expected_owners[str(class_id)] = model_id
        generation_metrics = dict(generation.get("metrics") or {})
        if (
            model.get("role") != "class_incremental_expert"
            or model.get("incremental_mode") != "class_incremental"
            or set(int(value) for value in model.get("owns_classes", []))
            != set(round_spec["new_class_ids"])
            or local_to_global != round_spec["specialist"]["local_to_global"]
            or not model_path.is_file()
            or sha256_file(model_path) != model.get("sha256")
            or evidence.get("phase") != "incremental_learning"
            or evidence.get("round_id") != round_spec["round_id"]
            or evidence.get("parent_generation_id") != previous_generation_id
            or evidence.get("generation_id") != generation_id
            or evidence.get("training_data_scope")
            != "incremental_dataset_only"
            or evidence.get("validation_data_scope")
            != "incremental_dataset_only"
            or int(evidence.get("old_raw_image_count", -1)) != 0
            or int(evidence.get("old_raw_label_count", -1)) != 0
            or evidence.get("base_detector_weights_frozen") is not True
            or evidence.get("historical_incremental_expert_weights_frozen")
            is not True
            or model.get("acceptance", {}).get("competition_gates_passed")
            is not True
            or generation.get("parent") != previous_generation_id
            or set(int(value) for value in generation.get("classes", []))
            != set(round_spec["learned_class_ids"])
            or set(int(value) for value in generation.get("old_class_ids", []))
            != set(round_spec["old_class_ids"])
            or set(int(value) for value in generation.get("new_class_ids", []))
            != set(round_spec["new_class_ids"])
            or generation.get("class_owners") != expected_owners
            or generation.get("model_members") != expected_members
            or generation.get("status") not in {"registered_candidate", "active"}
            or any(
                key not in generation_metrics
                for key in ("new_map50", "krr", "full_map50")
            )
            or generation.get("acceptance", {}).get(
                "competition_gates_passed"
            )
            is not True
        ):
            raise ValueError(
                f"严格轮次登记或冻结证据不完整：{round_spec['round_id']}"
            )
        previous_generation_id = generation_id
    if (
        str(raw.get("channels", {}).get("candidate"))
        != target_round["generation_id"]
        or generations[target_round["generation_id"]].get("status")
        != "registered_candidate"
    ):
        raise ValueError("candidate 频道尚未指向待晋级的最后一轮冻结代际")
    return models, generations


def validate_round_evidence_identity(
    payload: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Any]],
    generations: Mapping[str, Mapping[str, Any]],
    active_rounds: list[dict[str, Any]],
) -> None:
    rows = {str(row["round_id"]): row for row in payload["rounds"]}
    registrations = {
        str(row["generation_id"]): row for row in payload["registrations"]
    }
    for round_spec in active_rounds:
        round_id = str(round_spec["round_id"])
        generation_id = str(round_spec["generation_id"])
        model_id = str(round_spec["specialist"]["model_id"])
        row = rows[round_id]
        registration = registrations[generation_id]
        generation = generations[generation_id]
        model = models[model_id]
        evidence = dict(model.get("training_evidence") or {})
        metrics = dict(row["metrics"])
        registered_metrics = dict(generation.get("metrics") or {})
        if (
            registration.get("model_id") != model_id
            or registered_path(registration.get("model_path") or "")
            != registered_path(model.get("path") or "")
            or registered_path(registration.get("evaluation_source") or "")
            != registered_path(evidence.get("evaluation_source") or "")
            or registration.get("generation_status")
            != generation.get("status")
            or any(
                abs(float(metrics[key]) - float(registered_metrics.get(key, -1.0)))
                > 1e-12
                for key in ("new_map50", "krr", "full_map50")
            )
        ):
            raise ValueError(f"顺序增量证据与已登记资产不一致：{round_id}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def mark_system_calibration(payload: dict[str, Any]) -> dict[str, Any]:
    payload.update(
        {
            "schema_version": max(2, int(payload.get("schema_version", 1))),
            "phase": "system_calibration",
            "counted_as_incremental_learning": False,
            "detector_weights_updated": False,
            "data_scope": dict(SYSTEM_CALIBRATION_DATA_SCOPE),
        }
    )
    return payload


def scoped_dev_report(source: str) -> str:
    note = (
        "本步骤属于 system_calibration，不计入 incremental_learning，"
        "也不更新 Base 或 Increment 检测器权重。"
    )
    if note in source:
        return source
    lines = source.splitlines()
    insert_at = 2 if lines and lines[0].startswith("#") else 0
    lines[insert_at:insert_at] = [note, ""]
    return "\n".join(lines).rstrip() + "\n"


def scoped_scene_report(source: str) -> str:
    note = (
        "该模型属于 system_calibration 功能模型，不计入竞赛口径的 "
        "incremental_learning，且不会更新任何检测器权重。"
    )
    if note in source:
        return source
    lines = source.splitlines()
    insert_at = 2 if lines and lines[0].startswith("#") else 0
    lines[insert_at:insert_at] = [note, ""]
    return "\n".join(lines).rstrip() + "\n"


def scoped_selection_report(source: str, note: str) -> str:
    if note in source:
        return source
    lines = source.splitlines()
    insert_at = 2 if lines and lines[0].startswith("#") else 0
    lines[insert_at:insert_at] = [note, ""]
    return "\n".join(lines).rstrip() + "\n"


def update_release_checksums(
    path: Path,
    models: Mapping[str, Mapping[str, Any]],
    active_model_ids: list[str],
) -> None:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        digest, separator, relative = value.partition("  ")
        if not separator or len(digest) != 64 or not relative:
            raise ValueError(f"模型校验清单格式非法：{line}")
        entries[relative] = digest
    for relative in list(entries):
        if (
            relative.startswith("production/incremental_detection/")
            and relative.endswith(".pt")
        ) or relative.startswith("candidates/incremental_detection/"):
            entries.pop(relative)
    models_root = (ROOT / "models").resolve()
    for model_id in active_model_ids:
        model = models[model_id]
        model_path = registered_path(str(model["path"]))
        try:
            relative = model_path.relative_to(models_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"production 模型必须位于 models/：{model_id}"
            ) from exc
        entries[relative] = str(model["sha256"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            f"{digest}  {relative}\n"
            for relative, digest in sorted(entries.items())
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def adapt_split_metrics(raw: Mapping[str, Any], base_image_count: int) -> dict[str, Any]:
    per_class = {str(key): dict(value) for key, value in raw["per_class"].items()}
    new_quality = {
        str(class_id): {
            key: per_class[str(class_id)][key]
            for key in (
                "class_name",
                "map50",
                "precision",
                "recall",
                "f1",
                "tp",
                "fp",
                "targets",
                "threshold",
                "max_scene_penalty",
            )
        }
        for class_id in NEW_IDS
    }
    false_activation = {
        str(class_id): {
            "class_name": per_class[str(class_id)]["class_name"],
            "negative_image_count": per_class[str(class_id)]["negative_image_count"],
            "false_activation_image_count": per_class[str(class_id)][
                "false_activation_image_count"
            ],
            "false_activation_rate": per_class[str(class_id)][
                "false_activation_rate"
            ],
        }
        for class_id in NEW_IDS
    }
    return {
        "image_count": int(raw["image_count"]),
        "base_image_count": int(base_image_count),
        "base_map50": float(raw["base_map50"]),
        "base_per_class_ap50": dict(raw["base_per_class_ap50"]),
        "old_map50_before": float(raw["old_map50_before"]),
        "old_map50_after": float(raw["old_map50_after"]),
        "krr": float(raw["krr"]),
        "old_prediction_equivalent": False,
        "new_map50": float(raw["new_map50"]),
        "new_per_class_ap50": dict(raw["new_per_class_ap50"]),
        "full_map50": float(raw["full_map50"]),
        "full_per_class_ap50": {
            str(class_id): float(per_class[str(class_id)]["map50"])
            for class_id in CLASS_NAMES
        },
        "new_class_quality": new_quality,
        "false_activation": false_activation,
        "all_class_quality": per_class,
        "overall_quality": dict(raw["overall"]),
        "new_class_summary": dict(raw["new_classes"]),
        "prediction_counts": dict(raw["prediction_counts"]),
    }


def context_gate(
    source: Path,
    penalties: Mapping[int, float],
    scope: str,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "policy": "soft_threshold_penalty",
        "hard_routing": False,
        "learning_data_scope": scope,
        "dimensions": ["scene"],
        "max_threshold_penalty": max(float(value) for value in penalties.values()),
        "max_threshold_penalties": {
            str(key): float(value) for key, value in penalties.items()
        },
        "prior_source": rel(source),
        "prior_sha256": sha256_file(source),
        "online_input": "scene_sensor_net_probabilities",
    }


def diagnostics_markdown(
    baseline: Mapping[str, Any],
    lock: Mapping[str, Any],
    thresholds: Mapping[int, float],
    penalties: Mapping[int, float],
) -> str:
    lines = [
        "# strict 4+2 场景感知 production 运行点诊断",
        "",
        "该运行点使用 Scene-SensorNet 的实际场景概率与六类逐类训练先验做软阈值惩罚。",
        "线上不读取文件名或真值标签，不做场景硬路由；Base 与 Increment 仍对每张图执行。",
        "",
        "## lock 汇总",
        "",
        "| 运行点 | Base mAP50 | New-mAP50 | KRR | 六类 TP | 六类 FP | 六类 precision | 新类 TP | 新类 FP | 新类 precision | 误激活图像 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in (("旧运行点（含运行时 NMS）", baseline), ("场景感知 production", lock)):
        lines.append(
            f"| {name} | {row['base_map50']:.6f} | {row['new_map50']:.6f} | "
            f"{row['krr']:.6f} | {row['overall']['tp']} | {row['overall']['fp']} | "
            f"{row['overall']['precision']:.6f} | {row['new_classes']['tp']} | "
            f"{row['new_classes']['fp']} | {row['new_classes']['precision']:.6f} | "
            f"{row['overall']['false_activation_image_count']} / {row['image_count']} |"
        )
    lines.extend(
        [
            "",
            "## 六类逐类结果",
            "",
            "| 类别 | 基础阈值 | 最大场景惩罚 | mAP50 | TP | FP | precision | recall | 误激活率 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for class_id in CLASS_NAMES:
        row = lock["per_class"][str(class_id)]
        lines.append(
            f"| {class_id} {CLASS_NAMES[class_id]} | {thresholds[class_id]:.2f} | "
            f"{penalties[class_id]:.2f} | {row['map50']:.6f} | {row['tp']} | "
            f"{row['fp']} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['false_activation_image_count']} / {row['negative_image_count']} "
            f"= {row['false_activation_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "比赛硬门禁仍只使用 Base mAP50、New-mAP50 和 KRR；precision、FP 与误激活率为运行质量诊断。",
            "候选参数只由 mixed dev 选择，随后冻结并一次性复核 mixed lock。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="晋级已冻结并通过 lock 的场景感知 4+2 候选。")
    parser.add_argument(
        "--round-registry", type=Path, default=ROOT / DEFAULT_ROUND_REGISTRY
    )
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--lock-result", type=Path, required=True)
    parser.add_argument("--dev-search", type=Path, required=True)
    parser.add_argument("--dev-report", type=Path, required=True)
    parser.add_argument("--round-evidence", type=Path, required=True)
    args = parser.parse_args()
    round_registry, target_round, active_rounds = configure_round_contract(
        args.round_registry, args.round_id
    )
    round_registry_ref = rel(Path(round_registry["path"]))

    candidate_path = args.candidate.expanduser().resolve()
    lock_result_path = args.lock_result.expanduser().resolve()
    dev_search_path = args.dev_search.expanduser().resolve()
    dev_report_path = args.dev_report.expanduser().resolve()
    round_evidence_path = args.round_evidence.expanduser().resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    result = json.loads(lock_result_path.read_text(encoding="utf-8"))
    dev_search = json.loads(dev_search_path.read_text(encoding="utf-8"))
    dev_report = dev_report_path.read_text(encoding="utf-8")
    round_evidence = json.loads(round_evidence_path.read_text(encoding="utf-8"))
    validate_round_evidence(
        round_evidence, round_registry, target_round, active_rounds
    )
    lineage_keys = (
        "round_id",
        "round_index",
        "parent_generation_id",
        "generation_id",
    )
    if (
        candidate.get("selection_source") != "mixed_dev_only"
        or candidate.get("phase") not in (None, "system_calibration")
        or result.get("phase") not in (None, "joint_evaluation")
        or dev_search.get("phase") not in (None, "system_calibration")
        or result.get("competition_accepted") is not True
        or result.get("candidate_frozen_before_lock_labels") is not True
        or result.get("candidate", {}).get("candidate_id") != candidate.get("candidate_id")
        or not all(result.get("score_gates", {}).values())
        or any(candidate.get(key) != target_round[key] for key in lineage_keys)
        or any(result.get(key) != target_round[key] for key in lineage_keys)
        or any(dev_search.get(key) != target_round[key] for key in lineage_keys)
    ):
        raise ValueError("候选未通过冻结 dev→lock 晋级契约")

    generations_path = ROOT / "models" / "generations.json"
    generations = json.loads(generations_path.read_text(encoding="utf-8"))
    previous_production_id = str(generations["channels"]["production"])
    registered_models, registered_generations = validate_registered_lineage(
        generations, round_registry, target_round, active_rounds
    )
    validate_round_evidence_identity(
        round_evidence,
        registered_models,
        registered_generations,
        active_rounds,
    )
    candidate["round_registry"] = round_registry_ref
    result["round_registry"] = round_registry_ref
    dev_search["round_registry"] = round_registry_ref
    round_evidence["round_registry"] = round_registry_ref
    round_evidence["generation_registry"] = "models/generations.json"

    mark_system_calibration(candidate)
    for dev_candidate in dict(dev_search.get("candidates") or {}).values():
        mark_system_calibration(dev_candidate)
    dev_search.update(
        {
            "schema_version": max(2, int(dev_search.get("schema_version", 1))),
            "phase": "system_calibration",
            "counted_as_incremental_learning": False,
            "detector_weights_updated": False,
            "data_scope": dict(SYSTEM_CALIBRATION_DATA_SCOPE),
        }
    )
    result.update(
        {
            "schema_version": max(2, int(result.get("schema_version", 1))),
            "phase": "joint_evaluation",
            "counted_as_incremental_learning": False,
            "detector_weights_updated": False,
            "model_selection_allowed": False,
            "candidate": candidate,
        }
    )

    scene_metrics_path = ROOT / "models" / "context" / "scene_sensor_metrics.json"
    scene_report_path = ROOT / "models" / "context" / "scene_sensor_report.md"
    old_scene_metrics_hash = sha256_file(scene_metrics_path)
    scene_metrics = json.loads(scene_metrics_path.read_text(encoding="utf-8"))
    scene_metrics.update(
        {
            "schema_version": max(2, int(scene_metrics.get("schema_version", 1))),
            "phase": "system_calibration",
            "counted_as_incremental_learning": False,
            "detector_weights_updated": False,
            "data_scope": {
                "training": "base_and_incremental_train",
                "model_selection": "base_and_incremental_dev",
                "functional_model_recheck": "base_and_incremental_lock",
            },
        }
    )
    atomic_json(scene_metrics_path, scene_metrics)
    scene_report_path.write_text(
        scoped_scene_report(scene_report_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    new_scene_metrics_hash = sha256_file(scene_metrics_path)

    production = ROOT / "models" / "production" / "incremental_detection"
    evidence = production / "evidence"
    production_expert_root = production / "round_experts"
    for round_spec in active_rounds:
        model_id = str(round_spec["specialist"]["model_id"])
        model = registered_models[model_id]
        source_weight = registered_path(model["path"])
        production_weight = production_expert_root / f"{model_id}.pt"
        copy_immutable_weight(source_weight, production_weight)
        model["path"] = rel(production_weight)
        model["sha256"] = sha256_file(production_weight)
    base_prior_path = production / "base_context_prior.json"
    incremental_prior_path = production / "incremental_context_prior.json"
    atomic_json(base_prior_path, candidate["base_context_prior"])
    atomic_json(incremental_prior_path, candidate["incremental_context_prior"])
    atomic_json(evidence / "scene_aware_candidate.json", candidate)
    atomic_json(evidence / "scene_aware_lock_recheck.json", result)
    atomic_json(evidence / "scene_aware_dev_search.json", dev_search)
    sequential_evidence_path = evidence / "sequential_round_evidence.json"
    atomic_json(sequential_evidence_path, round_evidence)
    round_evidence_report = round_evidence_path.with_suffix(".md")
    if round_evidence_report.is_file():
        (evidence / "sequential_round_evidence.md").write_text(
            round_evidence_report.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (evidence / "scene_aware_dev_search.md").write_text(
        scoped_dev_report(dev_report), encoding="utf-8"
    )
    round_selection_rows = []
    for round_spec in active_rounds:
        model_id = str(round_spec["specialist"]["model_id"])
        training_evidence = dict(
            registered_models[model_id].get("training_evidence") or {}
        )
        round_evidence_root = evidence / "rounds" / str(round_spec["round_id"])
        selection_source = registered_path(training_evidence["selection_source"])
        evaluation_source = registered_path(training_evidence["evaluation_source"])
        calibration_source = registered_path(training_evidence["calibration_source"])
        selection_payload = json.loads(
            selection_source.read_text(encoding="utf-8")
        )
        evaluation_payload = json.loads(
            evaluation_source.read_text(encoding="utf-8")
        )
        calibration_payload = json.loads(
            calibration_source.read_text(encoding="utf-8")
        )
        atomic_json(round_evidence_root / "incremental_selection.json", selection_payload)
        atomic_json(round_evidence_root / "metrics.json", evaluation_payload)
        atomic_json(round_evidence_root / "calibration.json", calibration_payload)
        selection_report = selection_source.with_suffix(".md")
        if selection_report.is_file():
            (round_evidence_root / "incremental_selection.md").write_text(
                selection_report.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        round_selection_rows.append(
            {
                "round_id": round_spec["round_id"],
                "round_index": round_spec["round_index"],
                "parent_generation_id": round_spec["parent_generation_id"],
                "generation_id": round_spec["generation_id"],
                "model_id": model_id,
                "new_class_ids": round_spec["new_class_ids"],
                "selection": selection_payload,
                "production_evidence_root": rel(round_evidence_root),
            }
        )
    atomic_json(
        evidence / "incremental_selection.json",
        {
            "schema_version": 3,
            "phase": "incremental_learning",
            "counted_as_incremental_learning": True,
            "detector_weights_updated": False,
            "component": "sequential_incremental_detector_candidate_selection",
            "training_data_scope": "incremental_dataset_only",
            "validation_data_scope": "incremental_dataset_only",
            "base_detector_weights_frozen": True,
            "historical_incremental_expert_weights_frozen": True,
            "rounds": round_selection_rows,
        },
    )
    incremental_selection_lines = [
        "# 严格顺序增量专家选模",
        "",
        "每轮只读取当轮 Increment dev；Base 与历史专家权重冻结，lock 不参与选模。",
        "",
        "| 轮次 | 父代 | 子代 | 专家 | 新类 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in round_selection_rows:
        incremental_selection_lines.append(
            f"| {row['round_id']} | {row['parent_generation_id']} | "
            f"{row['generation_id']} | {row['model_id']} | {row['new_class_ids']} |"
        )
    (evidence / "incremental_selection.md").write_text(
        "\n".join(incremental_selection_lines) + "\n", encoding="utf-8"
    )
    selection_specs = (
        (
            "base_selection",
            "base_learning",
            False,
            "base_detector_candidate_selection",
            "本步骤属于 base_learning，不计入 incremental_learning，且选模本身不更新检测器权重。",
        ),
    )
    for stem, phase, counted, component, note in selection_specs:
        selection_path = evidence / f"{stem}.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection.update(
            {
                "schema_version": max(2, int(selection.get("schema_version", 1))),
                "phase": phase,
                "counted_as_incremental_learning": counted,
                "detector_weights_updated": False,
                "component": component,
            }
        )
        atomic_json(selection_path, selection)
        selection_report_path = evidence / f"{stem}.md"
        selection_report_path.write_text(
            scoped_selection_report(
                selection_report_path.read_text(encoding="utf-8"), note
            ),
            encoding="utf-8",
        )

    thresholds = {
        int(key): float(value) for key, value in candidate["thresholds"].items()
    }
    penalties = {
        int(key): float(value)
        for key, value in candidate["max_threshold_penalties"].items()
    }
    base_thresholds = {class_id: thresholds[class_id] for class_id in BASE_IDS}
    new_thresholds = {class_id: thresholds[class_id] for class_id in NEW_IDS}
    base_penalties = {class_id: penalties[class_id] for class_id in BASE_IDS}
    new_penalties = {class_id: penalties[class_id] for class_id in NEW_IDS}
    base_gate = context_gate(base_prior_path, base_penalties, "base_train_only")
    incremental_gate = context_gate(
        incremental_prior_path, new_penalties, "incremental_train_only"
    )
    now = datetime.now().astimezone().isoformat()
    run_id = "strict-4plus2-scene-aware-" + datetime.now().astimezone().strftime(
        "%Y%m%d-%H%M%S"
    )
    old_metrics = json.loads((production / "metrics.json").read_text(encoding="utf-8"))
    base_dev_count = int(dev_search.get("splits", {}).get("base_dev") or 0)
    base_lock_count = int(result.get("splits", {}).get("base_lock") or 0)
    if not base_dev_count or not base_lock_count:
        raise ValueError("候选缺少注册表驱动的 Base dev/lock 数量")
    dev_metrics = adapt_split_metrics(candidate["dev_metrics"], base_dev_count)
    lock_metrics = adapt_split_metrics(result["lock"], base_lock_count)
    metrics = {
        **old_metrics,
        "schema_version": 5,
        "run_id": run_id,
        "created_at": now,
        "protocol": "strict-4plus2-parallel-specialist-class-specific-scene-gate",
        "round_registry": round_registry_ref,
        "round_id": target_round["round_id"],
        "round_index": target_round["round_index"],
        "parent_generation_id": target_round["parent_generation_id"],
        "generation_id": target_round["generation_id"],
        "phase": "joint_evaluation",
        "counted_as_incremental_learning": False,
        "detector_weights_updated": False,
        "model_selection_allowed": False,
        "incremental_learning_data_scope": "incremental_dataset_only",
        "inference": {
            **dict(old_metrics.get("inference") or {}),
            "fusion": "fixed_class_owners_with_scene_gate_and_cross_class_suppression",
            "fusion_iou": float(candidate.get("fusion_iou", 0.60)),
            "context_gate_enabled": True,
            "hard_scene_routing": False,
            "per_class_thresholds": {
                str(key): value for key, value in thresholds.items()
            },
            "max_threshold_penalties": {
                str(key): value for key, value in penalties.items()
            },
            "conflict_policy": dict(candidate["conflict_policy"]),
            "cross_class_suppression": dict(
                candidate.get("cross_class_suppression") or {"enabled": False}
            ),
        },
        "dev": dev_metrics,
        "lock": lock_metrics,
        "score_gates": dict(result["score_gates"]),
        "competition_accepted": True,
        "context_accepted": bool(old_metrics.get("context_accepted", True)),
        "deployment_accepted": True,
        "accepted": True,
        "predictions_frozen_before_lock_labels": True,
        "candidate_frozen_before_lock_labels": True,
    }
    metrics.pop("learning_data_scope", None)
    atomic_json(production / "metrics.json", metrics)

    calibration = {
        "schema_version": 4,
        "phase": "system_calibration",
        "counted_as_incremental_learning": False,
        "detector_weights_updated": False,
        "data_scope": dict(SYSTEM_CALIBRATION_DATA_SCOPE),
        "source_split": "mixed_dev_only",
        "selection_metric": "precision_subject_to_map50_krr_guardrails_with_scene_probabilities",
        "deployment_policy": "competition_map50_dev_calibrated",
        "per_class_thresholds": {
            str(class_id): new_thresholds[class_id] for class_id in NEW_IDS
        },
        "max_threshold_penalties": {
            str(class_id): new_penalties[class_id] for class_id in NEW_IDS
        },
        "per_class": {
            str(class_id): {
                "class_name": CLASS_NAMES[class_id],
                "selected": {
                    key: candidate["dev_metrics"]["per_class"][str(class_id)][key]
                    for key in (
                        "threshold",
                        "map50",
                        "precision",
                        "recall",
                        "f1",
                        "tp",
                        "fp",
                        "targets",
                        "max_scene_penalty",
                    )
                },
                "curve": [],
            }
            for class_id in NEW_IDS
        },
        "selected_new_map50": float(candidate["dev_metrics"]["new_map50"]),
        "context_policy": {
            "policy": "class_specific_scene_soft_threshold_penalty",
            "hard_routing": False,
            "online_input": "scene_sensor_net_probabilities",
            "base_prior_source": rel(base_prior_path),
            "incremental_prior_source": rel(incremental_prior_path),
        },
        "selection_constraints": dict(candidate["selection_constraints"]),
    }
    atomic_json(production / "calibration.json", calibration)

    new_quality = {
        class_id: lock_metrics["new_class_quality"][str(class_id)]
        for class_id in NEW_IDS
    }
    new_false_activation = {
        class_id: lock_metrics["false_activation"][str(class_id)]
        for class_id in NEW_IDS
    }
    advisory_warnings = []
    if min(float(row["precision"]) for row in new_quality.values()) < 0.90:
        advisory_warnings.append("lock_precision_below_0_90")
    if max(
        float(row["false_activation_rate"])
        for row in new_false_activation.values()
    ) > 0.05:
        advisory_warnings.append("false_activation_rate_above_0_05")
    profile = json.loads((production / "profile.json").read_text(encoding="utf-8"))
    profile.update(
        {
            "schema_version": max(2, int(profile.get("schema_version", 1))),
            "run_id": run_id,
            "deployment": "multi_specialist",
            "phase_contract": PHASE_CONTRACT,
            "round_registry": round_registry_ref,
            "round_id": target_round["round_id"],
            "round_index": target_round["round_index"],
            "parent_generation_id": target_round["parent_generation_id"],
            "generation_id": target_round["generation_id"],
            "new_global_ids": list(NEW_IDS),
            "new_classes": {
                str(class_id): CLASS_NAMES[class_id] for class_id in NEW_IDS
            },
            "specialist_models": [
                {
                    "round_id": row["round_id"],
                    "model_id": row["specialist"]["model_id"],
                    "weight": registered_models[
                        str(row["specialist"]["model_id"])
                    ]["path"],
                    "sha256": registered_models[
                        str(row["specialist"]["model_id"])
                    ]["sha256"],
                    "global_class_ids": list(row["new_class_ids"]),
                    "local_to_global": {
                        str(key): value
                        for key, value in row["specialist"][
                            "local_to_global"
                        ].items()
                    },
                    "activation_thresholds": {
                        str(class_id): new_thresholds[class_id]
                        for class_id in row["new_class_ids"]
                    },
                    "calibration_sources": {
                        str(class_id): rel(production / "calibration.json")
                        for class_id in row["new_class_ids"]
                    },
                    "new_map50": sum(
                        float(new_quality[class_id]["map50"])
                        for class_id in row["new_class_ids"]
                    )
                    / len(row["new_class_ids"]),
                }
                for row in active_rounds
            ],
            "runtime_registry_source": "models/generations.json",
            "agent_structure": {
                **dict(profile["agent_structure"]),
                "architecture": "parallel_base_registered_incremental_experts",
                "new_class_owner": "registered_incremental_specialists",
                "scene_soft_gating": True,
                "scene_hard_routing": False,
                "label_aware_routing": False,
                "filename_class_routing": False,
            },
            "activation_thresholds": {
                str(key): value for key, value in new_thresholds.items()
            },
            "calibration_source": rel(production / "calibration.json"),
            "calibration_sources": {
                str(key): rel(production / "calibration.json")
                for key in new_thresholds
            },
            "base_activation_thresholds": {
                str(key): value for key, value in base_thresholds.items()
            },
            "evaluation_semantics": "all_owners_scene_probability_gated_on_mixed_dev_then_frozen_mixed_lock_v2",
            "rechecked_at": now,
            "base_test_map50": lock_metrics["base_map50"],
            "base_dev_map50": dev_metrics["base_map50"],
            "old_map50_before": lock_metrics["old_map50_before"],
            "old_map50_after": lock_metrics["old_map50_after"],
            "new_map50": lock_metrics["new_map50"],
            "full_map50": lock_metrics["full_map50"],
            "krr": lock_metrics["krr"],
            "lock_precision": min(
                float(row["precision"]) for row in new_quality.values()
            ),
            "lock_recall": min(float(row["recall"]) for row in new_quality.values()),
            "lock_false_activation_rate": max(
                float(row["false_activation_rate"])
                for row in new_false_activation.values()
            ),
            "lock_precision_by_class": {
                str(key): value["precision"] for key, value in new_quality.items()
            },
            "lock_recall_by_class": {
                str(key): value["recall"] for key, value in new_quality.items()
            },
            "lock_false_activation_rate_by_class": {
                str(key): value["false_activation_rate"]
                for key, value in new_false_activation.items()
            },
            "fusion_policy": {
                "enabled": bool(candidate["conflict_policy"].get("enabled")),
                "iou": float(candidate["conflict_policy"]["iou"]),
                "base_confidence": float(
                    candidate["conflict_policy"]["base_confidence"]
                ),
                "incremental_margin": float(
                    candidate["conflict_policy"]["specialist_margin"]
                ),
                "preserve_base_class_owners": True,
                "class_aware_nms_iou": float(candidate.get("fusion_iou", 0.60)),
                "cross_class_suppression": dict(
                    candidate.get("cross_class_suppression")
                    or {"enabled": False}
                ),
            },
            "context_prior": dict(candidate["incremental_context_prior"]),
            "context_gate": incremental_gate,
            "context_prior_source": rel(incremental_prior_path),
            "context_prior_sha256": sha256_file(incremental_prior_path),
            "base_context_prior": dict(candidate["base_context_prior"]),
            "base_context_gate": base_gate,
            "base_context_prior_source": rel(base_prior_path),
            "base_context_prior_sha256": sha256_file(base_prior_path),
            "diagnostic_warnings": advisory_warnings,
        }
    )
    for legacy_key in (
        "specialist_weight",
        "specialist_sha256",
        "specialist_local_to_global",
    ):
        profile.pop(legacy_key, None)
    atomic_json(production / "profile.json", profile)
    active_profile = ROOT / "models" / "profiles" / "incremental-detection" / "active.json"
    atomic_json(active_profile, profile)

    registry_path = ROOT / "models" / "profiles" / "registry.json"
    profile_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registered = profile_registry["verified_profiles"][0]
    registered.update(
        {
            "run_id": run_id,
            "generation_id": target_round["generation_id"],
            "round_registry": round_registry_ref,
            "round_id": target_round["round_id"],
            "sequential_class_incremental_verified": True,
            "sequential_round_evidence_source": rel(
                sequential_evidence_path
            ),
            "specialist_models": list(profile["specialist_models"]),
            "phase_contract": PHASE_CONTRACT,
            "evaluation_semantics": profile["evaluation_semantics"],
            "rechecked_at": now,
            "base_test_map50": lock_metrics["base_map50"],
            "new_map50": lock_metrics["new_map50"],
            "new_per_class_ap50": dict(lock_metrics["new_per_class_ap50"]),
            "krr": lock_metrics["krr"],
            "activation_thresholds": {
                str(key): value for key, value in new_thresholds.items()
            },
            "base_activation_thresholds": {
                str(key): value for key, value in base_thresholds.items()
            },
            "scene_soft_gating": True,
            "diagnostic_warnings": advisory_warnings,
        }
    )
    atomic_json(registry_path, profile_registry)

    models = {str(row["id"]): row for row in generations["models"]}
    base_model = models[str(round_registry["base"]["model_id"])]
    base_model.update(
        {
            "per_class_thresholds": {
                str(key): value for key, value in base_thresholds.items()
            },
            "context_prior": dict(candidate["base_context_prior"]),
            "context_gate": base_gate,
        }
    )
    for round_spec in active_rounds:
        model_id = str(round_spec["specialist"]["model_id"])
        owned_ids = [int(value) for value in round_spec["new_class_ids"]]
        incremental_model = models[model_id]
        incremental_model.update(
            {
                "per_class_thresholds": {
                    str(class_id): new_thresholds[class_id]
                    for class_id in owned_ids
                },
                "context_prior": dict(candidate["incremental_context_prior"]),
                "context_gate": incremental_gate,
                "metrics": {
                    "new_map50": sum(
                        float(new_quality[class_id]["map50"])
                        for class_id in owned_ids
                    )
                    / len(owned_ids),
                    "per_class": {
                        str(class_id): {
                            "map50": new_quality[class_id]["map50"],
                            "precision": new_quality[class_id]["precision"],
                            "recall": new_quality[class_id]["recall"],
                            "threshold": new_thresholds[class_id],
                            "max_scene_penalty": new_penalties[class_id],
                        }
                        for class_id in owned_ids
                    },
                },
                "deployment_metrics": {
                    "evaluation_semantics": profile["evaluation_semantics"],
                    "overall_precision": result["lock"]["overall"]["precision"],
                    "new_class_precision": result["lock"]["new_classes"]["precision"],
                    "false_activation_rate_by_class": {
                        str(class_id): new_false_activation[class_id][
                            "false_activation_rate"
                        ]
                        for class_id in owned_ids
                    },
                    "diagnostic_only": True,
                },
                "acceptance": {
                    "official_score_gates_passed": True,
                    "competition_gates_passed": True,
                    "deployment_quality_gates_passed": not advisory_warnings,
                    "integrity_gates_passed": True,
                    "passed": True,
                    "diagnostic_warnings": advisory_warnings,
                },
                "status": "active",
            }
        )
    generation_rows = {str(row["id"]): row for row in generations["generations"]}
    base_generation = generation_rows[round_registry["base"]["generation_id"]]
    base_generation["metrics"].update(
        {
            "base_dev_map50": dev_metrics["base_map50"],
            "base_test_map50": lock_metrics["base_map50"],
            "raw_base_test_map50": result["baseline"]["base_map50"],
            "scene_soft_gating": True,
        }
    )
    generation = generation_rows[target_round["generation_id"]]
    generation.update(
        {
            "run_id": run_id,
            "status": "active",
            "phase_contract": PHASE_CONTRACT,
            "sequential_round_evidence": True,
            "sequential_round_evidence_source": rel(
                sequential_evidence_path
            ),
            "metrics": {
                "base_test_map50": lock_metrics["base_map50"],
                "old_map50_before": lock_metrics["old_map50_before"],
                "old_map50_after": lock_metrics["old_map50_after"],
                "new_map50": lock_metrics["new_map50"],
                "krr": lock_metrics["krr"],
                "full_map50": lock_metrics["full_map50"],
                "old_prediction_equivalent": False,
                "per_class": {
                    str(class_id): {
                        "map50": new_quality[class_id]["map50"],
                        "precision": new_quality[class_id]["precision"],
                        "recall": new_quality[class_id]["recall"],
                        "false_activation_rate": new_false_activation[class_id][
                            "false_activation_rate"
                        ],
                        "negative_image_count": new_false_activation[class_id][
                            "negative_image_count"
                        ],
                        "false_activation_image_count": new_false_activation[class_id][
                            "false_activation_image_count"
                        ],
                        "threshold": new_thresholds[class_id],
                        "max_scene_penalty": new_penalties[class_id],
                    }
                    for class_id in NEW_IDS
                },
                "cumulative_per_class": dict(lock_metrics["full_per_class_ap50"]),
                "overall_precision": result["lock"]["overall"]["precision"],
                "new_class_precision": result["lock"]["new_classes"]["precision"],
                "overall_false_activation_image_count": result["lock"]["overall"][
                    "false_activation_image_count"
                ],
                "image_count": result["lock"]["image_count"],
                "base_test_image_count": base_lock_count,
                "new_class_image_count": int(
                    result.get("splits", {}).get("incremental_lock") or 0
                ),
                "specialist_count": len(active_rounds),
            },
            "acceptance": {
                "core_metrics_passed": True,
                "competition_gates_passed": True,
                "deployment_quality_gates_passed": not advisory_warnings,
                "deployment_recheck_passed": True,
                "status": "passed",
                "diagnostic_warnings": advisory_warnings,
            },
            "lock_recheck": {
                "status": "completed",
                "rechecked_at": now,
                "evaluation_semantics": profile["evaluation_semantics"],
                "input_image_count": result["lock"]["image_count"],
                "label_aware_routing": False,
                "filename_class_routing": False,
                "scene_hard_routing": False,
                "scene_soft_gating": True,
                "activation_threshold_source": "mixed_dev_only",
                "context_gate_enabled": True,
                "fusion_inputs": "fixed_class_owners_with_runtime_class_aware_nms",
                "candidate_frozen_before_lock_labels": True,
            },
        }
    )
    if previous_production_id != target_round["generation_id"]:
        previous_generation = generation_rows[previous_production_id]
        if previous_production_id != round_registry["base"]["generation_id"]:
            previous_generation["status"] = "retired_baseline"
        target_members = set(generation["model_members"])
        for model_id in previous_generation.get(
            "model_members", previous_generation["class_owners"].values()
        ):
            if (
                str(model_id) not in target_members
                and models[str(model_id)].get("role")
                in {"class_incremental_expert", "target_incremental_expert"}
            ):
                models[str(model_id)]["status"] = "retired_baseline"
    generations["channels"]["production"] = target_round["generation_id"]
    generations["channels"]["candidate"] = target_round["generation_id"]
    atomic_json(generations_path, generations)
    update_release_checksums(
        ROOT / "models" / "SHA256SUMS.txt",
        models,
        list(generation["model_members"]),
    )

    manifest_path = ROOT / "models" / "manifest.json"
    old_manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "release": (
                "competition-4plus2-sequential-scene-aware-v3-"
                + datetime.now().astimezone().strftime("%Y%m%d")
            ),
            "created_at": now,
            "run_id": run_id,
            "generation_id": target_round["generation_id"],
            "round_registry": round_registry_ref,
            "sequential_class_incremental_verified": True,
            "sequential_round_evidence_source": rel(
                sequential_evidence_path
            ),
            "phase_contract": PHASE_CONTRACT,
        }
    )
    manifest["base_model"].update(
        {
            "training_phase": "base_learning",
            "counted_as_incremental_learning": False,
            "raw_base_test_map50": result["baseline"]["base_map50"],
            "base_dev_map50": dev_metrics["base_map50"],
            "base_test_map50": lock_metrics["base_map50"],
            "per_class_thresholds": {
                str(key): value for key, value in base_thresholds.items()
            },
            "max_threshold_penalties": {
                str(key): value for key, value in base_penalties.items()
            },
            "context_prior_source": rel(base_prior_path),
        }
    )
    for functional in manifest["functional_models"]:
        if functional["id"] == "scene_sensor_net_v1":
            functional.update(
                {
                    "competition_phase": "system_calibration",
                    "counted_as_incremental_learning": False,
                    "detector_weights_updated": False,
                    "metrics_sha256": new_scene_metrics_hash,
                }
            )
        elif functional["id"] == "four_class_base_detector":
            functional.update(
                {
                    "competition_phase": "base_learning",
                    "counted_as_incremental_learning": False,
                    "base_test_map50": lock_metrics["base_map50"],
                    "raw_base_test_map50": result["baseline"]["base_map50"],
                    "scene_soft_gating": True,
                }
            )
        elif functional["id"] == "incremental_model_bank_v1":
            functional.update(
                {
                    "competition_phase": "incremental_learning",
                    "counted_as_incremental_learning": True,
                    "model_count": len(active_rounds),
                    "protocol_count": len(active_rounds),
                    "passed_protocols": len(active_rounds),
                    "total_protocols": len(active_rounds),
                    "true_class_incremental_verified": True,
                }
            )
    manifest_protocols = []
    for round_spec in active_rounds:
        model_id = str(round_spec["specialist"]["model_id"])
        registered_model = registered_models[model_id]
        owned_ids = [int(value) for value in round_spec["new_class_ids"]]
        manifest_protocols.append(
            {
                "protocol": model_id,
                "display_name": "{} 增量专家".format(round_spec["round_id"]),
                "task_type": "incremental_object_detection",
                "incremental_mode": "class_incremental",
                "learning_data_scope": "incremental_dataset_only",
                "class_names": {
                    str(class_id): CLASS_NAMES[class_id]
                    for class_id in owned_ids
                },
                "global_class_ids": owned_ids,
                "base_model_id": round_registry["base"]["model_id"],
                "base_class_ids": list(BASE_IDS),
                "round_id": round_spec["round_id"],
                "round_index": round_spec["round_index"],
                "parent_generation_id": round_spec["parent_generation_id"],
                "generation_id": round_spec["generation_id"],
                "activation_thresholds": {
                    str(class_id): new_thresholds[class_id]
                    for class_id in owned_ids
                },
                "calibration_source": rel(production / "calibration.json"),
                "calibration_sources": {
                    str(class_id): rel(production / "calibration.json")
                    for class_id in owned_ids
                },
                "calibration_sha256": sha256_file(
                    production / "calibration.json"
                ),
                "metrics_source": rel(production / "metrics.json"),
                "metrics_sha256": sha256_file(production / "metrics.json"),
                "evidence_level": "verified",
                "available": True,
                "context_prior": dict(candidate["incremental_context_prior"]),
                "context_gate": incremental_gate,
                "path": registered_model["path"],
                "sha256": registered_model["sha256"],
                "evaluation_semantics": profile["evaluation_semantics"],
                "rechecked_at": now,
                "base_test_map50": lock_metrics["base_map50"],
                "old_map50_before": lock_metrics["old_map50_before"],
                "old_map50_after": lock_metrics["old_map50_after"],
                "new_map50": sum(
                    float(new_quality[class_id]["map50"])
                    for class_id in owned_ids
                )
                / len(owned_ids),
                "new_per_class_ap50": {
                    str(class_id): new_quality[class_id]["map50"]
                    for class_id in owned_ids
                },
                "full_map50": lock_metrics["full_map50"],
                "krr": lock_metrics["krr"],
                "lock_precision_by_class": {
                    str(class_id): new_quality[class_id]["precision"]
                    for class_id in owned_ids
                },
                "lock_recall_by_class": {
                    str(class_id): new_quality[class_id]["recall"]
                    for class_id in owned_ids
                },
                "false_activation_rate_by_class": {
                    str(class_id): new_false_activation[class_id][
                        "false_activation_rate"
                    ]
                    for class_id in owned_ids
                },
                "competition_accepted": True,
                "deployment_accepted": True,
                "acceptance": "passed",
                "diagnostic_warnings": advisory_warnings,
                "training_phase": "incremental_learning",
                "counted_as_incremental_learning": True,
                "training_data_scope": "incremental_dataset_only",
                "base_detector_weights_frozen": True,
                "historical_incremental_expert_weights_frozen": True,
                "system_calibration_source": rel(production / "calibration.json"),
            }
        )
    manifest["incremental_models"] = manifest_protocols
    manifest["notes"] = [
        "所有检测选择和验收硬指标均为 mAP50。",
        "incremental_learning 按轮次注册表只使用当轮 Increment train/dev；Base 与历史专家权重保持冻结。",
        "两轮不同新增类别的父子代际、逐轮 New-mAP50/KRR/Full-mAP50 与专家登记均由 sequential_round_evidence.json 固化。",
        "Scene-SensorNet 训练和六类场景门控搜索属于 system_calibration，可使用 Base/Increment train/dev，但不更新任何检测器权重。",
        "六类检测的 mixed lock 只用于参数冻结后的 joint_evaluation，不训练也不选参；Scene-SensorNet 的 context lock 仅做冻结功能模型复核。",
        "Base 与截至当前轮的注册专家对每张图并行推理，类别 owner 保持固定。",
        "六类逐类阈值和场景惩罚只由 mixed dev 选择；线上只读取 Scene-SensorNet 概率，不读取文件名或标签。",
        "precision 与误激活率是非阻断诊断；当前场景感知运行点已显著改善二者。",
        "Ascend310B v2 使用独立四类 Base、二类 Specialist 与 Scene-SensorNet 三个 OM，并由内容门控调度增量专家。",
    ]
    atomic_json(manifest_path, manifest)
    new_manifest_hash = sha256_file(manifest_path)
    functional_path = ROOT / "configs" / "functional_models.yaml"
    functional_text = functional_path.read_text(encoding="utf-8")
    if old_manifest_hash not in functional_text:
        raise ValueError("功能模型注册表未引用晋级前 manifest，拒绝静默改写")
    if old_scene_metrics_hash not in functional_text:
        raise ValueError("功能模型注册表未引用晋级前 Scene-SensorNet 证据，拒绝静默改写")
    functional_registry = yaml.safe_load(functional_text)
    functional_models = {
        str(row["id"]): row for row in functional_registry["models"]
    }
    scene_functional = functional_models["scene_sensor_net_v1"]
    base_functional = functional_models["four_class_base_detector"]
    incremental_functional = functional_models["incremental_model_bank_v1"]
    scene_functional["evidence"]["sha256"] = new_scene_metrics_hash
    base_functional["evidence"]["sha256"] = new_manifest_hash
    incremental_functional.update(
        {
            "display_name": "顺序类别增量目标检测模型库",
            "implementation": (
                "parallel_frozen_base_and_registered_incremental_experts"
            ),
            "artifacts": [
                {
                    "path": models[model_id]["path"],
                    "sha256": models[model_id]["sha256"],
                }
                for model_id in generation["model_members"]
                if models[model_id].get("role")
                in {"class_incremental_expert", "target_incremental_expert"}
            ],
        }
    )
    incremental_functional["evidence"]["sha256"] = new_manifest_hash
    for edge in functional_registry.get("collaboration", []):
        direction = (str(edge.get("from")), str(edge.get("to")))
        if direction == (
            "four_class_base_detector",
            "incremental_model_bank_v1",
        ):
            edge["purpose"] = (
                "Base owner 与两个注册单类专家对每张未知图并行推理"
            )
        elif direction == (
            "incremental_model_bank_v1",
            "four_class_base_detector",
        ):
            edge["purpose"] = (
                "Base 固定负责全局类0至3，两个注册单类专家分别负责4和5，"
                "按dev逐类阈值及固定类别 owner 融合"
            )
    temporary_functional = functional_path.with_suffix(".yaml.tmp")
    temporary_functional.write_text(
        yaml.safe_dump(
            functional_registry,
            allow_unicode=True,
            sort_keys=False,
            width=1000,
        ),
        encoding="utf-8",
    )
    temporary_functional.replace(functional_path)

    (evidence / "operating_point_diagnostics.md").write_text(
        diagnostics_markdown(result["baseline"], result["lock"], thresholds, penalties),
        encoding="utf-8",
    )
    summary = {
        "run_id": run_id,
        "candidate_id": candidate["candidate_id"],
        "generation_id": target_round["generation_id"],
        "sequential_class_incremental_verified": True,
        "base_map50": lock_metrics["base_map50"],
        "new_map50": lock_metrics["new_map50"],
        "krr": lock_metrics["krr"],
        "overall_precision": result["lock"]["overall"]["precision"],
        "new_class_precision": result["lock"]["new_classes"]["precision"],
        "overall_false_activation_image_count": result["lock"]["overall"][
            "false_activation_image_count"
        ],
        "diagnostic_warnings": advisory_warnings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
