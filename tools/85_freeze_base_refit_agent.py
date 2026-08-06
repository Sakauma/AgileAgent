#!/usr/bin/env python3
"""Freeze an OOF-selected base policy and its fully trained refit passes."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import sha256_file


SELECTION_SCOPE = "base_train_dev_oof_policy_and_refit_weights"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def canonical_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def reject_test_reference(value: str | Path, role: str) -> None:
    lowered = str(value).replace("\\", "/").lower()
    forbidden = ("mixed_test", "base_test", "/test/", "lock_split")
    if any(token in lowered for token in forbidden):
        raise ValueError(f"{role} 不得引用 lock/test：{value}")


def required_base_models(config: Mapping[str, Any]) -> set[str]:
    policies = config["base_fusion"]["class_policies"].values()
    return {
        str(policy["primary"]) for policy in policies
    } | {
        str(name)
        for policy in policies
        for name in dict(policy.get("secondary_scales", {}))
    }


def validate_training_report(
    path: Path,
    model_name: str,
    model_config: Mapping[str, Any],
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    reject_test_reference(path, f"{model_name} training report")
    report = read_json(path)
    if (
        report.get("selection_scope") != "base_train_and_dev_only"
        or bool(report.get("lock_data_access", True))
    ):
        raise ValueError(f"{model_name} 必须只由 base_train+base_dev refit")

    dataset = dict(report.get("dataset_audit", {}))
    if (
        bool(dataset.get("lock_data_access", True))
        or bool(dataset.get("source_declared_test_split", True))
        or bool(dataset.get("training_declared_test_split", True))
        or list(dataset.get("train_dev_overlap", ["missing"]))
        or int(dataset.get("train_count", 0)) <= 0
        or int(dataset.get("dev_count", 0)) <= 0
    ):
        raise ValueError(f"{model_name} refit 数据审计未通过")

    training = dict(report.get("training", {}))
    arguments = dict(training.get("arguments", {}))
    required_epochs = int(requirements["required_epochs"])
    required_batch = int(requirements["required_batch"])
    required_patience = int(requirements["required_patience"])
    if (
        int(training.get("requested_epochs", -1)) != required_epochs
        or int(training.get("completed_epochs", -1)) != required_epochs
        or bool(training.get("stopped_early", True))
        or int(arguments.get("epochs", -1)) != required_epochs
        or int(arguments.get("batch", -1)) != required_batch
        or int(arguments.get("patience", -1)) != required_patience
    ):
        raise ValueError(
            f"{model_name} 未按 epochs={required_epochs}, batch={required_batch}, "
            f"patience={required_patience} 跑满"
        )
    best_epoch = int(training.get("best_epoch", 0))
    if not 1 <= best_epoch <= required_epochs:
        raise ValueError(f"{model_name} best_epoch 超出完整训练预算")

    configured_weight = resolve_path(model_config["weights"])
    reported_weight = Path(str(report.get("best_weight", ""))).resolve()
    if configured_weight != reported_weight:
        raise ValueError(f"{model_name} 配置权重与 refit 报告不一致")
    reported_hash = str(report.get("best_weight_sha256", ""))
    if sha256_file(configured_weight) != reported_hash:
        raise ValueError(f"{model_name} best.pt 哈希不一致")

    evaluation = dict(report.get("evaluation", {}))
    return {
        "training_report": str(path),
        "training_report_sha256": sha256_file(path),
        "weight": str(configured_weight),
        "weight_sha256": reported_hash,
        "imgsz": int(model_config["imgsz"]),
        "inference_mode": str(model_config.get("inference_mode", "full_frame")),
        "requested_epochs": required_epochs,
        "completed_epochs": int(training["completed_epochs"]),
        "best_epoch": best_epoch,
        "best_metric_value": float(training["best_metric_value"]),
        "checkpoint_holdout_eval_imgsz": int(evaluation["imgsz"]),
        "checkpoint_holdout_map50": float(evaluation["map50"]),
    }


def validate_oof_source(
    source_name: str,
    source_entry: Mapping[str, Any],
    model_name: str,
    model_config: Mapping[str, Any],
    manifest_sha256: str,
    required_epochs: int,
) -> dict[str, Any]:
    source_path = Path(str(source_entry.get("path", ""))).resolve()
    reject_test_reference(source_path, f"OOF {source_name} source")
    source = read_json(source_path)
    if (
        sha256_file(source_path) != str(source_entry.get("sha256", ""))
        or source.get("selection_scope") != "base_train_and_dev_oof_only"
        or bool(source.get("lock_data_access", True))
        or str(source.get("manifest_sha256", "")) != manifest_sha256
        or str(source.get("inference_mode", ""))
        != str(model_config.get("inference_mode", "full_frame"))
    ):
        raise ValueError(f"OOF {source_name} 源报告范围、模式或哈希不一致")
    folds = dict(source.get("models", {}))
    expected_folds = {f"fold_{index}" for index in range(5)}
    if set(folds) != expected_folds:
        raise ValueError(f"OOF {source_name} 必须完整覆盖五折")
    for fold_name, fold in folds.items():
        if (
            int(fold.get("completed_epochs", -1)) != required_epochs
            or int(fold.get("imgsz", -1)) != int(model_config["imgsz"])
            or str(fold.get("inference_mode", ""))
            != str(model_config.get("inference_mode", "full_frame"))
        ):
            raise ValueError(f"OOF {source_name}/{fold_name} 预算或推理配置不一致")
    return {
        "path": str(source_path),
        "sha256": sha256_file(source_path),
        "mapped_model": model_name,
        "fold_count": len(folds),
        "all_folds_completed_epochs": required_epochs,
        "imgsz": int(model_config["imgsz"]),
        "inference_mode": str(model_config.get("inference_mode", "full_frame")),
        "image_count": int(source["image_count"]),
    }


def validate_oof_evidence(
    candidate_path: Path,
    policy_path: Path,
    owner_mapping: Mapping[str, str],
    policies: Mapping[int, Mapping[str, Any]],
    models: Mapping[str, Mapping[str, Any]],
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    reject_test_reference(candidate_path, "OOF candidate selection")
    reject_test_reference(policy_path, "OOF policy")
    candidate = read_json(candidate_path)
    policy = read_json(policy_path)
    selected = dict(candidate.get("selected", {}))
    gates = dict(candidate.get("gates", {}))
    if (
        candidate.get("selection_scope")
        != "base_train_and_dev_oof_candidate_selection"
        or bool(candidate.get("lock_data_access", True))
        or str(candidate.get("selection_basis", "")) != "tuning_folds_only"
        or not gates
        or not all(bool(value) for value in gates.values())
        or Path(str(selected.get("source_report", ""))).resolve() != policy_path
        or str(selected.get("source_report_sha256", "")) != sha256_file(policy_path)
    ):
        raise ValueError("最终 OOF candidate selection 未通过无泄露审计")
    if (
        policy.get("selection_scope") != "base_train_and_dev_oof_tune_validate"
        or bool(policy.get("lock_data_access", True))
        or not bool(policy.get("validation_labels_opened_after_policy_selection", False))
        or not bool(policy.get("validation_predictions_frozen_before_labels", False))
    ):
        raise ValueError("最终 OOF policy 没有在选定并冻结预测后才打开验证标签")

    selected_policy = dict(policy.get("selected_policy", {}))
    if canonical_json(selected.get("policy")) != canonical_json(selected_policy):
        raise ValueError("candidate selection 与源 OOF policy 不一致")
    secondaries = list(map(str, selected_policy.get("secondaries", [])))
    if list(map(str, policy.get("secondaries", []))) != secondaries:
        raise ValueError("OOF policy 的 secondary 注册不一致")
    expected_oof_names = {"generic", *secondaries}
    mapping = {str(key): str(value) for key, value in owner_mapping.items()}
    if set(mapping) != expected_oof_names:
        raise ValueError("OOF owner mapping 必须恰好覆盖选中的推理 pass")

    focus = dict(policies[0])
    secondary_scales = {
        str(name): float(scale)
        for name, scale in dict(focus.get("secondary_scales", {})).items()
    }
    if (
        str(focus.get("primary")) != mapping["generic"]
        or set(secondary_scales) != {mapping[name] for name in secondaries}
        or any(
            abs(value - float(selected_policy["secondary_scale"])) > 1e-12
            for value in secondary_scales.values()
        )
        or abs(float(focus["iou"]) - float(selected_policy["fusion_iou"])) > 1e-12
        or abs(
            float(focus.get("agreement_bonus", 0.0))
            - float(selected_policy["agreement_bonus"])
        )
        > 1e-12
        or bool(focus.get("weighted_boxes", False))
        != bool(selected_policy["weighted_boxes"])
        or bool(focus.get("scale_before_clustering", True))
    ):
        raise ValueError("最终 refit 融合配置与 OOF 选定 policy 不一致")
    tuning_folds = {int(value) for value in policy.get("tuning_folds", [])}
    validation_folds = {int(value) for value in policy.get("validation_folds", [])}
    if (
        not tuning_folds
        or not validation_folds
        or tuning_folds & validation_folds
        or sorted(tuning_folds) != list(candidate.get("tuning_folds", []))
        or sorted(validation_folds)
        != list(candidate.get("validation_folds_opened_after_selection", []))
    ):
        raise ValueError("OOF tuning/validation fold 边界不一致")

    tuning = dict(policy.get("tuning", {}))
    validation = dict(policy.get("validation", {}))
    diagnostic = dict(policy.get("all_oof_diagnostic", {}))
    if (
        float(diagnostic.get("fused_map50", 0.0))
        < float(requirements["min_oof_map50"])
        or float(validation.get("fused_map50", 0.0))
        < float(requirements["min_validation_fold_map50"])
        or int(selected_policy.get("degraded_tuning_fold_count", -1))
        > int(requirements["max_degraded_tuning_folds"])
        or float(tuning.get("delta_map50", 0.0)) <= 0.0
        or float(validation.get("delta_map50", 0.0)) <= 0.0
    ):
        raise ValueError("OOF policy 未达到 0.85 安全线或存在折退化")

    manifest_sha256 = str(policy.get("manifest_sha256", ""))
    if not manifest_sha256 or manifest_sha256 != str(candidate.get("manifest_sha256", "")):
        raise ValueError("OOF candidate 与 policy 的 fold manifest 不一致")
    source_entries = dict(policy.get("sources", {}))
    source_audit = {}
    for source_name in sorted(expected_oof_names):
        if source_name not in source_entries:
            raise ValueError(f"OOF policy 缺少选中源：{source_name}")
        model_name = mapping[source_name]
        source_audit[source_name] = validate_oof_source(
            source_name,
            dict(source_entries[source_name]),
            model_name,
            dict(models[model_name]),
            manifest_sha256,
            int(requirements["required_epochs"]),
        )
    if {row["image_count"] for row in source_audit.values()} != {
        int(diagnostic["image_count"])
    }:
        raise ValueError("选中 OOF 源没有覆盖相同的完整非测试样本")

    return {
        "candidate_selection_path": str(candidate_path),
        "candidate_selection_sha256": sha256_file(candidate_path),
        "selected_candidate": str(selected["name"]),
        "policy_path": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "selection_basis": "tuning_folds_only",
        "validation_predictions_frozen_before_labels": True,
        "owner_mapping": mapping,
        "selected_policy": selected_policy,
        "tuning_map50": float(tuning["fused_map50"]),
        "validation_map50": float(validation["fused_map50"]),
        "all_oof_map50": float(diagnostic["fused_map50"]),
        "all_oof_per_class_ap50": diagnostic["fused_per_class_ap50"],
        "image_count": int(diagnostic["image_count"]),
        "sources": source_audit,
    }


def validate_artifact(entry: Mapping[str, Any], role: str) -> dict[str, Any]:
    artifact_path = resolve_path(str(entry.get("path", "")))
    reject_test_reference(artifact_path, role)
    if sha256_file(artifact_path) != str(entry.get("sha256", "")):
        raise ValueError(f"{role} 哈希不一致")
    return {
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "row_count": int(entry["row_count"]),
    }


def validate_checkpoint_holdout(
    path: Path,
    config_path: Path,
    config: Mapping[str, Any],
    model_audits: Mapping[str, Mapping[str, Any]],
    oof: Mapping[str, Any],
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    reject_test_reference(path, "checkpoint holdout report")
    report = read_json(path)
    if (
        report.get("selection_scope") != "final_refit_checkpoint_holdout"
        or bool(report.get("lock_data_access", True))
        or bool(report.get("performance_evidence", True))
        or bool(report.get("independent_test_evidence", True))
        or str(report.get("evidence_limitation", ""))
        != "used_during_best_checkpoint_selection"
        or not bool(report.get("predictions_frozen_before_labels", False))
    ):
        raise ValueError("checkpoint holdout 必须明确标记为选 checkpoint 用的非独立 sanity")
    reject_test_reference(str(report.get("split", "")), "checkpoint holdout split")
    if (
        str(report.get("candidate_id", "")) != str(config["candidate_id"])
        or canonical_json(report.get("base_fusion"))
        != canonical_json(config.get("base_fusion"))
        or str(report.get("config_sha256", "")) != sha256_file(config_path)
    ):
        raise ValueError("checkpoint holdout 的候选、配置或融合规则已漂移")

    selection = dict(report.get("selection_audit", {}))
    if (
        str(selection.get("selection_sha256", ""))
        != str(oof["candidate_selection_sha256"])
        or str(selection.get("policy_sha256", "")) != str(oof["policy_sha256"])
        or canonical_json(selection.get("selected_policy"))
        != canonical_json(oof["selected_policy"])
        or canonical_json(selection.get("owner_mapping"))
        != canonical_json(oof["owner_mapping"])
        or canonical_json(selection.get("class_owner_selection"))
        != canonical_json(oof["class_owner_selection"])
    ):
        raise ValueError("checkpoint holdout 没有使用已冻结的 OOF 选择")

    report_models = dict(report.get("models", {}))
    if set(report_models) != set(model_audits):
        raise ValueError("checkpoint holdout 必须运行每个且仅运行选中的基础 pass")
    for name, audit in model_audits.items():
        observed = dict(report_models[name])
        if (
            str(observed.get("weight_sha256", "")) != str(audit["weight_sha256"])
            or int(observed.get("imgsz", -1)) != int(audit["imgsz"])
            or str(observed.get("inference_mode", "")) != str(audit["inference_mode"])
        ):
            raise ValueError(f"checkpoint holdout 模型与冻结 refit pass 不一致：{name}")

    freeze_entry = dict(report.get("freeze_manifest", {}))
    freeze_path = resolve_path(str(freeze_entry.get("path", "")))
    if sha256_file(freeze_path) != str(freeze_entry.get("sha256", "")):
        raise ValueError("checkpoint prediction freeze manifest 哈希不一致")
    freeze = read_json(freeze_path)
    expected_receivers = sorted(model_audits)
    if (
        str(freeze.get("config_sha256", "")) != sha256_file(config_path)
        or canonical_json(freeze.get("selection_audit")) != canonical_json(selection)
        or canonical_json(freeze.get("base_fusion"))
        != canonical_json(config.get("base_fusion"))
        or sorted(map(str, freeze.get("models_receiving_identical_complete_input", [])))
        != expected_receivers
        or bool(freeze.get("label_aware_routing", True))
        or bool(freeze.get("scene_hard_routing", True))
        or bool(freeze.get("filename_class_routing", True))
        or not bool(freeze.get("predictions_frozen_before_labels", False))
    ):
        raise ValueError("checkpoint holdout 的预测冻结顺序或无路由约束未通过")
    artifacts = dict(freeze.get("artifacts", {}))
    if set(artifacts) != {*expected_receivers, "fused"}:
        raise ValueError("checkpoint holdout 冻结产物不完整")
    artifact_audit = {
        name: validate_artifact(dict(entry), f"checkpoint artifact {name}")
        for name, entry in artifacts.items()
    }

    per_class = dict(report.get("per_class_ap50", {}))
    if set(map(int, per_class)) != {int(value) for value in config["old_class_ids"]}:
        raise ValueError("checkpoint holdout 指标必须且只能包含三个旧类别")
    map50 = float(report["map50"])
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "selection_scope": report["selection_scope"],
        "performance_evidence": False,
        "independent_test_evidence": False,
        "evidence_limitation": report["evidence_limitation"],
        "predictions_frozen_before_labels": True,
        "all_base_passes_receive_every_image": True,
        "image_count": int(report["image_count"]),
        "map50": map50,
        "per_class_ap50": per_class,
        "meets_engineering_line_but_is_not_test_evidence": map50
        >= float(requirements["min_oof_map50"]),
        "freeze_manifest": {
            "path": str(freeze_path),
            "sha256": sha256_file(freeze_path),
            "artifacts": artifact_audit,
        },
    }


def freeze_selection(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if (
        config.get("selection_scope") != SELECTION_SCOPE
        or bool(config.get("lock_data_access_during_selection", True))
    ):
        raise ValueError("最终配置必须声明 OOF policy + refit weights 且无 lock/test 选模")
    if bool(config.get("incremental_fusion", {}).get("cross_class", {}).get("enabled", True)):
        raise ValueError("最终增量融合必须关闭跨类别压制以保护旧类 owner")

    policies = {
        int(class_id): dict(policy)
        for class_id, policy in config["base_fusion"]["class_policies"].items()
    }
    if set(policies) != {0, 1, 3}:
        raise ValueError("最终基础 policy 必须且只能覆盖三个旧类别")
    required_models = required_base_models(config)
    paths = dict(config["paths"])
    training_reports = {
        str(name): resolve_path(path)
        for name, path in dict(paths["training_reports"]).items()
    }
    if set(training_reports) != required_models:
        raise ValueError("training_reports 必须恰好覆盖全部基础推理 pass")
    models = {str(name): dict(value) for name, value in config["models"].items()}
    if not required_models <= set(models):
        raise ValueError("最终模型配置缺少基础推理 pass")

    requirements = dict(config["selection_requirements"])
    model_audits = {
        name: validate_training_report(
            training_reports[name], name, models[name], requirements
        )
        for name in sorted(required_models)
    }
    unique_weight_hashes = sorted(
        {str(audit["weight_sha256"]) for audit in model_audits.values()}
    )

    owner_mapping = {
        str(key): str(value) for key, value in config["oof_owner_mapping"].items()
    }
    oof = validate_oof_evidence(
        resolve_path(paths["oof_candidate_selection_report"]),
        resolve_path(paths["oof_policy_report"]),
        owner_mapping,
        policies,
        models,
        requirements,
    )
    final_evaluator = runpy.run_path(
        str(ROOT / "tools" / "87_evaluate_final_refit_policy.py")
    )
    oof["class_owner_selection"] = final_evaluator[
        "verify_class_owner_selection"
    ](config, oof)
    checkpoint = validate_checkpoint_holdout(
        resolve_path(paths["checkpoint_holdout_report"]),
        config_path,
        config,
        model_audits,
        oof,
        requirements,
    )

    report = {
        "schema_version": 2,
        "candidate_id": str(config["candidate_id"]),
        "selection_scope": SELECTION_SCOPE,
        "lock_data_access": False,
        "performance_evidence": False,
        "selection_evidence": "base_train_dev_oof_only",
        "independent_test_evidence": False,
        "image_count": int(oof["image_count"]),
        "config": str(config_path),
        "config_sha256_at_selection": sha256_file(config_path),
        "models": model_audits,
        "unique_weight_count": len(unique_weight_hashes),
        "unique_weight_sha256": unique_weight_hashes,
        "base_fusion": config["base_fusion"],
        "oof_policy_evidence": oof,
        "checkpoint_holdout_sanity": checkpoint,
        "selection_requirements": requirements,
        "no_label_or_scene_routing": True,
        "all_base_owners_receive_every_image": True,
        "parameter_freezing_is_training_time_only": True,
    }
    destination = resolve_path(paths["selection_report"])
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已冻结最终 Agent：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report = freeze_selection(args.config)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
