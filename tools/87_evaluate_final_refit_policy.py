#!/usr/bin/env python3
"""Run a frozen final-refit base policy on its non-performance holdout."""

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

from fair_agent.modules.strict_incremental import (
    evaluate_ap50,
    read_split,
    sha256_file,
    yolo_ground_truth,
)


FORBIDDEN_MARKERS = ("mixed_test", "base_test", "lock")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def canonical_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def reject_test_reference(path: Path, role: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"{role} 不得引用 test/lock：{path}")


def verify_oof_selection(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = dict(config["paths"])
    selection_path = resolve_path(paths["oof_candidate_selection_report"])
    policy_path = resolve_path(paths["oof_policy_report"])
    reject_test_reference(selection_path, "OOF candidate selection")
    reject_test_reference(policy_path, "OOF policy")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    policy_report = json.loads(policy_path.read_text(encoding="utf-8"))
    selected = dict(selection.get("selected", {}))
    if (
        selection.get("selection_scope")
        != "base_train_and_dev_oof_candidate_selection"
        or bool(selection.get("lock_data_access", True))
        or str(selection.get("selection_basis")) != "tuning_folds_only"
        or not all(bool(value) for value in dict(selection.get("gates", {})).values())
        or Path(str(selected.get("source_report", ""))).resolve() != policy_path
        or str(selected.get("source_report_sha256", "")) != sha256_file(policy_path)
    ):
        raise ValueError("最终 OOF candidate selection 未通过无泄露审计")
    if (
        policy_report.get("selection_scope")
        != "base_train_and_dev_oof_tune_validate"
        or bool(policy_report.get("lock_data_access", True))
        or int(policy_report.get("focus_class_id", 0)) != 0
        or not bool(policy_report.get("validation_labels_opened_after_policy_selection", False))
        or not bool(policy_report.get("validation_predictions_frozen_before_labels", False))
    ):
        raise ValueError("最终 OOF policy 没有在选定并冻结预测后才打开验证标签")

    selected_policy = dict(policy_report["selected_policy"])
    mapping = {str(key): str(value) for key, value in config["oof_owner_mapping"].items()}
    expected_oof_names = {"generic", *map(str, selected_policy["secondaries"])}
    if set(mapping) != expected_oof_names:
        raise ValueError("OOF owner mapping 必须恰好覆盖 primary 与选中的 secondaries")
    focus = dict(config["base_fusion"]["class_policies"][0])
    expected_secondaries = {mapping[name] for name in selected_policy["secondaries"]}
    secondary_scales = {
        str(name): float(scale)
        for name, scale in dict(focus.get("secondary_scales", {})).items()
    }
    if (
        str(focus.get("primary")) != mapping["generic"]
        or set(secondary_scales) != expected_secondaries
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
    return {
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "policy_path": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "selected_candidate": str(selected["name"]),
        "selected_policy": selected_policy,
        "owner_mapping": mapping,
        "all_oof_map50": float(selection["all_oof_diagnostic"]["fused_map50"]),
        "post_selection_validation_map50": float(
            selection["post_selection_validation"]["fused_map50"]
        ),
    }


def verify_class_fusion_policy(
    config: Mapping[str, Any],
    focus_audit: Mapping[str, Any],
    class_id: int,
    path: Path,
) -> dict[str, Any]:
    """Verify one non-focus class policy selected without validation labels."""
    reject_test_reference(path, f"class {class_id} fusion policy")
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = dict(report.get("selected_policy", {}))
    if (
        report.get("selection_scope")
        != "base_train_and_dev_oof_tune_validate"
        or bool(report.get("lock_data_access", True))
        or int(report.get("focus_class_id", -1)) != int(class_id)
        or not bool(report.get("validation_labels_opened_after_policy_selection", False))
        or not bool(report.get("validation_predictions_frozen_before_labels", False))
        or int(selected.get("degraded_tuning_fold_count", -1))
        > int(config["selection_requirements"]["max_degraded_tuning_folds"])
    ):
        raise ValueError(f"class {class_id} fusion policy 未通过无泄露审计")

    focus_report = json.loads(
        Path(str(focus_audit["policy_path"])).read_text(encoding="utf-8")
    )
    manifest_sha256 = str(report.get("manifest_sha256", ""))
    if not manifest_sha256 or manifest_sha256 != str(
        focus_report.get("manifest_sha256", "")
    ):
        raise ValueError(f"class {class_id} fusion 没有使用相同 OOF manifest")

    mapping = {str(key): str(value) for key, value in config["oof_owner_mapping"].items()}
    secondaries = list(map(str, selected.get("secondaries", [])))
    expected_sources = {"generic", *secondaries}
    if not expected_sources <= set(mapping):
        raise ValueError(f"class {class_id} fusion owner mapping 不完整")
    policy = dict(config["base_fusion"]["class_policies"][class_id])
    secondary_scales = {
        str(name): float(scale)
        for name, scale in dict(policy.get("secondary_scales", {})).items()
    }
    if (
        str(policy.get("primary")) != mapping["generic"]
        or set(secondary_scales) != {mapping[name] for name in secondaries}
        or any(
            abs(value - float(selected["secondary_scale"])) > 1e-12
            for value in secondary_scales.values()
        )
        or abs(float(policy["iou"]) - float(selected["fusion_iou"])) > 1e-12
        or abs(
            float(policy.get("agreement_bonus", 0.0))
            - float(selected["agreement_bonus"])
        )
        > 1e-12
        or bool(policy.get("weighted_boxes", False))
        != bool(selected["weighted_boxes"])
        or bool(policy.get("scale_before_clustering", True))
    ):
        raise ValueError(f"class {class_id} 最终配置与 OOF fusion policy 不一致")

    sources = dict(report.get("sources", {}))
    source_audit = {}
    for source_name in sorted(expected_sources):
        entry = dict(sources.get(source_name, {}))
        source_path = Path(str(entry.get("path", ""))).resolve()
        reject_test_reference(source_path, f"class {class_id} OOF source {source_name}")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        model_name = mapping[source_name]
        model_config = dict(config["models"][model_name])
        folds = dict(source.get("models", {}))
        if (
            sha256_file(source_path) != str(entry.get("sha256", ""))
            or source.get("selection_scope") != "base_train_and_dev_oof_only"
            or bool(source.get("lock_data_access", True))
            or str(source.get("manifest_sha256", "")) != manifest_sha256
            or str(source.get("inference_mode", ""))
            != str(model_config.get("inference_mode", "full_frame"))
            or set(folds) != {f"fold_{index}" for index in range(5)}
            or any(
                int(row.get("completed_epochs", -1))
                != int(config["selection_requirements"]["required_epochs"])
                or int(row.get("imgsz", -1)) != int(model_config["imgsz"])
                for row in folds.values()
            )
        ):
            raise ValueError(f"class {class_id} OOF fusion source 无效：{source_name}")
        source_audit[source_name] = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "mapped_model": model_name,
            "imgsz": int(model_config["imgsz"]),
        }

    tuning = dict(report.get("tuning", {}))
    validation = dict(report.get("validation", {}))
    all_oof = dict(report.get("all_oof_diagnostic", {}))
    if (
        float(tuning.get("delta_map50", 0.0)) <= 0.0
        or float(validation.get("delta_map50", 0.0)) <= 0.0
        or str(class_id) not in dict(tuning.get("fused_per_class_ap50", {}))
        or str(class_id) not in dict(validation.get("fused_per_class_ap50", {}))
        or str(class_id) not in dict(all_oof.get("fused_per_class_ap50", {}))
    ):
        raise ValueError(f"class {class_id} fusion 未在调参与后置验证同时改善")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "focus_class_id": int(class_id),
        "selected_policy": selected,
        "sources": source_audit,
        "tuning_ap50": float(tuning["fused_per_class_ap50"][str(class_id)]),
        "post_selection_validation_ap50": float(
            validation["fused_per_class_ap50"][str(class_id)]
        ),
        "all_oof_ap50": float(all_oof["fused_per_class_ap50"][str(class_id)]),
        "validation_predictions_frozen_before_labels": True,
    }


def verify_class_owner_selection(
    config: Mapping[str, Any], focus_audit: Mapping[str, Any]
) -> dict[str, Any]:
    path = resolve_path(config["paths"]["class_owner_selection_report"])
    reject_test_reference(path, "class owner selection")
    report = json.loads(path.read_text(encoding="utf-8"))
    gates = dict(report.get("gates", {}))
    if (
        report.get("selection_scope")
        != "base_train_and_dev_oof_class_owner_selection"
        or bool(report.get("lock_data_access", True))
        or str(report.get("selection_basis", ""))
        != "maximin_tuning_fold_ap50_then_pooled_tuning_ap50"
        or not bool(report.get("validation_labels_opened_after_owner_selection", False))
        or not bool(report.get("validation_predictions_frozen_before_labels", False))
        or not gates
        or not all(all(bool(value) for value in dict(row).values()) for row in gates.values())
    ):
        raise ValueError("非 focus 类 owner selection 未通过无泄露后置验证")
    selected = {int(key): str(value) for key, value in report["selected_owners"].items()}
    if set(selected) != {1, 3}:
        raise ValueError("class owner selection 必须且只能覆盖 aircraft/tank")
    mapping = {
        str(key): str(value) for key, value in config["class_owner_mapping"].items()
    }
    required_sources = {str(report["baseline"]), *selected.values()}
    if not required_sources <= set(mapping):
        raise ValueError("class owner mapping 缺少 baseline 或选中 owner")
    class_results = {int(key): dict(value) for key, value in report["class_results"].items()}
    fusion_paths = {
        int(key): resolve_path(value)
        for key, value in dict(
            config["paths"].get("class_fusion_policy_reports", {})
        ).items()
    }
    if not set(fusion_paths) <= set(selected):
        raise ValueError("class fusion policy 只能覆盖已选择 owner 的非 focus 类")
    class_fusions = {
        class_id: verify_class_fusion_policy(
            config, focus_audit, class_id, fusion_path
        )
        for class_id, fusion_path in sorted(fusion_paths.items())
    }
    for class_id, source_name in selected.items():
        policy = dict(config["base_fusion"]["class_policies"][class_id])
        if (
            str(policy.get("primary")) != mapping[source_name]
            or (
                class_id not in class_fusions
                and dict(policy.get("secondary_scales", {}))
            )
            or str(class_results[class_id].get("selected_owner")) != source_name
        ):
            raise ValueError(f"class {class_id} 最终 owner 与 OOF 选择不一致")

    manifest_sha256 = str(report.get("manifest_sha256", ""))
    policy_report = json.loads(Path(str(focus_audit["policy_path"])).read_text(encoding="utf-8"))
    if not manifest_sha256 or manifest_sha256 != str(policy_report.get("manifest_sha256", "")):
        raise ValueError("focus fusion 与 class owner selection 没有使用同一 OOF manifest")
    sources = dict(report.get("sources", {}))
    source_audit = {}
    for source_name in sorted(required_sources):
        source_entry = dict(sources.get(source_name, {}))
        source_path = Path(str(source_entry.get("path", ""))).resolve()
        reject_test_reference(source_path, f"class owner OOF source {source_name}")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        model_name = mapping[source_name]
        model_config = dict(config["models"][model_name])
        folds = dict(source.get("models", {}))
        if (
            sha256_file(source_path) != str(source_entry.get("sha256", ""))
            or source.get("selection_scope") != "base_train_and_dev_oof_only"
            or bool(source.get("lock_data_access", True))
            or str(source.get("manifest_sha256", "")) != manifest_sha256
            or set(folds) != {f"fold_{index}" for index in range(5)}
            or any(
                int(row.get("completed_epochs", -1))
                != int(config["selection_requirements"]["required_epochs"])
                or int(row.get("imgsz", -1)) != int(model_config["imgsz"])
                or str(row.get("inference_mode", ""))
                != str(model_config.get("inference_mode", "full_frame"))
                for row in folds.values()
            )
        ):
            raise ValueError(f"class owner OOF source 无效：{source_name}")
        source_audit[source_name] = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "mapped_model": model_name,
            "imgsz": int(model_config["imgsz"]),
        }

    focus_policy = dict(focus_audit["selected_policy"])
    tuning_map50 = sum(
        [
            float(focus_policy["tuning_per_class_ap50"]["0"]),
            float(class_results[1]["tuning"]["ap50"]),
            (
                float(class_fusions[3]["tuning_ap50"])
                if 3 in class_fusions
                else float(class_results[3]["tuning"]["ap50"])
            ),
        ]
    ) / 3.0
    focus_report = policy_report
    validation_map50 = sum(
        [
            float(focus_report["validation"]["fused_per_class_ap50"]["0"]),
            float(class_results[1]["validation"]["ap50"]),
            (
                float(class_fusions[3]["post_selection_validation_ap50"])
                if 3 in class_fusions
                else float(class_results[3]["validation"]["ap50"])
            ),
        ]
    ) / 3.0
    all_oof_map50 = sum(
        [
            float(focus_report["all_oof_diagnostic"]["fused_per_class_ap50"]["0"]),
            float(class_results[1]["all_oof"]["ap50"]),
            (
                float(class_fusions[3]["all_oof_ap50"])
                if 3 in class_fusions
                else float(class_results[3]["all_oof"]["ap50"])
            ),
        ]
    ) / 3.0
    requirements = dict(config["selection_requirements"])
    if (
        validation_map50 < float(requirements["min_validation_fold_map50"])
        or all_oof_map50 < float(requirements["min_oof_map50"])
    ):
        raise ValueError("逐类 OOF policy 未达到 0.85 工程线")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "selection_basis": report["selection_basis"],
        "selected_owners": {str(key): value for key, value in selected.items()},
        "owner_mapping": mapping,
        "sources": source_audit,
        "class_fusion_policies": {
            str(key): value for key, value in class_fusions.items()
        },
        "tuning_map50": tuning_map50,
        "post_selection_validation_map50": validation_map50,
        "all_oof_map50": all_oof_map50,
        "validation_predictions_frozen_before_labels": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    if not str(args.device).isdigit():
        raise ValueError("device 必须是明确的单卡编号")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    selection_audit = verify_oof_selection(config)
    selection_audit["class_owner_selection"] = verify_class_owner_selection(
        config, selection_audit
    )

    manifest_path = resolve_path(config["paths"]["checkpoint_holdout_manifest"])
    reject_test_reference(manifest_path, "checkpoint holdout manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("purpose")
        != "leak_free_category_agnostic_small_object_training"
        or manifest.get("selection_scope") != "base_train_and_base_dev_only"
        or bool(manifest.get("lock_data_access", True))
        or list(manifest.get("source_train_val_overlap", ["missing"]))
    ):
        raise ValueError("checkpoint holdout manifest 不是无 test/lock 的 refit 数据边界")
    source_path = Path(str(manifest["source_split"])).resolve()
    split_path = Path(str(manifest["val_source_split"])).resolve()
    reject_test_reference(source_path, "final refit training split")
    reject_test_reference(split_path, "checkpoint holdout split")
    if sha256_file(source_path) != str(manifest["source_split_sha256"]):
        raise ValueError("final refit training split 哈希不一致")
    if sha256_file(split_path) != str(manifest["val_source_split_sha256"]):
        raise ValueError("checkpoint holdout split 哈希不一致")
    images = read_split(split_path)
    if len(images) != int(manifest["val_source_count"]):
        raise ValueError("checkpoint holdout 图像数与 manifest 不一致")
    if canonical_json(manifest.get("base_local_to_global")) != canonical_json(
        config["base_local_to_global"]
    ):
        raise ValueError("checkpoint holdout 类别映射与最终 Agent 不一致")
    expected_local_classes = {str(key) for key in config["base_local_to_global"]}
    if set(map(str, manifest.get("val_label_counts", {}))) != expected_local_classes:
        raise ValueError("checkpoint holdout 必须且只能包含三个基础局部类别")

    report_path = resolve_path(config["paths"]["checkpoint_holdout_report"])
    if report_path.parent.exists():
        raise FileExistsError(f"拒绝覆盖已有 checkpoint holdout：{report_path.parent}")
    runner = runpy.run_path(str(ROOT / "tools" / "70_run_strict_3plus1.py"))
    evaluator = runpy.run_path(str(ROOT / "tools" / "73_evaluate_base_ensemble.py"))
    ensemble = runpy.run_path(str(ROOT / "tools" / "72_select_base_ensemble.py"))
    small_object = runpy.run_path(
        str(ROOT / "tools" / "76_evaluate_small_object_inference.py")
    )
    from ultralytics import YOLO

    policies = {
        int(class_id): dict(policy)
        for class_id, policy in config["base_fusion"]["class_policies"].items()
    }
    model_names = sorted(
        {str(policy["primary"]) for policy in policies.values()}
        | {
            str(name)
            for policy in policies.values()
            for name in dict(policy.get("secondary_scales", {}))
        }
    )
    predict_config = {"common": {"imgsz": 0}, "predict": dict(config["predict"])}
    mapping = {
        int(key): int(value) for key, value in config["base_local_to_global"].items()
    }
    predictions = {}
    model_audit = {}
    for name in model_names:
        model_config = dict(config["models"][name])
        weight = resolve_path(model_config["weights"])
        rows, inference_ms, inference_audit = evaluator["predict_base_owner"](
            YOLO(str(weight)),
            images,
            mapping,
            predict_config,
            str(args.device),
            name,
            model_config,
            runner,
            small_object,
        )
        predictions[name] = rows
        model_audit[name] = {
            "weight": str(weight),
            "weight_sha256": sha256_file(weight),
            "imgsz": int(model_config["imgsz"]),
            "inference_mode": str(model_config.get("inference_mode", "full_frame")),
            "prediction_count": len(rows),
            "inference_ms": float(inference_ms),
            **inference_audit,
        }
    fused = evaluator["fuse_base_classes"](
        predictions, policies, ensemble["fuse_focus_class"]
    )

    # Freeze every prediction artifact before opening even this non-performance
    # holdout's labels.
    prediction_dir = report_path.parent / "predictions"
    artifacts = {
        name: runner["write_jsonl_artifact"](
            prediction_dir / f"{name}_unlabeled.jsonl", predictions[name]
        )
        for name in model_names
    }
    artifacts["fused"] = runner["write_jsonl_artifact"](
        prediction_dir / "fused_unlabeled.jsonl", fused
    )
    freeze_manifest = runner["write_json_artifact"](
        report_path.parent / "freeze_manifest.json",
        {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "selection_audit": selection_audit,
            "models": model_audit,
            "base_fusion": config["base_fusion"],
            "image_count": len(images),
            "models_receiving_identical_complete_input": model_names,
            "label_aware_routing": False,
            "scene_hard_routing": False,
            "filename_class_routing": False,
            "artifacts": artifacts,
            "predictions_frozen_before_labels": True,
        },
    )

    targets = yolo_ground_truth(images)
    metrics = evaluate_ap50(fused, targets, [int(value) for value in config["old_class_ids"]])
    report = {
        "schema_version": 1,
        "candidate_id": str(config["candidate_id"]),
        "selection_scope": "final_refit_checkpoint_holdout",
        "lock_data_access": False,
        "performance_evidence": False,
        "independent_test_evidence": False,
        "evidence_limitation": "used_during_best_checkpoint_selection",
        "split": str(split_path),
        "split_sha256": sha256_file(split_path),
        "image_count": len(images),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "selection_audit": selection_audit,
        "models": model_audit,
        "base_fusion": config["base_fusion"],
        "freeze_manifest": freeze_manifest,
        "predictions_frozen_before_labels": True,
        "map50": float(metrics["map50"]),
        "per_class_ap50": metrics["per_class_ap50"],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
