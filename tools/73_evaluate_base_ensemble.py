#!/usr/bin/env python3
"""One-shot, label-blind lock evaluation of a frozen base ensemble."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import (
    evaluate_ap50,
    fuse_old_new_predictions,
    read_split,
    retention_metrics,
    sha256_file,
    subset_rows,
    yolo_ground_truth,
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def scaled_rows(rows: Sequence[Mapping[str, Any]], scale: float) -> list[Dict[str, Any]]:
    return [
        {**dict(row), "confidence": min(1.0, float(row["confidence"]) * float(scale))}
        for row in rows
    ]


def canonical_json(value: Any) -> Any:
    """Normalize YAML integer keys to their JSON string representation."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def predict_base_owner(
    detector: Any,
    images: Sequence[Path],
    local_to_global: Mapping[int, int],
    predict_config: Mapping[str, Any],
    device: str,
    name: str,
    model_config: Mapping[str, Any],
    runner: Mapping[str, Any],
    small_object: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], float, Dict[str, Any]]:
    inference_mode = str(model_config.get("inference_mode", "full_frame"))
    imgsz = int(model_config["imgsz"])
    if inference_mode == "full_frame":
        rows, inference_ms = runner["predict_records"](
            detector,
            images,
            local_to_global,
            predict_config,
            str(device),
            name,
            imgsz,
        )
        return rows, float(inference_ms), {"inference_mode": inference_mode}
    if inference_mode != "sliding_window":
        raise ValueError(f"未知基础 owner 推理模式：{name} -> {inference_mode}")

    tile = dict(model_config.get("tile", {}))
    required = {"width", "height", "overlap", "focus_class_local"}
    missing = sorted(required - set(tile))
    if missing:
        raise ValueError(f"滑窗 owner 缺少配置：{name} -> {missing}")
    focus_local = int(tile["focus_class_local"])
    mapping = {int(key): int(value) for key, value in local_to_global.items()}
    if focus_local not in mapping:
        raise ValueError(f"滑窗 owner focus 类别未注册：{name} -> {focus_local}")
    predict = dict(predict_config["predict"])
    local_rows, inference_ms, tile_audit = small_object["predict_tiles"](
        detector,
        images,
        (int(tile["width"]), int(tile["height"])),
        float(tile["overlap"]),
        imgsz,
        int(predict.get("batch", 32)),
        str(device),
        focus_local,
        float(predict["conf"]),
        float(predict["iou"]),
        int(predict["max_det"]),
        runner["evaluation_predictor_class"](),
        name,
    )
    rows = []
    for row in local_rows:
        local_id = int(row["class_id"])
        if local_id not in mapping:
            raise RuntimeError(f"滑窗 owner 输出未注册类别：{name} -> {local_id}")
        rows.append({**dict(row), "class_id": mapping[local_id]})
    return rows, float(inference_ms), {
        "inference_mode": inference_mode,
        "tile": tile_audit,
    }


def fuse_base_classes(
    predictions_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    policies: Mapping[int, Mapping[str, Any]],
    fuse_focus_class: Any,
) -> list[Dict[str, Any]]:
    output: list[Dict[str, Any]] = []
    for class_id, raw_policy in sorted(policies.items()):
        policy = dict(raw_policy)
        primary = str(policy["primary"])
        secondary_scales = {
            str(name): float(scale)
            for name, scale in dict(policy.get("secondary_scales", {})).items()
        }
        required = {primary, *secondary_scales}
        missing = sorted(required - set(predictions_by_model))
        if missing:
            raise ValueError(f"基础融合策略引用未执行模型：class={class_id} missing={missing}")
        if not secondary_scales:
            output.extend(
                dict(row)
                for row in predictions_by_model[primary]
                if int(row["class_id"]) == int(class_id)
            )
            continue
        scale_before_clustering = bool(policy.get("scale_before_clustering", False))
        prepared = {primary: [dict(row) for row in predictions_by_model[primary]]}
        if scale_before_clustering:
            for name, scale in secondary_scales.items():
                prepared[name] = scaled_rows(predictions_by_model[name], scale)
            fusion_scale = 1.0
        else:
            unique_scales = set(secondary_scales.values())
            if len(unique_scales) != 1:
                raise ValueError(
                    "聚类后缩放要求同一类别的 secondary_scales 完全一致："
                    f"class={class_id} scales={sorted(unique_scales)}"
                )
            for name in secondary_scales:
                prepared[name] = [dict(row) for row in predictions_by_model[name]]
            fusion_scale = next(iter(unique_scales))
        fused = fuse_focus_class(
            prepared,
            primary,
            list(secondary_scales),
            int(class_id),
            float(policy["iou"]),
            fusion_scale,
            float(policy.get("agreement_bonus", 0.0)),
            bool(policy.get("weighted_boxes", False)),
        )
        output.extend(row for row in fused if int(row["class_id"]) == int(class_id))
    output.sort(
        key=lambda row: (
            str(row["image_id"]),
            int(row["class_id"]),
            -float(row["confidence"]),
        )
    )
    return output


def verify_selection(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    report_path = resolve_path(config["paths"]["selection_report"])
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selection_scope = str(report.get("selection_scope", ""))
    allowed_scopes = {
        "base_dev_only",
        "base_train_dev_oof_policy_and_refit_weights",
    }
    lowered = str(report.get("split", "")).lower()
    if (
        selection_scope not in allowed_scopes
        or selection_scope != str(config.get("selection_scope", selection_scope))
        or bool(report.get("lock_data_access", True))
        or "mixed_test" in lowered
        or "lock" in lowered
        or "base_test" in lowered
    ):
        raise ValueError("基础 ensemble 候选必须由允许的非测试证据选择且不得访问 lock/test")
    if str(report.get("candidate_id")) != str(config.get("candidate_id")):
        raise ValueError("基础选择报告与待评测 candidate_id 不一致")
    if str(report.get("config_sha256_at_selection")) != sha256_file(config_path):
        raise ValueError("候选配置在策略与权重冻结后发生变化")
    if canonical_json(report.get("base_fusion")) != canonical_json(config.get("base_fusion")):
        raise ValueError("待评测基础融合规则与冻结规则不一致")
    if selection_scope == "base_train_dev_oof_policy_and_refit_weights":
        if bool(report.get("independent_test_evidence", True)):
            raise ValueError("OOF/refit 选择报告不得冒充独立测试证据")
        if not bool(report.get("all_base_owners_receive_every_image", False)):
            raise ValueError("冻结报告必须保证每个基础 owner 接收每张未知图片")
        if not bool(report.get("no_label_or_scene_routing", False)):
            raise ValueError("冻结报告不得包含标签、场景或新旧图片路由")
    selected_models = dict(report.get("models", {}))
    required_model_names = {
        str(policy["primary"])
        for policy in config["base_fusion"]["class_policies"].values()
    } | {
        str(name)
        for policy in config["base_fusion"]["class_policies"].values()
        for name in dict(policy.get("secondary_scales", {}))
    }
    for name in required_model_names:
        if name not in selected_models:
            raise ValueError(f"基础选择报告缺少模型：{name}")
        configured_model = dict(config["models"][name])
        selected_model = dict(selected_models[name])
        weight = resolve_path(configured_model["weights"])
        if sha256_file(weight) != str(selected_models[name]["weight_sha256"]):
            raise ValueError(f"基础模型在冻结后发生变化：{name}")
        if int(configured_model["imgsz"]) != int(
            selected_model.get("imgsz", configured_model["imgsz"])
        ):
            raise ValueError(f"基础模型推理尺度在冻结后发生变化：{name}")
        configured_mode = str(configured_model.get("inference_mode", "full_frame"))
        if configured_mode != str(selected_model.get("inference_mode", "full_frame")):
            raise ValueError(f"基础模型推理模式在冻结后发生变化：{name}")
        configured_tile = configured_model.get("tile")
        selected_tile = selected_model.get("tile")
        if configured_mode == "sliding_window" and canonical_json(configured_tile) != canonical_json(
            {
                "width": selected_tile["crop_size"]["width"],
                "height": selected_tile["crop_size"]["height"],
                "overlap": selected_tile["requested_overlap"],
                "focus_class_local": selected_tile["focus_class_local"],
            }
        ):
            raise ValueError(f"滑窗配置在 base-dev 冻结后发生变化：{name}")
    return {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "selection_scope": selection_scope,
        "lock_data_access": bool(report["lock_data_access"]),
        "image_count": int(report["image_count"]),
        "independent_test_evidence": bool(report.get("independent_test_evidence", False)),
    }


def evaluation_evidence(config: Mapping[str, Any]) -> Dict[str, Any]:
    protocol = dict(config.get("evaluation_protocol", {}))
    kind = str(protocol.get("evidence_kind", "independent_lock_test"))
    allowed = {
        "independent_lock_test": True,
        "local_regression_reuse_not_independent": False,
    }
    if kind not in allowed:
        raise ValueError(f"未知评测证据类型：{kind}")
    independent = bool(protocol.get("independent_test_evidence", allowed[kind]))
    if independent != allowed[kind]:
        raise ValueError("评测证据类型与 independent_test_evidence 声明矛盾")
    if not bool(protocol.get("labels_may_only_be_read_after_prediction_freeze", True)):
        raise ValueError("评测必须先冻结全部预测，再读取标签")
    return {
        "evidence_kind": kind,
        "independent_test_evidence": independent,
        "performance_claim_allowed": independent,
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
    candidate_id = str(config["candidate_id"])
    report_dir = resolve_path(config["paths"]["report_root"]) / candidate_id
    if report_dir.exists():
        raise FileExistsError(f"拒绝覆盖已冻结 ensemble 评测：{report_dir}")
    selection_audit = verify_selection(config, config_path)
    evidence = evaluation_evidence(config)

    runner = runpy.run_path(str(ROOT / "tools" / "70_run_strict_3plus1.py"))
    ensemble_module = runpy.run_path(str(ROOT / "tools" / "72_select_base_ensemble.py"))
    small_object_module = runpy.run_path(
        str(ROOT / "tools" / "76_evaluate_small_object_inference.py")
    )
    from ultralytics import YOLO

    # The only lock input available before prediction freeze is the image list.
    lock_images = read_split(resolve_path(config["paths"]["lock_split"]))
    input_stems = [path.stem for path in lock_images]
    if not input_stems or len(input_stems) != len(set(input_stems)):
        raise ValueError("mixed_test 图像清单必须非空且 stem 唯一")

    predict_config = {
        "common": {"imgsz": 0},
        "predict": dict(config["predict"]),
    }
    base_mapping = {int(key): int(value) for key, value in config["base_local_to_global"].items()}
    incremental_mapping = {
        int(key): int(value) for key, value in config["incremental_local_to_global"].items()
    }
    base_model_names = sorted(
        {
            str(policy["primary"])
            for policy in config["base_fusion"]["class_policies"].values()
        }
        | {
            str(name)
            for policy in config["base_fusion"]["class_policies"].values()
            for name in dict(policy.get("secondary_scales", {}))
        }
    )
    predictions_by_model: dict[str, list[Dict[str, Any]]] = {}
    model_audit = {}
    for name in base_model_names:
        model_config = dict(config["models"][name])
        weight = resolve_path(model_config["weights"])
        detector = YOLO(str(weight))
        rows, inference_ms, inference_audit = predict_base_owner(
            detector,
            lock_images,
            base_mapping,
            predict_config,
            str(args.device),
            name,
            model_config,
            runner,
            small_object_module,
        )
        predictions_by_model[name] = rows
        model_audit[name] = {
            "weights": str(weight),
            "weights_sha256": sha256_file(weight),
            "imgsz": int(model_config["imgsz"]),
            "prediction_count": len(rows),
            "inference_ms": float(inference_ms),
            **inference_audit,
        }

    policies = {
        int(class_id): dict(policy)
        for class_id, policy in config["base_fusion"]["class_policies"].items()
    }
    base_predictions = fuse_base_classes(
        predictions_by_model,
        policies,
        ensemble_module["fuse_focus_class"],
    )

    specialist_config = dict(config["models"]["incremental_specialist"])
    specialist_weight = resolve_path(specialist_config["weights"])
    specialist = YOLO(str(specialist_weight))
    incremental_predictions, incremental_inference_ms = runner["predict_records"](
        specialist,
        lock_images,
        incremental_mapping,
        predict_config,
        str(args.device),
        "incremental_specialist",
        int(specialist_config["imgsz"]),
    )
    combined_predictions, fusion_decisions = fuse_old_new_predictions(
        base_predictions,
        incremental_predictions,
        nms_iou=float(config["incremental_fusion"]["nms_iou"]),
        cross_class=config["incremental_fusion"].get("cross_class"),
    )

    # Freeze and hash every unlabeled prediction artifact before any label helper
    # is called.  No path below this point may influence model or fusion choices.
    prediction_dir = report_dir / "predictions"
    artifacts = {
        "inputs": runner["write_json_artifact"](
            prediction_dir / "lock_unlabeled_inputs.json",
            {
                "candidate_id": candidate_id,
                "input_mode": "unlabeled_images",
                "image_count": len(lock_images),
                "image_stems": input_stems,
                "models_receiving_identical_complete_input": base_model_names
                + ["incremental_specialist"],
                "label_aware_routing": False,
                "scene_hard_routing": False,
                "filename_class_routing": False,
            },
        ),
        "base_members": {},
    }
    for name in base_model_names:
        artifacts["base_members"][name] = runner["write_jsonl_artifact"](
            prediction_dir / f"lock_{name}_unlabeled.jsonl",
            predictions_by_model[name],
        )
    artifacts["base_ensemble"] = runner["write_jsonl_artifact"](
        prediction_dir / "lock_base_ensemble_unlabeled.jsonl", base_predictions
    )
    artifacts["incremental_raw"] = runner["write_jsonl_artifact"](
        prediction_dir / "lock_incremental_unlabeled.jsonl", incremental_predictions
    )
    artifacts["fusion_decisions"] = runner["write_jsonl_artifact"](
        prediction_dir / "lock_fusion_decisions.jsonl", fusion_decisions
    )
    artifacts["fused"] = runner["write_jsonl_artifact"](
        prediction_dir / "lock_fused_unlabeled.jsonl", combined_predictions
    )
    freeze_manifest = runner["write_json_artifact"](
        report_dir / "freeze_manifest.json",
        {
            "candidate_id": candidate_id,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "selection_audit": selection_audit,
            "models": model_audit,
            "specialist": {
                "weights": str(specialist_weight),
                "weights_sha256": sha256_file(specialist_weight),
                "imgsz": int(specialist_config["imgsz"]),
                "prediction_count": len(incremental_predictions),
                "inference_ms": float(incremental_inference_ms),
            },
            "base_fusion": config["base_fusion"],
            "incremental_fusion": config["incremental_fusion"],
            "evaluation_evidence": evidence,
            "artifacts": artifacts,
            "predictions_frozen_before_labels": True,
        },
    )

    # First and only lock-label access for this frozen candidate.
    ground_truth = yolo_ground_truth(lock_images)
    new_class_id = int(config["new_class_id"])
    old_class_ids = [int(value) for value in config["old_class_ids"]]
    base_test_ids = runner["base_test_image_ids"](lock_images, ground_truth, new_class_id)
    base_test = evaluate_ap50(
        subset_rows(base_predictions, base_test_ids),
        subset_rows(ground_truth, base_test_ids),
        old_class_ids,
    )
    retention = retention_metrics(
        base_predictions, combined_predictions, ground_truth, old_class_ids
    )
    new_metrics = evaluate_ap50(combined_predictions, ground_truth, [new_class_id])
    full_metrics = evaluate_ap50(
        combined_predictions, ground_truth, old_class_ids + [new_class_id]
    )
    thresholds = dict(config["acceptance"])
    gates = {
        "base_test_map50": float(base_test["map50"])
        >= float(thresholds["min_base_test_map50"]),
        "new_map50": float(new_metrics["map50"]) >= float(thresholds["min_new_map50"]),
        "krr": float(retention["krr"]) >= float(thresholds["min_krr"]),
    }
    metrics = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "evaluation_evidence": evidence,
        "independent_test_evidence": bool(evidence["independent_test_evidence"]),
        "performance_claim_allowed": bool(evidence["performance_claim_allowed"]),
        "predictions_frozen_before_labels": True,
        "freeze_manifest": freeze_manifest,
        "base_test_image_count": len(base_test_ids),
        "base_test_map50": float(base_test["map50"]),
        "base_test_per_class_ap50": base_test["per_class_ap50"],
        "old_map50_before": float(retention["old_map50_before"]),
        "old_map50_after": float(retention["old_map50_after"]),
        "krr": float(retention["krr"]),
        "old_prediction_equivalent": bool(retention["old_prediction_equivalent"]),
        "new_map50": float(new_metrics["map50"]),
        "full_map50": float(full_metrics["map50"]),
        "full_per_class_ap50": full_metrics["per_class_ap50"],
        "thresholds": thresholds,
        "gates": gates,
        "accepted": all(gates.values()),
    }
    metrics_path = report_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))
    return 0 if metrics["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
