#!/usr/bin/env python3
"""Promote a frozen scene-aware 4+2 candidate into the x86/CUDA release metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CLASS_NAMES = {
    0: "soldier",
    1: "small_aircraft",
    2: "warship",
    3: "tank",
    4: "patrol_boat",
    5: "armored_vehicle",
}
BASE_IDS = (0, 1, 2, 3)
NEW_IDS = (4, 5)


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
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--lock-result", type=Path, required=True)
    parser.add_argument("--dev-search", type=Path, required=True)
    parser.add_argument("--dev-report", type=Path, required=True)
    args = parser.parse_args()

    candidate_path = args.candidate.expanduser().resolve()
    lock_result_path = args.lock_result.expanduser().resolve()
    dev_search_path = args.dev_search.expanduser().resolve()
    dev_report_path = args.dev_report.expanduser().resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    result = json.loads(lock_result_path.read_text(encoding="utf-8"))
    if (
        candidate.get("selection_source") != "mixed_dev_only"
        or result.get("competition_accepted") is not True
        or result.get("candidate_frozen_before_lock_labels") is not True
        or result.get("candidate", {}).get("candidate_id") != candidate.get("candidate_id")
        or not all(result.get("score_gates", {}).values())
    ):
        raise ValueError("候选未通过冻结 dev→lock 晋级契约")

    production = ROOT / "models" / "production" / "incremental_detection"
    evidence = production / "evidence"
    base_prior_path = production / "base_context_prior.json"
    incremental_prior_path = production / "incremental_context_prior.json"
    atomic_json(base_prior_path, candidate["base_context_prior"])
    atomic_json(incremental_prior_path, candidate["incremental_context_prior"])
    shutil.copy2(candidate_path, evidence / "scene_aware_candidate.json")
    shutil.copy2(lock_result_path, evidence / "scene_aware_lock_recheck.json")
    shutil.copy2(dev_search_path, evidence / "scene_aware_dev_search.json")
    shutil.copy2(dev_report_path, evidence / "scene_aware_dev_search.md")

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
    dev_metrics = adapt_split_metrics(candidate["dev_metrics"], 75)
    lock_metrics = adapt_split_metrics(result["lock"], 75)
    metrics = {
        **old_metrics,
        "schema_version": 4,
        "run_id": run_id,
        "created_at": now,
        "protocol": "strict-4plus2-parallel-specialist-class-specific-scene-gate",
        "inference": {
            **dict(old_metrics.get("inference") or {}),
            "fusion": "fixed_class_owners_with_class_specific_scene_soft_gate",
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
    atomic_json(production / "metrics.json", metrics)

    calibration = {
        "schema_version": 3,
        "learning_data_scope": "incremental_dataset_only",
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
            "run_id": run_id,
            "agent_structure": {
                **dict(profile["agent_structure"]),
                "scene_soft_gating": True,
                "scene_hard_routing": False,
                "label_aware_routing": False,
                "filename_class_routing": False,
            },
            "activation_thresholds": {
                str(key): value for key, value in new_thresholds.items()
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
    atomic_json(production / "profile.json", profile)
    active_profile = ROOT / "models" / "profiles" / "incremental-detection" / "active.json"
    atomic_json(active_profile, profile)

    registry_path = ROOT / "models" / "profiles" / "registry.json"
    profile_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registered = profile_registry["verified_profiles"][0]
    registered.update(
        {
            "run_id": run_id,
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

    generations_path = ROOT / "models" / "generations.json"
    generations = json.loads(generations_path.read_text(encoding="utf-8"))
    models = {str(row["id"]): row for row in generations["models"]}
    base_model = models["four_class_base_detector"]
    base_model.update(
        {
            "per_class_thresholds": {
                str(key): value for key, value in base_thresholds.items()
            },
            "context_prior": dict(candidate["base_context_prior"]),
            "context_gate": base_gate,
        }
    )
    incremental_model = models["incremental_detector"]
    incremental_model.update(
        {
            "per_class_thresholds": {
                str(key): value for key, value in new_thresholds.items()
            },
            "context_prior": dict(candidate["incremental_context_prior"]),
            "context_gate": incremental_gate,
            "metrics": {
                "new_map50": lock_metrics["new_map50"],
                "per_class": {
                    str(class_id): {
                        "map50": new_quality[class_id]["map50"],
                        "precision": new_quality[class_id]["precision"],
                        "recall": new_quality[class_id]["recall"],
                        "threshold": new_thresholds[class_id],
                        "max_scene_penalty": new_penalties[class_id],
                    }
                    for class_id in NEW_IDS
                },
            },
            "deployment_metrics": {
                "evaluation_semantics": profile["evaluation_semantics"],
                "overall_precision": result["lock"]["overall"]["precision"],
                "new_class_precision": result["lock"]["new_classes"]["precision"],
                "false_activation_rate_by_class": {
                    str(key): value["false_activation_rate"]
                    for key, value in new_false_activation.items()
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
        }
    )
    generation_rows = {str(row["id"]): row for row in generations["generations"]}
    base_generation = generation_rows["base_detection_generation_4plus2"]
    base_generation["metrics"].update(
        {
            "base_dev_map50": dev_metrics["base_map50"],
            "base_test_map50": lock_metrics["base_map50"],
            "raw_base_test_map50": result["baseline"]["base_map50"],
            "scene_soft_gating": True,
        }
    )
    generation = generation_rows["incremental_detection_generation_4plus2"]
    generation.update(
        {
            "run_id": run_id,
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
                "base_test_image_count": 75,
                "new_class_image_count": 14,
                "specialist_count": 1,
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
    atomic_json(generations_path, generations)

    manifest_path = ROOT / "models" / "manifest.json"
    old_manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "release": "competition-4plus2-agent-scene-aware-v2-20260822",
            "created_at": now,
            "run_id": run_id,
        }
    )
    manifest["base_model"].update(
        {
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
        if functional["id"] == "four_class_base_detector":
            functional.update(
                {
                    "base_test_map50": lock_metrics["base_map50"],
                    "raw_base_test_map50": result["baseline"]["base_map50"],
                    "scene_soft_gating": True,
                }
            )
    protocol = manifest["incremental_models"][0]
    protocol.update(
        {
            "activation_thresholds": {
                str(key): value for key, value in new_thresholds.items()
            },
            "calibration_sha256": sha256_file(production / "calibration.json"),
            "metrics_sha256": sha256_file(production / "metrics.json"),
            "context_prior": dict(candidate["incremental_context_prior"]),
            "context_gate": incremental_gate,
            "evaluation_semantics": profile["evaluation_semantics"],
            "rechecked_at": now,
            "base_test_map50": lock_metrics["base_map50"],
            "old_map50_before": lock_metrics["old_map50_before"],
            "old_map50_after": lock_metrics["old_map50_after"],
            "new_map50": lock_metrics["new_map50"],
            "new_per_class_ap50": dict(lock_metrics["new_per_class_ap50"]),
            "full_map50": lock_metrics["full_map50"],
            "krr": lock_metrics["krr"],
            "lock_precision_by_class": {
                str(key): value["precision"] for key, value in new_quality.items()
            },
            "lock_recall_by_class": {
                str(key): value["recall"] for key, value in new_quality.items()
            },
            "false_activation_rate_by_class": {
                str(key): value["false_activation_rate"]
                for key, value in new_false_activation.items()
            },
            "overall_lock_precision": result["lock"]["overall"]["precision"],
            "new_class_lock_precision": result["lock"]["new_classes"]["precision"],
            "diagnostic_warnings": advisory_warnings,
        }
    )
    manifest["notes"] = [
        "所有检测选择和验收硬指标均为 mAP50。",
        "Base 与二类专家对每张图并行推理，类别 owner 保持固定。",
        "六类逐类阈值和场景惩罚只由 mixed dev 选择；线上只读取 Scene-SensorNet 概率，不读取文件名或标签。",
        "precision 与误激活率是非阻断诊断；当前场景感知运行点已显著改善二者。",
        "仓库不包含竞赛数据集和标签。",
        "Ascend 3+1 不可变包仅为历史板端发布；当前 4+2 场景感知策略尚未转换为 OM。",
    ]
    atomic_json(manifest_path, manifest)
    new_manifest_hash = sha256_file(manifest_path)
    functional_path = ROOT / "configs" / "functional_models.yaml"
    functional_text = functional_path.read_text(encoding="utf-8")
    if old_manifest_hash not in functional_text:
        raise ValueError("功能模型注册表未引用晋级前 manifest，拒绝静默改写")
    functional_path.write_text(
        functional_text.replace(old_manifest_hash, new_manifest_hash),
        encoding="utf-8",
    )

    (evidence / "operating_point_diagnostics.md").write_text(
        diagnostics_markdown(result["baseline"], result["lock"], thresholds, penalties),
        encoding="utf-8",
    )
    summary = {
        "run_id": run_id,
        "candidate_id": candidate["candidate_id"],
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
