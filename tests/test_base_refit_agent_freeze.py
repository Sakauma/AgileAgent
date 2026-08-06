from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
FREEZER = runpy.run_path(str(ROOT / "tools" / "85_freeze_base_refit_agent.py"))
EVALUATOR = runpy.run_path(str(ROOT / "tools" / "73_evaluate_base_ensemble.py"))
FINAL_EVALUATOR = runpy.run_path(
    str(ROOT / "tools" / "87_evaluate_final_refit_policy.py")
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    weights = {
        "generic": tmp_path / "runs" / "generic" / "weights" / "best.pt",
        "crop": tmp_path / "runs" / "crop" / "weights" / "best.pt",
        "recent": tmp_path / "runs" / "recent" / "weights" / "best.pt",
    }
    for name, path in weights.items():
        path.parent.mkdir(parents=True)
        path.write_bytes(name.encode())
    weight_hashes = {
        name: FREEZER["sha256_file"](path) for name, path in weights.items()
    }

    reports = {
        "candidate": tmp_path / "reports" / "candidate.json",
        "policy": tmp_path / "reports" / "policy.json",
        "oof_generic": tmp_path / "reports" / "oof_generic.json",
        "oof_generic896": tmp_path / "reports" / "oof_generic896.json",
        "oof_crop": tmp_path / "reports" / "oof_crop.json",
        "oof_crop896": tmp_path / "reports" / "oof_crop896.json",
        "oof_recent": tmp_path / "reports" / "oof_recent.json",
        "class_owner": tmp_path / "reports" / "class_owner.json",
        "training_generic": tmp_path / "reports" / "training_generic.json",
        "training_crop": tmp_path / "reports" / "training_crop.json",
        "training_recent": tmp_path / "reports" / "training_recent.json",
        "checkpoint": tmp_path / "reports" / "checkpoint.json",
        "freeze": tmp_path / "reports" / "checkpoint_freeze.json",
        "selection": tmp_path / "reports" / "selection.json",
    }
    source_specs = {
        "oof_generic": 1024,
        "oof_generic896": 896,
        "oof_crop": 640,
        "oof_crop896": 896,
        "oof_recent": 896,
    }
    manifest_hash = "fixture-manifest-sha256"
    for source_name, imgsz in source_specs.items():
        write_json(
            reports[source_name],
            {
                "selection_scope": "base_train_and_dev_oof_only",
                "lock_data_access": False,
                "inference_mode": "full_frame",
                "manifest_sha256": manifest_hash,
                "image_count": 475,
                "models": {
                    f"fold_{index}": {
                        "completed_epochs": 160,
                        "imgsz": imgsz,
                        "inference_mode": "full_frame",
                    }
                    for index in range(5)
                },
            },
        )

    selected_policy = {
        "secondaries": ["crop_full", "recent", "generic896"],
        "fusion_iou": 0.35,
        "secondary_scale": 0.8,
        "agreement_bonus": 0.1,
        "weighted_boxes": True,
        "degraded_tuning_fold_count": 0,
        "tuning_per_class_ap50": {"0": 0.6652, "1": 0.9934, "3": 0.9056},
    }
    sources = {
        "generic": reports["oof_generic"],
        "generic896": reports["oof_generic896"],
        "crop_full": reports["oof_crop"],
        "recent": reports["oof_recent"],
    }
    write_json(
        reports["policy"],
        {
            "selection_scope": "base_train_and_dev_oof_tune_validate",
            "lock_data_access": False,
            "manifest_sha256": manifest_hash,
            "tuning_folds": [0, 1, 2],
            "validation_folds": [3, 4],
            "validation_labels_opened_after_policy_selection": True,
            "validation_predictions_frozen_before_labels": True,
            "secondaries": selected_policy["secondaries"],
            "sources": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": FREEZER["sha256_file"](path),
                }
                for name, path in sources.items()
            },
            "selected_policy": selected_policy,
            "tuning": {"fused_map50": 0.8548, "delta_map50": 0.03},
            "validation": {
                "fused_map50": 0.8731,
                "delta_map50": 0.028,
                "fused_per_class_ap50": {"0": 0.6853, "1": 0.9904, "3": 0.9437},
            },
            "all_oof_diagnostic": {
                "image_count": 475,
                "fused_map50": 0.8622,
                "fused_per_class_ap50": {"0": 0.672, "1": 0.992, "3": 0.922},
            },
        },
    )
    policy_hash = FREEZER["sha256_file"](reports["policy"])
    write_json(
        reports["candidate"],
        {
            "selection_scope": "base_train_and_dev_oof_candidate_selection",
            "lock_data_access": False,
            "selection_basis": "tuning_folds_only",
            "manifest_sha256": manifest_hash,
            "tuning_folds": [0, 1, 2],
            "validation_folds_opened_after_selection": [3, 4],
            "gates": {"validation_map50": True, "all_oof_map50": True},
            "selected": {
                "name": "generic1024",
                "source_report": str(reports["policy"].resolve()),
                "source_report_sha256": policy_hash,
                "policy": selected_policy,
            },
        },
    )
    write_json(
        reports["class_owner"],
        {
            "selection_scope": "base_train_and_dev_oof_class_owner_selection",
            "lock_data_access": False,
            "selection_basis": "maximin_tuning_fold_ap50_then_pooled_tuning_ap50",
            "manifest_sha256": manifest_hash,
            "validation_labels_opened_after_owner_selection": True,
            "validation_predictions_frozen_before_labels": True,
            "baseline": "generic1024",
            "sources": {
                "generic1024": {
                    "path": str(reports["oof_generic"].resolve()),
                    "sha256": FREEZER["sha256_file"](reports["oof_generic"]),
                },
                "generic896": {
                    "path": str(reports["oof_generic896"].resolve()),
                    "sha256": FREEZER["sha256_file"](reports["oof_generic896"]),
                },
                "crop896": {
                    "path": str(reports["oof_crop896"].resolve()),
                    "sha256": FREEZER["sha256_file"](reports["oof_crop896"]),
                },
            },
            "selected_owners": {"1": "generic896", "3": "crop896"},
            "class_results": {
                "1": {
                    "selected_owner": "generic896",
                    "tuning": {"ap50": 0.9945},
                    "validation": {"ap50": 0.9928},
                    "all_oof": {"ap50": 0.9937},
                },
                "3": {
                    "selected_owner": "crop896",
                    "tuning": {"ap50": 0.9246},
                    "validation": {"ap50": 0.9648},
                    "all_oof": {"ap50": 0.9406},
                },
            },
            "gates": {
                "1": {"tuning": True, "validation": True},
                "3": {"tuning": True, "validation": True},
            },
        },
    )

    def training_report(weight_name: str, eval_imgsz: int) -> dict:
        return {
            "selection_scope": "base_train_and_dev_only",
            "lock_data_access": False,
            "dataset_audit": {
                "lock_data_access": False,
                "source_declared_test_split": False,
                "training_declared_test_split": False,
                "train_dev_overlap": [],
                "train_count": 427,
                "dev_count": 48,
            },
            "training": {
                "requested_epochs": 160,
                "completed_epochs": 160,
                "stopped_early": False,
                "best_epoch": 73,
                "best_metric_value": 0.88,
                "arguments": {"epochs": 160, "batch": 32, "patience": 0},
            },
            "evaluation": {"imgsz": eval_imgsz, "map50": 0.90},
            "best_weight": str(weights[weight_name].resolve()),
            "best_weight_sha256": weight_hashes[weight_name],
        }

    write_json(reports["training_generic"], training_report("generic", 896))
    write_json(reports["training_crop"], training_report("crop", 640))
    write_json(reports["training_recent"], training_report("recent", 896))

    model_specs = {
        "generic_b_1024": ("generic", 1024),
        "generic_b_896": ("generic", 896),
        "crop_a_full_640": ("crop", 640),
        "crop_a_full_896": ("crop", 896),
        "recent_crop_a_896": ("recent", 896),
    }
    config = {
        "candidate_id": "fixture-v2",
        "selection_scope": "base_train_dev_oof_policy_and_refit_weights",
        "lock_data_access_during_selection": False,
        "paths": {
            "selection_report": str(reports["selection"]),
            "oof_candidate_selection_report": str(reports["candidate"]),
            "oof_policy_report": str(reports["policy"]),
            "class_owner_selection_report": str(reports["class_owner"]),
            "checkpoint_holdout_report": str(reports["checkpoint"]),
            "training_reports": {
                "generic_b_1024": str(reports["training_generic"]),
                "generic_b_896": str(reports["training_generic"]),
                "crop_a_full_640": str(reports["training_crop"]),
                "crop_a_full_896": str(reports["training_crop"]),
                "recent_crop_a_896": str(reports["training_recent"]),
            },
        },
        "models": {
            name: {
                "weights": str(weights[weight_name]),
                "imgsz": imgsz,
                "inference_mode": "full_frame",
            }
            for name, (weight_name, imgsz) in model_specs.items()
        },
        "oof_owner_mapping": {
            "generic": "generic_b_1024",
            "generic896": "generic_b_896",
            "crop_full": "crop_a_full_640",
            "recent": "recent_crop_a_896",
        },
        "class_owner_mapping": {
            "generic1024": "generic_b_1024",
            "generic896": "generic_b_896",
            "crop896": "crop_a_full_896",
        },
        "old_class_ids": [0, 1, 3],
        "base_fusion": {
            "class_policies": {
                0: {
                    "primary": "generic_b_1024",
                    "secondary_scales": {
                        "crop_a_full_640": 0.8,
                        "recent_crop_a_896": 0.8,
                        "generic_b_896": 0.8,
                    },
                    "scale_before_clustering": False,
                    "iou": 0.35,
                    "agreement_bonus": 0.1,
                    "weighted_boxes": True,
                },
                1: {"primary": "generic_b_896", "secondary_scales": {}},
                3: {"primary": "crop_a_full_896", "secondary_scales": {}},
            }
        },
        "incremental_fusion": {"cross_class": {"enabled": False}},
        "selection_requirements": {
            "required_epochs": 160,
            "required_batch": 32,
            "required_patience": 0,
            "max_degraded_tuning_folds": 0,
            "min_oof_map50": 0.85,
            "min_validation_fold_map50": 0.85,
        },
    }
    config_path = tmp_path / "base_ensemble.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    config_hash = FREEZER["sha256_file"](config_path)

    checkpoint_models = {
        name: {
            "weight_sha256": weight_hashes[weight_name],
            "imgsz": imgsz,
            "inference_mode": "full_frame",
        }
        for name, (weight_name, imgsz) in model_specs.items()
    }
    artifact_entries = {}
    for name in [*model_specs, "fused"]:
        artifact = tmp_path / "reports" / "predictions" / f"{name}.jsonl"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}\n", encoding="utf-8")
        artifact_entries[name] = {
            "path": str(artifact.resolve()),
            "sha256": FREEZER["sha256_file"](artifact),
            "row_count": 1,
        }
    selection_audit = {
        "selection_sha256": FREEZER["sha256_file"](reports["candidate"]),
        "policy_sha256": policy_hash,
        "selected_policy": selected_policy,
        "owner_mapping": config["oof_owner_mapping"],
    }
    selection_audit["class_owner_selection"] = FINAL_EVALUATOR[
        "verify_class_owner_selection"
    ](
        config,
        {"policy_path": str(reports["policy"]), "selected_policy": selected_policy},
    )
    write_json(
        reports["freeze"],
        {
            "config_sha256": config_hash,
            "selection_audit": selection_audit,
            "base_fusion": config["base_fusion"],
            "models_receiving_identical_complete_input": sorted(model_specs),
            "label_aware_routing": False,
            "scene_hard_routing": False,
            "filename_class_routing": False,
            "predictions_frozen_before_labels": True,
            "artifacts": artifact_entries,
        },
    )
    write_json(
        reports["checkpoint"],
        {
            "selection_scope": "final_refit_checkpoint_holdout",
            "lock_data_access": False,
            "performance_evidence": False,
            "independent_test_evidence": False,
            "evidence_limitation": "used_during_best_checkpoint_selection",
            "predictions_frozen_before_labels": True,
            "candidate_id": config["candidate_id"],
            "config_sha256": config_hash,
            "base_fusion": config["base_fusion"],
            "split": str(tmp_path / "checkpoint_val.txt"),
            "image_count": 48,
            "selection_audit": selection_audit,
            "models": checkpoint_models,
            "freeze_manifest": {
                "path": str(reports["freeze"].resolve()),
                "sha256": FREEZER["sha256_file"](reports["freeze"]),
            },
            "map50": 0.9257,
            "per_class_ap50": {"0": 0.8474, "1": 0.9802, "3": 0.9496},
        },
    )
    return config_path, reports


def test_freeze_accepts_four_pass_oof_policy_and_full_refits(tmp_path: Path) -> None:
    config_path, reports = build_fixture(tmp_path)

    frozen = FREEZER["freeze_selection"](config_path)

    assert frozen["selection_scope"] == "base_train_dev_oof_policy_and_refit_weights"
    assert frozen["oof_policy_evidence"]["all_oof_map50"] == pytest.approx(0.8622)
    assert frozen["oof_policy_evidence"]["validation_map50"] == pytest.approx(0.8731)
    assert set(frozen["models"]) == {
        "generic_b_1024",
        "generic_b_896",
        "crop_a_full_640",
        "crop_a_full_896",
        "recent_crop_a_896",
    }
    assert frozen["unique_weight_count"] == 3
    assert all(row["completed_epochs"] == 160 for row in frozen["models"].values())
    assert frozen["checkpoint_holdout_sanity"]["map50"] == pytest.approx(0.9257)
    assert frozen["checkpoint_holdout_sanity"]["performance_evidence"] is False
    assert frozen["independent_test_evidence"] is False
    assert reports["selection"].is_file()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit = EVALUATOR["verify_selection"](config, config_path)
    assert audit["selection_scope"] == frozen["selection_scope"]


def test_freeze_rejects_training_that_did_not_run_full_budget(tmp_path: Path) -> None:
    config_path, reports = build_fixture(tmp_path)
    report = json.loads(reports["training_generic"].read_text(encoding="utf-8"))
    report["training"]["completed_epochs"] = 159
    write_json(reports["training_generic"], report)

    with pytest.raises(ValueError, match="未按 epochs=160"):
        FREEZER["freeze_selection"](config_path)


def test_freeze_rejects_image_routing_in_prediction_manifest(tmp_path: Path) -> None:
    config_path, reports = build_fixture(tmp_path)
    freeze = json.loads(reports["freeze"].read_text(encoding="utf-8"))
    freeze["label_aware_routing"] = True
    write_json(reports["freeze"], freeze)
    checkpoint = json.loads(reports["checkpoint"].read_text(encoding="utf-8"))
    checkpoint["freeze_manifest"]["sha256"] = FREEZER["sha256_file"](reports["freeze"])
    write_json(reports["checkpoint"], checkpoint)

    with pytest.raises(ValueError, match="无路由约束"):
        FREEZER["freeze_selection"](config_path)


def test_regression_evidence_cannot_claim_independent_test() -> None:
    evidence = EVALUATOR["evaluation_evidence"](
        {
            "evaluation_protocol": {
                "evidence_kind": "local_regression_reuse_not_independent",
                "independent_test_evidence": False,
                "labels_may_only_be_read_after_prediction_freeze": True,
            }
        }
    )
    assert evidence == {
        "evidence_kind": "local_regression_reuse_not_independent",
        "independent_test_evidence": False,
        "performance_claim_allowed": False,
    }


def test_regression_evidence_rejects_false_independent_claim() -> None:
    with pytest.raises(ValueError, match="声明矛盾"):
        EVALUATOR["evaluation_evidence"](
            {
                "evaluation_protocol": {
                    "evidence_kind": "local_regression_reuse_not_independent",
                    "independent_test_evidence": True,
                }
            }
        )
