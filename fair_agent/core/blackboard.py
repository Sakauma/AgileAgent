from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from .config import rel_path, resolve_path, runtime_platform_info
from .hashes import hash_if_exists, verify_sha256s
from fair_agent.modules.functional_models import validate_functional_models
from fair_agent.modules.model_generations import load_generation_registry
from fair_agent.modules.status import fingerprints


DEFAULT_DEMO_EVIDENCE = "configs/agent_demo_state.json"
LEGACY_DEMO_EVIDENCE = "demo_artifacts/agent_demo_state.json"


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_if_exists(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def count_metadata(rows: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    rows = list(rows)
    sensor = Counter(row.get("sensor", "") for row in rows)
    scene = Counter(row.get("scene", "") for row in rows)
    sensor_scene = Counter(f"{row.get('sensor', '')}/{row.get('scene', '')}" for row in rows)
    class_presence = Counter()
    total_objects = 0
    for row in rows:
        total_objects += int(row.get("num_objects") or 0)
        for name in (row.get("classes_present") or "").split(";"):
            if name:
                class_presence[name] += 1
    return {
        "image_count": len(rows),
        "object_count": total_objects,
        "sensor": dict(sorted(sensor.items())),
        "scene": dict(sorted(scene.items())),
        "sensor_scene": dict(sorted(sensor_scene.items())),
        "class_presence_images": dict(sorted(class_presence.items())),
    }


def artifact_status(paths: Iterable[str]) -> Dict[str, bool]:
    return {path: resolve_path(path).exists() for path in paths}


def resolve_demo_evidence(config: Dict[str, Any]) -> Path:
    configured = str(config.get("blackboard", {}).get("demo_evidence") or DEFAULT_DEMO_EVIDENCE)
    path = resolve_path(configured)
    if path.exists() or configured != LEGACY_DEMO_EVIDENCE:
        return path
    return resolve_path(DEFAULT_DEMO_EVIDENCE)


def build_blackboard(config: Dict[str, Any]) -> Dict[str, Any]:
    inputs = config.get("inputs", {})
    demo_path = resolve_demo_evidence(config)
    demo = read_json_if_exists(demo_path)
    detector = dict(config.get("detector", {}))
    model_cfg = dict(config.get("model", {}))
    assets_cfg = dict(config.get("assets", {}))
    functional_cfg = dict(config.get("functional_models", {}))
    functional_models = validate_functional_models(functional_cfg.get("registry", "configs/functional_models.yaml"))
    detector["weights"] = model_cfg.get("weights")
    detector["expected_sha256"] = model_cfg.get("expected_sha256")
    manifest_path = resolve_path(assets_cfg.get("manifest", "models/manifest.json"))
    frozen_manifest = read_json_if_exists(manifest_path)
    frozen_candidate = frozen_manifest.get("frozen_candidate") or frozen_manifest.get("base_model", {})
    if frozen_candidate:
        selected_imgsz = int(frozen_candidate.get("imgsz", 640))
        architecture = str(frozen_candidate.get("architecture") or "yolo").lower()
        detector["name"] = f"{architecture}_imgsz{selected_imgsz}"
        detector["imgsz"] = selected_imgsz
        detector["candidate_status"] = frozen_candidate.get("status", detector.get("candidate_status"))
        for key in ["base_test_map50", "evaluation_split"]:
            if key in frozen_candidate:
                detector[key] = frozen_candidate[key]

    metadata_rows = read_csv_if_exists(resolve_path(inputs.get("metadata", "reports/metadata.csv")))
    data_audit = read_json_if_exists(resolve_path(inputs.get("data_audit", "reports/data_audit_summary.json")))
    evidence_sources = {
        "dataset": "live" if metadata_rows else "demo",
        "data_audit": "live" if data_audit else "demo",
    }
    dataset_summary = count_metadata(metadata_rows) if metadata_rows else dict(demo.get("dataset", {}))
    if not data_audit:
        data_audit = dict(demo.get("data_audit", {}))

    weights_path = resolve_path(
        model_cfg.get("weights")
        or "models/production/incremental_detection/four_class_base_detector.pt"
    )
    weights_hash = hash_if_exists(weights_path)
    expected_hash = model_cfg.get("expected_sha256")
    weights_hash["matches_expected"] = bool(expected_hash and weights_hash.get("sha256") == expected_hash)

    inference_config_path = resolve_path(
        detector.get("config", "configs/submission_infer_base_4class.yaml")
    )
    inference_config: Dict[str, Any] = {}
    if inference_config_path.exists():
        loaded = yaml.safe_load(inference_config_path.read_text(encoding="utf-8"))
        inference_config = loaded if isinstance(loaded, dict) else {}
    inference_weights_path = resolve_path(inference_config.get("model", "__missing_inference_model__"))
    inference_weights_hash = hash_if_exists(inference_weights_path)
    inference_weights_hash["matches_expected"] = bool(
        expected_hash and inference_weights_hash.get("sha256") == expected_hash
    )
    inference_weights_hash["same_frozen_path"] = inference_weights_path.resolve() == weights_path.resolve()

    actions = config.get("decision", {}).get("actions", {})
    generation_registry_path = assets_cfg.get("generation_registry", "models/generations.json")
    generation_summary: Dict[str, Any] = {"valid": False, "path": str(generation_registry_path)}
    incremental: Dict[str, Any] = {
        "protocols": [],
        "complete": False,
        "passed": False,
        "source": str(generation_registry_path),
        "compliance_required": True,
        "compliance_verified": False,
        "warnings": [],
        "task_type": "incremental_object_detection",
        "primary_mode": config.get("incremental", {}).get("primary_mode", "class_incremental"),
        "supported_modes": list(config.get("incremental", {}).get("supported_modes", [])),
        "learning_data_scope": config.get("incremental", {}).get("learning_data_scope"),
        "freshness": {"freshness": "missing", "reason": "generation_registry_invalid", "missing": []},
    }
    production_incremental_verified = False
    try:
        generation_registry = load_generation_registry(generation_registry_path)
        production_id = str(generation_registry["channels"]["production"])
        production = generation_registry["generations_by_id"][production_id]
        production_model_ids = set(production["class_owners"].values())
        production_experts = [
            generation_registry["models_by_id"][model_id]
            for model_id in production_model_ids
            if generation_registry["models_by_id"][model_id].get("role") == "class_incremental_expert"
        ]
        production_incremental_verified = bool(production_experts) and all(
            model.get("acceptance", {}).get("passed") is True for model in production_experts
        )
        production_metrics = dict(production.get("metrics", {}))
        core_metrics_passed = bool(production.get("acceptance", {}).get("core_metrics_passed"))
        class_map = generation_registry.get("class_map", {})
        protocols = []
        for model in production_experts:
            class_ids = sorted(int(value) for value in model.get("owns_classes", []))
            model_passed = model.get("acceptance", {}).get("passed") is True and core_metrics_passed
            protocols.append(
                {
                    "protocol": model.get("id"),
                    "new_class": ",".join(
                        str(class_map.get(class_id, class_id))
                        for class_id in class_ids
                    ),
                    "new_class_ids": class_ids,
                    "task_type": "incremental_object_detection",
                    "incremental_mode": "class_incremental",
                    "learning_data_scope": config.get("incremental", {}).get("learning_data_scope"),
                    "learning_scope_verified": model_passed,
                    "new_map50": model.get("metrics", {}).get("new_map50", production_metrics.get("new_map50")),
                    "krr": production_metrics.get("krr"),
                    "compliant": model_passed,
                    "passed": model_passed,
                }
            )
        incremental.update(
            {
                "protocols": protocols,
                "complete": bool(protocols),
                "passed": bool(protocols) and all(row["passed"] for row in protocols),
                "compliance_verified": bool(protocols) and core_metrics_passed,
                "freshness": {"freshness": "current", "reason": "production_generation", "missing": []},
            }
        )
        generation_summary = {
            "valid": True,
            "path": str(generation_registry_path),
            "production": production_id,
            "classes": sorted(int(value) for value in production["classes"]),
            "incremental_verified": production_incremental_verified,
            "metrics": production_metrics,
        }
        evidence_sources["model_generation"] = "live"
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        generation_summary["error"] = str(exc)
        evidence_sources["model_generation"] = "missing"

    fixed_artifacts = list(assets_cfg.get("required", [model_cfg.get("weights")]))
    checksum_path = resolve_path(assets_cfg.get("checksums", "models/SHA256SUMS.txt"))

    submission_cfg = config.get("submission", {})
    official_ready = bool(submission_cfg.get("official_test_ready", False))
    official_format = submission_cfg.get("official_format")
    format_confirmed = bool(submission_cfg.get("official_format_confirmed", False))
    source_path = resolve_path(submission_cfg.get("official_test_dir", "datasets_r1_base_test"))

    blockers = []
    if not weights_hash["matches_expected"]:
        blockers.append("final_weight_hash_not_verified")
    if not inference_weights_hash["matches_expected"] or not inference_weights_hash["same_frozen_path"]:
        blockers.append("inference_weight_not_frozen_or_verified")
    if not official_ready:
        blockers.append("official_test_not_ready")
    if not format_confirmed or not official_format:
        blockers.append("official_format_not_confirmed")
    if official_ready and not source_path.exists():
        blockers.append("official_test_dir_missing")
    strict_class_incremental_verified = any(
        item.get("function") == "incremental_object_detection"
        and item.get("evidence", {}).get("summary", {}).get("true_class_incremental_verified") is True
        for item in functional_models.get("models", [])
    )
    if (
        incremental.get("compliance_required")
        and incremental.get("complete")
        and not incremental.get("passed")
        and not strict_class_incremental_verified
        and not production_incremental_verified
    ):
        blockers.append("incremental_compliant_threshold_not_met")
    if not generation_summary["valid"]:
        blockers.append("model_generation_registry_invalid")
    if not functional_models.get("valid"):
        blockers.append("functional_models_invalid")
    elif not functional_models.get("all_x86_gpu_ready"):
        blockers.append("functional_models_x86_not_ready")
    if not functional_models.get("all_ascend_310b_ready"):
        blockers.append("ascend_310b_not_ready")

    config_path = resolve_path(config.get("_config_path", "configs/agent_pipeline.yaml"))
    tracked_inputs: List[str] = []
    for value in inputs.values():
        if isinstance(value, str):
            tracked_inputs.append(value)
        elif isinstance(value, list):
            tracked_inputs.extend(str(item) for item in value)
    tracked_inputs.append(rel_path(demo_path))
    tracked_inputs.append(str(functional_cfg.get("registry", "configs/functional_models.yaml")))
    tracked_inputs.append(str(config.get("incremental", {}).get("policy", "configs/incremental_detection_policy.yaml")))

    source_values = set(evidence_sources.values())
    evidence_mode = "live" if source_values == {"live"} else ("demo" if source_values == {"demo"} else "mixed")

    state = {
        "schema_version": 4,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evidence": {
            "mode": evidence_mode,
            "sources": evidence_sources,
            "demo_path": rel_path(demo_path),
            "demo_hash": hash_if_exists(demo_path),
        },
        "config": {"path": rel_path(config_path), **hash_if_exists(config_path)},
        "input_fingerprints": fingerprints(tracked_inputs),
        "runtime": {
            **dict(config.get("runtime", {})),
            "platform": runtime_platform_info(config),
        },
        "dataset": dataset_summary,
        "data_audit": {
            "total_images": data_audit.get("total_images"),
            "total_labels": data_audit.get("total_labels"),
            "total_objects": data_audit.get("total_objects"),
            "missing_images": data_audit.get("missing_images"),
            "missing_labels": data_audit.get("missing_labels"),
            "invalid_errors": data_audit.get("invalid_errors"),
        },
        "detector": detector,
        "functional_models": functional_models,
        "frozen_assets": {
            "weights": {"path": rel_path(weights_path), **weights_hash},
            "inference_weights": {"path": rel_path(inference_weights_path), **inference_weights_hash},
            "inference_config": {"path": rel_path(inference_config_path), **hash_if_exists(inference_config_path)},
            "manifest": {"path": rel_path(manifest_path), **hash_if_exists(manifest_path)},
            "checksums": {"path": rel_path(checksum_path), **verify_sha256s(checksum_path)},
            "artifacts": artifact_status(fixed_artifacts),
        },
        "incremental_learning": incremental,
        "model_generation": generation_summary,
        "submission": {
            "official_test_ready": official_ready,
            "official_format_confirmed": format_confirmed,
            "official_format": official_format,
            "official_test_dir": rel_path(source_path),
        },
        "available_actions": sorted(actions),
        "current_blockers": blockers,
    }
    if not state["frozen_assets"]["checksums"]["valid"]:
        state["current_blockers"].append("frozen_asset_checksums_invalid")
    return state


def write_blackboard(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Path]:
    blackboard_cfg = config.get("blackboard", {})
    out_dir = resolve_path(blackboard_cfg.get("output_dir", "reports/agent_blackboard"))
    state_path = out_dir / blackboard_cfg.get("state_json", "blackboard_state.json")
    report_path = out_dir / blackboard_cfg.get("report_md", "agent_preparation_report.md")
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_blackboard_report(state), encoding="utf-8")
    return {"state": state_path, "report": report_path}


def render_blackboard_report(state: Dict[str, Any]) -> str:
    detector = state.get("detector", {})
    dataset = state.get("dataset", {})
    weights = state.get("frozen_assets", {}).get("weights", {})
    blockers = state.get("current_blockers", [])
    lines = [
        "# 智能体黑板报告",
        "",
        f"生成时间：{state.get('generated_at')}",
        f"证据模式：`{state.get('evidence', {}).get('mode')}`",
        "",
        "## 总体状态",
        "",
        f"- 当前阻塞项：`{blockers or ['无']}`",
        f"- 数据量：`{dataset.get('image_count')}` 张图像 / `{dataset.get('object_count')}` 个目标",
        f"- 传感器分布：`{dataset.get('sensor')}`",
        f"- 场景分布：`{dataset.get('scene')}`",
        "",
        "## 最终检测器",
        "",
        f"- 名称：`{detector.get('name')}`",
        f"- 候选状态：`{detector.get('candidate_status')}`",
        f"- 权重：`{weights.get('path')}`",
        f"- SHA256 匹配：`{weights.get('matches_expected')}`",
        f"- 基础测试 mAP50：`{detector.get('base_test_map50')}`",
        f"- 基础测试清单：`{detector.get('evaluation_split')}`",
        "",
        "## 三个功能模型",
        "",
        f"- 注册表有效：`{state.get('functional_models', {}).get('valid')}`",
        f"- 不同功能数量：`{state.get('functional_models', {}).get('distinct_function_count')}`",
        f"- x86 GPU 全部就绪：`{state.get('functional_models', {}).get('all_x86_gpu_ready')}`",
        f"- Ascend 310B 全部就绪：`{state.get('functional_models', {}).get('all_ascend_310b_ready')}`",
        "",
        "## 当前生产代际",
        "",
        f"- 代际：`{state.get('model_generation', {}).get('production')}`",
        f"- 类别：`{state.get('model_generation', {}).get('classes')}`",
        f"- 增量模型已验证：`{state.get('model_generation', {}).get('incremental_verified')}`",
        f"- 增量协议数：`{len(state.get('incremental_learning', {}).get('protocols', []))}`",
        "",
        "## 提交状态",
        "",
        f"- 本地评测输入已配置：`{state.get('submission', {}).get('official_test_ready')}`",
        f"- 提交格式已确认：`{state.get('submission', {}).get('official_format_confirmed')}`",
        f"- 提交格式：`{state.get('submission', {}).get('official_format')}`",
    ]
    return "\n".join(lines) + "\n"
