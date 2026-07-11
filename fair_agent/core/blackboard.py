from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from .config import rel_path, resolve_path
from .hashes import hash_if_exists, verify_sha256s
from fair_agent.modules.functional_models import validate_functional_models
from fair_agent.modules.status import fingerprints, output_freshness, parse_incremental, parse_specialist


CLASSES = ["soldier", "small_aircraft", "warship", "tank"]


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


def summarize_case_bank(rows: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    rows = list(rows)
    actions = Counter(row.get("recommended_action", "") for row in rows)
    scenes = Counter(row.get("scene", "") for row in rows)
    statuses = Counter()
    top_cases = []
    for row in rows:
        for status in (row.get("statuses") or "").split("+"):
            if status:
                statuses[status] += 1
    for row in rows[:10]:
        top_cases.append(
            {
                "rank": row.get("rank"),
                "image_path": row.get("image_path"),
                "scene": row.get("scene"),
                "statuses": row.get("statuses"),
                "priority": row.get("priority"),
                "recommended_action": row.get("recommended_action"),
            }
        )
    return {
        "case_count": len(rows),
        "scene": dict(sorted(scenes.items())),
        "status": dict(sorted(statuses.items())),
        "recommended_action": dict(sorted(actions.items())),
        "top_cases": top_cases,
    }


def artifact_status(paths: Iterable[str]) -> Dict[str, bool]:
    return {path: resolve_path(path).exists() for path in paths}


def build_blackboard(config: Dict[str, Any]) -> Dict[str, Any]:
    inputs = config.get("inputs", {})
    blackboard_cfg = config.get("blackboard", {})
    demo_path = resolve_path(blackboard_cfg.get("demo_evidence", "demo_artifacts/agent_demo_state.json"))
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
        detector["name"] = f"yolo11s_imgsz{selected_imgsz}"
        detector["imgsz"] = selected_imgsz
        detector["candidate_status"] = frozen_candidate.get("status", detector.get("candidate_status"))
        for key in ["combined_all_map50", "combined_sar_map50", "combined_soldier_map50", "bootstrap_delta_ci95"]:
            if key in frozen_candidate:
                detector[key] = frozen_candidate[key]

    metadata_rows = read_csv_if_exists(resolve_path(inputs.get("metadata", "reports/metadata.csv")))
    data_audit = read_json_if_exists(resolve_path(inputs.get("data_audit", "reports/data_audit_summary.json")))
    case_bank = read_csv_if_exists(resolve_path(inputs.get("sar_soldier_case_bank", "reports/agent_blackboard/sar_soldier_case_bank.csv")))
    dryrun = read_json_if_exists(resolve_path(inputs.get("submission_dryrun_manifest", "runs/submission/dryrun_yolo11s_imgsz640_lock_val_20260710/manifest.json")))
    smoke = read_json_if_exists(resolve_path(inputs.get("submission_smoke_manifest", "runs/submission/smoke_yolo11s_imgsz640_lock_sar_20260710/manifest.json")))
    evidence_sources = {
        "dataset": "live" if metadata_rows else "demo",
        "data_audit": "live" if data_audit else "demo",
        "sar_soldier": "live" if case_bank else "demo",
        "incremental_learning": "live",
        "submission_dryrun": "live" if dryrun else "demo",
        "submission_smoke": "live" if smoke else "demo",
    }
    dataset_summary = count_metadata(metadata_rows) if metadata_rows else dict(demo.get("dataset", {}))
    if not data_audit:
        data_audit = dict(demo.get("data_audit", {}))
    if not dryrun:
        dryrun = dict(demo.get("submission", {}).get("dryrun", {}))
    if not smoke:
        smoke = dict(demo.get("submission", {}).get("smoke", {}))

    weights_path = resolve_path(model_cfg.get("weights") or "models/base/yolo11s_ir_sar_imgsz640.pt")
    weights_hash = hash_if_exists(weights_path)
    expected_hash = model_cfg.get("expected_sha256")
    weights_hash["matches_expected"] = bool(expected_hash and weights_hash.get("sha256") == expected_hash)

    inference_config_path = resolve_path(detector.get("config", "configs/submission_infer_yolo11s_imgsz640.yaml"))
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
    diagnose_cfg = actions.get("diagnose_sar_soldier", {})
    case_freshness = output_freshness(diagnose_cfg.get("inputs", []), diagnose_cfg.get("outputs", []))
    incremental = parse_incremental(config)
    incremental_cfg = actions.get("review_incremental_learning", {})
    incremental["freshness"] = output_freshness(
        incremental_cfg.get("inputs", []), incremental_cfg.get("outputs", [])
    )
    specialist = parse_specialist(config)
    case_summary = summarize_case_bank(case_bank)
    sar_reason = "案例库专用模型略微改善了 SAR soldier，但降低了 lock_all 和 IR soldier 指标，因此主线继续使用统一 YOLO11s。"
    if not case_bank and demo:
        demo_sar = demo.get("sar_soldier", {})
        case_summary = dict(demo_sar.get("case_bank", {}))
        case_freshness = {"freshness": "current", "reason": "demo_snapshot", "missing": []}
        sar_reason = str(demo_sar.get("reason") or sar_reason)
    if specialist.get("status") == "not_run" and demo:
        specialist = dict(demo.get("sar_soldier", {}).get("specialist", specialist))
    if not incremental.get("complete") and demo.get("incremental_learning"):
        incremental = dict(demo["incremental_learning"])
        evidence_sources["incremental_learning"] = "demo"

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
    dryrun_ok = bool(dryrun.get("image_count")) and dryrun.get("model_sha256") == expected_hash
    smoke_ok = bool(smoke.get("image_count")) and smoke.get("model_sha256") == expected_hash
    if not dryrun_ok:
        blockers.append("submission_dryrun_missing_or_invalid")
    if not smoke_ok:
        blockers.append("submission_smoke_missing_or_invalid")
    if incremental.get("compliance_required") and incremental.get("complete") and not incremental.get("passed"):
        blockers.append("incremental_compliant_threshold_not_met")
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
        "runtime": config.get("runtime", {}),
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
        "sar_soldier": {
            "case_bank": case_summary,
            "case_bank_freshness": case_freshness,
            "specialist_status": specialist.get("status"),
            "specialist": specialist,
            "reason": sar_reason,
        },
        "incremental_learning": incremental,
        "submission": {
            "official_test_ready": official_ready,
            "official_format_confirmed": format_confirmed,
            "official_format": official_format,
            "official_test_dir": rel_path(source_path),
            "dryrun": dryrun,
            "smoke": smoke,
            "dryrun_valid": dryrun_ok,
            "smoke_valid": smoke_ok,
        },
        "available_actions": [
            "refresh_blackboard",
            "diagnose_sar_soldier",
            "review_incremental_learning",
            "dryrun_submission",
            "formal_submission",
        ],
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
        f"- lock_all mAP50：`{detector.get('lock_all_map50')}`",
        f"- lock_sar mAP50：`{detector.get('lock_sar_map50')}`",
        f"- lock_sar soldier mAP50：`{detector.get('lock_sar_soldier_map50')}`",
        "",
        "## 三个功能模型",
        "",
        f"- 注册表有效：`{state.get('functional_models', {}).get('valid')}`",
        f"- 不同功能数量：`{state.get('functional_models', {}).get('distinct_function_count')}`",
        f"- x86 GPU 全部就绪：`{state.get('functional_models', {}).get('all_x86_gpu_ready')}`",
        f"- Ascend 310B 全部就绪：`{state.get('functional_models', {}).get('all_ascend_310b_ready')}`",
        "",
        "## SAR Soldier 分析",
        "",
        f"- 案例库样本数：`{state.get('sar_soldier', {}).get('case_bank', {}).get('case_count')}`",
        f"- 专用模型状态：`{state.get('sar_soldier', {}).get('specialist_status')}`",
        f"- 案例库新鲜度：`{state.get('sar_soldier', {}).get('case_bank_freshness', {}).get('freshness')}`",
        f"- 结论：{state.get('sar_soldier', {}).get('reason')}",
        "",
        "## 提交状态",
        "",
        f"- 隐藏测试集就绪：`{state.get('submission', {}).get('official_test_ready')}`",
        f"- 提交格式已确认：`{state.get('submission', {}).get('official_format_confirmed')}`",
        f"- 提交格式：`{state.get('submission', {}).get('official_format')}`",
    ]
    return "\n".join(lines) + "\n"
