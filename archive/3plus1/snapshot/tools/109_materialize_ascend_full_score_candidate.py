#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import validate_config  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label}必须是YAML mapping：{path}")
    return payload


def _absolute_strings(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_absolute_strings(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_absolute_strings(item, f"{prefix}[{index}]"))
    elif isinstance(value, str) and (
        value.startswith(("/home/", "/root/", "/mnt/")) or WINDOWS_ABSOLUTE.match(value)
    ):
        found.append(prefix)
    return found


def validate_method(method: Mapping[str, Any]) -> None:
    if (
        method.get("schema_version") != 1
        or method.get("kind") != "ascend310b_full_score_method"
    ):
        raise ValueError("满分方法配置schema/kind不受支持")
    absolute = _absolute_strings(method)
    if absolute:
        raise ValueError("满分方法配置禁止板端/本机绝对路径：" + ", ".join(absolute))
    target = method.get("target") or {}
    if target.get("candidate_port") != 8502 or target.get("formal_port") != 8501:
        raise ValueError("满分方法必须固定8502候选和8501正式端口")
    training = method.get("training") or {}
    if (
        training.get("method") != "residual_adapter"
        or training.get("checkpoint_metric") != "map50"
        or training.get("export_checkpoints") != ["best", "last"]
        or training.get("reference_export_checkpoint") != "last"
    ):
        raise ValueError("满分方法必须固化residual adapter及best/last导出契约")
    export = method.get("export") or {}
    if (
        export.get("model_layout") != "shared_backbone_dual_head_v1"
        or export.get("output_contract") != "raw_dual_head_v1"
    ):
        raise ValueError(
            "满分方法必须使用shared_backbone_dual_head_v1/raw_dual_head_v1"
        )
    if export.get("input_name") != "images" or export.get("input_shape_nchw") != [
        1,
        3,
        736,
        896,
    ]:
        raise ValueError("满分方法输入契约必须为images:[1,3,736,896]")
    benchmark = method.get("benchmark") or {}
    competition = method.get("competition") or {}
    accuracy = competition.get("accuracy_gates") or {}
    performance = competition.get("performance_gate") or {}
    if (
        set(accuracy)
        != {"base_map50_min", "new_map50_min", "krr_min"}
        or any(not 0.0 <= float(value) <= 1.0 for value in accuracy.values())
        or benchmark.get("base_url")
        != f"http://127.0.0.1:{target.get('candidate_port')}"
        or benchmark.get("official_url")
        != f"http://127.0.0.1:{target.get('formal_port')}"
        or benchmark.get("batch_probe_size")
        != performance.get("batch_image_count")
        or benchmark.get("batch_rounds") != performance.get("batch_rounds")
        or float(benchmark.get("target_batch_fps", -1.0))
        != float(performance.get("median_fps_min", -2.0))
    ):
        raise ValueError("benchmark协议必须与competition performance_gate一致")


def structural_logical_heads(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("logical_heads必须是mapping")
    heads = copy.deepcopy(dict(value))
    for head in heads.values():
        if not isinstance(head, dict):
            raise ValueError("logical head必须是mapping")
        head.pop("candidate_confidence", None)
        head.pop("output_shape", None)
    return heads


def _artifact_by_role(manifest: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("build manifest缺少artifacts")
    rows = [
        row
        for row in artifacts.values()
        if isinstance(row, Mapping) and row.get("role") == role
    ]
    if len(rows) != 1:
        raise ValueError(f"build manifest要求恰好一个{role}资产，实际{len(rows)}")
    return rows[0]


def _verify_manifest_asset(entry: Mapping[str, Any], path: Path, label: str) -> str:
    recorded = entry.get("om")
    if not isinstance(recorded, Mapping):
        raise ValueError(f"build manifest缺少{label}.om")
    digest = sha256_file(path)
    if recorded.get("sha256") != digest:
        raise ValueError(f"{label} OM SHA256与build manifest不一致")
    manifest_path = Path(str(recorded.get("path") or ""))
    if manifest_path.name != path.name:
        raise ValueError(f"{label} OM文件名与build manifest不一致")
    return digest


def _verify_manifest_evidence(manifest: Mapping[str, Any], key: str) -> None:
    entry = manifest.get(key)
    if not isinstance(entry, Mapping):
        raise ValueError(f"build manifest缺少{key}")
    path = Path(str(entry.get("path") or ""))
    digest = str(entry.get("sha256") or "")
    if not path.is_file() or len(digest) != 64 or sha256_file(path) != digest:
        raise ValueError(f"build manifest {key}缺失或SHA256不一致")


def build_candidate_config(
    base: Mapping[str, Any],
    method: Mapping[str, Any],
    *,
    dual_om: Path,
    context_om: Path,
    build_manifest: Path,
    old_threshold: float,
    new_threshold: float,
    report_root: str,
    method_config: Path | None = None,
) -> dict[str, Any]:
    validate_method(method)
    for path in (dual_om, context_om, build_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0.0 <= old_threshold < 1.0 or not 0.0 <= new_threshold < 1.0:
        raise ValueError("logical head candidate_confidence必须位于[0,1)")

    manifest = json.loads(build_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("build manifest schema_version必须为1")
    target = method["target"]
    for key in ("soc_version", "cann_version", "precision"):
        if manifest.get(key) != target.get(key):
            raise ValueError(f"build manifest {key}与方法配置不一致")
    if manifest.get("model_layout") != method["export"]["model_layout"]:
        raise ValueError("build manifest model_layout与方法配置不一致")
    if method_config is not None:
        resolved_method_config = method_config.resolve()
        _verify_manifest_evidence(manifest, "method_config")
        manifest_method = manifest["method_config"]
        if (
            Path(str(manifest_method["path"])).name != resolved_method_config.name
            or manifest_method["sha256"] != sha256_file(resolved_method_config)
        ):
            raise ValueError("满分方法配置与build manifest不一致")
    _verify_manifest_evidence(manifest, "training_report")
    _verify_manifest_evidence(manifest, "export_manifest")

    dual_artifact = _artifact_by_role(manifest, "dual_detector")
    dual_digest = _verify_manifest_asset(dual_artifact, dual_om, "dual_detector")
    context_digest = _verify_manifest_asset(
        _artifact_by_role(manifest, "context"), context_om, "context"
    )
    structural_heads = structural_logical_heads(method["export"]["logical_heads"])
    heads = copy.deepcopy(structural_heads)
    heads["old"]["candidate_confidence"] = float(old_threshold)
    heads["new"]["candidate_confidence"] = float(new_threshold)
    if dual_artifact.get("output_contract") != method["export"]["output_contract"]:
        raise ValueError("dual_detector output_contract与方法配置不一致")
    if structural_logical_heads(dual_artifact.get("logical_heads")) != structural_heads:
        raise ValueError("dual_detector logical_heads结构与build manifest不一致")

    result = copy.deepcopy(dict(base))
    result.pop("_config_path", None)
    result.pop("_config_sha256", None)
    result.pop("_config_overrides", None)
    result.setdefault("runtime", {})["server_port"] = int(target["candidate_port"])
    result.setdefault("inference", {}).update(
        {
            "backend": "ascend_acl",
            "imgsz": int(method["training"]["input_size"]),
            "specialist_imgsz": int(method["training"]["input_size"]),
        }
    )
    result.setdefault("routing", {})["neutral_context_score"] = float(
        method["runtime"]["neutral_context_score"]
    )
    result.setdefault("performance", {}).update(
        {
            "target_api_fps": float(method["benchmark"]["target_batch_fps"]),
            "warmup_requests": int(method["benchmark"]["warmup_requests"]),
            "batch_probe_size": int(method["benchmark"]["batch_probe_size"]),
            "report_root": str(report_root),
        }
    )
    source_key = "models/production/incremental_detection/three_class_base_detector.pt"
    ascend = result.setdefault("ascend_backend", {})
    ascend.update(
        {
            "model_layout": method["export"]["model_layout"],
            "soc_version": target["soc_version"],
            "cann_version": target["cann_version"],
            "precision": target["precision"],
            "execution_mode": method["runtime"]["execution_mode"],
            "encoded_preprocessing": method["runtime"]["encoded_preprocessing"],
            "memory_mode": method["runtime"]["memory_mode"],
            "schedule_mode": method["runtime"]["schedule_mode"],
            "context_mode": method["runtime"]["context_mode"],
            "detailed_event_timing": method["runtime"]["detailed_event_timing"],
            "validated": False,
            "validation_candidate": True,
            "validation_report": None,
            "validation_report_sha256": None,
            "build_manifest": str(build_manifest.resolve()),
            "build_manifest_sha256": sha256_file(build_manifest),
            "models": {
                source_key: {
                    "path": str(dual_om.resolve()),
                    "sha256": dual_digest,
                    "output_contract": method["export"]["output_contract"],
                    "logical_heads": heads,
                }
            },
            "context_model": {
                "path": str(context_om.resolve()),
                "sha256": context_digest,
            },
        }
    )
    validate_config(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="生成隔离的Ascend310B满分候选配置。")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "configs/agent_pipeline_ascend310b.yaml",
    )
    parser.add_argument(
        "--method-config",
        type=Path,
        default=ROOT / "configs/ascend310b/full_score_method.yaml",
    )
    parser.add_argument("--dual-om", type=Path, required=True)
    parser.add_argument("--context-om", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--old-threshold", type=float)
    parser.add_argument("--new-threshold", type=float)
    parser.add_argument(
        "--report-root", default="reports/ascend310b/full-score-candidate"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"候选配置已存在，拒绝覆盖：{args.output}")

    method = read_yaml_mapping(args.method_config.resolve(), "满分方法配置")
    base = read_yaml_mapping(args.base_config.resolve(), "基础Agent配置")
    seed = method.get("threshold_search", {}).get("current_seed", {})
    result = build_candidate_config(
        base,
        method,
        dual_om=args.dual_om.resolve(),
        context_om=args.context_om.resolve(),
        build_manifest=args.build_manifest.resolve(),
        old_threshold=float(
            args.old_threshold if args.old_threshold is not None else seed["old"]
        ),
        new_threshold=float(
            args.new_threshold if args.new_threshold is not None else seed["new"]
        ),
        report_root=args.report_root,
        method_config=args.method_config.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(result, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "output": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "port": result["runtime"]["server_port"],
        "old_threshold": result["ascend_backend"]["models"][
            next(iter(result["ascend_backend"]["models"]))
        ]["logical_heads"]["old"]["candidate_confidence"],
        "new_threshold": result["ascend_backend"]["models"][
            next(iter(result["ascend_backend"]["models"]))
        ]["logical_heads"]["new"]["candidate_confidence"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
