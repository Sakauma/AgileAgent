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


def read_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label}必须是mapping：{path}")
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
        value.startswith(("/home/", "/root/", "/mnt/"))
        or WINDOWS_ABSOLUTE.match(value)
    ):
        found.append(prefix)
    return found


def validate_method(method: Mapping[str, Any]) -> None:
    if (
        method.get("schema_version") != 1
        or method.get("kind") != "ascend310b_full_score_method"
    ):
        raise ValueError("4+2满分方法schema/kind非法")
    absolute = _absolute_strings(method)
    if absolute:
        raise ValueError("4+2满分方法禁止机器绝对路径：" + ", ".join(absolute))
    target = method.get("target") or {}
    export = method.get("export") or {}
    runtime = method.get("runtime") or {}
    if target.get("candidate_port") != 8502 or target.get("formal_port") != 8501:
        raise ValueError("4+2满分方法必须固定8502候选和8501正式端口")
    if (
        export.get("model_layout") != "independent_yolo26_e2e_v1"
        or export.get("output_contract") != "yolo26_e2e_v1"
        or export.get("input_name") != "images"
        or export.get("input_shape_nchw") != [1, 3, 608, 736]
        or export.get("input_shape_aipp_nhwc") != [1, 608, 736, 3]
        or export.get("max_det") != 300
    ):
        raise ValueError("4+2满分方法的YOLO26 E2E导出契约非法")
    models = export.get("models") or {}
    expected_maps = {
        "base": {"0": 0, "1": 1, "2": 2, "3": 3},
        "specialist": {"0": 4, "1": 5},
    }
    if set(models) != set(expected_maps):
        raise ValueError("4+2满分方法必须包含Base与Specialist导出模型")
    for name, expected_map in expected_maps.items():
        row = models[name]
        if (
            row.get("class_map") != expected_map
            or row.get("class_count") != len(expected_map)
            or row.get("output_shape") != [1, 300, 6]
        ):
            raise ValueError(f"4+2满分方法{name}类别/输出契约非法")
    if (
        runtime.get("encoded_preprocessing") != "dvpp"
        or runtime.get("execution_mode") != "async_stream"
        or runtime.get("context_mode") != "fixed_neutral_v1"
    ):
        raise ValueError("4+2满分方法必须使用DVPP/async/fixed-neutral运行契约")
    competition = method.get("competition") or {}
    accuracy = competition.get("accuracy_gates") or {}
    performance = competition.get("performance_gate") or {}
    benchmark = method.get("benchmark") or {}
    if (
        accuracy
        != {
            "base_map50_min": 0.80,
            "new_map50_min": 0.60,
            "krr_min": 0.95,
        }
        or performance
        != {
            "batch_image_count": 20,
            "batch_rounds": 3,
            "median_fps_min": 30.0,
        }
        or benchmark.get("batch_probe_size") != 20
        or benchmark.get("batch_rounds") != 3
        or float(benchmark.get("target_batch_fps", -1.0)) != 30.0
    ):
        raise ValueError("4+2满分方法计分门禁非法")


def _artifact_by_role(manifest: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    rows = [
        row
        for row in (manifest.get("artifacts") or {}).values()
        if isinstance(row, Mapping) and row.get("role") == role
    ]
    if len(rows) != 1:
        raise ValueError(f"build manifest要求恰好一个{role}资产，实际{len(rows)}")
    return rows[0]


def _verify_artifact(
    artifact: Mapping[str, Any], path: Path, label: str
) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    entry = artifact.get("om") or {}
    digest = sha256_file(path)
    if entry.get("sha256") != digest or Path(str(entry.get("path") or "")).name != path.name:
        raise ValueError(f"{label} OM与build manifest不一致")
    return digest


def materialize_registry(
    source: Mapping[str, Any], old_threshold: float, new_threshold: float
) -> dict[str, Any]:
    registry = copy.deepcopy(dict(source))
    models = registry.get("models") or []
    by_id = {
        str(row.get("id")): row for row in models if isinstance(row, dict)
    }
    expected = {
        "four_class_base_detector": (range(4), float(old_threshold)),
        "incremental_detector": ((4, 5), float(new_threshold)),
    }
    for model_id, (class_ids, threshold) in expected.items():
        row = by_id.get(model_id)
        if row is None:
            raise ValueError(f"代际注册表缺少模型：{model_id}")
        row["per_class_thresholds"] = {
            str(class_id): threshold for class_id in class_ids
        }
        gate = dict(row.get("context_gate") or {})
        gate.update(
            {
                "enabled": False,
                "policy": "fixed_neutral_no_penalty",
                "hard_routing": False,
                "max_threshold_penalty": 0.0,
                "max_threshold_penalties": {
                    str(class_id): 0.0 for class_id in class_ids
                },
            }
        )
        row["context_gate"] = gate
        row.pop("positive_prototype", None)
        row.pop("positive_prototypes", None)
    return registry


def build_candidate_config(
    base: Mapping[str, Any],
    method: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    base_om: Path,
    specialist_om: Path,
    context_om: Path,
    build_manifest: Path,
    registry_path: Path,
    report_root: str,
) -> dict[str, Any]:
    validate_method(method)
    if manifest.get("model_layout") != method["export"]["model_layout"]:
        raise ValueError("build manifest与方法配置模型布局不一致")
    target = method["target"]
    for key in ("soc_version", "cann_version", "precision"):
        if manifest.get(key) != target[key]:
            raise ValueError(f"build manifest {key}不一致")
    artifacts = {
        role: _artifact_by_role(manifest, role)
        for role in ("base", "specialist", "context")
    }
    for role in ("base", "specialist"):
        if artifacts[role].get("output_contract") != "yolo26_e2e_v1":
            raise ValueError(f"{role} build manifest输出契约非法")
    digests = {
        "base": _verify_artifact(artifacts["base"], base_om, "base"),
        "specialist": _verify_artifact(
            artifacts["specialist"], specialist_om, "specialist"
        ),
        "context": _verify_artifact(artifacts["context"], context_om, "context"),
    }

    result = copy.deepcopy(dict(base))
    for key in ("_config_path", "_config_sha256", "_config_overrides"):
        result.pop(key, None)
    result["runtime"]["server_port"] = 8502
    result["web"]["generation_registry"] = str(registry_path.resolve())
    result["generation"]["registry"] = str(registry_path.resolve())
    result["inference"].update(
        {
            "backend": "ascend_acl",
            "imgsz": 736,
            "specialist_imgsz": 736,
            "max_det": 300,
            "confidence_min": 0.00001,
            "confidence_default": float(
                method["threshold_search"]["scoring_request_confidence"]
            ),
            "warmup_width": 640,
            "warmup_height": 512,
            "preload_specialists": True,
        }
    )
    result["routing"].update(
        {
            "parallel_model_execution": True,
            "parallel_context_execution": True,
            "max_model_workers": 3,
            "neutral_context_score": float(
                method["runtime"]["neutral_context_score"]
            ),
            "preserve_base_class_owners": True,
        }
    )
    result["performance"].update(
        {
            "target_api_fps": 30.0,
            "warmup_requests": 30,
            "batch_probe_size": 20,
            "report_root": str(report_root),
        }
    )
    base_key = "models/production/incremental_detection/four_class_base_detector.pt"
    specialist_key = "models/production/incremental_detection/incremental_detector.pt"
    result["model"]["weights"] = base_key
    result["model"]["expected_sha256"] = sha256_file(ROOT / base_key)
    ascend = result["ascend_backend"]
    ascend.update(
        {
            "model_layout": "independent_yolo26_e2e_v1",
            "soc_version": target["soc_version"],
            "cann_version": target["cann_version"],
            "precision": target["precision"],
            "execution_mode": method["runtime"]["execution_mode"],
            "encoded_preprocessing": method["runtime"]["encoded_preprocessing"],
            "context_mode": method["runtime"]["context_mode"],
            "memory_mode": method["runtime"]["memory_mode"],
            "schedule_mode": method["runtime"]["schedule_mode"],
            "detailed_event_timing": method["runtime"]["detailed_event_timing"],
            "validated": False,
            "validation_candidate": True,
            "validation_report": None,
            "validation_report_sha256": None,
            "build_manifest": str(build_manifest.resolve()),
            "build_manifest_sha256": sha256_file(build_manifest),
            "models": {
                base_key: {
                    "path": str(base_om.resolve()),
                    "sha256": digests["base"],
                    "output_contract": "yolo26_e2e_v1",
                    "max_det": 300,
                    "class_count": 4,
                },
                specialist_key: {
                    "path": str(specialist_om.resolve()),
                    "sha256": digests["specialist"],
                    "output_contract": "yolo26_e2e_v1",
                    "max_det": 300,
                    "class_count": 2,
                },
            },
            "context_model": {
                "path": str(context_om.resolve()),
                "sha256": digests["context"],
            },
        }
    )
    validate_config(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="物化独立YOLO26 E2E的Ascend310B 4+2候选配置。"
    )
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
    parser.add_argument(
        "--generation-registry",
        type=Path,
        default=ROOT / "models/generations.json",
    )
    parser.add_argument("--base-om", type=Path, required=True)
    parser.add_argument("--specialist-om", type=Path, required=True)
    parser.add_argument("--context-om", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--old-threshold", type=float)
    parser.add_argument("--new-threshold", type=float)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-registry", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.output_registry):
        if path.exists():
            raise FileExistsError(f"候选产物已存在，拒绝覆盖：{path}")

    method = read_mapping(args.method_config.resolve(), "满分方法")
    validate_method(method)
    seed = method["threshold_search"]["current_seed"]
    old_threshold = float(
        args.old_threshold if args.old_threshold is not None else seed["old"]
    )
    new_threshold = float(
        args.new_threshold if args.new_threshold is not None else seed["new"]
    )
    if not (0.0 < old_threshold < 1.0 and 0.0 < new_threshold < 1.0):
        raise ValueError("候选阈值必须位于(0,1)")
    registry_source = json.loads(
        args.generation_registry.resolve().read_text(encoding="utf-8")
    )
    registry = materialize_registry(
        registry_source, old_threshold, new_threshold
    )
    args.output_registry.parent.mkdir(parents=True, exist_ok=True)
    args.output_registry.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(args.build_manifest.resolve().read_text(encoding="utf-8"))
    config = build_candidate_config(
        read_mapping(args.base_config.resolve(), "基础Agent配置"),
        method,
        manifest,
        base_om=args.base_om.resolve(),
        specialist_om=args.specialist_om.resolve(),
        context_om=args.context_om.resolve(),
        build_manifest=args.build_manifest.resolve(),
        registry_path=args.output_registry,
        report_root=args.report_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "config": str(args.output.resolve()),
        "config_sha256": sha256_file(args.output),
        "generation_registry": str(args.output_registry.resolve()),
        "generation_registry_sha256": sha256_file(args.output_registry),
        "old_threshold": old_threshold,
        "new_threshold": new_threshold,
        "port": config["runtime"]["server_port"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
