#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import load_config, resolve_path  # noqa: E402
from fair_agent.web.app import AtomicEngineProvider, build_web_settings  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_record(result: Dict[str, Any]) -> Dict[str, Any]:
    agent = result.get("agent") or {}
    return {
        "image_id": Path(result["filename"]).stem,
        "filename": result["filename"],
        "context": result["context"],
        "detections": result["detections"],
        "class_counts": result["class_counts"],
        "detection_count": result["detection_count"],
        "models_used": agent.get("models_used") or [],
        "decision": agent.get("decision") or {},
        "timings": result.get("timings") or {},
    }


def prepare_calibration_probe(
    config: Dict[str, Any],
    temporary_root: Path,
    confidence_floor: float,
) -> Dict[str, Any]:
    """Neutralize policy layers so dev freezing retains both OM candidate streams.

    The probe still uses the real Base, Specialist and Scene OM files.  It only
    disables learned thresholds, scene gates and cross-model suppression.  The
    resulting unlabeled JSONL can therefore be replayed repeatedly by the
    constrained search without rerunning either detector.
    """

    registry_source = resolve_path(config["web"]["generation_registry"])
    registry = json.loads(registry_source.read_text(encoding="utf-8"))
    model_audit: Dict[str, Any] = {}
    for model in registry.get("models", []):
        owned = [int(value) for value in model.get("owns_classes", [])]
        if owned:
            model["per_class_thresholds"] = {
                str(class_id): confidence_floor for class_id in owned
            }
        model["context_prior"] = {}
        model["context_gate"] = {"enabled": False}
        if model.get("role") in {
            "class_incremental_expert",
            "target_incremental_expert",
        }:
            model["content_execution_gate"] = {"enabled": False}
        model_audit[str(model.get("id"))] = {
            "owns_classes": owned,
            "threshold": confidence_floor,
            "context_gate_enabled": False,
            "content_execution_gate_enabled": False,
        }

    probe_registry = temporary_root / "probe-generations.json"
    probe_registry.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config["web"]["generation_registry"] = str(probe_registry)
    config["generation"]["registry"] = str(probe_registry)
    config["inference"]["confidence_min"] = min(
        float(config["inference"]["confidence_min"]), confidence_floor
    )
    config["inference"]["confidence_default"] = confidence_floor
    routing = config["routing"]
    routing["fusion_iou"] = 1.0
    routing["conflict_iou"] = 1.0
    routing["conflict_incremental_coverage"] = None
    routing["conflict_base_confidence"] = 1.0
    routing["specialist_margin"] = 0.0
    routing["score_calibration"] = {"enabled": False}
    routing["cross_class_suppression"] = {
        "enabled": False,
        "strategy": "highest_confidence",
        "scope": "all_classes",
        "iou": 1.0,
        "smaller_box_coverage": None,
        "incremental_over_base_margin": 0.0,
    }
    return {
        "registry_source": str(registry_source),
        "confidence_floor": confidence_floor,
        "policy_layers_neutralized": [
            "per_class_thresholds",
            "known_scene_soft_gates",
            "content_execution_gate",
            "score_calibration",
            "old_new_conflict_arbitration",
            "cross_class_suppression",
        ],
        "models": model_audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在隔离进程中冻结指定图集的Ascend候选响应，用于评分与诊断。"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--expected-images", type=int, default=89)
    parser.add_argument(
        "--encoded",
        action="store_true",
        help="直接传入PNG字节，验证DVPP encoded路径；默认使用CPU解码输入。",
    )
    parser.add_argument(
        "--calibration-probe",
        action="store_true",
        help=(
            "冻结低阈值Base/Specialist原始候选与Scene概率；仅用于mixed dev约束搜索。"
        ),
    )
    parser.add_argument(
        "--context-om",
        type=Path,
        help="使用指定的候选 Scene-SensorNet OM。",
    )
    parser.add_argument(
        "--base-om",
        type=Path,
        help="使用指定的四类 Base 候选 OM。",
    )
    parser.add_argument(
        "--specialist-om",
        type=Path,
        help="仅用于隔离数值实验，覆盖incremental_detector对应OM。",
    )
    args = parser.parse_args()

    if args.output.exists() or args.summary.exists():
        raise FileExistsError("冻结预测或摘要已存在，拒绝覆盖。")
    if args.expected_images <= 0:
        raise ValueError("expected-images必须为正整数。")
    confidence = (
        float(args.confidence)
        if args.confidence is not None
        else (0.00001 if args.calibration_probe else 0.5)
    )
    if not 0.00001 <= confidence <= 1.0:
        raise ValueError("confidence必须位于[0.00001, 1.0]。")
    if args.calibration_probe and confidence > 0.01:
        raise ValueError("calibration-probe要求confidence不高于0.01。")
    paths = sorted(args.image_root.glob("*.png"))
    if (
        len(paths) != args.expected_images
        or len({path.stem for path in paths}) != args.expected_images
    ):
        raise RuntimeError(f"冻结输入必须是{args.expected_images}张stem唯一PNG。")

    config = copy.deepcopy(load_config(args.config))
    model_overrides = {}
    for label, suffix, override in (
        ("base", "four_class_base_detector.pt", args.base_om),
        ("specialist", "incremental_detector.pt", args.specialist_om),
    ):
        if override is None:
            continue
        resolved = override.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        keys = [
            key
            for key in config["ascend_backend"]["models"]
            if str(key).endswith(suffix)
        ]
        if len(keys) != 1:
            raise RuntimeError(f"无法唯一定位{label}模型配置：{keys}")
        entry = dict(config["ascend_backend"]["models"][keys[0]])
        entry.update({"path": str(resolved), "sha256": sha256(resolved)})
        config["ascend_backend"]["models"][keys[0]] = entry
        model_overrides[label] = {"path": str(resolved), "sha256": sha256(resolved)}
    context_override = None
    if args.context_om is not None:
        context_override = args.context_om.resolve()
        if not context_override.is_file():
            raise FileNotFoundError(context_override)
        config["ascend_backend"]["context_model"] = {
            "path": str(context_override),
            "sha256": sha256(context_override),
        }

    records = []
    wall_ms = []
    probe_audit = None
    with tempfile.TemporaryDirectory(prefix="agileagent-ascend-probe-") as directory:
        if args.calibration_probe:
            probe_audit = prepare_calibration_probe(
                config,
                Path(directory),
                confidence,
            )
        engine = AtomicEngineProvider._build_engine(build_web_settings(config))
        try:
            for path in paths:
                started = time.perf_counter_ns()
                if args.encoded:
                    data = path.read_bytes()
                    if not engine.accepts_encoded(data):
                        raise RuntimeError(f"候选拒绝固定PNG encoded契约：{path}")
                    result = engine.predict_encoded(
                        data,
                        path.name,
                        confidence=confidence,
                        incremental_protocol="auto",
                    )
                else:
                    with Image.open(path) as opened:
                        image = opened.convert("RGB")
                    result = engine.predict(
                        image,
                        path.name,
                        confidence=confidence,
                        incremental_protocol="auto",
                    )
                wall_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
                records.append(public_record(result))
        finally:
            engine.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "input_mode": "unlabeled_images",
        "labels_read": False,
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "context_override": (
            {
                "path": str(context_override),
                "sha256": sha256(context_override),
            }
            if context_override is not None
            else None
        ),
        "model_overrides": model_overrides,
        "confidence": confidence,
        "calibration_probe": bool(args.calibration_probe),
        "probe_audit": probe_audit,
        "encoded": args.encoded,
        "image_count": len(records),
        "prediction_count": sum(len(row["detections"]) for row in records),
        "mean_wall_ms": sum(wall_ms) / len(wall_ms),
        "predictions": str(args.output.resolve()),
        "predictions_sha256": sha256(args.output),
        "passed": len(records) == args.expected_images,
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
