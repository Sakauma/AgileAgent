#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import load_config  # noqa: E402
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在隔离进程中冻结指定图集的Ascend候选响应，用于评分与诊断。"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--expected-images", type=int, default=89)
    parser.add_argument(
        "--encoded",
        action="store_true",
        help="直接传入PNG字节，验证DVPP encoded路径；默认使用CPU解码输入。",
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

    engine = AtomicEngineProvider._build_engine(build_web_settings(config))
    records = []
    wall_ms = []
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
                    confidence=args.confidence,
                    incremental_protocol="auto",
                )
            else:
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                result = engine.predict(
                    image,
                    path.name,
                    confidence=args.confidence,
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
        "confidence": args.confidence,
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
