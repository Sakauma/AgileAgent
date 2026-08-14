#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import load_config
from fair_agent.web.app import AtomicEngineProvider, build_web_settings


TIMING_KEYS = (
    "dvpp_enqueue_ms",
    "context_total_ms",
    "context_inference_ms",
    "detector_total_ms",
    "detector_inference_ms",
    "specialist_inference_ms",
    "routing_fusion_ms",
    "engine_total_ms",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> Dict[str, float | int]:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
    }


def git_snapshot() -> Dict[str, Any]:
    def value(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = value("status", "--porcelain=v1")
    return {
        "root": str(ROOT),
        "head": value("rev-parse", "HEAD"),
        "branch": value("branch", "--show-current"),
        "status_porcelain": status,
        "clean": status == "" if status is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="供msprof启动的Ascend生产encoded请求采集应用。"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-count", type=int, required=True)
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"profile应用报告已存在，拒绝覆盖：{args.output}")
    if args.request_count <= 0 or args.warmup_count < 0:
        raise ValueError("request-count必须为正数，warmup-count不得为负数。")
    paths = sorted(args.image_root.glob("*.png"))
    if len(paths) != 89 or len({path.stem for path in paths}) != 89:
        raise RuntimeError("profile输入必须是固定的89张stem唯一PNG。")

    config = load_config(args.config)
    engine = AtomicEngineProvider._build_engine(build_web_settings(config))
    rows = []
    started_all = time.perf_counter_ns()
    try:
        for index in range(args.warmup_count):
            path = paths[index % len(paths)]
            data = path.read_bytes()
            if not engine.accepts_encoded(data):
                raise RuntimeError(f"profile输入不满足固定encoded契约：{path}")
            engine.predict_encoded(
                data,
                path.name,
                confidence=args.confidence,
                incremental_protocol="auto",
            )
        for index in range(args.request_count):
            path = paths[index % len(paths)]
            data = path.read_bytes()
            request_started = time.perf_counter_ns()
            result = engine.predict_encoded(
                data,
                path.name,
                confidence=args.confidence,
                incremental_protocol="auto",
            )
            wall_ms = (time.perf_counter_ns() - request_started) / 1_000_000.0
            timings = result.get("timings") or {}
            row: Dict[str, Any] = {
                "index": index,
                "image": path.name,
                "wall_ms": wall_ms,
                "inference_ms": float(result["inference_ms"]),
                "detection_count": int(result["detection_count"]),
            }
            row.update({key: float(timings.get(key, 0.0)) for key in TIMING_KEYS})
            rows.append(row)
    finally:
        engine.close()
    application_wall_ms = (time.perf_counter_ns() - started_all) / 1_000_000.0

    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profile_scope": {
            "input_mode": "encoded_png",
            "labels_read": False,
            "confidence": args.confidence,
            "warmup_count": args.warmup_count,
            "request_count": args.request_count,
            "application_wall_ms": application_wall_ms,
        },
        "distributions": {
            "wall": distribution([float(row["wall_ms"]) for row in rows]),
            "inference": distribution([float(row["inference_ms"]) for row in rows]),
            **{
                key: distribution([float(row[key]) for row in rows])
                for key in TIMING_KEYS
            },
        },
        "requests": rows,
        "environment": {
            "python": sys.version,
            "git": git_snapshot(),
            "config": str(args.config.resolve()),
            "config_sha256": sha256(args.config),
        },
        "passed": len(rows) == args.request_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "requests"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
