#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


MEMCPY_DIRECTIONS = {
    1: "host_to_device",
    2: "device_to_host",
    3: "device_to_device",
    4: "host_to_host",
}


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


def distribution(values: Iterable[float]) -> Dict[str, float | int]:
    rows = list(values)
    if not rows:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(rows),
        "mean_ms": statistics.fmean(rows),
        "p50_ms": percentile(rows, 0.50),
        "p95_ms": percentile(rows, 0.95),
        "p99_ms": percentile(rows, 0.99),
        "max_ms": max(rows),
    }


def one_path(paths: Iterable[Path], label: str) -> Path:
    rows = list(paths)
    if len(rows) != 1:
        raise RuntimeError(f"{label}必须唯一，实际为：{rows}")
    return rows[0]


def read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def model_names(profiler_dir: Path) -> Dict[int, str]:
    database = profiler_dir / "host/sqlite/ge_model_info.db"
    with sqlite3.connect(database) as connection:
        return {
            int(model_id): str(name)
            for model_id, name in connection.execute(
                "SELECT model_id, model_name FROM GeModelLoad"
            )
        }


def measured_model_times(
    profiler_dir: Path,
    export_dir: Path,
    request_count: int,
) -> Dict[str, Any]:
    names = model_names(profiler_dir)
    step_path = one_path(export_dir.glob("step_trace_*.csv"), "step trace CSV")
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in read_csv(step_path):
        model_id = int(row["Model ID"])
        grouped[model_id].append(
            (int(row["Iteration ID"]), float(row["Iteration Time(us)"]) / 1000.0)
        )
    result = {}
    for model_id, values in grouped.items():
        ordered = sorted(values)
        if len(ordered) < request_count:
            raise RuntimeError(
                f"模型{model_id}迭代数不足：{len(ordered)} < {request_count}"
            )
        measured = ordered[-request_count:]
        result[names.get(model_id, f"model_{model_id}")] = {
            "model_id": model_id,
            "all_iteration_count": len(ordered),
            "measured_iteration_ids": [measured[0][0], measured[-1][0]],
            "duration": distribution(value for _, value in measured),
        }
    return result


def runtime_summary(profiler_dir: Path) -> Dict[str, Any]:
    database = profiler_dir / "host/sqlite/runtime.db"
    with sqlite3.connect(database) as connection:
        api_rows = list(
            connection.execute(
                """
                SELECT api, COUNT(*), SUM(exit_time-entry_time) / 1000.0,
                       AVG(exit_time-entry_time) / 1000.0,
                       MAX(exit_time-entry_time) / 1000.0
                FROM ApiCall
                GROUP BY api
                ORDER BY SUM(exit_time-entry_time) DESC
                """
            )
        )
        memcpy_rows = list(
            connection.execute(
                """
                SELECT memcpy_direction, COUNT(*), SUM(data_size),
                       SUM(exit_time-entry_time) / 1000.0
                FROM ApiCall
                WHERE api LIKE '%Memcpy%'
                GROUP BY memcpy_direction
                ORDER BY memcpy_direction
                """
            )
        )
    by_api = [
        {
            "name": str(name),
            "count": int(count),
            "total_us": float(total_us or 0.0),
            "mean_us": float(mean_us or 0.0),
            "max_us": float(max_us or 0.0),
        }
        for name, count, total_us, mean_us, max_us in api_rows
    ]
    categories: dict[str, dict[str, float | int]] = {}
    predicates = {
        "stream_wait_and_synchronize": lambda name: "Synchronize" in name or "Wait" in name,
        "model_execute_enqueue": lambda name: "ModelExecute" in name,
        "memcpy": lambda name: "Memcpy" in name,
        "event": lambda name: "Event" in name,
    }
    for category, predicate in predicates.items():
        selected = [row for row in by_api if predicate(str(row["name"]))]
        categories[category] = {
            "count": sum(int(row["count"]) for row in selected),
            "total_us": sum(float(row["total_us"]) for row in selected),
        }
    memcpy = {}
    for direction, count, size, total_us in memcpy_rows:
        numeric_direction = int(direction or 0)
        memcpy[MEMCPY_DIRECTIONS.get(numeric_direction, f"direction_{numeric_direction}")] = {
            "count": int(count),
            "bytes": int(size or 0),
            "host_api_total_us": float(total_us or 0.0),
        }
    return {"categories": categories, "memcpy_by_direction": memcpy, "top_apis": by_api[:20]}


def ai_core_summary(export_dir: Path) -> Dict[str, Any]:
    statistic_path = one_path(export_dir.glob("op_statistic_*.csv"), "op statistic CSV")
    rows = read_csv(statistic_path)
    models = sorted({row["Model Name"] for row in rows})
    top = sorted(rows, key=lambda row: float(row["Total Time(us)"]), reverse=True)[:20]
    return {
        "exported_models": models,
        "note": "CANN 7.0.RC1自动导出默认模型/迭代；完整模型时长使用step_trace。",
        "top_operator_types": [
            {
                "model": row["Model Name"],
                "op_type": row["OP Type"],
                "core_type": row["Core Type"],
                "count": int(row["Count"]),
                "total_us": float(row["Total Time(us)"]),
                "ratio_percent": float(row["Ratio(%)"]),
            }
            for row in top
        ],
    }


def dvpp_summary(export_dir: Path) -> Dict[str, Any]:
    path = one_path(export_dir.glob("dvpp_*.csv"), "DVPP CSV")
    rows = read_csv(path)
    return {
        "rows": rows,
        "reported_total_us": sum(float(row["All Time(us)"]) for row in rows),
        "note": "此CANN汇总表仅枚举VDEC/VENC；PNGD/VPC耗时以应用dvpp_enqueue_ms和原始trace为准。",
    }


def artifact(path: Path, root: Path) -> Dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总Ascend msprof原始采集为机器可读P2报告。")
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.profile_root.resolve()
    if args.output.exists():
        raise FileExistsError(f"profile摘要已存在，拒绝覆盖：{args.output}")
    application_path = root / "application-report.json"
    application = json.loads(application_path.read_text(encoding="utf-8"))
    request_count = int(application["profile_scope"]["request_count"])
    profiler_dir = one_path((root / "raw").glob("PROF_*"), "PROF目录")
    export_dir = profiler_dir / "mindstudio_profiler_output"
    models = measured_model_times(profiler_dir, export_dir, request_count)
    engine_mean = float(application["distributions"]["engine_total_ms"]["mean_ms"])
    routing_mean = float(application["distributions"]["routing_fusion_ms"]["mean_ms"])
    critical_model = max(
        models,
        key=lambda name: float(models[name]["duration"]["mean_ms"]),
    )
    critical_model_mean = float(models[critical_model]["duration"]["mean_ms"])
    scene_mean = float(models.get("scene_sensor_net", {}).get("duration", {}).get("mean_ms", 0.0))
    all_files = [path for path in profiler_dir.rglob("*") if path.is_file()]
    analyzed = [
        application_path,
        profiler_dir / "host/sqlite/ge_model_info.db",
        profiler_dir / "host/sqlite/runtime.db",
        one_path(export_dir.glob("step_trace_*.csv"), "step trace CSV"),
        one_path(export_dir.glob("op_statistic_*.csv"), "op statistic CSV"),
        one_path(export_dir.glob("dvpp_*.csv"), "DVPP CSV"),
    ]
    report = {
        "schema_version": 1,
        "profile_root": str(root),
        "profiler_dir": str(profiler_dir),
        "capture_scope": "full_process_including_model_load_and_engine_warmup",
        "application": application,
        "model_execution": models,
        "runtime": runtime_summary(profiler_dir),
        "dvpp": dvpp_summary(export_dir),
        "ai_core": ai_core_summary(export_dir),
        "critical_path": {
            "engine_mean_ms": engine_mean,
            "routing_fusion_mean_ms": routing_mean,
            "critical_model": critical_model,
            "critical_model_mean_ms": critical_model_mean,
            "critical_model_share_of_engine": critical_model_mean / engine_mean,
            "scene_model_mean_ms": scene_mean,
            "scene_share_of_engine": scene_mean / engine_mean,
        },
        "decision_inputs": {
            "model_execution_dominates_host_routing": critical_model_mean > routing_mean * 5.0,
            "aoe_evaluation_allowed": critical_model_mean > routing_mean * 5.0,
            "scene_aoe_eligible": scene_mean > 1.0 or scene_mean / engine_mean > 0.05,
        },
        "artifacts": {
            "raw_file_count": len(all_files),
            "raw_total_bytes": sum(path.stat().st_size for path in all_files),
            "analyzed": [artifact(path, root) for path in analyzed],
        },
        "passed": bool(application.get("passed")) and bool(models),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
