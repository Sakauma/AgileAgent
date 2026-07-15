#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.hashes import sha256_file


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _scenario_summary(config_path: Path, run_id: str) -> tuple[Path, Dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    path = resolve_path(config["experiment"]["report_root"]) / run_id / "summary.json"
    if not path.is_file():
        return path, {}
    return path, json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# 多批次小样本压力测试矩阵",
        "",
        f"- 矩阵编号：`{summary['run_id']}`",
        f"- 通过场景：{summary['passed_scenarios']}/{len(summary['scenarios'])}",
        f"- 总轮数：{summary['completed_rounds']}/{summary['expected_rounds']}",
        f"- 最终状态：`{summary['status']}`",
        "",
        "| 场景 | 轮数 | 连续晋升 | 最低New-mAP50 | 最低KRR | 结果 |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in summary["scenarios"]:
        metrics = [item.get("metrics") or {} for item in row.get("rounds", [])]
        new_values = [float(item["new_map50"]) for item in metrics if "new_map50" in item]
        krr_values = [float(item["krr"]) for item in metrics if "krr" in item]
        minimum_new = f"{min(new_values):.4f}" if new_values else "N/A"
        minimum_krr = f"{min(krr_values):.4f}" if krr_values else "N/A"
        lines.append(
            f"| {row['id']} | {len(row.get('rounds', []))}/{row['expected_rounds']} | "
            f"{'是' if row.get('all_rounds_promoted') else '否'} | "
            f"{minimum_new} | {minimum_krr} | {row['status']} |"
        )
    lines.extend(["", "## 逐轮结果", ""])
    lines.extend([
        "| 场景 | 轮次 | train/dev/lock | New-mAP50 | KRR | 组合mAP50 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for scenario in summary["scenarios"]:
        for row in scenario.get("rounds", []):
            metrics = row.get("metrics") or {}
            counts = row.get("split_counts") or {}
            lines.append(
                f"| {scenario['id']} | {row['round']} | "
                f"{counts.get('train', 0)}/{counts.get('val', 0)}/{counts.get('lock', 0)} | "
                f"{metrics.get('new_map50', 0):.4f} | {metrics.get('krr', 0):.4f} | "
                f"{metrics.get('combined_map50', 0):.4f} | {row['status']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(matrix_path: Path, run_id: str | None = None) -> Dict[str, Any]:
    raw = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    matrix = raw["matrix"]
    run_id = run_id or f"{matrix['id']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_root = resolve_path(matrix["output_root"]) / run_id
    output_root.mkdir(parents=True, exist_ok=False)
    summary: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matrix_config": rel_path(matrix_path),
        "matrix_config_sha256": sha256_file(matrix_path),
        "scenarios": [],
        "status": "RUNNING",
    }
    _atomic_json(output_root / "summary.json", summary)
    for item in matrix["scenarios"]:
        scenario_id = str(item["id"])
        config_path = resolve_path(item["config"])
        scenario_run_id = f"{run_id}-{scenario_id}"
        command = [
            sys.executable,
            str(ROOT / "tools" / "81_validate_multibatch_incremental.py"),
            "--config",
            str(config_path),
            "--run-id",
            scenario_run_id,
        ]
        print(f"[matrix] 开始 {scenario_id}: {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        scenario_summary_path, scenario = _scenario_summary(config_path, scenario_run_id)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        row = {
            "id": scenario_id,
            "config": rel_path(config_path),
            "config_sha256": sha256_file(config_path),
            "command": command,
            "return_code": completed.returncode,
            "summary": rel_path(scenario_summary_path),
            "expected_rounds": int(config["experiment"]["rounds"]),
            "status": str(scenario.get("status") or "PROCESS_FAILED"),
            "all_rounds_promoted": bool(scenario.get("all_rounds_promoted")),
            "rounds": list(scenario.get("rounds") or []),
        }
        summary["scenarios"].append(row)
        _atomic_json(output_root / "summary.json", summary)
        print(
            f"[matrix] 完成 {scenario_id}: {row['status']} "
            f"({len(row['rounds'])}/{row['expected_rounds']})",
            flush=True,
        )
        if completed.returncode and not bool(matrix.get("continue_after_scenario_failure")):
            break
    summary["expected_rounds"] = sum(row["expected_rounds"] for row in summary["scenarios"])
    summary["completed_rounds"] = sum(len(row["rounds"]) for row in summary["scenarios"])
    summary["passed_scenarios"] = sum(bool(row["all_rounds_promoted"]) for row in summary["scenarios"])
    summary["status"] = (
        "PASSED"
        if len(summary["scenarios"]) == len(matrix["scenarios"])
        and summary["passed_scenarios"] == len(matrix["scenarios"])
        else "STRESS_LIMIT_FOUND"
    )
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(output_root / "summary.json", summary)
    _write_report(output_root / "report.md", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="顺序执行多组小样本持续学习压力测试。")
    parser.add_argument(
        "--config",
        default="configs/incremental/multibatch_stress_matrix.yaml",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    summary = run(resolve_path(args.config), args.run_id)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
