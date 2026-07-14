#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.modules.strict_incremental import sha256_file


REQUIRED_METHODS = {"duet_yolo11s", "yolo_iod_lite"}


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("比较配置必须是 YAML 对象")
    return loaded


def validate_comparison_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    errors: list[str] = []
    protocols = list(config.get("protocols", []))
    modes = {str(item.get("adaptation_mode")) for item in protocols}
    if modes != REQUIRED_METHODS:
        errors.append(f"protocols 必须且只比较 {sorted(REQUIRED_METHODS)}")
    if len(protocols) != 2:
        errors.append("比较实验必须声明两个协议")
    signatures = {
        (
            tuple(item.get("base_classes", [])),
            item.get("new_class"),
            int(item.get("new_global_id", -1)),
            tuple(sorted((int(k), int(v)) for k, v in item.get("base_local_to_global", {}).items())),
            tuple(sorted(item.get("expected_incremental_counts", {}).items())),
        )
        for item in protocols
    }
    if len(signatures) != 1:
        errors.append("两种方法的数据协议、类别映射或样本数不一致")
    devices = [str(item.get("preferred_device")) for item in protocols]
    if len(set(devices)) != len(devices):
        errors.append("两种方法必须使用不同 GPU，避免显存和时序干扰")
    missing_methods = REQUIRED_METHODS - set(config.get("methods", {}))
    if missing_methods:
        errors.append(f"缺少方法配置：{sorted(missing_methods)}")
    if not config.get("paths", {}).get("shared_base_checkpoint"):
        errors.append("必须声明 paths.shared_base_checkpoint，保证基础权重完全一致")
    return {"valid": not errors, "errors": errors, "method_devices": dict(zip(sorted(modes), devices))}


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_metrics(config: Mapping[str, Any], run_id: str) -> list[Dict[str, Any]]:
    root = resolve_path(config["paths"]["report_root"]) / run_id
    rows = []
    for protocol in config["protocols"]:
        metrics_path = root / str(protocol["id"]) / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"方法指标不存在：{metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "protocol": str(protocol["id"]),
                "method": str(protocol["adaptation_mode"]),
                "metrics_path": rel_path(metrics_path),
                "metrics_sha256": sha256_file(metrics_path),
                "base_weight_sha256": metrics["base_weight_sha256"],
                "student_weight_sha256": metrics["student_weight_sha256"],
                "new_map50": float(metrics["new_map50"]),
                "krr": float(metrics["krr"]),
                "full_map50": float(metrics["full_map50"]),
                "calibration_precision": float(metrics["calibration"]["selected"]["precision"]),
                "calibration_recall": float(metrics["calibration"]["selected"]["recall"]),
                "lock_precision": float(metrics["lock_deployment_metrics"]["precision"]),
                "lock_recall": float(metrics["lock_deployment_metrics"]["recall"]),
                "old_channel_max_abs_drift": float(metrics["old_channel_max_abs_drift"]),
                "shared_parameter_relative_drift": float(metrics.get("shared_parameter_relative_drift", 0.0)),
                "training_seconds": float(metrics["training_seconds"]),
                "inference_ms_total": float(metrics["student_inference_ms_total"]),
                "accepted": bool(metrics["accepted"]),
                "failed_gates": sorted(key for key, value in metrics["gates"].items() if not value),
            }
        )
    if len({row["base_weight_sha256"] for row in rows}) != 1:
        raise RuntimeError("两种方法没有使用同一基础权重，比较无效")
    return rows


def _load_baseline(config: Mapping[str, Any]) -> Dict[str, Any] | None:
    value = config.get("paths", {}).get("row_only_baseline_metrics")
    if not value:
        return None
    path = resolve_path(value)
    if not path.is_file():
        return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return {
        "method": "new_classification_channel_only",
        "metrics_path": rel_path(path),
        "metrics_sha256": sha256_file(path),
        "new_map50": float(metrics["new_map50"]),
        "krr": float(metrics["krr"]),
        "full_map50": float(metrics["full_map50"]),
        "training_seconds": float(metrics["training_seconds"]),
        "accepted": bool(metrics["accepted"]),
    }


def build_comparison(
    config_path: Path,
    config: Mapping[str, Any],
    run_id: str,
) -> Dict[str, Any]:
    rows = _load_metrics(config, run_id)
    baseline = _load_baseline(config)
    verified = [row for row in rows if row["accepted"]]
    winner = (
        max(verified, key=lambda row: (row["full_map50"], row["new_map50"], -row["training_seconds"]))
        if verified
        else None
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "verified_winner" if winner else "completed_no_verified_winner",
        "winner": winner["method"] if winner else None,
        "selection_rule": "只在全部门禁通过的方法中按 full_mAP50、New-mAP50、训练时间排序",
        "config_path": rel_path(config_path),
        "config_sha256": _config_sha256(config_path),
        "same_base_weight_verified": True,
        "base_weight_sha256": rows[0]["base_weight_sha256"],
        "methods": rows,
        "row_only_baseline": baseline,
        "lock_results_must_not_be_used_for_this_run_tuning": True,
    }


def write_comparison(config: Mapping[str, Any], comparison: Mapping[str, Any]) -> Path:
    output = resolve_path(config["paths"]["report_root"]) / str(comparison["run_id"])
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "comparison.json"
    csv_path = output / "comparison.csv"
    md_path = output / "comparison.md"
    for path in (json_path, csv_path, md_path):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖比较报告：{path}")
    json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = [
        "method",
        "new_map50",
        "krr",
        "full_map50",
        "calibration_precision",
        "lock_precision",
        "lock_recall",
        "training_seconds",
        "inference_ms_total",
        "accepted",
        "failed_gates",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in comparison["methods"]:
            writer.writerow({key: row.get(key) for key in fields})

    baseline = comparison.get("row_only_baseline")
    lines = [
        "# DuET-YOLO11s 与 YOLO-IOD-lite 比较报告",
        "",
        f"- run_id：`{comparison['run_id']}`",
        f"- 基础权重 SHA256：`{comparison['base_weight_sha256']}`",
        f"- 状态：`{comparison['status']}`",
        f"- 通过门禁的推荐方法：`{comparison['winner'] or '无'}`",
        "",
        "| 方法 | New-mAP50 | KRR | 四类 mAP50 | 校准P | lock P/R | 更新耗时 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison["methods"]:
        lines.append(
            f"| {row['method']} | {row['new_map50']:.5f} | {row['krr']:.5f} | "
            f"{row['full_map50']:.5f} | {row['calibration_precision']:.5f} | "
            f"{row['lock_precision']:.5f}/{row['lock_recall']:.5f} | "
            f"{row['training_seconds'] / 60:.2f} min | {'通过' if row['accepted'] else '未通过'} |"
        )
    if baseline:
        lines.extend(
            [
                "",
                "## 当前行冻结基线",
                "",
                f"New-mAP50 `{baseline['new_map50']:.5f}`，KRR `{baseline['krr']:.5f}`，"
                f"四类 mAP50 `{baseline['full_map50']:.5f}`。",
            ]
        )
    lines.extend(
        [
            "",
            "## 判定约束",
            "",
            "只有通过全部数据、校准、New-mAP50、KRR、四类 mAP50 和误激活门禁的方法才可成为候选。",
            "本次 lock 结果只用于一次性方法验收，不得据此修改本 run 的损失权重或超参数。",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def run_training(config_path: Path, run_id: str, check_only: bool) -> int:
    command = [
        sys.executable,
        str(ROOT / "tools" / "70_run_strict_3plus1.py"),
        "--config",
        str(config_path),
        "--run-id",
        run_id,
    ]
    if check_only:
        command.append("--check-only")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="比较 DuET-YOLO11s 与 YOLO-IOD-lite。")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "incremental_method_comparison.yaml",
    )
    parser.add_argument("--run-id", help="覆盖 YAML 中的 run_id。")
    parser.add_argument("--check-only", action="store_true", help="只执行数据、环境和配置预检。")
    parser.add_argument("--report-only", action="store_true", help="只汇总已经完成的两种方法。")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    validation = validate_comparison_config(config)
    if not validation["valid"]:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 1
    run_id = str(args.run_id or config.get("experiment", {}).get("run_id") or "")
    if not run_id:
        raise ValueError("配置必须声明 experiment.run_id")
    if args.check_only:
        return run_training(config_path, run_id, check_only=True)
    if not args.report_only:
        training_code = run_training(config_path, run_id, check_only=False)
        if training_code != 0:
            print(f"训练入口返回 {training_code}，继续检查是否已生成完整门禁指标。")
    comparison = build_comparison(config_path, config, run_id)
    output = write_comparison(config, comparison)
    print(rel_path(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
