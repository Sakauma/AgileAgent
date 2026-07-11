#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from fair_agent.modules.incremental_compliance import verify_class_incremental_learning_scope


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def split_values(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_learning_scope(protocol: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    dataset_name = row.get("learning_dataset") or protocol.get("historical_learning_data") or protocol["learning_data"]
    dataset_path = resolve(dataset_name)
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    result = verify_class_incremental_learning_scope(
        split_values(resolve(dataset["train"])),
        split_values(resolve(dataset["val"])),
        split_values(resolve(protocol["new_train_split"])),
        split_values(resolve(protocol["new_val_split"])),
        protocol["base_classes"],
        protocol["new_classes"],
        verify_content=True,
    )
    result["audited_learning_dataset"] = dataset_path.relative_to(ROOT).as_posix()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "incremental_no_old_distill_yolo11s.yaml")
    args = parser.parse_args()
    config: Dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report_root = resolve(config["report_dir"])
    rows = []
    for protocol in config["protocols"]:
        path = report_root / protocol["name"] / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing protocol metrics: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        scope = audit_learning_scope(protocol, row)
        row.update({
            "task_type": scope["task_type"],
            "learning_data_scope": scope["learning_data_scope"],
            "learning_scope_verified": scope["learning_scope_verified"],
            "old_raw_image_count": scope["old_raw_image_count"],
            "validation_old_raw_image_count": scope["validation"]["old_raw_image_count"],
            "audited_learning_dataset": scope["audited_learning_dataset"],
        })
        row["decision"]["compliant"] = bool(scope["compliant"])
        row["decision"]["passed"] = bool(row["decision"]["passed"] and scope["compliant"])
        rows.append(row)
    fields = ["protocol", "task_type", "learning_data_scope", "learning_scope_verified", "method", "new_classes", "old_map50_before", "old_map50_after", "new_map50_after", "full_map50_after", "krr", "training_seconds", "training_image_count", "old_raw_image_count", "validation_old_raw_image_count", "frozen_parameter_max_abs_drift", "passed"]
    flat_rows = []
    for row in rows:
        flat = {key: row.get(key) for key in fields[:-1]}
        flat["new_classes"] = ";".join(row.get("new_classes", []))
        flat["passed"] = row["decision"]["passed"]
        flat_rows.append(flat)
    with (report_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
    (report_root / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# 合规的无旧数据增量学习汇总",
        "",
        "任务类型为类别增量目标检测。训练、验证、早停和调参仅使用增量数据；冻结的基础模型负责旧类别，增量专用模型负责新类别。旧类测试数据只在权重冻结后的评分阶段读取。",
        "",
        "| 协议 | 数据边界 | New-mAP50 | KRR | 完整集 mAP50 | 旧类原始图像数 | 冻结参数漂移 | 耗时（秒） | 结果 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['protocol']} | {'通过' if row['learning_scope_verified'] else '失败'} | {row['new_map50_after']:.5f} | {row['krr']:.5f} | {row['full_map50_after']:.5f} | "
            f"{row['old_raw_image_count']} | {row['frozen_parameter_max_abs_drift']:.3g} | {row['training_seconds']:.1f} | "
            f"{'通过' if row['decision']['passed'] else '未通过'} |"
        )
    overall = all(row["decision"]["passed"] for row in rows)
    lines += ["", f"总体结论：**{'通过' if overall else '未通过'}**。"]
    (report_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((report_root / "summary.md").relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
