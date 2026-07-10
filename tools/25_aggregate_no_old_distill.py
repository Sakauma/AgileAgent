#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    fields = ["protocol", "method", "new_classes", "old_map50_before", "old_map50_after", "new_map50_after", "full_map50_after", "krr", "training_seconds", "training_image_count", "old_raw_image_count", "frozen_parameter_max_abs_drift", "passed"]
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
        "训练仅使用增量图像。冻结的基础模型负责旧类别，基于新增数据训练的专用模型负责新类别；整个过程不回放旧类原始图像。",
        "",
        "| 协议 | New-mAP50 | KRR | 完整集 mAP50 | 旧类原始图像数 | 冻结参数漂移 | 耗时（秒） | 结果 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['protocol']} | {row['new_map50_after']:.5f} | {row['krr']:.5f} | {row['full_map50_after']:.5f} | "
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
