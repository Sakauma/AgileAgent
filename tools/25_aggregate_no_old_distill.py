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
        "# Compliant No-Old-Data Incremental Summary",
        "",
        "Training uses incremental images only. A frozen base handles old classes and new-data specialists handle new classes; no old raw image is replayed.",
        "",
        "| protocol | New-mAP50 | KRR | full mAP50 | old raw images | frozen drift | seconds | result |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['protocol']} | {row['new_map50_after']:.5f} | {row['krr']:.5f} | {row['full_map50_after']:.5f} | "
            f"{row['old_raw_image_count']} | {row['frozen_parameter_max_abs_drift']:.3g} | {row['training_seconds']:.1f} | "
            f"{'PASS' if row['decision']['passed'] else 'FAIL'} |"
        )
    overall = all(row["decision"]["passed"] for row in rows)
    lines += ["", f"Overall decision: **{'PASS' if overall else 'FAIL'}**."]
    (report_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((report_root / "summary.md").relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
