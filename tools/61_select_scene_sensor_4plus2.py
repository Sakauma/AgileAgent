#!/usr/bin/env python3
"""Select a Scene-SensorNet candidate using dev metrics only."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds 必须是非空且不重复的整数列表")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description="按 dev 指标选择 4+2 Scene-SensorNet。")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    args = parser.parse_args()

    project = args.project.expanduser().resolve()
    candidates: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_dir = project / f"seed_{seed}"
        weight = run_dir / "scene_sensor_net.pt"
        metrics_path = run_dir / "scene_sensor_metrics.json"
        if not weight.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"场景候选未完成：seed={seed}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        dev = metrics["dev"]
        candidates.append(
            {
                "seed": seed,
                "weight": weight.as_posix(),
                "metrics": metrics_path.as_posix(),
                "best_epoch": int(metrics["best_epoch"]),
                "dev_sensor_accuracy": float(dev["sensor_accuracy"]),
                "dev_scene_accuracy": float(dev["scene_accuracy"]),
                "dev_joint_accuracy": float(dev["joint_accuracy"]),
                "dev_selection_score": 0.35 * float(dev["sensor_accuracy"])
                + 0.65 * float(dev["scene_accuracy"]),
            }
        )
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["dev_selection_score"],
            row["dev_joint_accuracy"],
            row["dev_scene_accuracy"],
        ),
        reverse=True,
    )
    selected = ranked[0]
    selected_dir = project / "selection" / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_weight = selected_dir / "scene_sensor_net.pt"
    selected_metrics = selected_dir / "scene_sensor_metrics.json"
    shutil.copy2(selected["weight"], selected_weight)
    shutil.copy2(selected["metrics"], selected_metrics)
    report = Path(selected["metrics"]).with_name("scene_sensor_report.md")
    if report.is_file():
        shutil.copy2(report, selected_dir / report.name)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_scope": "dev_only",
        "ranking": ranked,
        "selected": {
            **selected,
            "promoted_weight": selected_weight.as_posix(),
            "promoted_metrics": selected_metrics.as_posix(),
        },
    }
    output = project / "selection" / "scene_selection.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["selected"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
