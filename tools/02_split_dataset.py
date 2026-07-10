#!/usr/bin/env python3
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from math import floor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fair_agent.dataset_utils import REPORTS_DIR, SPLITS_DIR, count_class_presence, count_values, read_metadata

SEED = 20260705
TARGET_COUNTS = {"train": 560, "dev_val": 95, "lock_val": 95}


def group_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    detailed = defaultdict(list)
    for row in rows:
        key = f"{row['sensor']}|{row['scene']}|{row['classes_present']}"
        detailed[key].append(row)

    grouped = defaultdict(list)
    for key, group in detailed.items():
        sensor, scene, _classes_present = key.split("|", 2)
        final_key = key if len(group) >= 3 else f"{sensor}|{scene}"
        grouped[final_key].extend(group)
    return dict(grouped)


def allocate_exact(group_sizes: Dict[str, int], target: int, total: int) -> Dict[str, int]:
    raw = []
    for key, size in group_sizes.items():
        quota = size * target / total
        base = floor(quota)
        raw.append((key, base, quota - base))
    allocation = {key: base for key, base, _rem in raw}
    remaining = target - sum(allocation.values())
    for key, _base, _rem in sorted(raw, key=lambda item: (-item[2], item[0]))[:remaining]:
        allocation[key] += 1
    return allocation


def split_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    rng = random.Random(SEED)
    groups = group_rows(rows)
    for group in groups.values():
        rng.shuffle(group)

    total = len(rows)
    group_sizes = {key: len(group) for key, group in groups.items()}
    train_alloc = allocate_exact(group_sizes, TARGET_COUNTS["train"], total)

    remaining_sizes = {key: group_sizes[key] - train_alloc[key] for key in groups}
    dev_alloc = allocate_exact(remaining_sizes, TARGET_COUNTS["dev_val"], total - TARGET_COUNTS["train"])

    splits = {"train": [], "dev_val": [], "lock_val": []}
    for key in sorted(groups):
        group = groups[key]
        train_count = train_alloc[key]
        dev_count = dev_alloc[key]
        splits["train"].extend(group[:train_count])
        splits["dev_val"].extend(group[train_count : train_count + dev_count])
        splits["lock_val"].extend(group[train_count + dev_count :])

    for split_name in splits:
        splits[split_name].sort(key=lambda row: row["image_path"])
    return splits


def write_split(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.write_text("\n".join(row["image_path"] for row in rows) + "\n", encoding="utf-8")


def format_counter(counter: Counter) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}: {value}" for key, value in sorted(counter.items()))


def split_report(splits: Dict[str, List[Dict[str, str]]]) -> str:
    all_paths = []
    lines = [
        "# 数据划分报告",
        "",
        f"随机种子：`{SEED}`",
        "",
        "| 划分 | 图片数 | 传感器 | 场景 | 包含类别 |",
        "|---|---:|---|---|---|",
    ]
    for split_name in ["train", "dev_val", "lock_val"]:
        rows = splits[split_name]
        all_paths.extend(row["image_path"] for row in rows)
        lines.append(
            f"| {split_name} | {len(rows)} | {format_counter(count_values(rows, 'sensor'))} | "
            f"{format_counter(count_values(rows, 'scene'))} | {format_counter(count_class_presence(rows))} |"
        )

    duplicates = len(all_paths) - len(set(all_paths))
    lines.extend(
        [
            "",
            "## 完整性检查",
            "",
            f"- 总图片数：{len(all_paths)}",
            f"- 重复图片数：{duplicates}",
            f"- 互斥划分：{'是' if duplicates == 0 else '否'}",
            "",
            "## 说明",
            "",
            "划分优先按 `sensor + scene + classes_present` 分层；样本过少的小组退化为 `sensor + scene`。`lock_val` 仅用于阶段性验证，日常调参使用 `dev_val`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    metadata_path = REPORTS_DIR / "metadata.csv"
    rows = read_metadata(metadata_path)
    if len(rows) != sum(TARGET_COUNTS.values()):
        print(f"Expected {sum(TARGET_COUNTS.values())} metadata rows, got {len(rows)}")
        return 1

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    splits = split_rows(rows)

    for split_name, split_rows_value in splits.items():
        write_split(SPLITS_DIR / f"{split_name}.txt", split_rows_value)

    for sensor in ["ir", "sar"]:
        for split_name in ["train", "dev_val", "lock_val"]:
            sensor_rows = [row for row in splits[split_name] if row["sensor"] == sensor]
            write_split(SPLITS_DIR / f"{split_name}_{sensor}.txt", sensor_rows)

    (REPORTS_DIR / "split_report.md").write_text(split_report(splits), encoding="utf-8")
    print(" ".join(f"{name}={len(value)}" for name, value in splits.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
