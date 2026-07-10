#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
IMAGE_CASES = ROOT / "reports" / "error_analysis_sar_soldier_v2" / "image_error_cases.csv"
GT_CASES = ROOT / "reports" / "error_analysis_sar_soldier_v2" / "gt_error_cases.csv"
OUT_DIR = ROOT / "reports" / "agent_blackboard"
OUT_CSV = OUT_DIR / "sar_soldier_case_bank.csv"
OUT_JSON = OUT_DIR / "sar_soldier_case_bank.json"
OUT_MD = OUT_DIR / "sar_soldier_case_bank_report.md"


FIELDS = [
    "rank",
    "split",
    "image_path",
    "scene",
    "image_id",
    "statuses",
    "priority",
    "gt_soldier",
    "tp",
    "fp",
    "fn",
    "mean_matched_iou",
    "max_candidate_conf",
    "gt_status_counts",
    "recommended_action",
    "rationale",
    "visualization_hint",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_priority(value: str) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def status_set(value: str) -> set[str]:
    return {item for item in (value or "").split("+") if item}


def choose_action(statuses: set[str], scene: str) -> tuple[str, str]:
    if "FN_NO_CANDIDATE" in statuses:
        return (
            "add_hard_positive_replay",
            "几乎没有有效候选，需要作为高优先级正样本回放并人工复核标注可见性。",
        )
    if "LOW_IOU" in statuses:
        return (
            "localization_replay",
            "已有候选但定位偏移，应优先进入小目标定位复训或高分辨率定位复核。",
        )
    if "LOW_CONF" in statuses and scene == "forest":
        return (
            "forest_confidence_stabilization",
            "forest 场景低置信度更集中，应进入置信度稳定性样本池。",
        )
    if "LOW_CONF" in statuses:
        return (
            "confidence_stabilization",
            "候选 IoU 足够但置信度不足，应保留为阈值与置信度校准样本。",
        )
    if "FP" in statuses:
        return (
            "false_positive_review",
            "存在额外 soldier 预测，应检查背景纹理或相邻目标误触发。",
        )
    if "TP_LOW_QUALITY" in statuses:
        return (
            "localization_quality_review",
            "达到 TP 但 IoU 不高，适合用于小目标框质量优化。",
        )
    return ("monitor_only", "当前样本不是主要错误来源，保留观察即可。")


def visualization_hint(split: str, image_path: str) -> str:
    stem = Path(image_path).stem
    return f"reports/error_analysis_sar_soldier_v2/visualizations/{split}/*{stem}*.png"


def build_case_bank(image_rows: List[Dict[str, str]], gt_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    gt_by_image: Dict[str, Counter] = {}
    for row in gt_rows:
        image_path = row["image_path"]
        gt_by_image.setdefault(image_path, Counter())[row.get("status", "")] += 1

    sorted_rows = sorted(image_rows, key=lambda row: parse_priority(row.get("priority", "")), reverse=True)
    cases: List[Dict[str, Any]] = []
    for rank, row in enumerate(sorted_rows, start=1):
        statuses = status_set(row.get("statuses", ""))
        action, rationale = choose_action(statuses, row.get("scene", ""))
        gt_counts = dict(sorted(gt_by_image.get(row["image_path"], Counter()).items()))
        cases.append(
            {
                "rank": rank,
                "split": row.get("split", ""),
                "image_path": row.get("image_path", ""),
                "scene": row.get("scene", ""),
                "image_id": row.get("image_id", ""),
                "statuses": row.get("statuses", ""),
                "priority": parse_priority(row.get("priority", "")),
                "gt_soldier": row.get("gt_soldier", ""),
                "tp": row.get("tp", ""),
                "fp": row.get("fp", ""),
                "fn": row.get("fn", ""),
                "mean_matched_iou": row.get("mean_matched_iou", ""),
                "max_candidate_conf": row.get("max_candidate_conf", ""),
                "gt_status_counts": ";".join(f"{key}:{value}" for key, value in gt_counts.items() if key),
                "recommended_action": action,
                "rationale": rationale,
                "visualization_hint": visualization_hint(row.get("split", ""), row.get("image_path", "")),
            }
        )
    return cases


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    action_counts = Counter(row["recommended_action"] for row in rows)
    scene_counts = Counter(row["scene"] for row in rows)
    lines = [
        "# SAR Soldier 案例库报告",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 结论",
        "",
        "已将 SAR soldier 错误分析结果整理为智能体可读取的案例库。样本按既有 `priority` 从高到低排序，并附带推荐处理动作。",
        "",
        "## 汇总",
        "",
        f"- 样本数：`{len(rows)}`",
        f"- 场景分布：`{dict(sorted(scene_counts.items()))}`",
        f"- 推荐动作分布：`{dict(sorted(action_counts.items()))}`",
        "",
        "## 重点样本",
        "",
        "| 排名 | 图像 | 场景 | 状态 | 优先级 | 推荐动作 |",
        "|---:|---|---|---|---:|---|",
    ]
    for row in rows[:10]:
        lines.append(
            f"| {row['rank']} | `{row['image_path']}` | `{row['scene']}` | `{row['statuses']}` | {row['priority']} | `{row['recommended_action']}` |"
        )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            f"- CSV：`{rel(OUT_CSV)}`",
            f"- JSON：`{rel(OUT_JSON)}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    image_rows = read_csv(IMAGE_CASES)
    gt_rows = read_csv(GT_CASES)
    cases = build_case_bank(image_rows, gt_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_CSV, cases)
    OUT_JSON.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(OUT_MD, cases)
    print(rel(OUT_CSV))
    print(rel(OUT_JSON))
    print(rel(OUT_MD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
