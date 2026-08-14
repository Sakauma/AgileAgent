#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fair_agent.dataset_utils import REPORTS_DIR, scan_dataset, write_json


def markdown_table(title, mapping):
    lines = [f"### {title}", "", "| 项目 | 数量 |", "|---|---:|"]
    for key, value in mapping.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def main() -> int:
    summary = scan_dataset()
    distributions = summary["distributions"]
    total_objects = summary["total_objects"]

    report_lines = [
        "# 数据体检报告",
        "",
        "## 总览",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| 图片数量 | {summary['total_images']} |",
        f"| 标签文件数量 | {summary['total_labels']} |",
        f"| 标注框数量 | {total_objects} |",
        f"| 缺失标签 | {summary['missing_labels']} |",
        f"| 缺失图片 | {summary['missing_images']} |",
        f"| 标签格式/范围错误 | {summary['invalid_errors']} |",
        "",
        "## 统计口径说明",
        "",
        f"脚本直接读取 YOLO 标签，统计得到 `{total_objects}` 个非空标注框。实验、报告与划分统一使用该结果。",
        "",
        markdown_table("传感器分布", distributions["sensor"]),
        "",
        markdown_table("场景分布", distributions["scene"]),
        "",
        markdown_table("传感器/场景交叉分布", distributions["sensor_scene"]),
        "",
        markdown_table("类别目标数", distributions["class_objects"]),
        "",
        markdown_table("包含类别的图片数", distributions["class_images"]),
        "",
        markdown_table("每图目标数分布", distributions["objects_per_image"]),
        "",
        "## IR/SAR 类别目标数",
        "",
    ]

    for sensor, counts in distributions["class_by_sensor"].items():
        report_lines.extend([markdown_table(sensor, counts), ""])

    report_lines.extend(["## 场景类别目标数", ""])
    for scene, counts in distributions["class_by_scene"].items():
        report_lines.extend([markdown_table(scene, counts), ""])

    if summary["errors"]:
        report_lines.extend(["## 错误明细", "", "| 文件 | 行 | 类型 | 内容 |", "|---|---:|---|---|"])
        for error in summary["errors"]:
            report_lines.append(f"| {error['file']} | {error['line']} | {error['type']} | `{error['value']}` |")
    else:
        report_lines.extend(["## 错误明细", "", "未发现缺失图片、缺失标签、类别越界、坐标越界或标签列数错误。"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "data_audit_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_json(REPORTS_DIR / "data_audit_summary.json", {key: value for key, value in summary.items() if key != "rows"})

    print(f"images={summary['total_images']} labels={summary['total_labels']} objects={total_objects} errors={len(summary['errors'])}")
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
