from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fair_agent.core.config import resolve_path


def render_incremental_review(summary: Dict[str, Any]) -> str:
    acceptance = summary.get("acceptance", {})
    lines = [
        "# 合规增量学习复核报告",
        "",
        f"- 协议完整：`{summary.get('complete')}`",
        f"- 合规证据已验证：`{summary.get('compliance_verified')}`",
        f"- 总体验收：`{'通过' if summary.get('passed') else '未通过'}`",
        f"- New-mAP50 门槛：`{acceptance.get('min_new_class_map50')}`",
        f"- KRR 门槛：`{acceptance.get('min_krr')}`",
        "",
        "| 协议 | 新类别 | New-mAP50 | KRR | 旧类原始图像数 | 合规 | 结果 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for item in summary.get("protocols", []):
        lines.append(
            f"| {item.get('protocol')} | {item.get('new_class')} | "
            f"{float(item.get('new_map50') or 0):.5f} | {float(item.get('krr') or 0):.5f} | "
            f"{item.get('old_raw_image_count')} | {item.get('compliant')} | "
            f"{'通过' if item.get('passed') else '未通过'} |"
        )
    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend(["", "## 警告", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def write_incremental_review(config: Dict[str, Any], summary: Dict[str, Any]) -> Path:
    action = config.get("decision", {}).get("actions", {}).get("review_incremental_learning", {})
    outputs = list(action.get("outputs", []))
    if len(outputs) != 1:
        raise ValueError("review_incremental_learning 必须声明且只声明一个输出文件")
    output = resolve_path(outputs[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_incremental_review(summary), encoding="utf-8")
    return output
