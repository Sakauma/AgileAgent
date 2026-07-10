from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fair_agent.core.config import resolve_path


def _action(config: Dict[str, Any], name: str, status: str, freshness: str, reason: str, score: int, warnings: List[str] | None = None) -> Dict[str, Any]:
    action_cfg = config.get("decision", {}).get("actions", {}).get(name, {})
    argv = [str(value).replace("{python}", str(config.get("runtime", {}).get("local_python", "python"))) for value in action_cfg.get("argv", [])]
    risk_level = action_cfg.get("risk_level", "high")
    allowed_actions = set(config.get("automation", {}).get("allowed_actions", []))
    allowed_risks = set(config.get("automation", {}).get("allowed_risk_levels", ["low"]))
    return {
        "action": name,
        "status": status,
        "freshness": freshness,
        "reason": reason,
        "argv": argv,
        "command": " ".join(argv) if argv else action_cfg.get("handler", ""),
        "handler": action_cfg.get("handler"),
        "required_artifacts": list(action_cfg.get("inputs", [])),
        "risk_level": risk_level,
        "can_execute": status == "ready" and name in allowed_actions and risk_level in allowed_risks,
        "warnings": warnings or [],
        "timeout_seconds": action_cfg.get("timeout_seconds"),
        "score": score,
    }


def build_decision(config: Dict[str, Any], state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    blockers = set(state.get("current_blockers", []))
    sar_case_count = int(state.get("sar_soldier", {}).get("case_bank", {}).get("case_count") or 0)
    sar_specialist_status = state.get("sar_soldier", {}).get("specialist_status")
    class_focus = context.get("class_focus", "soldier")
    sensor = context.get("sensor", "sar")
    scene = context.get("scene", "all")

    case_freshness = state.get("sar_soldier", {}).get("case_bank_freshness", {})
    case_current = case_freshness.get("freshness") == "current" and sar_case_count > 0
    sar_focus = sensor == "sar" and class_focus == "soldier"
    diagnose_status = "completed" if case_current else ("ready" if sar_focus and case_freshness.get("reason") != "missing_inputs" else "blocked")
    incremental = state.get("incremental_learning", {})
    incremental_freshness_state = incremental.get("freshness", {})
    incremental_freshness = incremental_freshness_state.get("freshness", "missing")
    if incremental.get("complete") and incremental_freshness == "current":
        incremental_status = "completed" if incremental.get("passed") else "blocked"
    elif incremental_freshness_state.get("reason") == "missing_inputs":
        incremental_status = "blocked"
    else:
        incremental_status = "ready"

    candidates = [
        _action(config, "formal_submission", "blocked" if blockers else "ready", "current", "正式推理只提供人工审计命令；当前前置条件尚未全部满足。" if blockers else "正式推理前置条件已满足，但 v1 仍要求人工执行。", 100 if not blockers else 10, sorted(blockers)),
        _action(config, "diagnose_sar_soldier", diagnose_status, str(case_freshness.get("freshness", "missing")), "SAR soldier 案例库已生成且输入未变化。" if case_current else "SAR soldier 诊断产物缺失或过期，需要重新生成。", 70 if sar_focus else 20),
        _action(config, "review_incremental_learning", incremental_status, incremental_freshness, "p01-p04 指标已解析并满足验收阈值。" if incremental.get("passed") else "增量指标缺失、过期或未达到阈值，需要重新复核。", 45, list(incremental.get("warnings", []))),
        _action(config, "refresh_blackboard", "completed", "current", "本次决策已基于实时重建的黑板。", 15),
    ]
    candidates.append({
        "action": "reject_casebank_specialist", "status": "rejected" if sar_specialist_status == "rejected" else "blocked",
        "freshness": "current" if sar_specialist_status != "not_run" else "missing",
        "reason": "专用模型未满足全局、IR soldier 和 SAR soldier 的联合采纳阈值。",
        "argv": [], "command": "", "handler": None,
        "required_artifacts": [state.get("sar_soldier", {}).get("specialist", {}).get("metrics_path")],
        "risk_level": "low", "can_execute": False, "warnings": [], "timeout_seconds": None,
        "score": 65 if sar_specialist_status == "rejected" else 0,
    })
    if scene in {"forest", "urban"} and sensor == "sar" and class_focus == "soldier":
        candidates[1]["score"] += 10
        candidates[1]["reason"] += f" 当前场景=`{scene}`，更适合展示定位和置信度错误案例。"
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    recommended = next((item for item in ranked if item["status"] == "ready" and item.get("can_execute")), None)
    if recommended is None:
        recommended = {
            "action": "wait_for_external_input", "status": "blocked", "freshness": "current",
            "reason": "现有低风险动作均已完成或被指标门禁阻塞；等待官方输入或新的合规实验结果。",
            "argv": [], "command": "", "handler": None, "required_artifacts": [],
            "risk_level": "low", "can_execute": False, "warnings": sorted(blockers), "score": 0,
        }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "context": context,
        "recommended_action": recommended,
        "candidates": ranked,
        "current_blockers": sorted(blockers),
        "sar_soldier_evidence": state.get("sar_soldier", {}).get("case_bank", {}),
    }


def write_decision(config: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Path]:
    outputs = config.get("decision", {}).get("outputs", {})
    json_path = resolve_path(outputs.get("decision_json", "reports/agent_blackboard/agent_decision.json"))
    md_path = resolve_path(outputs.get("decision_md", "reports/agent_blackboard/agent_decision_report.md"))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_decision_report(decision), encoding="utf-8")
    return {"json": json_path, "report": md_path}


def render_decision_report(decision: Dict[str, Any]) -> str:
    rec = decision["recommended_action"]
    context = decision["context"]
    lines = [
        "# 智能体决策报告",
        "",
        f"生成时间：{decision.get('generated_at')}",
        "",
        "## 决策上下文",
        "",
        f"- 传感器：`{context.get('sensor')}`",
        f"- 场景：`{context.get('scene')}`",
        f"- 关注类别：`{context.get('class_focus')}`",
        "",
        "## 推荐动作",
        "",
        f"- 动作：`{rec.get('action')}`",
        f"- 状态：`{rec.get('status')}`",
        f"- 风险等级：`{rec.get('risk_level')}`",
        f"- 新鲜度：`{rec.get('freshness')}`",
        f"- 允许执行：`{rec.get('can_execute')}`",
        f"- 原因：{rec.get('reason')}",
        "",
        "```bash",
        rec.get("command", ""),
        "```",
        "",
        "## 候选动作",
        "",
    ]
    for item in decision.get("candidates", []):
        lines.append(
            f"- `{item['action']}` 状态=`{item['status']}` 新鲜度=`{item.get('freshness')}` 风险=`{item['risk_level']}` 允许执行=`{item.get('can_execute')}` 得分=`{item['score']}`"
        )
    evidence = decision.get("sar_soldier_evidence", {})
    lines.extend(
        [
            "",
            "## SAR Soldier 证据",
            "",
            f"- 案例数：`{evidence.get('case_count')}`",
            f"- 推荐动作分布：`{evidence.get('recommended_action')}`",
            "",
            "## 关键结论",
            "",
            "案例库专用模型已被记录为 `rejected`：它对 SAR soldier 的提升不足以抵消全局 mAP50 和 IR soldier 的下降，因此主线继续采用统一 YOLO11s。",
        ]
    )
    return "\n".join(lines) + "\n"
