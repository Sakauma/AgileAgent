from __future__ import annotations

import json
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fair_agent.core.config import configured_python, resolve_path


def _action(config: Dict[str, Any], name: str, status: str, freshness: str, reason: str, score: int, warnings: List[str] | None = None) -> Dict[str, Any]:
    action_cfg = config.get("decision", {}).get("actions", {}).get(name, {})
    python = str(configured_python(config))
    argv = [str(value).replace("{python}", python) for value in action_cfg.get("argv", [])]
    risk_level = action_cfg.get("risk_level", "high")
    allowed_actions = set(config.get("automation", {}).get("allowed_actions", []))
    allowed_risks = set(config.get("automation", {}).get("allowed_risk_levels", ["low"]))
    return {
        "action": name,
        "status": status,
        "freshness": freshness,
        "reason": reason,
        "argv": argv,
        "command": shlex.join(argv) if argv else action_cfg.get("handler", ""),
        "handler": action_cfg.get("handler"),
        "required_artifacts": list(action_cfg.get("inputs", [])),
        "outputs": list(action_cfg.get("outputs", [])),
        "risk_level": risk_level,
        "can_execute": status == "ready" and name in allowed_actions and risk_level in allowed_risks,
        "warnings": warnings or [],
        "timeout_seconds": action_cfg.get("timeout_seconds"),
        "score": score,
    }


def build_decision(config: Dict[str, Any], state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    blockers = set(state.get("current_blockers", []))
    incremental = state.get("incremental_learning", {})

    candidates = [
        _action(config, "formal_submission", "blocked" if blockers else "ready", "current", "正式推理只提供人工审计命令；当前前置条件尚未全部满足。" if blockers else "正式推理前置条件已满足，但 v1 仍要求人工执行。", 100 if not blockers else 10, sorted(blockers)),
        _action(config, "refresh_blackboard", "completed", "current", "本次决策已基于实时重建的黑板。", 15),
    ]
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
        "incremental_evidence": incremental,
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
    evidence = decision.get("incremental_evidence", {})
    lines.extend(
        [
            "",
            "## 增量生产证据",
            "",
            f"- 协议数：`{len(evidence.get('protocols', []))}`",
            f"- 合规验证：`{evidence.get('compliance_verified')}`",
            f"- 当前通过：`{evidence.get('passed')}`",
            f"- 来源：`{evidence.get('source')}`",
        ]
    )
    return "\n".join(lines) + "\n"
