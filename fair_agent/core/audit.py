from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import ROOT, rel_path, resolve_path
from .hashes import hash_if_exists


def make_run_dir(prefix: str = "run", run_root: str = "reports/agent_runs") -> Path:
    root = resolve_path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for suffix in range(100):
        extra = "" if suffix == 0 else f"_{suffix:02d}"
        path = root / f"{prefix}_{run_id}{extra}"
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise RuntimeError("无法创建唯一的智能体运行目录")


def write_pipeline_artifacts(run_dir: Path, plan: Dict[str, Any], state: Dict[str, Any], decision: Dict[str, Any], action_results: list[Dict[str, Any]] | None = None) -> Dict[str, Path]:
    plan_path = run_dir / "plan.json"
    manifest_path = run_dir / "manifest.json"
    report_path = run_dir / "agent_run_report.md"
    log_path = run_dir / "action_log.jsonl"

    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": rel_path(run_dir),
        "mode": plan.get("mode"),
        "state_generated_at": state.get("generated_at"),
        "recommended_action": decision.get("recommended_action", {}).get("action"),
        "action_results": action_results or [],
        "weights": state.get("frozen_assets", {}).get("weights", {}),
        "key_artifacts": {
            "blackboard": hash_if_exists(resolve_path("reports/agent_blackboard/blackboard_state.json")),
            "decision": hash_if_exists(resolve_path("reports/agent_blackboard/agent_decision.json")),
            "model_manifest": state.get("frozen_assets", {}).get("manifest", {}),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_run_report(plan, state, decision, manifest), encoding="utf-8")
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")
    return {"plan": plan_path, "manifest": manifest_path, "report": report_path, "log": log_path}


def render_run_report(plan: Dict[str, Any], state: Dict[str, Any], decision: Dict[str, Any], manifest: Dict[str, Any]) -> str:
    rec = decision.get("recommended_action", {})
    lines = [
        "# 智能体运行报告",
        "",
        f"- 运行目录：`{manifest.get('run_dir')}`",
        f"- 运行模式：`{plan.get('mode')}`",
        f"- 生成时间：`{manifest.get('created_at')}`",
        f"- 推荐动作：`{rec.get('action')}`",
        f"- 动作状态：`{rec.get('status')}`",
        f"- 风险等级：`{rec.get('risk_level')}`",
        f"- 终止原因：`{plan.get('termination')}`",
        "",
        "## 阻塞项",
        "",
    ]
    blockers = state.get("current_blockers") or ["无"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "## 计划步骤",
            "",
        ]
    )
    for step in plan.get("steps", []):
        lines.append(f"- `{step['name']}` 执行=`{step['execute']}` 原因={step['reason']}")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "试运行只写入审计产物。执行模式仅允许运行配置中的低风险动作，不会启动训练或正式提交。",
        ]
    )
    return "\n".join(lines) + "\n"
