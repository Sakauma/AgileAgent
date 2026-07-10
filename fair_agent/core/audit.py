from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import ROOT, rel_path, resolve_path
from .hashes import hash_if_exists


def make_run_dir(prefix: str = "run", run_root: str = "reports/agent_runs") -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = resolve_path(run_root) / f"{prefix}_{run_id}"
    path.mkdir(parents=True, exist_ok=False)
    return path


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
            "final_assets_manifest": hash_if_exists(resolve_path("final_submission_assets/manifest.json")),
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
        "# Agent Run Report",
        "",
        f"- run_dir：`{manifest.get('run_dir')}`",
        f"- mode：`{plan.get('mode')}`",
        f"- generated_at：`{manifest.get('created_at')}`",
        f"- recommended_action：`{rec.get('action')}`",
        f"- action_status：`{rec.get('status')}`",
        f"- risk_level：`{rec.get('risk_level')}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = state.get("current_blockers") or ["none"]
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "## Planned Steps",
            "",
        ]
    )
    for step in plan.get("steps", []):
        lines.append(f"- `{step['name']}` execute=`{step['execute']}` reason={step['reason']}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "Dry-run only writes audit artifacts. Execute mode is restricted to configured low-risk actions and never runs training or formal submission.",
        ]
    )
    return "\n".join(lines) + "\n"
