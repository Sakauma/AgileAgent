from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from fair_agent.core.config import resolve_path
from fair_agent.core.hashes import sha256_file


def refresh_frozen_assets(selected_imgsz: int, status: str, stability_report: Optional[Path] = None, candidate_metrics: Optional[Dict[str, Any]] = None) -> Path:
    assets = resolve_path("final_submission_assets")
    freeze_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    history = assets / "history" / freeze_id
    history.mkdir(parents=True, exist_ok=False)
    for name in ["manifest.json", "SHA256SUMS.txt", "submission_infer_yolo11s_imgsz640.yaml", "42_predict_submission.py", "final_candidate_card.md"]:
        source = assets / name
        if source.exists():
            shutil.copy2(source, history / name)

    active_config = resolve_path("configs/submission_infer_yolo11s_imgsz640.yaml")
    config: Dict[str, Any] = yaml.safe_load(active_config.read_text(encoding="utf-8"))
    config["model"] = "final_submission_assets/best.pt"
    config["predict"]["imgsz"] = int(selected_imgsz)
    config["output"]["package_name"] = f"yolo11s_imgsz{selected_imgsz}_submission"
    active_config.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    shutil.copy2(active_config, assets / "submission_infer_yolo11s_imgsz640.yaml")
    shutil.copy2(resolve_path("tools/42_predict_submission.py"), assets / "42_predict_submission.py")
    shutil.copy2(resolve_path("reports/final_candidate_card.md"), assets / "final_candidate_card.md")
    synchronized_reports = {
        "reports/agent_blackboard/agent_decision_report.md": "agent_decision_report.md",
        "reports/incremental_learning_p01_p04_summary.md": "incremental_learning_p01_p04_summary.md",
        "reports/incremental_no_old_distill/summary.md": "incremental_no_old_distill_summary.md",
        "reports/incremental_no_old_distill/summary.csv": "incremental_no_old_distill_summary.csv",
        "reports/submission_dryrun_lock_val_report.md": "submission_dryrun_lock_val_report.md",
        "reports/submission_inference_smoke_report.md": "submission_inference_smoke_report.md",
        "最终材料总览.md": "最终材料总览.md",
        "提交前检查清单.md": "提交前检查清单.md",
    }
    for source_name, target_name in synchronized_reports.items():
        source = resolve_path(source_name)
        if source.exists():
            shutil.copy2(source, assets / target_name)
    if stability_report and stability_report.exists():
        shutil.copy2(stability_report, assets / "inference_size_stability_report.md")

    previous = json.loads((history / "manifest.json").read_text(encoding="utf-8")) if (history / "manifest.json").exists() else {}
    candidate = dict(previous.get("frozen_candidate", {}))
    if int(candidate.get("imgsz", selected_imgsz)) != int(selected_imgsz):
        candidate = {key: value for key, value in candidate.items() if not key.startswith("lock_")}
    candidate.update({"weights": "best.pt", "imgsz": int(selected_imgsz), "metric": "mAP50", "status": status})
    candidate.update(candidate_metrics or {})
    tracked = sorted(path for path in assets.iterdir() if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS.txt"})
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project": previous.get("project", "tiaozhanbei"),
        "final_model": f"YOLO11s unified imgsz={selected_imgsz}",
        "frozen_candidate": candidate,
        "files": [{"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in tracked],
    }
    (assets / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums = [f"{sha256_file(path)}  {path.name}" for path in tracked]
    sums.append(f"{sha256_file(assets / 'manifest.json')}  manifest.json")
    (assets / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return history
