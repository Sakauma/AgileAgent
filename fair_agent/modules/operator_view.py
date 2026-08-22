from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


BLOCKER_LABELS = {
    "official_test_not_ready": "本地赛题评测输入未配置（不阻断源码交付）",
    "official_format_not_confirmed": "官方提交格式尚未确认",
    "official_test_dir_missing": "已配置的本地赛题评测目录不存在",
    "incremental_compliant_threshold_not_met": "增量协议未全部达到门槛",
    "ascend_310b_not_ready": "Ascend 310B 尚未完成验收",
    "final_weight_hash_not_verified": "冻结权重哈希未通过",
    "inference_weight_not_frozen_or_verified": "推理权重未冻结或未通过哈希校验",
    "functional_models_invalid": "功能模型注册表无效",
    "functional_models_x86_not_ready": "功能模型尚未完成 x86 GPU 验收",
    "frozen_asset_checksums_invalid": "冻结资产校验失败",
}


def _protocol_counts(protocols: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    rows = list(protocols)
    return {
        "total": len(rows),
        "passed": sum(bool(row.get("passed")) for row in rows),
        "compliant": sum(bool(row.get("compliant")) for row in rows),
    }


def build_operator_snapshot(state: Dict[str, Any], decision: Dict[str, Any] | None = None) -> Dict[str, Any]:
    decision = decision or {}
    functional = state.get("functional_models", {})
    incremental = state.get("incremental_learning", {})
    submission = state.get("submission", {})
    blockers = list(state.get("current_blockers", []))
    critical_blockers = [
        item
        for item in blockers
        if item
        not in {
            "official_test_not_ready",
            "official_format_not_confirmed",
            "ascend_310b_not_ready",
        }
    ]
    model_rows: List[Dict[str, Any]] = []
    for item in functional.get("models", []):
        model_rows.append(
            {
                "id": item.get("id"),
                "name": item.get("display_name"),
                "function": item.get("function"),
                "status": item.get("status"),
                "x86_gpu": bool(item.get("x86_gpu")),
                "ascend_310b": bool(item.get("ascend_310b")),
            }
        )
    recommended = decision.get("recommended_action") or {
        "action": "not_generated",
        "status": "missing",
        "reason": "尚未生成策略决策。",
        "can_execute": False,
    }
    return {
        "schema_version": 1,
        "generated_at": state.get("generated_at"),
        "evidence_mode": state.get("evidence", {}).get("mode", "unknown"),
        "health": "attention" if critical_blockers else "ready_with_external_gates",
        "blockers": [
            {"code": item, "label": BLOCKER_LABELS.get(item, item), "external": item not in critical_blockers}
            for item in blockers
        ],
        "detector": {
            "name": state.get("detector", {}).get("name"),
            "map50": state.get("detector", {}).get("base_test_map50"),
            "evaluation_split": state.get("detector", {}).get("evaluation_split"),
            "weights": state.get("frozen_assets", {}).get("weights", {}).get("path"),
            "hash_verified": bool(state.get("frozen_assets", {}).get("weights", {}).get("matches_expected")),
        },
        "dataset": state.get("dataset", {}),
        "models": model_rows,
        "incremental": {
            "task_type": incremental.get("task_type"),
            "primary_mode": incremental.get("primary_mode"),
            "supported_modes": incremental.get("supported_modes", []),
            "counts": _protocol_counts(incremental.get("protocols", [])),
            "passed": bool(incremental.get("passed")),
            "compliance_verified": bool(incremental.get("compliance_verified")),
        },
        "submission": {
            "official_test_ready": bool(submission.get("official_test_ready")),
            "official_format_confirmed": bool(submission.get("official_format_confirmed")),
        },
        "deployment": {
            "x86_nvidia_gpu": "ready" if functional.get("all_x86_gpu_ready") else "blocked",
            "ascend_310b": "ready" if functional.get("all_ascend_310b_ready") else "waiting_for_hardware",
            "ascend_gates": ["ATC/OM 转换", "AscendCL 推理", "精度回归", "FPS 与内存实测"],
        },
        "recommended_action": recommended,
    }


def render_console(snapshot: Dict[str, Any]) -> str:
    detector = snapshot["detector"]
    incremental = snapshot["incremental"]
    counts = incremental["counts"]
    deployment = snapshot["deployment"]
    recommended = snapshot["recommended_action"]
    lines = [
        "灵动Agent终端工作台",
        "=" * 72,
        f"状态        {snapshot['health']}    证据 {snapshot['evidence_mode']}    更新时间 {snapshot.get('generated_at')}",
        "-" * 72,
        f"基础检测    {detector.get('name')}    base-test mAP50={detector.get('map50')}",
        f"冻结权重    SHA256={'通过' if detector.get('hash_verified') else '失败'}    {detector.get('weights')}",
        f"增量检测    {incremental.get('primary_mode')}    通过 {counts['passed']}/{counts['total']}    合规={'是' if incremental.get('compliance_verified') else '否'}",
        f"部署目标    x86 NVIDIA={deployment['x86_nvidia_gpu']}    Ascend 310B={deployment['ascend_310b']}",
        "-" * 72,
        f"推荐动作    {recommended.get('action')} [{recommended.get('status')}]",
        f"原因        {recommended.get('reason')}",
        "-" * 72,
        "当前门禁",
    ]
    blockers = snapshot.get("blockers", [])
    lines.extend(f"  {'[外部]' if item['external'] else '[内部]'} {item['label']} ({item['code']})" for item in blockers)
    if not blockers:
        lines.append("  无")
    lines.extend(
        [
            "-" * 72,
            "310B 后续门禁  " + " -> ".join(deployment["ascend_gates"]),
            "提示：使用 `agile-agent status --format json` 获取机器可读状态。",
        ]
    )
    return "\n".join(lines)


def render_snapshot(snapshot: Dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(snapshot, ensure_ascii=False, indent=2)
    return render_console(snapshot)
