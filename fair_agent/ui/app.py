from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str) -> Dict[str, Any]:
    target = ROOT / path
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def load_csv(path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    target = ROOT / path
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def markdown_file(path: str) -> str:
    target = ROOT / path
    return target.read_text(encoding="utf-8") if target.exists() else f"`{path}` not found."


def metric_row(cols, items):
    for col, (label, value) in zip(cols, items):
        col.metric(label, value if value is not None else "N/A")


st.set_page_config(page_title="IR/SAR Agent Workbench", layout="wide")
st.title("IR/SAR 快速学习 Agent Workbench")

state = load_json("reports/agent_blackboard/blackboard_state.json")
decision = load_json("reports/agent_blackboard/agent_decision.json")

if not state:
    st.warning("未找到黑板状态。请先运行 `python -m fair_agent.cli refresh`。")
    st.stop()

tabs = st.tabs(["首页", "数据", "错误分析", "策略", "增量学习", "提交"])

with tabs[0]:
    st.subheader("Agent 状态")
    st.caption(f"黑板生成时间：{state.get('generated_at')} · schema v{state.get('schema_version', 1)}")
    detector = state.get("detector", {})
    weights = state.get("frozen_assets", {}).get("weights", {})
    cols = st.columns(4)
    metric_row(
        cols,
        [
            ("all mAP50", detector.get("combined_all_map50", detector.get("lock_all_map50"))),
            ("SAR mAP50", detector.get("combined_sar_map50", detector.get("lock_sar_map50"))),
            ("soldier mAP50", detector.get("combined_soldier_map50", detector.get("lock_sar_soldier_map50"))),
            ("权重哈希匹配", weights.get("matches_expected")),
        ],
    )
    blockers = state.get("current_blockers") or ["none"]
    st.write("当前阻塞项：", blockers)
    st.write("最终权重：", weights.get("path"))
    st.write("候选状态：", detector.get("candidate_status"))
    if detector.get("combined_all_map50") is not None:
        st.write("稳定性复核：", {"imgsz": detector.get("imgsz"), "combined_mAP50": detector.get("combined_all_map50"), "bootstrap_delta_ci95": detector.get("bootstrap_delta_ci95")})

with tabs[1]:
    st.subheader("数据集黑板")
    dataset = state.get("dataset", {})
    cols = st.columns(2)
    metric_row(cols, [("图像数", dataset.get("image_count")), ("目标数", dataset.get("object_count"))])
    st.write("sensor 分布")
    st.json(dataset.get("sensor", {}))
    st.write("scene 分布")
    st.json(dataset.get("scene", {}))
    metadata = load_csv("reports/metadata.csv", limit=100)
    if metadata:
        st.dataframe(metadata, use_container_width=True)

with tabs[2]:
    st.subheader("SAR Soldier Case Bank")
    sar = state.get("sar_soldier", {})
    case_bank = sar.get("case_bank", {})
    cols = st.columns(3)
    metric_row(
        cols,
        [
            ("case 数", case_bank.get("case_count")),
            ("specialist 状态", sar.get("specialist_status")),
            ("主线策略", "统一 YOLO11s"),
        ],
    )
    st.info(sar.get("reason"))
    rows = load_csv("reports/agent_blackboard/sar_soldier_case_bank.csv", limit=50)
    if rows:
        st.dataframe(rows, use_container_width=True)

with tabs[3]:
    st.subheader("策略选择")
    if not decision:
        st.warning("未找到策略报告。请运行 `python -m fair_agent.cli decide --sensor sar --class-focus soldier`。")
    else:
        rec = decision.get("recommended_action", {})
        st.metric("推荐动作", rec.get("action"))
        st.write("状态：", rec.get("status"))
        st.write("风险：", rec.get("risk_level"))
        st.write("原因：", rec.get("reason"))
        st.write("新鲜度：", rec.get("freshness"))
        st.write("允许自动执行：", rec.get("can_execute"))
        st.code(rec.get("command", ""), language="bash")
        st.dataframe(decision.get("candidates", []), use_container_width=True)

with tabs[4]:
    st.subheader("3+1 增量学习")
    compliant = "reports/incremental_no_old_distill/summary.md"
    legacy = "reports/incremental_learning_p01_p04_summary.md"
    st.markdown(markdown_file(compliant) if (ROOT / compliant).exists() else markdown_file(legacy))

with tabs[5]:
    st.subheader("提交准备")
    submission = state.get("submission", {})
    cols = st.columns(3)
    metric_row(
        cols,
        [
            ("hidden test ready", submission.get("official_test_ready")),
            ("format confirmed", submission.get("official_format_confirmed")),
            ("format", submission.get("official_format")),
        ],
    )
    st.write("官方测试目录：", submission.get("official_test_dir"))
    st.markdown(markdown_file("reports/submission_dryrun_lock_val_report.md"))
    st.markdown("### 正式提交模板")
    st.markdown(markdown_file("reports/final_submission_report_template.md"))
