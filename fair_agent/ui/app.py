from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str) -> Dict[str, Any]:
    target = ROOT / path
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def metric_row(cols, items):
    for col, (label, value) in zip(cols, items):
        col.metric(label, value if value is not None else "暂无")


def run_cli(arguments: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "fair_agent.cli", *arguments]
    return subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True, timeout=timeout)


def remember_result(label: str, result: subprocess.CompletedProcess[str]) -> None:
    st.session_state["last_command"] = {
        "label": label,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


st.set_page_config(page_title="IR/SAR 快速学习智能体工作台", layout="wide")
st.title("IR/SAR 快速学习智能体工作台")

state = load_json("reports/agent_blackboard/blackboard_state.json")
decision = load_json("reports/agent_blackboard/agent_decision.json")

if not state:
    st.warning("未找到黑板状态。请先运行 `python -m fair_agent.cli refresh`。")
    st.stop()

tabs = st.tabs(["首页", "数据", "错误分析", "策略", "增量学习", "提交"])

with tabs[0]:
    st.subheader("智能体状态")
    st.caption(f"黑板生成时间：{state.get('generated_at')} · 架构版本 {state.get('schema_version', 1)}")
    evidence = state.get("evidence", {})
    st.caption(f"证据模式：{evidence.get('mode', 'unknown')} · demo 表示脱敏快照，live 表示本机真实实验产物")
    detector = state.get("detector", {})
    weights = state.get("frozen_assets", {}).get("weights", {})
    cols = st.columns(4)
    metric_row(
        cols,
        [
            ("整体 mAP50", detector.get("combined_all_map50", detector.get("lock_all_map50"))),
            ("SAR mAP50", detector.get("combined_sar_map50", detector.get("lock_sar_map50"))),
            ("soldier mAP50", detector.get("combined_soldier_map50", detector.get("lock_sar_soldier_map50"))),
            ("权重哈希匹配", weights.get("matches_expected")),
        ],
    )
    blockers = state.get("current_blockers") or ["无"]
    st.write("当前阻塞项：", blockers)
    st.write("最终权重：", weights.get("path"))
    st.write("候选状态：", detector.get("candidate_status"))
    st.write("证据来源：", evidence.get("sources", {}))
    if detector.get("imgsz") is not None:
        st.write("冻结模型配置：", {"imgsz": detector.get("imgsz"), "状态": detector.get("candidate_status")})
    functional = state.get("functional_models", {})
    st.markdown("### 三个功能模型")
    function_labels = {
        "context_perception": "场景与传感器认知",
        "multimodal_target_detection": "IR/SAR 统一目标检测",
        "new_class_incremental_learning": "新类别快速学习",
    }
    status_labels = {
        "verified": "已验证",
        "partially_verified": "部分验证",
    }
    model_rows = [
        {
            "模型": item.get("display_name"),
            "功能": function_labels.get(item.get("function"), item.get("function")),
            "实现": item.get("implementation"),
            "状态": status_labels.get(item.get("status"), item.get("status")),
            "x86 GPU": item.get("x86_gpu"),
            "Ascend 310B": item.get("ascend_310b"),
        }
        for item in functional.get("models", [])
    ]
    if model_rows:
        st.dataframe(model_rows, width="stretch", hide_index=True)
    st.caption(f"不同功能数量：{functional.get('distinct_function_count')} · 注册表有效：{functional.get('valid')}")
    if functional.get("collaboration"):
        st.write("模型协同链路：", functional.get("collaboration"))
    if st.button("刷新黑板与默认决策", width="content"):
        refresh_result = run_cli(["refresh"])
        if refresh_result.returncode == 0:
            refresh_result = run_cli(["decide"])
        remember_result("刷新黑板与默认决策", refresh_result)
        st.rerun()

with tabs[1]:
    st.subheader("数据集黑板")
    dataset = state.get("dataset", {})
    cols = st.columns(2)
    metric_row(cols, [("图像数", dataset.get("image_count")), ("目标数", dataset.get("object_count"))])
    st.write("传感器分布")
    st.json(dataset.get("sensor", {}))
    st.write("场景分布")
    st.json(dataset.get("scene", {}))
    distribution_rows = []
    for dimension, values in [("传感器", dataset.get("sensor", {})), ("场景", dataset.get("scene", {})), ("包含类别", dataset.get("class_presence_images", {}))]:
        distribution_rows.extend({"维度": dimension, "名称": key, "数量": value} for key, value in values.items())
    if distribution_rows:
        st.dataframe(distribution_rows, width="stretch", hide_index=True)

with tabs[2]:
    st.subheader("SAR Soldier 案例库")
    sar = state.get("sar_soldier", {})
    case_bank = sar.get("case_bank", {})
    cols = st.columns(3)
    metric_row(
        cols,
        [
            ("案例数", case_bank.get("case_count")),
            ("专用模型状态", sar.get("specialist_status")),
            ("主线策略", "统一 YOLO11s"),
        ],
    )
    st.info(sar.get("reason"))
    rows = case_bank.get("top_cases", [])
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)

with tabs[3]:
    st.subheader("策略选择")
    controls = st.columns(3)
    sensor = controls[0].selectbox("传感器", ["sar", "ir"], index=0)
    scene = controls[1].selectbox("场景", ["all", "air", "forest", "sea", "urban"], index=0)
    class_focus = controls[2].selectbox("关注类别", ["soldier", "small_aircraft", "warship", "tank"], index=0)
    command_args = ["--sensor", sensor, "--scene", scene, "--class-focus", class_focus]
    action_cols = st.columns(3)
    if action_cols[0].button("生成决策", width="stretch"):
        result = run_cli(["decide", *command_args])
        remember_result("生成决策", result)
        st.rerun()
    if action_cols[1].button("生成试运行计划", width="stretch"):
        result = run_cli(["pipeline", "--mode", "dryrun", *command_args])
        remember_result("生成试运行计划", result)
        st.rerun()
    execute_confirmed = st.checkbox("我确认只执行允许列表中的低风险动作")
    if action_cols[2].button("执行低风险动作", disabled=not execute_confirmed, width="stretch"):
        result = run_cli(["pipeline", "--mode", "execute", *command_args])
        remember_result("执行低风险动作", result)
        st.rerun()
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
        st.dataframe(decision.get("candidates", []), width="stretch")
    last_command = st.session_state.get("last_command")
    if last_command:
        if last_command["returncode"] == 0:
            st.success(f"{last_command['label']} 已完成")
        else:
            st.error(f"{last_command['label']} 失败，返回码 {last_command['returncode']}")
        output = (last_command.get("stdout") or "") + (last_command.get("stderr") or "")
        if output.strip():
            st.code(output.strip(), language="text")

with tabs[4]:
    st.subheader("3+1 增量学习")
    incremental = state.get("incremental_learning", {})
    cols = st.columns(3)
    metric_row(cols, [("协议完整", incremental.get("complete")), ("合规证据", incremental.get("compliance_verified")), ("总体验收", incremental.get("passed"))])
    st.write("验收阈值：", incremental.get("acceptance", {}))
    protocols = incremental.get("protocols", [])
    if protocols:
        st.dataframe(protocols, width="stretch", hide_index=True)
    if incremental.get("warnings"):
        st.warning(incremental.get("warnings"))

with tabs[5]:
    st.subheader("提交准备")
    submission = state.get("submission", {})
    cols = st.columns(3)
    metric_row(
        cols,
        [
            ("隐藏测试集就绪", submission.get("official_test_ready")),
            ("提交格式已确认", submission.get("official_format_confirmed")),
            ("提交格式", submission.get("official_format")),
        ],
    )
    st.write("官方测试目录：", submission.get("official_test_dir"))
    st.markdown("### 已验证运行")
    run_rows = []
    for name in ["smoke", "dryrun"]:
        item = submission.get(name, {})
        run_rows.append({
            "类型": name,
            "图像数": item.get("image_count"),
            "预测数": item.get("prediction_count"),
            "模型 SHA256": item.get("model_sha256"),
            "有效": submission.get(f"{name}_valid"),
        })
    st.dataframe(run_rows, width="stretch", hide_index=True)
    st.info("正式提交仍受官方测试目录和提交格式门禁控制，工作台不会自动提交。")
