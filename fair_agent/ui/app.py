from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import streamlit as st

from fair_agent.modules.operator_view import build_operator_snapshot


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: str | Path) -> Dict[str, Any]:
    target = ROOT / path
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def run_cli(arguments: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "fair_agent.cli", *arguments],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def remember_result(label: str, result: subprocess.CompletedProcess[str]) -> None:
    st.session_state["last_command"] = {
        "label": label,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def metric_row(items: Iterable[tuple[str, Any]]) -> None:
    values = list(items)
    for col, (label, value) in zip(st.columns(len(values)), values):
        col.metric(label, value if value is not None else "暂无")


def render_command_result() -> None:
    result = st.session_state.get("last_command")
    if not result:
        return
    if result["returncode"] == 0:
        st.success(f"{result['label']}已完成")
    else:
        st.error(f"{result['label']}失败，返回码 {result['returncode']}")
    output = (result.get("stdout") or "") + (result.get("stderr") or "")
    if output.strip():
        with st.expander("命令输出"):
            st.code(output.strip(), language="text")


def latest_agent_runs(limit: int = 10) -> list[Dict[str, Any]]:
    root = ROOT / "reports" / "agent_runs"
    rows = []
    if not root.exists():
        return rows
    for run_dir in sorted((item for item in root.iterdir() if item.is_dir()), reverse=True)[:limit]:
        plan = load_json(run_dir.relative_to(ROOT) / "plan.json")
        manifest = load_json(run_dir.relative_to(ROOT) / "manifest.json")
        rows.append(
            {
                "运行": run_dir.name,
                "模式": plan.get("mode"),
                "终止原因": plan.get("termination"),
                "步骤数": len(plan.get("steps", [])),
                "创建时间": manifest.get("created_at"),
            }
        )
    return rows


st.set_page_config(page_title="AgileAgent", page_icon=None, layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1500px;}
    [data-testid="stSidebar"] {border-right: 1px solid #d8dee6;}
    [data-testid="stMetric"] {background: #ffffff; border: 1px solid #d8dee6; border-radius: 6px; padding: 12px 14px;}
    [data-testid="stMetricLabel"] {color: #52606d !important;}
    [data-testid="stMetricValue"] {color: #17202a !important;}
    h1, h2, h3 {letter-spacing: 0;}
    .agent-kicker {color: #52606d; font-size: 0.86rem; margin-bottom: 0.2rem;}
    .agent-title {font-size: 2rem; font-weight: 700; line-height: 1.15; margin-bottom: 0.2rem;}
    .agent-meta {color: #66788a; font-size: 0.86rem; margin-bottom: 1.2rem;}
    .status-line {border-left: 4px solid #16845b; background: #f3f8f5; color: #17202a; padding: 10px 14px; margin: 8px 0 18px 0;}
    .status-line.attention {border-left-color: #c17b13; background: #fff8eb;}
    </style>
    """,
    unsafe_allow_html=True,
)

state = load_json("reports/agent_blackboard/blackboard_state.json")
decision = load_json("reports/agent_blackboard/agent_decision.json")
if not state:
    st.warning("黑板状态尚未生成。")
    if st.button("初始化工作台", type="primary"):
        result = run_cli(["refresh"])
        if result.returncode == 0:
            result = run_cli(["decide"])
        remember_result("初始化工作台", result)
        st.rerun()
    st.stop()

snapshot = build_operator_snapshot(state, decision)
page = st.sidebar.radio(
    "工作区",
    ["运行总览", "模型与协同", "数据与诊断", "策略运行", "增量学习", "审计与部署"],
)
st.sidebar.divider()
st.sidebar.caption(f"证据模式：{snapshot['evidence_mode']}")
st.sidebar.caption(f"黑板时间：{snapshot.get('generated_at')}")
st.sidebar.caption("运行目标：x86 NVIDIA GPU")
if st.sidebar.button("刷新状态", width="stretch"):
    result = run_cli(["refresh"])
    if result.returncode == 0:
        result = run_cli(["decide"])
    remember_result("刷新状态", result)
    st.rerun()

st.markdown('<div class="agent-kicker">IR / SAR 快速学习智能体</div>', unsafe_allow_html=True)
st.markdown(f'<div class="agent-title">{page}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="agent-meta">AgileAgent · 配置驱动 · 审计优先 · {snapshot.get("generated_at")}</div>',
    unsafe_allow_html=True,
)

if page == "运行总览":
    detector = snapshot["detector"]
    counts = snapshot["incremental"]["counts"]
    metric_row(
        [
            ("基础模型 mAP50", detector.get("map50")),
            ("SAR mAP50", detector.get("sar_map50")),
            ("增量协议通过", f"{counts['passed']} / {counts['total']}"),
            ("冻结哈希", "通过" if detector.get("hash_verified") else "失败"),
        ]
    )
    health_class = "attention" if snapshot["health"] == "attention" else ""
    action = snapshot["recommended_action"]
    st.markdown(
        f'<div class="status-line {health_class}"><strong>当前动作：{action.get("action")}</strong><br>{action.get("reason")}</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("门禁状态")
        blocker_rows = [
            {
                "类型": "外部输入" if item["external"] else "工程内部",
                "状态": item["label"],
                "代码": item["code"],
            }
            for item in snapshot["blockers"]
        ]
        if blocker_rows:
            st.dataframe(blocker_rows, width="stretch", hide_index=True)
        else:
            st.success("当前没有阻塞项。")
    with right:
        st.subheader("提交准备")
        submission = snapshot["submission"]
        st.dataframe(
            [
                {"检查项": "隐藏测试集", "通过": submission["official_test_ready"]},
                {"检查项": "提交格式", "通过": submission["official_format_confirmed"]},
                {"检查项": "Dry-run", "通过": submission["dryrun_valid"]},
                {"检查项": "Smoke test", "通过": submission["smoke_valid"]},
            ],
            width="stretch",
            hide_index=True,
        )

elif page == "模型与协同":
    function_labels = {
        "context_perception": "场景与传感器认知",
        "multimodal_target_detection": "IR/SAR 统一目标检测",
        "incremental_object_detection": "增量目标检测",
    }
    rows = []
    for item in snapshot["models"]:
        rows.append(
            {
                "模型": item["name"],
                "功能": function_labels.get(item["function"], item["function"]),
                "状态": item["status"],
                "x86 GPU": item["x86_gpu"],
                "Ascend 310B": item["ascend_310b"],
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)
    st.subheader("协同链路")
    collaboration = state.get("functional_models", {}).get("collaboration", [])
    st.dataframe(
        [
            {
                "上游": item.get("from"),
                "下游": item.get("to"),
                "载荷": item.get("payload"),
                "状态": item.get("status"),
                "用途": item.get("purpose"),
            }
            for item in collaboration
        ],
        width="stretch",
        hide_index=True,
    )
    st.subheader("冻结资产")
    st.code(snapshot["detector"].get("weights") or "", language="text")

elif page == "数据与诊断":
    dataset = snapshot["dataset"]
    metric_row([("图像数", dataset.get("image_count")), ("目标数", dataset.get("object_count")), ("错误案例", state.get("sar_soldier", {}).get("case_bank", {}).get("case_count"))])
    distributions = []
    for dimension, values in [
        ("传感器", dataset.get("sensor", {})),
        ("场景", dataset.get("scene", {})),
        ("类别出现图像", dataset.get("class_presence_images", {})),
    ]:
        distributions.extend({"维度": dimension, "名称": key, "数量": value} for key, value in values.items())
    st.subheader("数据分布")
    st.dataframe(distributions, width="stretch", hide_index=True)
    st.subheader("SAR Soldier 高优先级案例")
    st.caption(state.get("sar_soldier", {}).get("reason"))
    top_cases = state.get("sar_soldier", {}).get("case_bank", {}).get("top_cases", [])
    if top_cases:
        st.dataframe(top_cases, width="stretch", hide_index=True)
    else:
        st.info("当前证据中没有案例记录。")

elif page == "策略运行":
    controls = st.columns(3)
    sensor = controls[0].selectbox("传感器", ["sar", "ir"])
    scene = controls[1].selectbox("场景", ["all", "air", "forest", "sea", "urban"])
    class_focus = controls[2].selectbox("关注类别", ["soldier", "small_aircraft", "warship", "tank"])
    args = ["--sensor", sensor, "--scene", scene, "--class-focus", class_focus]
    buttons = st.columns(3)
    if buttons[0].button("生成决策", type="primary", width="stretch"):
        remember_result("生成决策", run_cli(["decide", *args]))
        st.rerun()
    if buttons[1].button("生成 Dry-run", width="stretch"):
        remember_result("生成 Dry-run", run_cli(["pipeline", "--mode", "dryrun", *args]))
        st.rerun()
    confirmed = st.checkbox("确认执行低风险允许列表")
    if buttons[2].button("执行低风险动作", disabled=not confirmed, width="stretch"):
        remember_result("执行低风险动作", run_cli(["pipeline", "--mode", "execute", *args]))
        st.rerun()
    action = snapshot["recommended_action"]
    metric_row([("推荐动作", action.get("action")), ("状态", action.get("status")), ("风险", action.get("risk_level")), ("允许执行", action.get("can_execute"))])
    st.write(action.get("reason"))
    if action.get("command"):
        st.code(action["command"], language="bash")
    candidates = decision.get("candidates", [])
    if candidates:
        st.subheader("候选动作")
        st.dataframe(candidates, width="stretch", hide_index=True)
    render_command_result()

elif page == "增量学习":
    incremental = state.get("incremental_learning", {})
    counts = snapshot["incremental"]["counts"]
    metric_row([("协议数", counts["total"]), ("通过", counts["passed"]), ("数据合规", counts["compliant"]), ("总体验收", incremental.get("passed"))])
    st.subheader("任务契约")
    st.dataframe(
        [
            {"字段": "任务类型", "值": incremental.get("task_type")},
            {"字段": "主模式", "值": incremental.get("primary_mode")},
            {"字段": "支持模式", "值": ", ".join(incremental.get("supported_modes", []))},
            {"字段": "学习数据边界", "值": incremental.get("learning_data_scope")},
        ],
        width="stretch",
        hide_index=True,
    )
    st.subheader("协议指标")
    st.dataframe(incremental.get("protocols", []), width="stretch", hide_index=True)
    st.caption(f"验收阈值：{incremental.get('acceptance', {})}")

elif page == "审计与部署":
    deployment = snapshot["deployment"]
    x86_label = "就绪" if deployment["x86_nvidia_gpu"] == "ready" else "阻塞"
    ascend_label = "就绪" if deployment["ascend_310b"] == "ready" else "等待板卡"
    evidence_label = "实时" if snapshot["evidence_mode"] == "live" else "演示"
    metric_row([("x86 NVIDIA GPU", x86_label), ("Ascend 310B", ascend_label), ("证据模式", evidence_label)])
    left, right = st.columns([1, 1])
    with left:
        st.subheader("310B 验收门禁")
        st.dataframe(
            [{"顺序": index, "门禁": gate, "状态": "待板卡"} for index, gate in enumerate(deployment["ascend_gates"], 1)],
            width="stretch",
            hide_index=True,
        )
        st.info("当前不将 x86 GPU 指标标记为 310B 证据。")
    with right:
        st.subheader("运行入口")
        st.code("./scripts/start_agent.sh", language="bash")
        st.code("./scripts/start_agent.sh --cli", language="bash")
        st.code("agile-agent status --format json --refresh", language="bash")
    st.subheader("最近审计运行")
    runs = latest_agent_runs()
    if runs:
        st.dataframe(runs, width="stretch", hide_index=True)
    else:
        st.info("尚无审计运行记录。")
