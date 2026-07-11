from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
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


def model_pipeline(models: list[Dict[str, Any]]) -> str:
    labels = {
        "context_perception": ("01", "环境认知", "IR / SAR 与场景识别"),
        "multimodal_target_detection": ("02", "统一检测", "YOLO11s · imgsz 640"),
        "incremental_object_detection": ("03", "增量模型库", "类别增量 / 目标增量"),
    }
    nodes = []
    for item in models:
        number, title, detail = labels.get(item.get("function"), ("--", item.get("name"), item.get("function")))
        status = "已验证" if item.get("status") == "verified" else "部分验证"
        status_class = "ok" if item.get("status") == "verified" else "warn"
        nodes.append(
            f'<div class="pipeline-node"><div class="node-index">{number}</div>'
            f'<div><div class="node-title">{title}</div><div class="node-detail">{detail}</div></div>'
            f'<span class="status-pill {status_class}">{status}</span></div>'
        )
    return '<div class="pipeline">' + '<div class="pipeline-arrow">→</div>'.join(nodes) + "</div>"


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
    :root {--ink:#18242D; --muted:#667783; --line:#DCE4E9; --teal:#087F8C; --green:#167A5A; --amber:#B66A09;}
    .stApp {background:#F4F7F9; color:var(--ink);}
    [data-testid="stHeader"] {background:transparent;}
    [data-testid="stToolbar"], footer {display:none !important;}
    .block-container {padding-top:2.2rem; padding-bottom:3rem; max-width:1420px;}
    [data-testid="stSidebar"] {background:#101A22; border-right:0; min-width:260px;}
    [data-testid="stSidebar"] * {color:#D8E1E7;}
    [data-testid="stSidebar"] hr {border-color:#2A3944;}
    [data-testid="stSidebar"] [role="radiogroup"] label {padding:0.62rem 0.75rem; border-radius:5px; margin:2px 0;}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#173943;}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color:#FFFFFF; font-weight:600;}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#8FA1AD;}
    [data-testid="stSidebar"] button {border-color:#3A4B57; background:#16232C;}
    [data-testid="stSidebar"] button:hover {border-color:#42A7AD; color:#FFFFFF;}
    [data-testid="stMetric"] {background:#FFFFFF; border:1px solid var(--line); border-top:3px solid #9FB1BC; border-radius:6px; padding:14px 16px; box-shadow:0 1px 2px rgba(20,39,51,.04); min-height:112px;}
    [data-testid="stMetricLabel"] {color:#60717C !important; font-size:.78rem; text-transform:uppercase; letter-spacing:.035em;}
    [data-testid="stMetricValue"] {color:var(--ink) !important; font-size:1.8rem;}
    [data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:6px; overflow:hidden;}
    .stButton button {border-radius:5px; font-weight:600;}
    h1, h2, h3 {letter-spacing:0; color:var(--ink);}
    h3 {font-size:1.12rem !important; margin-top:1.1rem !important;}
    .brand-mark {font-size:1.12rem; font-weight:750; color:#FFFFFF; letter-spacing:.03em; margin:0.2rem 0 0;}
    .brand-sub {font-size:.72rem; color:#7FD0D2; text-transform:uppercase; letter-spacing:.12em; margin-bottom:1.5rem;}
    .runtime-chip {display:inline-flex; align-items:center; gap:7px; font-size:.72rem; color:#AFC0CA; margin:.3rem 0 .8rem;}
    .runtime-chip:before {content:""; width:7px; height:7px; border-radius:50%; background:#36B37E; box-shadow:0 0 0 3px rgba(54,179,126,.14);}
    .agent-kicker {color:var(--teal); font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; margin-bottom:.35rem;}
    .agent-title {font-size:2rem; font-weight:750; line-height:1.15; color:var(--ink); margin-bottom:.3rem;}
    .agent-meta {color:#71828D; font-size:.8rem; margin-bottom:1.4rem;}
    .command-bar {display:flex; justify-content:space-between; align-items:center; gap:20px; border:1px solid #E3CF9F; border-left:4px solid var(--amber); background:#FFF9EC; color:#473817; padding:14px 16px; margin:16px 0 22px;}
    .command-label {font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:#95600F;}
    .command-title {font-size:1rem; font-weight:700; color:#2D2B25; margin-top:2px;}
    .command-reason {font-size:.82rem; color:#6D5A32; text-align:right; max-width:62%;}
    .section-label {font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:#71828D; margin:1.4rem 0 .65rem;}
    .pipeline {display:grid; grid-template-columns:1fr 32px 1fr 32px 1fr; align-items:stretch; gap:8px; margin-bottom:1.2rem;}
    .pipeline-node {display:grid; grid-template-columns:34px 1fr auto; align-items:center; gap:10px; background:#FFFFFF; border:1px solid var(--line); padding:14px; min-height:76px;}
    .node-index {font-size:.7rem; font-weight:700; color:var(--teal); border-right:1px solid var(--line); padding-right:9px;}
    .node-title {font-weight:700; color:var(--ink);}
    .node-detail {font-size:.74rem; color:#71828D; margin-top:2px;}
    .pipeline-arrow {display:flex; align-items:center; justify-content:center; color:#82949F; font-size:1.15rem;}
    .status-pill {font-size:.66rem; font-weight:700; padding:3px 7px; border-radius:12px; white-space:nowrap;}
    .status-pill.ok {color:#126245; background:#E9F7F0;}
    .status-pill.warn {color:#8A570C; background:#FFF2D8;}
    .gate-list {background:#FFFFFF; border:1px solid var(--line);}
    .gate-row {display:grid; grid-template-columns:10px 1fr auto; gap:10px; align-items:center; padding:11px 13px; border-bottom:1px solid #E8EEF1; font-size:.82rem;}
    .gate-row:last-child {border-bottom:0;}
    .gate-dot {width:7px; height:7px; border-radius:50%; background:#D08A20;}
    .gate-dot.external {background:#82949F;}
    .gate-type {font-size:.68rem; color:#7A8A94; text-transform:uppercase;}
    @media (max-width: 900px) {
      .block-container {padding-top:1.25rem;}
      .pipeline {grid-template-columns:1fr;}
      .pipeline-arrow {transform:rotate(90deg); min-height:18px;}
      .command-bar {display:block;}
      .command-reason {max-width:100%; text-align:left; margin-top:8px;}
    }
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
st.sidebar.markdown('<div class="brand-mark">AGILE AGENT</div><div class="brand-sub">IR / SAR Workbench</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "工作区",
    ["运行总览", "模型与协同", "数据与诊断", "策略运行", "增量学习", "审计与部署"],
)
st.sidebar.divider()
st.sidebar.markdown('<div class="runtime-chip">x86 NVIDIA GPU 在线</div>', unsafe_allow_html=True)
st.sidebar.caption(f"证据：{snapshot['evidence_mode']}  ·  {snapshot.get('generated_at')}")
if st.sidebar.button("刷新状态", icon=":material/refresh:", width="stretch"):
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
    action = snapshot["recommended_action"]
    st.markdown(
        f'<div class="command-bar"><div><div class="command-label">当前 Agent 行动</div>'
        f'<div class="command-title">{action.get("action")}</div></div>'
        f'<div class="command-reason">{action.get("reason")}</div></div>',
        unsafe_allow_html=True,
    )
    metric_row(
        [
            ("LOCK-ALL MAP50", detector.get("map50")),
            ("SAR MAP50", detector.get("sar_map50")),
            ("增量协议", f"{counts['passed']} / {counts['total']} 通过"),
            ("冻结资产", "SHA256 通过" if detector.get("hash_verified") else "校验失败"),
        ]
    )
    st.markdown('<div class="section-label">模型协同链路</div>', unsafe_allow_html=True)
    st.markdown(model_pipeline(snapshot["models"]), unsafe_allow_html=True)
    left, right = st.columns([1.08, 1], gap="large")
    with left:
        st.markdown('<div class="section-label">当前门禁</div>', unsafe_allow_html=True)
        if snapshot["blockers"]:
            gate_rows = "".join(
                f'<div class="gate-row"><span class="gate-dot {"external" if item["external"] else ""}"></span>'
                f'<span>{item["label"]}</span><span class="gate-type">{"外部输入" if item["external"] else "工程内部"}</span></div>'
                for item in snapshot["blockers"]
            )
            st.markdown(f'<div class="gate-list">{gate_rows}</div>', unsafe_allow_html=True)
        else:
            st.success("当前没有阻塞项。")
    with right:
        st.markdown('<div class="section-label">数据态势</div>', unsafe_allow_html=True)
        sensor_values = snapshot.get("dataset", {}).get("sensor", {})
        if sensor_values:
            sensor_frame = pd.DataFrame({"图像数": sensor_values}).rename_axis("传感器")
            st.bar_chart(sensor_frame, color="#087F8C", height=190)
        submission = snapshot["submission"]
        st.caption(
            f"提交链路  ·  Dry-run {'通过' if submission['dryrun_valid'] else '待完成'}  ·  "
            f"Smoke {'通过' if submission['smoke_valid'] else '待完成'}  ·  官方格式待确认"
        )

elif page == "模型与协同":
    function_labels = {
        "context_perception": "场景与传感器认知",
        "multimodal_target_detection": "IR/SAR 统一目标检测",
        "incremental_object_detection": "增量目标检测",
    }
    st.markdown('<div class="section-label">推理与学习流水线</div>', unsafe_allow_html=True)
    st.markdown(model_pipeline(snapshot["models"]), unsafe_allow_html=True)
    rows = []
    for item in snapshot["models"]:
        rows.append(
            {
                "模型": item["name"],
                "功能": function_labels.get(item["function"], item["function"]),
                "状态": "已验证" if item["status"] == "verified" else "部分验证",
                "x86 GPU": "就绪" if item["x86_gpu"] else "阻塞",
                "Ascend 310B": "就绪" if item["ascend_310b"] else "待板卡",
            }
        )
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="section-label">模型注册表</div>', unsafe_allow_html=True)
        st.dataframe(rows, width="stretch", hide_index=True)
    collaboration = state.get("functional_models", {}).get("collaboration", [])
    with right:
        st.markdown('<div class="section-label">协同契约</div>', unsafe_allow_html=True)
        st.dataframe(
            [
                {
                    "上游": item.get("from"),
                    "下游": item.get("to"),
                    "载荷": item.get("payload"),
                    "状态": item.get("status"),
                }
                for item in collaboration
            ],
            width="stretch",
            hide_index=True,
        )
    st.markdown('<div class="section-label">冻结资产</div>', unsafe_allow_html=True)
    st.code(snapshot["detector"].get("weights") or "", language="text")

elif page == "数据与诊断":
    dataset = snapshot["dataset"]
    metric_row([("图像数", dataset.get("image_count")), ("目标数", dataset.get("object_count")), ("错误案例", state.get("sar_soldier", {}).get("case_bank", {}).get("case_count"))])
    charts = st.columns(3, gap="large")
    for col, (title, values, color) in zip(
        charts,
        [
            ("传感器分布", dataset.get("sensor", {}), "#087F8C"),
            ("场景分布", dataset.get("scene", {}), "#3B6EA8"),
            ("类别出现图像", dataset.get("class_presence_images", {}), "#B66A09"),
        ],
    ):
        with col:
            st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)
            if values:
                frame = pd.DataFrame({"数量": values}).rename_axis("名称")
                st.bar_chart(frame, color=color, height=220)
    st.markdown('<div class="section-label">SAR Soldier 高优先级案例</div>', unsafe_allow_html=True)
    st.caption(state.get("sar_soldier", {}).get("reason"))
    top_cases = state.get("sar_soldier", {}).get("case_bank", {}).get("top_cases", [])
    if top_cases:
        st.dataframe(
            [
                {
                    "排序": item.get("rank"),
                    "图像": item.get("image_path"),
                    "场景": item.get("scene"),
                    "错误类型": item.get("statuses"),
                    "优先级": item.get("priority"),
                    "建议动作": item.get("recommended_action"),
                }
                for item in top_cases
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("当前证据中没有案例记录。")

elif page == "策略运行":
    controls = st.columns(3)
    sensor = controls[0].selectbox("传感器", ["sar", "ir"])
    scene = controls[1].selectbox("场景", ["all", "air", "forest", "sea", "urban"])
    class_focus = controls[2].selectbox("关注类别", ["soldier", "small_aircraft", "warship", "tank"])
    args = ["--sensor", sensor, "--scene", scene, "--class-focus", class_focus]
    buttons = st.columns(3)
    if buttons[0].button("生成决策", icon=":material/psychology:", type="primary", width="stretch"):
        remember_result("生成决策", run_cli(["decide", *args]))
        st.rerun()
    if buttons[1].button("生成 Dry-run", icon=":material/description:", width="stretch"):
        remember_result("生成 Dry-run", run_cli(["pipeline", "--mode", "dryrun", *args]))
        st.rerun()
    confirmed = st.checkbox("确认执行低风险允许列表")
    if buttons[2].button("执行低风险动作", icon=":material/play_arrow:", disabled=not confirmed, width="stretch"):
        remember_result("执行低风险动作", run_cli(["pipeline", "--mode", "execute", *args]))
        st.rerun()
    action = snapshot["recommended_action"]
    status_label = {"blocked": "已阻塞", "ready": "可执行", "completed": "已完成", "rejected": "已否决"}.get(action.get("status"), action.get("status"))
    risk_label = {"low": "低风险", "medium": "中风险", "high": "高风险"}.get(action.get("risk_level"), action.get("risk_level"))
    st.markdown(
        f'<div class="command-bar"><div><div class="command-label">推荐动作 · {status_label} · {risk_label}</div>'
        f'<div class="command-title">{action.get("action")}</div></div>'
        f'<div class="command-reason">{action.get("reason")}</div></div>',
        unsafe_allow_html=True,
    )
    if action.get("command"):
        st.code(action["command"], language="bash")
    candidates = decision.get("candidates", [])
    if candidates:
        st.markdown('<div class="section-label">候选动作</div>', unsafe_allow_html=True)
        st.dataframe(
            [
                {
                    "动作": item.get("action"),
                    "状态": item.get("status"),
                    "新鲜度": item.get("freshness"),
                    "风险": item.get("risk_level"),
                    "允许执行": "是" if item.get("can_execute") else "否",
                    "得分": item.get("score"),
                }
                for item in candidates
            ],
            width="stretch",
            hide_index=True,
        )
    render_command_result()

elif page == "增量学习":
    incremental = state.get("incremental_learning", {})
    counts = snapshot["incremental"]["counts"]
    metric_row([("协议数", counts["total"]), ("通过协议", counts["passed"]), ("数据合规", f"{counts['compliant']} / {counts['total']}"), ("总体验收", "通过" if incremental.get("passed") else "需改进")])
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
    st.dataframe(
        [
            {
                "协议": item.get("protocol"),
                "增量模式": item.get("incremental_mode"),
                "新增类别": item.get("new_class"),
                "New-mAP50": item.get("new_map50"),
                "KRR": item.get("krr"),
                "数据合规": "是" if item.get("compliant") else "否",
                "通过": "是" if item.get("passed") else "否",
            }
            for item in incremental.get("protocols", [])
        ],
        width="stretch",
        hide_index=True,
    )
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
