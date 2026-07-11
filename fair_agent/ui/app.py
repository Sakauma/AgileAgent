from __future__ import annotations

import json
import gc
import html
from pathlib import Path
from typing import Any, Dict

import streamlit as st
from PIL import Image

from fair_agent.modules.web_inference import (
    SCENE_LABELS,
    SENSOR_LABELS,
    MAX_BATCH_FILES,
    WebInferenceEngine,
    build_batch_zip,
    result_json_bytes,
    validate_batch_uploads,
    validate_image_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
DETECTOR_PATH = ROOT / "models" / "base" / "yolo11s_ir_sar_imgsz640.pt"
CONTEXT_PATH = ROOT / "models" / "context" / "scene_sensor_net.pt"


@st.cache_resource(show_spinner=False)
def inference_engine() -> WebInferenceEngine:
    return WebInferenceEngine(DETECTOR_PATH, CONTEXT_PATH, device_index="0")


SESSION_RESULT_KEYS = [
    "single_result",
    "single_original_bytes",
    "batch_results",
    "batch_signature",
    "task_history",
    "single_upload",
    "batch_uploads",
]


def read_uploaded(uploaded: Any) -> tuple[Image.Image, str, bytes]:
    data = uploaded.getvalue()
    image, task_id = validate_image_bytes(data, uploaded.name)
    return image, task_id, data


def public_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in result.items() if key != "annotated_png"}


def add_history(result: Dict[str, Any]) -> None:
    history = st.session_state.setdefault("task_history", [])
    history[:] = [item for item in history if item.get("task_id") != result.get("task_id")]
    history.insert(0, public_payload(result))
    del history[20:]


def clear_session_results() -> None:
    for key in SESSION_RESULT_KEYS:
        st.session_state.pop(key, None)
    gc.collect()


def workflow(stage: int) -> str:
    labels = [(1, "上传图像"), (2, "智能分析"), (3, "结果导出")]
    nodes = []
    for number, label in labels:
        state = "done" if number < stage else ("active" if number == stage else "pending")
        marker = "✓" if state == "done" else str(number)
        nodes.append(
            f'<div class="workflow-step {state}"><span>{marker}</span><strong>{label}</strong></div>'
        )
    return '<div class="workflow">' + '<div class="workflow-line"></div>'.join(nodes) + "</div>"


def render_context(result: Dict[str, Any]) -> None:
    context = result["context"]
    metrics = st.columns(4)
    metrics[0].metric(
        "传感器",
        SENSOR_LABELS.get(context.get("sensor"), context.get("sensor")),
        f"置信度 {float(context.get('sensor_confidence') or 0):.0%}",
        delta_color="off",
    )
    metrics[1].metric(
        "场景",
        SCENE_LABELS.get(context.get("scene"), context.get("scene")),
        f"置信度 {float(context.get('scene_confidence') or 0):.0%}",
        delta_color="off",
    )
    metrics[2].metric("检测目标", result["detection_count"])
    metrics[3].metric("处理耗时", f"{result['elapsed_ms']:.1f} ms")


def render_result(result: Dict[str, Any], original_bytes: bytes) -> None:
    st.markdown(
        f'<div class="result-banner"><div><span class="result-check">✓</span><strong>检测完成</strong></div>'
        f'<span>任务 {str(result.get("task_id"))[:12]} · 排队 {float(result.get("queue_wait_ms") or 0):.1f} ms</span></div>',
        unsafe_allow_html=True,
    )
    render_context(result)
    if result.get("class_counts"):
        chips = "".join(
            f'<span class="count-chip"><strong>{name}</strong><b>{count}</b></span>'
            for name, count in result["class_counts"].items()
        )
        st.markdown(f'<div class="count-strip">{chips}</div>', unsafe_allow_html=True)
    original_col, result_col = st.columns(2, gap="large")
    with original_col:
        st.markdown('<div class="section-label">原始图像</div>', unsafe_allow_html=True)
        st.image(original_bytes, width="stretch")
    with result_col:
        st.markdown('<div class="section-label">检测结果</div>', unsafe_allow_html=True)
        st.image(result["annotated_png"], width="stretch")
    footer_left, footer_right = st.columns([1.2, 1], gap="large")
    with footer_left:
        st.markdown('<div class="section-label">目标明细</div>', unsafe_allow_html=True)
        records = result["detections"]
        if records:
            st.dataframe(
                [
                    {
                        "类别": item["class_name"],
                        "置信度": item["confidence"],
                        "边界框": item["xyxy"],
                    }
                    for item in records
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("未检测到目标。")
    with footer_right:
        st.markdown('<div class="section-label">结果导出</div>', unsafe_allow_html=True)
        stem = Path(result["filename"]).stem
        st.download_button(
            "下载标注图",
            result["annotated_png"],
            file_name=f"{stem}_detected.png",
            mime="image/png",
            icon=":material/image:",
            width="stretch",
        )
        st.download_button(
            "下载 JSON",
            result_json_bytes(public_payload(result)),
            file_name=f"{stem}_result.json",
            mime="application/json",
            icon=":material/data_object:",
            width="stretch",
        )
        st.button(
            "清除本次结果",
            icon=":material/delete_outline:",
            on_click=clear_session_results,
            width="stretch",
        )


st.set_page_config(page_title="AgileAgent 智能检测", page_icon=None, layout="wide")
st.markdown(
    """
    <style>
    :root {--ink:#18242D; --muted:#667783; --line:#DCE4E9; --teal:#087F8C;}
    .stApp {background:#F4F7F9; color:var(--ink);}
    [data-testid="stHeader"] {background:transparent;}
    [data-testid="stToolbar"] {display:flex !important; background:transparent !important;}
    [data-testid="stToolbar"] button:not([data-testid="stExpandSidebarButton"]) {display:none !important;}
    body:has([data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stExpandSidebarButton"] {
      display:inline-flex !important; width:38px !important; height:38px !important;
      position:fixed !important; top:12px !important; left:12px !important; z-index:100000 !important;
      align-items:center !important; justify-content:center !important;
      color:#FFFFFF !important; background:#101A22 !important; border:1px solid #30434F !important;
      border-radius:5px !important; box-shadow:0 2px 8px rgba(16,26,34,.18) !important;
    }
    body:has([data-testid="stSidebar"][aria-expanded="true"]) [data-testid="stExpandSidebarButton"] {display:none !important;}
    footer {display:none !important;}
    .block-container {padding-top:2.2rem; padding-bottom:3rem; max-width:1380px;}
    [data-testid="stSidebar"] {background:#101A22; border-right:0; min-width:250px;}
    [data-testid="stSidebar"] * {color:#D8E1E7;}
    [data-testid="stSidebar"] [role="radiogroup"] label {padding:.65rem .75rem; border-radius:5px; margin:2px 0;}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#173943;}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color:#FFFFFF; font-weight:650;}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; color:#8FA1AD;}
    [data-testid="stMetric"] {background:#FFFFFF; border:1px solid var(--line); border-top:3px solid #8EB1B6; border-radius:6px; padding:13px 15px; min-height:105px;}
    [data-testid="stMetricLabel"] {color:#60717C !important; font-size:.75rem; text-transform:uppercase; letter-spacing:.035em;}
    [data-testid="stMetricValue"] {color:var(--ink) !important; font-size:1.65rem;}
    [data-testid="stFileUploader"] {background:#FFFFFF; border:1px solid var(--line); border-radius:6px; padding:12px;}
    [data-testid="stImage"] img {border:1px solid var(--line);}
    [data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:6px; overflow:hidden;}
    .stButton button, .stDownloadButton button {border-radius:5px; font-weight:650; min-height:42px;}
    h1, h2, h3 {letter-spacing:0; color:var(--ink);}
    .brand-mark {font-size:1.12rem; font-weight:750; color:#FFFFFF; letter-spacing:.03em; margin:.2rem 0 0;}
    .brand-sub {font-size:.72rem; color:#7FD0D2; text-transform:uppercase; letter-spacing:.12em; margin-bottom:1.6rem;}
    .runtime-chip {display:inline-flex; align-items:center; gap:7px; font-size:.72rem; color:#AFC0CA; margin:1.2rem 0 .5rem;}
    .runtime-chip:before {content:""; width:7px; height:7px; border-radius:50%; background:#36B37E; box-shadow:0 0 0 3px rgba(54,179,126,.14);}
    .page-kicker {color:var(--teal); font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; margin-bottom:.35rem;}
    .page-title {font-size:2rem; font-weight:750; line-height:1.15; color:var(--ink); margin-bottom:.35rem;}
    .page-subtitle {color:#71828D; font-size:.88rem; margin-bottom:1.6rem;}
    .section-label {font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:#71828D; margin:1.2rem 0 .6rem;}
    .class-strip {display:flex; flex-wrap:wrap; gap:8px; margin:1rem 0 1.5rem;}
    .class-chip {background:#E8F4F4; color:#23646B; border:1px solid #CDE3E4; border-radius:14px; padding:4px 10px; font-size:.72rem; font-weight:650;}
    .empty-state {background:#FFFFFF; border:1px solid var(--line); border-left:4px solid var(--teal); padding:18px 20px; color:#52636E; margin-top:1rem;}
    .workflow {display:grid; grid-template-columns:1fr 44px 1fr 44px 1fr; align-items:center; margin:0 0 1.2rem;}
    .workflow-step {display:flex; align-items:center; gap:9px; color:#82919A; font-size:.78rem;}
    .workflow-step span {display:inline-flex; align-items:center; justify-content:center; width:27px; height:27px; border-radius:50%; border:1px solid #CBD6DC; background:#FFFFFF; font-size:.7rem; font-weight:700;}
    .workflow-step.active {color:#1F5960;}.workflow-step.active span {background:#087F8C;color:#FFFFFF;border-color:#087F8C;box-shadow:0 0 0 4px rgba(8,127,140,.10);}
    .workflow-step.done {color:#28705C;}.workflow-step.done span {background:#E8F7F1;color:#176648;border-color:#BBDDCF;}
    .workflow-line {height:1px;background:#CBD6DC;}
    .upload-summary {display:grid;grid-template-columns:minmax(180px,320px) 1fr;gap:20px;align-items:center;background:#FFFFFF;border:1px solid var(--line);padding:16px;margin:.8rem 0;}
    .file-name {font-weight:700;color:var(--ink);word-break:break-all;}.file-meta {font-size:.76rem;color:#71828D;margin-top:6px;line-height:1.65;}
    .result-banner {display:flex;align-items:center;justify-content:space-between;gap:16px;background:#EAF7F2;border:1px solid #C5E5D8;color:#225D4A;padding:12px 14px;margin:1.2rem 0; font-size:.78rem;}
    .result-banner>div {display:flex;align-items:center;gap:8px;font-size:.9rem;}.result-check {display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;background:#167A5A;color:#FFFFFF;font-weight:800;}
    .count-strip {display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 2px;}.count-chip {display:inline-flex;align-items:center;gap:10px;background:#FFFFFF;border:1px solid var(--line);padding:6px 9px;font-size:.74rem;}.count-chip b {display:inline-flex;align-items:center;justify-content:center;min-width:22px;height:20px;background:#E8F4F4;color:#17606A;}
    .batch-summary {display:flex;align-items:center;justify-content:space-between;background:#FFFFFF;border:1px solid var(--line);padding:12px 14px;margin:.8rem 0;font-size:.8rem;}
    @media (max-width:900px) {
      .block-container{padding-top:4rem}.page-title{font-size:1.7rem}
      .workflow{grid-template-columns:1fr;gap:8px}.workflow-line{display:none}.upload-summary{grid-template-columns:1fr}
      .result-banner{display:block}.result-banner>span{display:block;margin-top:7px}.result-banner{word-break:break-word}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="brand-mark">AGILE AGENT</div><div class="brand-sub">IR / SAR Detection</div>', unsafe_allow_html=True)
page = st.sidebar.radio("工作区", ["智能检测", "批量检测", "任务记录"])
st.sidebar.markdown('<div class="runtime-chip">GPU推理队列可用</div>', unsafe_allow_html=True)
st.sidebar.caption("数据仅用于当前会话处理")
st.sidebar.button(
    "清空当前会话",
    icon=":material/delete_sweep:",
    on_click=clear_session_results,
    width="stretch",
)

st.markdown('<div class="page-kicker">IR / SAR 智能目标检测</div>', unsafe_allow_html=True)
st.markdown(f'<div class="page-title">{page}</div>', unsafe_allow_html=True)

if page == "智能检测":
    st.markdown('<div class="page-subtitle">上传一张红外或 SAR 图像，系统自动识别场景并完成目标检测。</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="class-strip"><span class="class-chip">soldier</span><span class="class-chip">small aircraft</span>'
        '<span class="class-chip">warship</span><span class="class-chip">tank</span></div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "选择图像",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="single_upload",
    )
    validated_single = None
    validation_error = None
    if uploaded:
        try:
            validated_single = read_uploaded(uploaded)
        except ValueError as exc:
            validation_error = str(exc)
    current_result = st.session_state.get("single_result")
    current_task_id = validated_single[1] if validated_single else None
    current_complete = bool(current_result and current_result.get("task_id") == current_task_id)
    st.markdown(workflow(3 if current_complete else (2 if validated_single else 1)), unsafe_allow_html=True)
    with st.expander("检测设置"):
        confidence = st.slider("最低置信度", min_value=0.01, max_value=0.80, value=0.15, step=0.01)
    if validation_error:
        st.error(validation_error)
    if validated_single:
        original, task_id, original_bytes = validated_single
        safe_filename = html.escape(uploaded.name)
        preview_col, action_col = st.columns([0.72, 1.28], gap="large")
        with preview_col:
            st.image(original_bytes, width="stretch")
        with action_col:
            st.markdown(
                f'<div class="upload-summary"><div><div class="file-name">{safe_filename}</div>'
                f'<div class="file-meta">{original.width} × {original.height} 像素<br>'
                f'{len(original_bytes) / (1024 * 1024):.2f} MB · 任务 {task_id[:12]}</div></div>'
                f'<div class="file-meta">内容哈希已确认<br>等待进入GPU推理队列</div></div>',
                unsafe_allow_html=True,
            )
        if st.button("开始检测", type="primary", icon=":material/search:", width="stretch"):
            try:
                with st.spinner("正在排队并分析图像..."):
                    result = inference_engine().predict(
                        original,
                        uploaded.name,
                        confidence,
                        task_id=task_id,
                    )
                st.session_state["single_result"] = result
                st.session_state["single_original_bytes"] = original_bytes
                add_history(result)
            except (RuntimeError, ValueError, OSError) as exc:
                st.error(f"检测失败：{exc}")
        result = st.session_state.get("single_result")
        if result and result.get("task_id") == task_id:
            render_result(result, st.session_state["single_original_bytes"])
    else:
        if not validation_error:
            st.markdown('<div class="empty-state">等待图像输入。单文件最大20MB，支持 PNG、JPEG、BMP 和 TIFF。</div>', unsafe_allow_html=True)

elif page == "批量检测":
    st.markdown(f'<div class="page-subtitle">一次处理多张图像并导出标注图与结构化结果，单批最多 {MAX_BATCH_FILES} 张。</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "选择图像",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="batch_uploads",
    )
    validated_batch = []
    batch_error = None
    batch_signature = None
    if uploads:
        try:
            validated_batch = validate_batch_uploads([(item.name, item.getvalue()) for item in uploads])
            batch_signature = ":".join(item[3][:12] for item in validated_batch)
        except ValueError as exc:
            batch_error = str(exc)
    batch_complete = bool(
        batch_signature and st.session_state.get("batch_signature") == batch_signature
    )
    st.markdown(workflow(3 if batch_complete else (2 if validated_batch else 1)), unsafe_allow_html=True)
    with st.expander("检测设置"):
        batch_confidence = st.slider("最低置信度", min_value=0.01, max_value=0.80, value=0.15, step=0.01, key="batch_confidence")
    if batch_error:
        st.error(batch_error)
    if validated_batch:
        total_mb = sum(len(item[1]) for item in validated_batch) / (1024 * 1024)
        st.markdown(
            f'<div class="batch-summary"><strong>已验证 {len(validated_batch)} 张图像</strong>'
            f'<span>合计 {total_mb:.2f} MB · 无重复内容</span></div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            [
                {
                    "文件": filename,
                    "尺寸": f"{image.width} × {image.height}",
                    "大小(MB)": round(len(data) / (1024 * 1024), 2),
                    "任务ID": task_id[:12],
                }
                for filename, data, image, task_id in validated_batch
            ],
            width="stretch",
            hide_index=True,
        )
    if validated_batch and st.button("开始批量检测", type="primary", icon=":material/batch_prediction:", width="stretch"):
        results = []
        progress = st.progress(0, text="准备检测")
        try:
            engine = inference_engine()
            for index, (filename, _data, image, task_id) in enumerate(validated_batch, 1):
                progress.progress((index - 1) / len(validated_batch), text=f"正在处理 {filename}")
                result = engine.predict(image, filename, batch_confidence, task_id=task_id)
                results.append(result)
                add_history(result)
            progress.progress(1.0, text="检测完成")
            st.session_state["batch_results"] = results
            st.session_state["batch_signature"] = batch_signature
        except (RuntimeError, ValueError, OSError) as exc:
            st.error(f"批量检测失败：{exc}")
    batch_results = (
        st.session_state.get("batch_results", [])
        if batch_signature and st.session_state.get("batch_signature") == batch_signature
        else []
    )
    if batch_results:
        st.markdown(
            f'<div class="result-banner"><div><span class="result-check">✓</span><strong>批量检测完成</strong></div>'
            f'<span>{len(batch_results)} 张图像 · 可下载完整结果包</span></div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            [
                {
                    "文件": item["filename"],
                    "传感器": SENSOR_LABELS.get(item["context"].get("sensor"), item["context"].get("sensor")),
                    "场景": SCENE_LABELS.get(item["context"].get("scene"), item["context"].get("scene")),
                    "目标数": item["detection_count"],
                    "耗时(ms)": item["elapsed_ms"],
                }
                for item in batch_results
            ],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "下载全部结果",
            build_batch_zip(batch_results),
            file_name="agile_agent_results.zip",
            mime="application/zip",
            icon=":material/download:",
            width="stretch",
        )
        st.button(
            "清除批量结果",
            icon=":material/delete_outline:",
            on_click=clear_session_results,
            width="stretch",
        )

else:
    st.markdown('<div class="page-subtitle">当前浏览器会话中的检测任务。</div>', unsafe_allow_html=True)
    history = st.session_state.get("task_history", [])
    if history:
        summary_cols = st.columns(3)
        summary_cols[0].metric("会话任务", len(history))
        summary_cols[1].metric("检测图像", len(history))
        summary_cols[2].metric("累计目标", sum(int(item.get("detection_count") or 0) for item in history))
        st.dataframe(
            [
                {
                    "文件": item["filename"],
                    "传感器": SENSOR_LABELS.get(item["context"].get("sensor"), item["context"].get("sensor")),
                    "场景": SCENE_LABELS.get(item["context"].get("scene"), item["context"].get("scene")),
                    "目标数": item["detection_count"],
                    "类别统计": json.dumps(item["class_counts"], ensure_ascii=False),
                    "耗时(ms)": item["elapsed_ms"],
                }
                for item in history
            ],
            width="stretch",
            hide_index=True,
        )
        st.button(
            "清空任务记录",
            icon=":material/delete_sweep:",
            on_click=clear_session_results,
            width="stretch",
        )
    else:
        st.markdown('<div class="empty-state">当前会话还没有检测任务。</div>', unsafe_allow_html=True)
