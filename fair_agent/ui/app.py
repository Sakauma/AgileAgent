from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st
from PIL import Image

from fair_agent.modules.web_inference import (
    SCENE_LABELS,
    SENSOR_LABELS,
    WebInferenceEngine,
    build_batch_zip,
    result_json_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
DETECTOR_PATH = ROOT / "models" / "base" / "yolo11s_ir_sar_imgsz640.pt"
CONTEXT_PATH = ROOT / "models" / "context" / "scene_sensor_net.pt"


@st.cache_resource(show_spinner=False)
def inference_engine() -> WebInferenceEngine:
    return WebInferenceEngine(DETECTOR_PATH, CONTEXT_PATH, device_index="0")


def open_uploaded(uploaded: Any) -> Image.Image:
    return Image.open(BytesIO(uploaded.getvalue())).convert("RGB")


def public_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"annotated_image", "annotated_png"}}


def render_context(result: Dict[str, Any]) -> None:
    context = result["context"]
    metrics = st.columns(4)
    metrics[0].metric("传感器", SENSOR_LABELS.get(context.get("sensor"), context.get("sensor")))
    metrics[1].metric("场景", SCENE_LABELS.get(context.get("scene"), context.get("scene")))
    metrics[2].metric("检测目标", result["detection_count"])
    metrics[3].metric("处理耗时", f"{result['elapsed_ms']:.1f} ms")


def render_result(result: Dict[str, Any], original: Image.Image) -> None:
    render_context(result)
    original_col, result_col = st.columns(2, gap="large")
    with original_col:
        st.markdown('<div class="section-label">原始图像</div>', unsafe_allow_html=True)
        st.image(original, width="stretch")
    with result_col:
        st.markdown('<div class="section-label">检测结果</div>', unsafe_allow_html=True)
        st.image(result["annotated_image"], width="stretch")
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
    @media (max-width:900px) {.block-container{padding-top:1.2rem}.page-title{font-size:1.7rem}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="brand-mark">AGILE AGENT</div><div class="brand-sub">IR / SAR Detection</div>', unsafe_allow_html=True)
page = st.sidebar.radio("工作区", ["智能检测", "批量检测", "任务记录"])
st.sidebar.markdown('<div class="runtime-chip">检测服务已就绪</div>', unsafe_allow_html=True)
st.sidebar.caption("数据仅用于当前会话处理")

st.markdown('<div class="page-kicker">IR / SAR 智能目标检测</div>', unsafe_allow_html=True)
st.markdown(f'<div class="page-title">{page}</div>', unsafe_allow_html=True)

if page == "智能检测":
    st.markdown('<div class="page-subtitle">上传一张红外或 SAR 图像，系统自动识别场景并完成目标检测。</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="class-strip"><span class="class-chip">soldier</span><span class="class-chip">small aircraft</span>'
        '<span class="class-chip">warship</span><span class="class-chip">tank</span></div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("选择图像", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    with st.expander("检测设置"):
        confidence = st.slider("最低置信度", min_value=0.01, max_value=0.80, value=0.15, step=0.01)
    if uploaded:
        original = open_uploaded(uploaded)
        if st.button("开始检测", type="primary", icon=":material/search:", width="stretch"):
            try:
                with st.spinner("正在分析图像..."):
                    result = inference_engine().predict(original, uploaded.name, confidence)
                st.session_state["single_result"] = result
                st.session_state["single_original"] = original
                history = st.session_state.setdefault("task_history", [])
                history.insert(0, public_payload(result))
            except (RuntimeError, ValueError, OSError) as exc:
                st.error(f"检测失败：{exc}")
        result = st.session_state.get("single_result")
        if result and result.get("filename") == uploaded.name:
            render_result(result, st.session_state["single_original"])
    else:
        st.markdown('<div class="empty-state">等待图像输入。支持 PNG、JPEG、BMP 和 TIFF。</div>', unsafe_allow_html=True)

elif page == "批量检测":
    st.markdown('<div class="page-subtitle">一次处理多张图像并导出标注图与结构化结果。</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "选择图像",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )
    with st.expander("检测设置"):
        batch_confidence = st.slider("最低置信度", min_value=0.01, max_value=0.80, value=0.15, step=0.01, key="batch_confidence")
    if uploads and st.button("开始批量检测", type="primary", icon=":material/batch_prediction:", width="stretch"):
        results = []
        progress = st.progress(0, text="准备检测")
        try:
            engine = inference_engine()
            for index, uploaded in enumerate(uploads, 1):
                progress.progress((index - 1) / len(uploads), text=f"正在处理 {uploaded.name}")
                results.append(engine.predict(open_uploaded(uploaded), uploaded.name, batch_confidence))
            progress.progress(1.0, text="检测完成")
            st.session_state["batch_results"] = results
            history = st.session_state.setdefault("task_history", [])
            history[0:0] = [public_payload(item) for item in results]
        except (RuntimeError, ValueError, OSError) as exc:
            st.error(f"批量检测失败：{exc}")
    batch_results = st.session_state.get("batch_results", [])
    if batch_results:
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

else:
    st.markdown('<div class="page-subtitle">当前浏览器会话中的检测任务。</div>', unsafe_allow_html=True)
    history = st.session_state.get("task_history", [])
    if history:
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
    else:
        st.markdown('<div class="empty-state">当前会话还没有检测任务。</div>', unsafe_allow_html=True)
