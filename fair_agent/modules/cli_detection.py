from __future__ import annotations

import csv
import http.client
import io
import json
import mimetypes
import os
import re
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw

from fair_agent.core.audit import make_run_dir
from fair_agent.core.config import resolve_path
from fair_agent.ui.terminal import panel, table


DEFAULT_RESULT_ROOT = "runs/cli_detections"
DEFAULT_BATCH_SIZE = 20
DEFAULT_ANNOTATION_WORKERS = 3
DEFAULT_ANNOTATION_FORMAT = "jpeg"
SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
CLASS_LABELS = {
    "soldier": "人员",
    "small_aircraft": "小型飞行器",
    "warship": "舰船",
    "tank": "坦克",
    "patrol_boat": "巡逻艇",
    "armored_vehicle": "装甲车辆",
}
SENSOR_LABELS = {"ir": "红外", "sar": "SAR"}
SCENE_LABELS = {
    "air": "空域",
    "forest": "林地",
    "sea": "海域",
    "urban": "城市场景",
}
BOX_COLORS = {
    0: "#18a999",
    1: "#3b82f6",
    2: "#f59e0b",
    3: "#8b5cf6",
    4: "#ef476f",
    5: "#22c55e",
}


@dataclass(frozen=True)
class DetectionInput:
    path: Path
    name: str


def normalize_user_path_text(
    value: str | Path,
    *,
    host_os: str | None = None,
) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    if not text:
        raise ValueError("输入路径不能为空")
    if (host_os or os.name) == "nt":
        return text
    wsl_unc = re.match(
        r"^[\\/]+(?:wsl\.localhost|wsl\$)[\\/]+[^\\/]+(?P<path>[\\/].*)?$",
        text,
        flags=re.IGNORECASE,
    )
    if wsl_unc:
        suffix = str(wsl_unc.group("path") or "/").replace("\\", "/")
        return "/" + suffix.lstrip("/")
    windows_drive = re.match(
        r"^(?P<drive>[A-Za-z]):[\\/](?P<path>.*)$",
        text,
    )
    if windows_drive:
        drive = str(windows_drive.group("drive")).lower()
        suffix = str(windows_drive.group("path")).replace("\\", "/")
        return f"/mnt/{drive}/{suffix.lstrip('/')}"
    return text.replace("\\", "/")


def resolve_user_path(value: str | Path) -> Path:
    path = Path(normalize_user_path_text(value)).expanduser()
    return (path if path.is_absolute() else resolve_path(path)).resolve()


def discover_detection_inputs(
    source: str | Path,
    *,
    recursive: bool = False,
) -> tuple[Path, list[DetectionInput]]:
    source_path = resolve_user_path(source)
    if not source_path.exists():
        raise ValueError(f"输入路径不存在：{source_path}")
    if source_path.is_file():
        return source_path, [DetectionInput(source_path, source_path.name)]
    if not source_path.is_dir():
        raise ValueError(f"输入不是普通文件或目录：{source_path}")
    iterator = source_path.rglob("*") if recursive else source_path.glob("*")
    paths = sorted(
        (
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda path: path.as_posix().lower(),
    )
    if not paths:
        scope = "及其子目录" if recursive else ""
        raise ValueError(f"目录{scope}中没有可识别图像：{source_path}")
    return source_path, [
        DetectionInput(path, path.relative_to(source_path).as_posix())
        for path in paths
    ]


def safe_stem(value: str) -> str:
    stem = Path(value).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "image"


def create_result_dir(output: str | Path | None, source: Path) -> Path:
    if output is None:
        prefix = f"detect_{safe_stem(source.name)}"
        return make_run_dir(prefix=prefix, run_root=DEFAULT_RESULT_ROOT)
    target = resolve_user_path(output)
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"结果路径不是目录：{target}")
        if any(target.iterdir()):
            raise ValueError(f"结果目录必须为空，拒绝覆盖：{target}")
        return target
    target.mkdir(parents=True, exist_ok=False)
    return target


def _service_host(config: Mapping[str, Any]) -> str:
    host = str(config.get("runtime", {}).get("server_host") or "127.0.0.1")
    if host in {"0.0.0.0", "::", "::0"}:
        return "127.0.0.1"
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def local_api_base_url(config: Mapping[str, Any]) -> str:
    host = _service_host(config)
    port = int(config.get("runtime", {}).get("server_port") or 8501)
    return f"http://{host}:{port}"


def probe_local_detection_api(
    config: Mapping[str, Any],
    *,
    timeout: float = 0.8,
) -> str | None:
    base_url = local_api_base_url(config)
    request = Request(
        f"{base_url}/api/health",
        headers={"Accept": "application/json", "User-Agent": "AgileAgent-CLI/1"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError):
        return None
    expected_backend = str(config.get("inference", {}).get("backend") or "")
    if (
        payload.get("status") != "ready"
        or str(payload.get("backend") or "") != expected_backend
        or payload.get("validated") is not True
    ):
        return None
    return base_url


def _multipart_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower() or ".bin"
    return safe_stem(filename) + suffix


def _multipart_body(
    rows: Sequence[tuple[str, bytes]],
    *,
    field_name: str,
    boundary: str,
) -> bytes:
    if not rows:
        raise ValueError("检测请求至少需要一张图像")
    chunks: list[bytes] = []
    for filename, data in rows:
        ascii_name = _multipart_filename(filename)
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                (
                    f'Content-Disposition: form-data; name="{field_name}"; '
                    f'filename="{ascii_name}"\r\n'
                ).encode("ascii"),
                f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
                data,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks)


class LocalDetectionApiClient:
    """Small keep-alive client used by the primary CLI inference path."""

    def __init__(self, base_url: str, timeout: float) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("本机推理服务必须使用明确的 http:// 地址")
        self._prefix = parsed.path.rstrip("/")
        self._connection = http.client.HTTPConnection(
            parsed.hostname,
            parsed.port or 80,
            timeout=timeout,
        )

    def __enter__(self) -> "LocalDetectionApiClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _post_multipart(
        self,
        path: str,
        rows: Sequence[tuple[str, bytes]],
        *,
        field_name: str,
    ) -> tuple[Dict[str, Any], float]:
        boundary = f"----AgileAgentCLI{uuid.uuid4().hex}"
        body = _multipart_body(
            rows,
            field_name=field_name,
            boundary=boundary,
        )
        started = time.perf_counter()
        try:
            self._connection.request(
                "POST",
                f"{self._prefix}{path}",
                body=body,
                headers={
                    "Accept": "application/json",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Content-Length": str(len(body)),
                    "Connection": "keep-alive",
                    "User-Agent": "AgileAgent-CLI/1",
                    "X-Agile-Agent-Execution-Profile": "speculative-low-latency",
                },
            )
            response = self._connection.getresponse()
            response_body = response.read()
        except (http.client.HTTPException, OSError, TimeoutError) as exc:
            raise RuntimeError(f"本机推理服务连接中断：{exc}") from exc
        wall_ms = (time.perf_counter() - started) * 1000
        try:
            payload = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("本机推理服务返回了无效 JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("本机推理服务返回的 JSON 不是对象")
        if not 200 <= response.status < 300:
            detail = str(payload.get("error") or response.reason or "请求失败")
            raise RuntimeError(f"本机推理服务返回 HTTP {response.status}：{detail}")
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return payload, wall_ms

    def detect(self, data: bytes, filename: str) -> tuple[Dict[str, Any], float]:
        payload, wall_ms = self._post_multipart(
            "/api/detect",
            [(filename, data)],
            field_name="file",
        )
        payload["filename"] = filename
        return payload, wall_ms

    def detect_batch(
        self,
        rows: Sequence[tuple[str, bytes]],
    ) -> tuple[Dict[str, Any], float]:
        payload, wall_ms = self._post_multipart(
            "/api/batch",
            rows,
            field_name="files",
        )
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != len(rows):
            raise RuntimeError("批量推理服务返回的结果数量与输入不一致")
        for result, (filename, _data) in zip(results, rows):
            if not isinstance(result, dict):
                raise RuntimeError("批量推理服务返回了无效的逐图结果")
            result["filename"] = filename
        return payload, wall_ms


def detect_via_local_api(
    base_url: str,
    data: bytes,
    filename: str,
    *,
    timeout: float,
) -> Dict[str, Any]:
    with LocalDetectionApiClient(base_url, timeout) as client:
        payload, _wall_ms = client.detect(data, filename)
    return payload


def public_detection_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"annotated_png", "annotated_image", "source_bytes"}
    }


def display_class_name(value: Any) -> str:
    name = str(value)
    label = CLASS_LABELS.get(name)
    return f"{label} ({name})" if label else name


def _annotated_image(
    data: bytes,
    detections: Iterable[Mapping[str, Any]],
    *,
    image_format: str,
) -> bytes:
    with Image.open(io.BytesIO(data)) as source:
        source.load()
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(min(canvas.size) / 320))
    for item in detections:
        class_id = int(item.get("class_id") or 0)
        color = BOX_COLORS.get(class_id, "#18a999")
        x1, y1, x2, y2 = [float(value) for value in item["xyxy"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f"{item.get('class_name', class_id)} {float(item.get('confidence') or 0):.1%}"
        text_box = draw.textbbox((0, 0), label)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0.0, y1 - text_height - 5)
        draw.rectangle(
            (x1, label_y, x1 + text_width + 6, label_y + text_height + 4),
            fill=color,
        )
        draw.text((x1 + 3, label_y + 2), label, fill="white")
    buffer = io.BytesIO()
    if image_format == "jpeg":
        canvas.save(
            buffer,
            format="JPEG",
            quality=90,
            subsampling=0,
            optimize=False,
        )
    elif image_format == "png":
        canvas.save(buffer, format="PNG", compress_level=1, optimize=False)
    else:
        raise ValueError(f"不支持的标注图格式：{image_format}")
    return buffer.getvalue()


def _prediction_lines(result: Mapping[str, Any]) -> list[str]:
    width = max(1.0, float(result.get("image_width") or 1))
    height = max(1.0, float(result.get("image_height") or 1))
    lines = []
    for item in result.get("detections", []):
        x1, y1, x2, y2 = [float(value) for value in item["xyxy"]]
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(
            f"{int(item['class_id'])} {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f} {float(item.get('confidence') or 0):.6f}"
        )
    return lines


def _owner_for(result: Mapping[str, Any], class_id: int) -> str:
    owners = (
        result.get("agent", {})
        .get("decision", {})
        .get("class_owners", {})
    )
    return str(owners.get(str(class_id)) or owners.get(class_id) or "")


def build_detection_statistics(
    results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    image_count = len(results)
    detection_counts = [
        int(result.get("detection_count") or 0)
        for result in results
    ]
    images_with_detections = sum(count > 0 for count in detection_counts)
    detection_count = sum(detection_counts)
    processing_times = [
        max(
            0.0,
            float(
                result.get("system_total_ms")
                or result.get("inference_ms")
                or 0.0
            ),
        )
        for result in results
    ]
    confidences = [
        float(detection.get("confidence") or 0.0)
        for result in results
        for detection in result.get("detections", [])
    ]
    sensor_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    for result in results:
        context = result.get("context", {})
        sensor_counts.update([str(context.get("sensor") or "unknown")])
        scene_counts.update([str(context.get("scene") or "unknown")])
    total_processing_ms = sum(processing_times)
    return {
        "images_with_detections": images_with_detections,
        "images_without_detections": image_count - images_with_detections,
        "image_detection_rate": round(
            images_with_detections / image_count if image_count else 0.0,
            6,
        ),
        "detections_per_image": round(
            detection_count / image_count if image_count else 0.0,
            6,
        ),
        "total_processing_ms": round(total_processing_ms, 3),
        "average_processing_ms": round(
            total_processing_ms / image_count if image_count else 0.0,
            3,
        ),
        "estimated_throughput_fps": (
            round(image_count * 1000.0 / total_processing_ms, 3)
            if total_processing_ms > 0.0
            else None
        ),
        "average_confidence": (
            round(sum(confidences) / len(confidences), 6)
            if confidences
            else None
        ),
        "minimum_confidence": round(min(confidences), 6) if confidences else None,
        "maximum_confidence": round(max(confidences), 6) if confidences else None,
        "sensor_counts": dict(sorted(sensor_counts.items())),
        "scene_counts": dict(sorted(scene_counts.items())),
    }


@dataclass(frozen=True)
class _SavedDetectionArtifact:
    result: Dict[str, Any]
    csv_rows: list[Dict[str, Any]]
    worker_ms: float


class DetectionArtifactWriter:
    """Materialize CLI artifacts while later inference batches are still running."""

    def __init__(
        self,
        run_dir: Path,
        *,
        annotation_format: str = DEFAULT_ANNOTATION_FORMAT,
        workers: int = DEFAULT_ANNOTATION_WORKERS,
    ) -> None:
        normalized_format = str(annotation_format).strip().lower()
        if normalized_format not in {"jpeg", "png"}:
            raise ValueError(f"不支持的标注图格式：{annotation_format}")
        self.run_dir = run_dir
        self.annotation_format = normalized_format
        self.annotation_suffix = ".jpg" if normalized_format == "jpeg" else ".png"
        self.workers = max(1, int(workers))
        self.annotated_dir = run_dir / "annotated"
        self.prediction_dir = run_dir / "predictions"
        self.annotated_dir.mkdir(parents=True, exist_ok=False)
        self.prediction_dir.mkdir(parents=True, exist_ok=False)
        self._executor = ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="agile-cli-save",
        )
        self._futures: list[Future[_SavedDetectionArtifact]] = []
        self._pipeline_started: float | None = None
        self._closed = False

    def submit(
        self,
        index: int,
        item: DetectionInput,
        data: bytes,
        raw_result: Mapping[str, Any],
    ) -> None:
        if self._closed:
            raise RuntimeError("结果写入器已经关闭")
        if self._pipeline_started is None:
            self._pipeline_started = time.perf_counter()
        self._futures.append(
            self._executor.submit(
                self._write_one,
                index,
                item,
                data,
                raw_result,
            )
        )

    def _write_one(
        self,
        index: int,
        item: DetectionInput,
        data: bytes,
        raw_result: Mapping[str, Any],
    ) -> _SavedDetectionArtifact:
        started = time.perf_counter()
        result = public_detection_result(raw_result)
        result["filename"] = item.name
        result["source_path"] = str(item.path)
        stem = f"{index:03d}_{safe_stem(item.name)}"
        annotated_relative = Path("annotated") / f"{stem}{self.annotation_suffix}"
        prediction_relative = Path("predictions") / f"{stem}.txt"
        (self.run_dir / annotated_relative).write_bytes(
            _annotated_image(
                data,
                result.get("detections", []),
                image_format=self.annotation_format,
            )
        )
        prediction_lines = _prediction_lines(result)
        (self.run_dir / prediction_relative).write_text(
            "\n".join(prediction_lines) + ("\n" if prediction_lines else ""),
            encoding="utf-8",
        )
        result["artifacts"] = {
            "annotated_image": annotated_relative.as_posix(),
            "prediction_txt": prediction_relative.as_posix(),
        }
        csv_rows: list[Dict[str, Any]] = []
        for detection_index, detection in enumerate(result.get("detections", []), 1):
            class_id = int(detection["class_id"])
            x1, y1, x2, y2 = [float(value) for value in detection["xyxy"]]
            csv_rows.append(
                {
                    "filename": item.name,
                    "detection_index": detection_index,
                    "class_id": class_id,
                    "class_name": detection.get("class_name", class_id),
                    "confidence": f"{float(detection.get('confidence') or 0):.6f}",
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                    "source": detection.get("source", ""),
                    "owner_model": _owner_for(result, class_id),
                }
            )
        return _SavedDetectionArtifact(
            result=result,
            csv_rows=csv_rows,
            worker_ms=(time.perf_counter() - started) * 1000,
        )

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def finalize(
        self,
        source: Path,
        *,
        transport: str,
        performance: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("结果写入器已经关闭")
        try:
            artifacts = [future.result() for future in self._futures]
        finally:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=False)
        pipeline_ms = (
            (time.perf_counter() - self._pipeline_started) * 1000
            if self._pipeline_started is not None
            else 0.0
        )
        public_results = [artifact.result for artifact in artifacts]
        csv_rows = [row for artifact in artifacts for row in artifact.csv_rows]
        class_counts = Counter()
        for result in public_results:
            class_counts.update(result.get("class_counts", {}))
        payload: Dict[str, Any] = {
            "schema_version": 2,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": str(source),
            "transport": transport,
            "image_count": len(public_results),
            "detection_count": sum(
                int(item.get("detection_count") or 0) for item in public_results
            ),
            "class_counts": dict(sorted(class_counts.items())),
            "statistics": build_detection_statistics(public_results),
            "performance": dict(performance or {}),
            "artifact_pipeline": {
                "annotation_format": self.annotation_format,
                "workers": self.workers,
                "pipeline_wall_ms": round(pipeline_ms, 3),
                "worker_total_ms": round(
                    sum(artifact.worker_ms for artifact in artifacts),
                    3,
                ),
            },
            "saved_to": str(self.run_dir),
            "results": public_results,
            "artifacts": {
                "results_json": "results.json",
                "detections_csv": "detections.csv",
                "summary_text": "summary.txt",
                "annotated_dir": "annotated",
                "predictions_dir": "predictions",
            },
        }
        fieldnames = [
            "filename",
            "detection_index",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "source",
            "owner_model",
        ]
        with (self.run_dir / "detections.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        write_detection_reports(payload)
        return payload


def write_detection_reports(payload: Mapping[str, Any]) -> None:
    run_dir = Path(str(payload["saved_to"]))
    (run_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.txt").write_text(
        render_detection_summary(payload) + "\n",
        encoding="utf-8",
    )


def save_detection_results(
    run_dir: Path,
    source: Path,
    inputs: Sequence[DetectionInput],
    source_bytes: Sequence[bytes],
    results: Sequence[Mapping[str, Any]],
    *,
    transport: str,
    performance: Mapping[str, Any] | None = None,
    annotation_format: str = "png",
    annotation_workers: int = 1,
) -> Dict[str, Any]:
    if not (len(inputs) == len(source_bytes) == len(results)):
        raise ValueError("输入、图像数据与检测结果数量不一致")
    writer = DetectionArtifactWriter(
        run_dir,
        annotation_format=annotation_format,
        workers=annotation_workers,
    )
    try:
        for index, (item, data, result) in enumerate(
            zip(inputs, source_bytes, results),
            1,
        ):
            writer.submit(index, item, data, result)
        return writer.finalize(
            source,
            transport=transport,
            performance=performance,
        )
    except Exception:
        writer.abort()
        raise


def _context_text(result: Mapping[str, Any]) -> str:
    context = result.get("context", {})
    sensor = SENSOR_LABELS.get(str(context.get("sensor")), str(context.get("sensor") or "未知"))
    scene = SCENE_LABELS.get(str(context.get("scene")), str(context.get("scene") or "未知"))
    return f"{sensor}/{scene}"


def render_detection_summary(
    payload: Mapping[str, Any],
    *,
    max_detection_rows: int = 80,
) -> str:
    results = list(payload.get("results", []))
    image_count = int(payload.get("image_count") or len(results))
    detection_count = int(payload.get("detection_count") or 0)
    is_batch = image_count > 1
    statistics = dict(
        payload.get("statistics")
        or build_detection_statistics(results)
    )
    image_rows = [
        [
            item.get("filename", ""),
            _context_text(item),
            int(item.get("detection_count") or 0),
        ]
        for item in results
    ]
    detection_rows = []
    for result in results:
        for item in result.get("detections", []):
            coordinates = ", ".join(f"{float(value):.1f}" for value in item["xyxy"])
            detection_rows.append(
                [
                    result.get("filename", ""),
                    int(item["class_id"]),
                    display_class_name(item.get("class_name", item["class_id"])),
                    f"{float(item.get('confidence') or 0):.1%}",
                    coordinates,
                ]
            )
    class_text = "、".join(
        f"{display_class_name(name)} × {count}"
        for name, count in payload.get("class_counts", {}).items()
    ) or "无"
    header = panel(
        f"灵动 Agent · {'批量识别完成' if is_batch else '识别完成'}",
        [
            f"输入：{payload.get('source')}",
            f"图像：{image_count}    目标：{detection_count}",
            f"类别：{class_text}",
            f"保存：{payload.get('saved_to') or '未保存'}",
        ],
    )
    images_with_detections = int(
        statistics.get("images_with_detections") or 0
    )
    images_without_detections = int(
        statistics.get("images_without_detections") or 0
    )
    average_confidence = statistics.get("average_confidence")
    minimum_confidence = statistics.get("minimum_confidence")
    maximum_confidence = statistics.get("maximum_confidence")
    confidence_range = (
        f"{float(minimum_confidence):.1%}–{float(maximum_confidence):.1%}"
        if minimum_confidence is not None and maximum_confidence is not None
        else "—"
    )
    performance = dict(payload.get("performance") or {})
    throughput = statistics.get("estimated_throughput_fps")
    sections = [
        header,
        "统计摘要",
        table(
            ["完成图像", "有目标", "未检出", "检测目标", "平均目标/图"],
            [
                [
                    image_count,
                    images_with_detections,
                    images_without_detections,
                    detection_count,
                    f"{float(statistics.get('detections_per_image') or 0):.2f}",
                ]
            ],
        ),
    ]
    if performance.get("end_to_end_inference_ms") is not None:
        end_to_end_fps = performance.get("end_to_end_inference_fps")
        sections.append(
            table(
                ["端到端推理时间", "端到端推理 FPS"],
                [
                    [
                        f"{float(performance.get('end_to_end_inference_ms') or 0):.1f} ms",
                        (
                            f"{float(end_to_end_fps):.2f} FPS"
                            if end_to_end_fps is not None
                            else "—"
                        ),
                    ]
                ],
            )
        )
    else:
        sections.append(
            table(
                ["处理耗时", "平均耗时/图", "估算吞吐", "平均置信度", "置信度范围"],
                [
                    [
                        f"{float(statistics.get('total_processing_ms') or 0):.1f} ms",
                        f"{float(statistics.get('average_processing_ms') or 0):.1f} ms",
                        f"{float(throughput):.2f} FPS" if throughput is not None else "—",
                        (
                            f"{float(average_confidence):.1%}"
                            if average_confidence is not None
                            else "—"
                        ),
                        confidence_range,
                    ]
                ],
            )
        )
    if is_batch:
        class_rows = [
            [
                display_class_name(name),
                int(count),
                f"{int(count) / detection_count:.1%}" if detection_count else "—",
            ]
            for name, count in payload.get("class_counts", {}).items()
        ]
        sections.extend(
            [
                "类别分布",
                (
                    table(
                        ["类别", "数量", "目标占比"],
                        class_rows,
                        max_widths=[28, 8, 10],
                    )
                    if class_rows
                    else "未识别到通过正式门控的目标。"
                ),
            ]
        )
        sensor_text = "、".join(
            f"{SENSOR_LABELS.get(str(name), str(name))} × {count}"
            for name, count in statistics.get("sensor_counts", {}).items()
        ) or "无"
        scene_text = "、".join(
            f"{SCENE_LABELS.get(str(name), str(name))} × {count}"
            for name, count in statistics.get("scene_counts", {}).items()
        ) or "无"
        sections.extend(
            [
                "上下文分布",
                table(
                    ["维度", "分布"],
                    [["传感器", sensor_text], ["场景", scene_text]],
                    max_widths=[8, 64],
                ),
                "逐图明细已保存到 results.json、detections.csv 与 predictions/。",
            ]
        )
        return "\n\n".join(sections)
    if image_rows:
        sections.extend(
            [
                "图像概览",
                table(
                    ["图像", "传感器/场景", "目标"],
                    image_rows,
                    max_widths=[34, 18, 6],
                ),
            ]
        )
    if detection_rows:
        visible = detection_rows[:max_detection_rows]
        sections.extend(
            [
                "检测明细",
                table(
                    ["图像", "ID", "类别", "置信度", "坐标 x1,y1,x2,y2"],
                    visible,
                    max_widths=[26, 4, 23, 8, 27],
                ),
            ]
        )
        if len(detection_rows) > len(visible):
            sections.append(
                f"另有 {len(detection_rows) - len(visible)} 条记录，请查看 results.json 或 detections.csv。"
            )
    else:
        sections.append("检测明细\n未识别到通过正式门控的目标。")
    return "\n\n".join(sections)


def list_recent_detection_runs(limit: int = 8) -> list[Dict[str, Any]]:
    root = resolve_path(DEFAULT_RESULT_ROOT)
    if not root.is_dir():
        return []
    rows = []
    for run_dir in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        result_path = run_dir / "results.json"
        if not result_path.is_file():
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "created_at": payload.get("created_at"),
                "image_count": int(payload.get("image_count") or 0),
                "detection_count": int(payload.get("detection_count") or 0),
                "path": str(run_dir),
            }
        )
        if len(rows) >= max(1, int(limit)):
            break
    return rows
