#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import platform
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping
from urllib.parse import urlsplit


ROUTING_TIMING_KEYS = (
    "routing_fusion_ms",
    "routing_conversion_ms",
    "routing_gate_ms",
    "routing_conflict_ms",
    "routing_nms_ms",
    "routing_decision_ms",
)
TIMING_KEYS = (
    "upload_parse_ms",
    "decode_ms",
    "queue_wait_ms",
    "engine_total_ms",
    *ROUTING_TIMING_KEYS,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> Dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "fps_from_mean": 0.0,
        }
    mean_ms = statistics.fmean(values)
    return {
        "count": len(values),
        "mean_ms": mean_ms,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
        "fps_from_mean": 1000.0 / mean_ms if mean_ms > 0.0 else 0.0,
    }


def command_snapshot(command: list[str]) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "available": False, "error": str(exc)}
    return {
        "command": command,
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[:8000],
        "stderr": completed.stderr.strip()[:8000],
    }


def git_evidence(root: Path) -> Dict[str, Any]:
    def value(*arguments: str) -> str | None:
        snapshot = command_snapshot(["git", "-C", str(root), *arguments])
        if not snapshot.get("available"):
            return None
        return str(snapshot.get("stdout") or "").strip()

    status = value("status", "--porcelain=v1")
    return {
        "root": str(root.resolve()),
        "head": value("rev-parse", "HEAD"),
        "branch": value("branch", "--show-current"),
        "origin": value("remote", "get-url", "origin"),
        "status_porcelain": status,
        "clean": status == "" if status is not None else None,
    }


def validate_png(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(33)
    if len(header) < 33 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"不是有效PNG：{path}")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", header[16:29]
    )
    if (width, height) != (640, 512):
        raise ValueError(f"PNG尺寸必须为640x512：{path}: {width}x{height}")
    if bit_depth != 8 or color_type not in (2, 6):
        raise ValueError(
            f"PNG必须为8位RGB/RGBA：{path}: bit_depth={bit_depth}, color_type={color_type}"
        )
    if compression != 0 or filtering != 0:
        raise ValueError(f"PNG压缩/过滤方法不受支持：{path}")
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": "rgb" if color_type == 2 else "rgba",
        "interlaced": bool(interlace),
    }


def multipart_body(path: Path, confidence: float, boundary: str) -> bytes:
    filename = path.name.replace('"', "_")
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode("utf-8")
    suffix = (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="confidence"\r\n\r\n'
        f"{confidence:.8g}\r\n"
        f"--{boundary}--\r\n"
    ).encode("ascii")
    return prefix + path.read_bytes() + suffix


class KeepAliveClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("板端基准只接受明确的http:// URL。")
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self._prefix = parsed.path.rstrip("/")
        self._connection = http.client.HTTPConnection(
            self._host, self._port, timeout=timeout
        )

    def close(self) -> None:
        self._connection.close()

    def _json_response(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        self._connection.request(method, f"{self._prefix}{path}", **kwargs)
        response = self._connection.getresponse()
        data = response.read()
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(
                f"HTTP {response.status} {response.reason}: {data[:500].decode('utf-8', 'replace')}"
            )
        value = json.loads(data)
        if not isinstance(value, dict):
            raise RuntimeError(f"API响应不是JSON对象：{path}")
        return value

    def health(self) -> Dict[str, Any]:
        return self._json_response("GET", "/api/health")

    def detect(self, body: bytes, boundary: str) -> tuple[Dict[str, Any], float]:
        started = time.perf_counter_ns()
        payload = self._json_response(
            "POST",
            "/api/detect",
            body=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
                "Connection": "keep-alive",
            },
        )
        wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        return payload, wall_ms


def artifact_evidence(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {"configured": False}
    resolved = path.resolve()
    return {
        "configured": True,
        "path": str(resolved),
        "exists": resolved.is_file(),
        "sha256": sha256(resolved) if resolved.is_file() else None,
    }


def request_row(
    client: KeepAliveClient,
    body: bytes,
    boundary: str,
    *,
    round_index: int,
    path: Path,
) -> Dict[str, Any]:
    payload, wall_ms = client.detect(body, boundary)
    if "annotated_base64" in payload:
        raise RuntimeError("检测API仍同步返回标注图，性能口径无效。")
    timings = payload.get("timings") or {}
    row: Dict[str, Any] = {
        "round": round_index,
        "image": path.name,
        "wall_ms": wall_ms,
        "server_ms": float(payload["system_total_ms"]),
        "inference_ms": float(payload["inference_ms"]),
        "detection_count": int(payload.get("detection_count", len(payload.get("detections") or []))),
    }
    row.update({key: float(timings.get(key, 0.0)) for key in TIMING_KEYS})
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在310B本机回环以HTTP keep-alive执行30预热+10x89单图API基准。"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8502")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--warmup-requests", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--expected-images", type=int, default=89)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--build-manifest", type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"性能报告已存在，拒绝覆盖：{args.output}")
    if args.warmup_requests < 0 or args.rounds <= 0:
        raise ValueError("warmup_requests必须非负且rounds必须为正数。")
    paths = sorted(args.image_root.glob("*.png"))
    if len(paths) != args.expected_images:
        raise ValueError(
            f"图像数量不符合固定协议：期望{args.expected_images}，实际{len(paths)}"
        )
    contracts = {path.name: validate_png(path) for path in paths}
    boundary = "AgileAgentAscend310BBenchmarkBoundary"
    bodies = {
        path: multipart_body(path, args.confidence, boundary) for path in paths
    }

    client = KeepAliveClient(args.base_url, args.timeout)
    try:
        health = client.health()
        if health.get("backend") != "ascend_acl":
            raise RuntimeError(f"基准服务不是ascend_acl：{health}")
        for index in range(args.warmup_requests):
            client.detect(bodies[paths[index % len(paths)]], boundary)
        rows = [
            request_row(
                client,
                bodies[path],
                boundary,
                round_index=round_index + 1,
                path=path,
            )
            for round_index in range(args.rounds)
            for path in paths
        ]
    finally:
        client.close()

    distributions = {
        "server": distribution([float(row["server_ms"]) for row in rows]),
        "client_wall": distribution([float(row["wall_ms"]) for row in rows]),
        "inference": distribution([float(row["inference_ms"]) for row in rows]),
        **{
            key: distribution([float(row[key]) for row in rows]) for key in TIMING_KEYS
        },
    }
    rounds = []
    for round_index in range(1, args.rounds + 1):
        subset = [row for row in rows if row["round"] == round_index]
        rounds.append(
            {
                "round": round_index,
                "server": distribution([float(row["server_ms"]) for row in subset]),
                "client_wall": distribution([float(row["wall_ms"]) for row in subset]),
            }
        )
    gates = {
        "sample_count": len(rows) == args.rounds * args.expected_images,
        "mean_server_ms": float(distributions["server"]["mean_ms"]) <= 40.0,
        "p95_server_ms": float(distributions["server"]["p95_ms"]) <= 42.0,
        "request_failures": True,
    }
    report = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": {
            "transport": "loopback_http_multipart_png_keep_alive",
            "base_url": args.base_url,
            "confidence": args.confidence,
            "image_count": len(paths),
            "warmup_requests": args.warmup_requests,
            "rounds": args.rounds,
            "sample_count": len(rows),
            "concurrency": 1,
            "png_contracts": {
                "width": 640,
                "height": 512,
                "bit_depth": 8,
                "color_types": sorted({row["color_type"] for row in contracts.values()}),
            },
        },
        "health": health,
        "distributions": distributions,
        "rounds": rounds,
        "requests": rows,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "git": git_evidence(Path(__file__).resolve().parents[1]),
            "config": artifact_evidence(args.config),
            "build_manifest": artifact_evidence(args.build_manifest),
            "commands": {
                "npu_smi": command_snapshot(["npu-smi", "info"]),
                "atc": command_snapshot(["atc", "--help"]),
                "msprof": command_snapshot(["msprof", "--help"]),
                "aoe": command_snapshot(["aoe", "-h"]),
            },
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "requests"}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
