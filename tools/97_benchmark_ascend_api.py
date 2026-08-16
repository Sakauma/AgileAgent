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
from typing import Any, Dict
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.hashes import business_payload_sha256  # noqa: E402
from fair_agent.modules.ascend_benchmark_guard import (  # noqa: E402
    collect_environment_snapshot,
    compare_environment_snapshots,
    evaluate_environment_snapshot,
)


ROUTING_TIMING_KEYS = (
    "routing_fusion_ms",
    "routing_conversion_ms",
    "routing_gate_ms",
    "routing_conflict_ms",
    "routing_nms_ms",
    "routing_decision_ms",
)
ASCEND_TIMING_KEYS = (
    "dvpp_enqueue_ms",
    "dvpp_device_ms",
    "ascend_submit_ms",
    "ascend_wait_ms",
    "ascend_input_copy_max_ms",
    "ascend_output_copy_max_ms",
)
TIMING_KEYS = (
    "upload_parse_ms",
    "decode_ms",
    "queue_wait_ms",
    "engine_total_ms",
    *ASCEND_TIMING_KEYS,
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


def batch_multipart_body(
    paths: list[Path], confidence: float, boundary: str
) -> bytes:
    if not paths:
        raise ValueError("batch性能探针至少需要一张图像。")
    chunks: list[bytes] = []
    for path in paths:
        filename = path.name.replace('"', "_")
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                (
                    'Content-Disposition: form-data; name="files"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                b"Content-Type: image/png\r\n\r\n",
                path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("ascii"),
            b'Content-Disposition: form-data; name="confidence"\r\n\r\n',
            f"{confidence:.8g}\r\n".encode("ascii"),
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )
    return b"".join(chunks)


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

    def batch(self, body: bytes, boundary: str) -> tuple[Dict[str, Any], float]:
        started = time.perf_counter_ns()
        payload = self._json_response(
            "POST",
            "/api/batch",
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


def load_json_object(path: Path, label: str) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label}不是JSON对象：{path}")
    return payload


def relative_difference_within(current: float, reference: float, limit: float) -> bool:
    if reference <= 0.0:
        raise ValueError("稳定性参考值必须为正数。")
    return abs(float(current) - float(reference)) / float(reference) <= float(limit)


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
        "business_sha256": business_payload_sha256(payload),
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
    parser.add_argument(
        "--gate-profile", choices=("p0", "p3", "p8", "score"), default="p0"
    )
    parser.add_argument("--baseline-mean-ms", type=float)
    parser.add_argument("--baseline-p99-ms", type=float)
    parser.add_argument("--environment-guard", action="store_true")
    parser.add_argument("--official-url", default="http://127.0.0.1:8501")
    parser.add_argument("--max-npu-temperature-c", type=int, default=65)
    parser.add_argument("--max-process-cpu-percent", type=float, default=10.0)
    parser.add_argument("--process-samples", type=int, default=3)
    parser.add_argument("--process-sample-interval", type=float, default=1.0)
    parser.add_argument("--p8-reference-report", type=Path)
    parser.add_argument("--batch-probe-size", type=int, default=20)
    parser.add_argument("--batch-rounds", type=int, default=3)
    parser.add_argument("--target-batch-fps", type=float, default=30.0)
    parser.add_argument(
        "--skip-single-requests",
        action="store_true",
        help="score门禁只执行30次预热和三轮20图batch，不采集单请求诊断。",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"性能报告已存在，拒绝覆盖：{args.output}")
    if args.warmup_requests < 0 or args.rounds < 0:
        raise ValueError("warmup_requests和rounds必须为非负数。")
    if args.skip_single_requests:
        if args.gate_profile != "score" or args.rounds != 0:
            raise ValueError("--skip-single-requests仅允许score门禁且要求--rounds=0。")
    elif args.rounds <= 0:
        raise ValueError("未跳过单请求诊断时rounds必须为正数。")
    if (
        args.batch_probe_size <= 0
        or args.batch_rounds <= 0
        or args.target_batch_fps <= 0.0
    ):
        raise ValueError("batch探针大小、轮数和目标FPS必须为正数。")
    if args.gate_profile == "p3" and (
        not args.baseline_mean_ms
        or args.baseline_mean_ms <= 0
        or not args.baseline_p99_ms
        or args.baseline_p99_ms <= 0
    ):
        raise ValueError("P3门禁要求正数baseline-mean-ms和baseline-p99-ms。")
    if args.environment_guard and (args.config is None or args.build_manifest is None):
        raise ValueError("环境守卫要求同时提供--config和--build-manifest。")
    if args.p8_reference_report is not None and args.gate_profile != "p8":
        raise ValueError("--p8-reference-report只能用于P8门禁。")
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

    guard_before: Dict[str, Any] | None = None
    guard_before_evaluation: Dict[str, Any] | None = None
    if args.environment_guard:
        guard_before = collect_environment_snapshot(
            repo_root=ROOT,
            config=args.config,
            build_manifest=args.build_manifest,
            official_url=args.official_url,
            candidate_url=args.base_url,
            process_sample_count=args.process_samples,
            process_sample_interval=args.process_sample_interval,
        )
        guard_before_evaluation = evaluate_environment_snapshot(
            guard_before,
            candidate_state="ready",
            max_npu_temperature_c=args.max_npu_temperature_c,
            max_process_cpu_percent=args.max_process_cpu_percent,
        )
        if not guard_before_evaluation["passed"]:
            raise RuntimeError(
                "P8环境守卫前置检查失败："
                + json.dumps(guard_before_evaluation, ensure_ascii=False)
            )

    client = KeepAliveClient(args.base_url, args.timeout)
    try:
        health = client.health()
        if health.get("backend") != "ascend_acl":
            raise RuntimeError(f"基准服务不是ascend_acl：{health}")
        for index in range(args.warmup_requests):
            client.detect(bodies[paths[index % len(paths)]], boundary)
        rows = (
            []
            if args.skip_single_requests
            else [
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
        )
        batch_paths = paths[: min(args.batch_probe_size, len(paths))]
        batch_boundary = boundary + "Batch"
        batch_body = batch_multipart_body(
            batch_paths, args.confidence, batch_boundary
        )
        batch_rounds = []
        for round_index in range(args.batch_rounds):
            payload, wall_ms = client.batch(batch_body, batch_boundary)
            system_total_ms = float(payload["system_total_ms"])
            if system_total_ms <= 0.0:
                raise RuntimeError("batch响应system_total_ms必须为正数。")
            batch_rounds.append(
                {
                    "round": round_index + 1,
                    "image_count": len(batch_paths),
                    "system_total_ms": system_total_ms,
                    "wall_ms": wall_ms,
                    "fps": len(batch_paths) * 1000.0 / system_total_ms,
                    "timings": dict(payload.get("timings") or {}),
                }
            )
    finally:
        client.close()

    median_batch = sorted(
        batch_rounds, key=lambda row: float(row["system_total_ms"])
    )[len(batch_rounds) // 2]

    guard_after: Dict[str, Any] | None = None
    guard_after_evaluation: Dict[str, Any] | None = None
    guard_run_consistency: Dict[str, Any] | None = None
    if args.environment_guard:
        guard_after = collect_environment_snapshot(
            repo_root=ROOT,
            config=args.config,
            build_manifest=args.build_manifest,
            official_url=args.official_url,
            candidate_url=args.base_url,
            process_sample_count=args.process_samples,
            process_sample_interval=args.process_sample_interval,
        )
        guard_after_evaluation = evaluate_environment_snapshot(
            guard_after,
            candidate_state="ready",
            max_npu_temperature_c=args.max_npu_temperature_c,
            max_process_cpu_percent=args.max_process_cpu_percent,
            # P8 requires the board to cool before each timed run.  The
            # post-run temperature is evidence of the load, not a start gate
            # applied retroactively to an otherwise valid measurement.
            require_temperature_limit=False,
        )
        guard_run_consistency = compare_environment_snapshots(guard_after, guard_before)

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
    p8_reference: Dict[str, Any] | None = None
    p8_environment_consistency: Dict[str, Any] | None = None
    if args.p8_reference_report is not None:
        p8_reference = load_json_object(args.p8_reference_report.resolve(), "P8参考报告")
        reference_guard = (
            p8_reference.get("environment", {})
            .get("guard", {})
            .get("before", {})
            .get("snapshot")
        )
        if not isinstance(reference_guard, dict) or guard_before is None:
            raise ValueError("P8参考报告缺少environment.guard.before.snapshot。")
        p8_environment_consistency = compare_environment_snapshots(
            guard_before, reference_guard
        )

    if args.gate_profile == "score":
        gates = {
            "sample_count": len(rows) == args.rounds * args.expected_images,
            "request_failures": True,
            "batch_fps": float(median_batch["fps"]) >= args.target_batch_fps,
        }
        if args.environment_guard:
            gates.update(
                {
                    "environment_before": bool(
                        guard_before_evaluation
                        and guard_before_evaluation["passed"]
                    ),
                    "environment_after": bool(
                        guard_after_evaluation
                        and guard_after_evaluation["passed"]
                    ),
                    "environment_unchanged_during_run": bool(
                        guard_run_consistency
                        and guard_run_consistency["passed"]
                    ),
                }
            )
    elif args.gate_profile == "p3":
        baseline_mean_ms = float(args.baseline_mean_ms)
        baseline_p99_ms = float(args.baseline_p99_ms)
        mean_ms = float(distributions["server"]["mean_ms"])
        gates = {
            "sample_count": len(rows) == args.rounds * args.expected_images,
            "mean_improvement_at_least_3pct": (
                baseline_mean_ms - mean_ms
            ) / baseline_mean_ms >= 0.03,
            "mean_server_ms": mean_ms <= 33.33,
            "p95_server_ms": float(distributions["server"]["p95_ms"]) <= 35.0,
            "p99_not_worse_than_2pct": float(
                distributions["server"]["p99_ms"]
            )
            <= baseline_p99_ms * 1.02,
            "request_failures": True,
        }
    elif args.gate_profile == "p8":
        gates = {
            "sample_count": len(rows) == args.rounds * args.expected_images,
            "request_failures": True,
            "environment_before": bool(
                guard_before_evaluation and guard_before_evaluation["passed"]
            ),
            "environment_after": bool(
                guard_after_evaluation and guard_after_evaluation["passed"]
            ),
            "environment_unchanged_during_run": bool(
                guard_run_consistency and guard_run_consistency["passed"]
            ),
        }
        if p8_reference is not None:
            reference_server = p8_reference.get("distributions", {}).get("server", {})
            if not isinstance(reference_server, dict):
                raise ValueError("P8参考报告缺少服务端分布。")
            gates.update(
                {
                    "reference_environment_identical": bool(
                        p8_environment_consistency
                        and p8_environment_consistency["passed"]
                    ),
                    "mean_within_2pct": relative_difference_within(
                        float(distributions["server"]["mean_ms"]),
                        float(reference_server["mean_ms"]),
                        0.02,
                    ),
                    "p95_within_2pct": relative_difference_within(
                        float(distributions["server"]["p95_ms"]),
                        float(reference_server["p95_ms"]),
                        0.02,
                    ),
                    "p99_within_2pct": relative_difference_within(
                        float(distributions["server"]["p99_ms"]),
                        float(reference_server["p99_ms"]),
                        0.02,
                    ),
                }
            )
    else:
        gates = {
            "sample_count": len(rows) == args.rounds * args.expected_images,
            "mean_server_ms": float(distributions["server"]["mean_ms"]) <= 40.0,
            "p95_server_ms": float(distributions["server"]["p95_ms"]) <= 42.0,
            "request_failures": True,
        }
    report = {
        "schema_version": 5,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": {
            "transport": "loopback_http_multipart_png_keep_alive",
            "base_url": args.base_url,
            "confidence": args.confidence,
            "image_count": len(paths),
            "warmup_requests": args.warmup_requests,
            "rounds": args.rounds,
            "single_requests_skipped": bool(args.skip_single_requests),
            "sample_count": len(rows),
            "concurrency": 1,
            "gate_profile": args.gate_profile,
            "baseline_mean_ms": args.baseline_mean_ms,
            "baseline_p99_ms": args.baseline_p99_ms,
            "p8_reference_report": (
                str(args.p8_reference_report.resolve())
                if args.p8_reference_report is not None
                else None
            ),
            "batch_probe_size": len(batch_paths),
            "batch_rounds": args.batch_rounds,
            "target_batch_fps": args.target_batch_fps,
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
        "competition": {
            "batch_fps": float(median_batch["fps"]),
            "batch_system_total_ms": float(median_batch["system_total_ms"]),
            "batch_wall_ms": float(median_batch["wall_ms"]),
            "batch_image_count": len(batch_paths),
            "batch_rounds": batch_rounds,
            "batch_fps_passed": (
                float(median_batch["fps"]) >= args.target_batch_fps
            ),
        },
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
            "guard": {
                "enabled": bool(args.environment_guard),
                "before": {
                    "snapshot": guard_before,
                    "evaluation": guard_before_evaluation,
                },
                "after": {
                    "snapshot": guard_after,
                    "evaluation": guard_after_evaluation,
                },
                "run_consistency": guard_run_consistency,
                "reference_consistency": p8_environment_consistency,
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
