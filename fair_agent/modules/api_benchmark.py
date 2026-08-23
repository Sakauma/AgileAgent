from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

import httpx

from fair_agent.core.config import config_sha256, rel_path, resolve_path
from fair_agent.core.hashes import business_payload_sha256, sha256_file


ROUTING_TIMING_KEYS = (
    "routing_fusion_ms",
    "routing_conversion_ms",
    "routing_gate_ms",
    "routing_conflict_ms",
    "routing_nms_ms",
    "routing_decision_ms",
)
def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _distribution(values: list[float]) -> Dict[str, float | int]:
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
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "max_ms": max(values),
        "fps_from_mean": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }


def _command_snapshot(command: list[str]) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(resolve_path(".")),
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


def _artifact_evidence(entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, Mapping) or not entry.get("path"):
        return {"configured": False}
    path = resolve_path(str(entry["path"]))
    return {
        "configured": True,
        "path": str(path),
        "exists": path.is_file(),
        "configured_sha256": entry.get("sha256"),
        "actual_sha256": sha256_file(path) if path.is_file() else None,
    }


def _runtime_evidence(config: Mapping[str, Any], health: Mapping[str, Any]) -> Dict[str, Any]:
    ascend = config.get("ascend_backend") or {}
    manifest_path = resolve_path(str(ascend.get("build_manifest") or ""))
    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = loaded if isinstance(loaded, dict) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            manifest = {}
    git_sha = str(manifest.get("git_sha") or "")
    if not git_sha:
        git_result = _command_snapshot(["git", "rev-parse", "HEAD"])
        git_sha = str(git_result.get("stdout") or "unknown")
    model_entries = {
        str(source): _artifact_evidence(entry)
        for source, entry in dict(ascend.get("models") or {}).items()
    }
    return {
        "git_sha": git_sha,
        "python": sys.version,
        "platform": platform.platform(),
        "health": dict(health),
        "ascend": {
            "device_id": ascend.get("device_id"),
            "soc_version": ascend.get("soc_version"),
            "cann_version": ascend.get("cann_version"),
            "precision": ascend.get("precision"),
            "execution_mode": ascend.get("execution_mode"),
            "encoded_preprocessing": ascend.get("encoded_preprocessing"),
            "memory_mode": ascend.get("memory_mode", "pageable"),
            "dvpp_scene_resize_stages": list(
                ascend.get("dvpp_scene_resize_stages") or []
            ),
            "models": model_entries,
            "context_model": _artifact_evidence(ascend.get("context_model")),
            "build_manifest": _artifact_evidence({
                "path": ascend.get("build_manifest"),
                "sha256": ascend.get("build_manifest_sha256"),
            }),
            "validation_report": _artifact_evidence({
                "path": ascend.get("validation_report"),
                "sha256": ascend.get("validation_report_sha256"),
            }),
        },
        "commands": {
            "npu_smi": _command_snapshot(["npu-smi", "info"]),
            "atc": _command_snapshot(["atc", "--help"]),
        },
    }


def _performance_assessment(
    summary: Mapping[str, Any],
    performance: Mapping[str, Any],
    concurrency: int,
) -> tuple[Dict[str, bool], Dict[str, bool]]:
    target_mean_ms = 1000.0 / float(performance["target_api_fps"])
    competition_gates = {
        "batch_fps": float(summary["batch_fps"]) >= float(performance["target_api_fps"]),
    }
    diagnostic_checks = {
        "mean_api_ms": float(summary["median_round_mean_server_ms"]) <= target_mean_ms,
        "p95_api_ms": float(summary["all_p95_server_ms"])
        <= float(performance["target_p95_ms"]),
        "concurrency": int(summary["concurrent_success_count"]) == concurrency,
    }
    return competition_gates, diagnostic_checks


def _post_one_with_client(client: httpx.Client, path: Path, confidence: float) -> Dict[str, float]:
    started = time.perf_counter()
    response = client.post(
        "/api/detect",
        files={"file": (path.name, path.read_bytes(), "application/octet-stream")},
        data={"confidence": str(confidence)},
    )
    wall_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    payload = response.json()
    if "annotated_base64" in payload:
        raise RuntimeError("检测API仍同步返回标注图，性能口径无效。")
    timings = payload.get("timings") or {}
    row = {
        "server_ms": float(payload["system_total_ms"]),
        "inference_ms": float(payload["inference_ms"]),
        "wall_ms": wall_ms,
        "upload_parse_ms": float(timings.get("upload_parse_ms", 0.0)),
        "decode_ms": float(timings.get("decode_ms", 0.0)),
        "queue_wait_ms": float(timings.get("queue_wait_ms", 0.0)),
        "engine_total_ms": float(timings.get("engine_total_ms", 0.0)),
    }
    row.update({key: float(timings.get(key, 0.0)) for key in ROUTING_TIMING_KEYS})
    return row


def _post_one(base_url: str, path: Path, confidence: float, timeout: float) -> Dict[str, float]:
    with httpx.Client(base_url=base_url, timeout=timeout, trust_env=False) as client:
        return _post_one_with_client(client, path, confidence)


def _health_available(base_url: str, timeout: float = 1.0) -> bool:
    try:
        return httpx.get(f"{base_url}/api/health", timeout=timeout, trust_env=False).is_success
    except httpx.HTTPError:
        return False


@contextmanager
def _server_session(config: Mapping[str, Any], base_url: str):
    performance = config["performance"]
    if _health_available(base_url):
        health = httpx.get(f"{base_url}/api/health", timeout=2.0, trust_env=False).json()
        if health.get("backend") != config["inference"]["backend"]:
            raise RuntimeError(
                f"已有检测服务后端为{health.get('backend')}，与待验收后端{config['inference']['backend']}不一致。"
            )
        yield "existing"
        return
    if not performance["auto_start_server"]:
        raise RuntimeError(f"检测服务未启动：{base_url}")

    runtime = config["runtime"]
    environment = dict(os.environ)
    environment["AGILE_AGENT_CONFIG"] = str(resolve_path(config["_config_path"]))
    environment["AGILE_AGENT_OVERRIDES"] = json.dumps(list(config.get("_config_overrides", [])))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "fair_agent.web.app:app",
            "--host",
            str(runtime["server_host"]),
            "--port",
            str(int(runtime["server_port"])),
            "--no-access-log",
        ],
        cwd=str(resolve_path(".")),
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + float(performance["server_start_timeout_seconds"])
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"临时检测服务启动失败，退出码：{process.returncode}")
            if _health_available(base_url):
                yield "temporary"
                return
            time.sleep(0.25)
        raise RuntimeError("等待临时检测服务就绪超时。")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _benchmark_running_server(
    config: Mapping[str, Any], base_url: str, server_mode: str
) -> Dict[str, Any]:
    performance = config["performance"]
    timeout = float(performance["request_timeout_seconds"])
    split = resolve_path(performance["benchmark_split"])
    paths = [resolve_path(line.strip()) for line in split.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("性能测试划分为空或包含缺失文件。")
    confidence = float(config["inference"]["confidence_default"])
    with httpx.Client(base_url=base_url, timeout=timeout, trust_env=False) as client:
        health = client.get("/api/health")
        health.raise_for_status()
        health_payload = health.json()
        generation_id = health_payload["generation_id"]
    for index in range(int(performance["warmup_requests"])):
        _post_one(base_url, paths[index % len(paths)], confidence, timeout)

    rounds = []
    all_server: list[float] = []
    all_wall: list[float] = []
    all_inference: list[float] = []
    all_upload: list[float] = []
    all_decode: list[float] = []
    all_queue: list[float] = []
    all_engine: list[float] = []
    routing_values = {key: [] for key in ROUTING_TIMING_KEYS}
    request_rows: list[Dict[str, Any]] = []
    for round_index in range(int(performance["benchmark_rounds"])):
        round_rows = [_post_one(base_url, path, confidence, timeout) for path in paths]
        server = [row["server_ms"] for row in round_rows]
        wall = [row["wall_ms"] for row in round_rows]
        inference = [row["inference_ms"] for row in round_rows]
        all_server.extend(server)
        all_wall.extend(wall)
        all_inference.extend(inference)
        all_upload.extend(row["upload_parse_ms"] for row in round_rows)
        all_decode.extend(row["decode_ms"] for row in round_rows)
        all_queue.extend(row["queue_wait_ms"] for row in round_rows)
        all_engine.extend(row["engine_total_ms"] for row in round_rows)
        for key in ROUTING_TIMING_KEYS:
            routing_values[key].extend(row[key] for row in round_rows)
        request_rows.extend(
            {
                "round": round_index + 1,
                "image": rel_path(path),
                **row,
            }
            for path, row in zip(paths, round_rows)
        )
        rounds.append({
            "round": round_index + 1,
            "mean_server_ms": statistics.fmean(server),
            "p95_server_ms": _percentile(server, 0.95),
            "mean_wall_ms": statistics.fmean(wall),
            "mean_inference_ms": statistics.fmean(inference),
        })

    batch_paths = paths[: min(int(performance["batch_probe_size"]), len(paths))]
    batch_rounds = []
    with httpx.Client(base_url=base_url, timeout=timeout, trust_env=False) as client:
        for round_index in range(int(performance["benchmark_rounds"])):
            files = [
                ("files", (path.name, path.read_bytes(), "application/octet-stream"))
                for path in batch_paths
            ]
            batch_response = client.post(
                "/api/batch", files=files, data={"confidence": str(confidence)}
            )
            batch_response.raise_for_status()
            batch_payload = batch_response.json()
            system_total_ms = float(batch_payload["system_total_ms"])
            batch_rounds.append({
                "round": round_index + 1,
                "system_total_ms": system_total_ms,
                "fps": len(batch_paths) / (system_total_ms / 1000.0),
                "timings": dict(batch_payload.get("timings", {})),
            })
    median_batch = sorted(
        batch_rounds, key=lambda row: row["system_total_ms"]
    )[len(batch_rounds) // 2]

    concurrency = int(performance["concurrent_requests"])
    concurrent_paths = [paths[index % len(paths)] for index in range(concurrency)]
    connection_limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    with httpx.Client(
        base_url=base_url,
        timeout=timeout,
        trust_env=False,
        limits=connection_limits,
    ) as concurrent_client:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            concurrent_rows = list(
                pool.map(
                    lambda path: _post_one_with_client(concurrent_client, path, confidence),
                    concurrent_paths,
                )
            )

    median_round = sorted(rounds, key=lambda row: row["mean_server_ms"])[len(rounds) // 2]
    summary = {
        "generation_id": generation_id,
        "image_count": len(paths),
        "round_count": len(rounds),
        "median_round_mean_server_ms": median_round["mean_server_ms"],
        "all_mean_server_ms": statistics.fmean(all_server),
        "all_p95_server_ms": _percentile(all_server, 0.95),
        "all_mean_wall_ms": statistics.fmean(all_wall),
        "all_mean_inference_ms": statistics.fmean(all_inference),
        "all_mean_upload_parse_ms": statistics.fmean(all_upload),
        "all_mean_decode_ms": statistics.fmean(all_decode),
        "all_mean_queue_wait_ms": statistics.fmean(all_queue),
        "all_mean_engine_total_ms": statistics.fmean(all_engine),
        "all_mean_routing_fusion_ms": statistics.fmean(routing_values["routing_fusion_ms"]),
        "server_distribution": _distribution(all_server),
        "client_wall_distribution": _distribution(all_wall),
        "routing_distributions": {
            key: _distribution(values) for key, values in routing_values.items()
        },
        "batch_image_count": len(batch_paths),
        "batch_system_total_ms": float(median_batch["system_total_ms"]),
        "batch_fps": float(median_batch["fps"]),
        "batch_timings": dict(median_batch["timings"]),
        "batch_rounds": batch_rounds,
        "concurrent_request_count": concurrency,
        "concurrent_success_count": len(concurrent_rows),
        "concurrent_p95_wall_ms": _percentile([row["wall_ms"] for row in concurrent_rows], 0.95),
        "concurrent_p95_server_ms": _percentile([row["server_ms"] for row in concurrent_rows], 0.95),
        "request_failure_count": 0,
    }
    competition_gates, diagnostic_checks = _performance_assessment(
        summary, performance, concurrency
    )
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = resolve_path(performance["report_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    clean_config = {key: value for key, value in config.items() if not str(key).startswith("_")}
    report = {
        "schema_version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "server_mode": server_mode,
        "config_sha256": config_sha256(clean_config),
        "split": rel_path(split),
        "split_sha256": sha256_file(split),
        "summary": summary,
        "rounds": rounds,
        "requests": request_rows,
        "environment": _runtime_evidence(config, health_payload),
        "competition_gates": competition_gates,
        "diagnostic_checks": diagnostic_checks,
        "warnings": [name for name, passed in diagnostic_checks.items() if not passed],
        "accepted": all(competition_gates.values()),
    }
    manifest = output / "benchmark.json"
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**report, "report": rel_path(manifest), "report_sha256": sha256_file(manifest)}


def benchmark_api(config: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = config["runtime"]
    base_url = f"http://{runtime['server_host']}:{int(runtime['server_port'])}"
    with _server_session(config, base_url) as server_mode:
        return _benchmark_running_server(config, base_url, server_mode)
