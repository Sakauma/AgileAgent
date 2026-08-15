from __future__ import annotations

import argparse
import json
import os
import platform
import pwd
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from fair_agent.core.hashes import sha256_file


DEFAULT_PROCESS_CPU_LIMIT = 10.0
DEFAULT_NPU_TEMPERATURE_LIMIT = 65
DEFAULT_PROCESS_SAMPLE_COUNT = 3
DEFAULT_PROCESS_SAMPLE_INTERVAL = 1.0
DEFAULT_PROCESS_WHITELIST = frozenset(
    {
        "Xorg",
        "sshd",
        "systemd",
        "systemd-journal",
        "systemd-logind",
    }
)


def command_snapshot(command: Sequence[str], timeout: float = 15.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=float(timeout),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "available": False,
            "error": str(exc),
        }
    return {
        "command": list(command),
        "available": completed.returncode == 0,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip()[:16000],
        "stderr": completed.stderr.strip()[:8000],
    }


def artifact_evidence(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"configured": False, "exists": False, "sha256": None}
    resolved = path.resolve()
    return {
        "configured": True,
        "path": str(resolved),
        "exists": resolved.is_file(),
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
    }


def git_evidence(root: Path) -> dict[str, Any]:
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


def manifest_artifact_evidence(path: Path | None) -> dict[str, Any]:
    evidence = artifact_evidence(path)
    evidence.update({"valid_json": False, "artifact_checks": [], "passed": False})
    if not evidence["exists"] or path is None:
        return evidence
    resolved = path.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        evidence["error"] = str(exc)
        return evidence
    if not isinstance(payload, Mapping):
        evidence["error"] = "top_level_not_mapping"
        return evidence
    evidence["valid_json"] = True
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        evidence["error"] = "artifacts_not_mapping"
        return evidence
    checks: list[dict[str, Any]] = []
    for model_id, raw_entry in sorted(artifacts.items()):
        if not isinstance(raw_entry, Mapping):
            checks.append({"model_id": str(model_id), "passed": False, "error": "invalid_entry"})
            continue
        for artifact_name in ("onnx", "aipp", "om"):
            raw_file = raw_entry.get(artifact_name)
            if not isinstance(raw_file, Mapping) or not raw_file.get("path"):
                checks.append(
                    {
                        "model_id": str(model_id),
                        "artifact": artifact_name,
                        "passed": False,
                        "error": "invalid_file_entry",
                    }
                )
                continue
            artifact_path = Path(str(raw_file["path"]))
            if not artifact_path.is_absolute():
                artifact_path = (resolved.parent / artifact_path).resolve()
            expected = str(raw_file.get("sha256") or "")
            actual = sha256_file(artifact_path) if artifact_path.is_file() else None
            checks.append(
                {
                    "model_id": str(model_id),
                    "artifact": artifact_name,
                    "path": str(artifact_path),
                    "expected_sha256": expected or None,
                    "actual_sha256": actual,
                    "passed": bool(actual and len(expected) == 64 and actual == expected),
                }
            )
    evidence["artifact_checks"] = checks
    evidence["passed"] = bool(checks) and all(bool(row.get("passed")) for row in checks)
    evidence["manifest_identity"] = {
        key: payload.get(key)
        for key in ("git_sha", "soc_version", "cann_version", "precision")
    }
    return evidence


def parse_npu_smi(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^\|\s*(?P<npu>\d+)\s+(?P<name>\S+)\s*"
        r"\|\s*(?P<health>[A-Za-z]+)\s*"
        r"\|\s*(?P<power>\d+(?:\.\d+)?)\s+(?P<temperature>\d+)\s+",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        if match.group("health") == "NA":
            continue
        rows.append(
            {
                "npu": int(match.group("npu")),
                "name": match.group("name"),
                "health": match.group("health"),
                "power_w": float(match.group("power")),
                "temperature_c": int(match.group("temperature")),
            }
        )
    return rows


def cpu_policy_snapshot(
    root: Path = Path("/sys/devices/system/cpu/cpufreq"),
) -> dict[str, Any]:
    policies: list[dict[str, Any]] = []
    for policy in sorted(root.glob("policy*")) if root.is_dir() else []:
        def read(name: str) -> str | None:
            target = policy / name
            try:
                return target.read_text(encoding="utf-8").strip()
            except OSError:
                return None

        policies.append(
            {
                "policy": policy.name,
                "governor": read("scaling_governor"),
                "available_governors": (read("scaling_available_governors") or "").split(),
                "driver": read("scaling_driver"),
            }
        )
    return {
        "supported": bool(policies),
        "state": "configured" if policies else "unsupported",
        "policies": policies,
    }


def parse_listeners(text: str) -> dict[str, dict[str, Any]]:
    listeners: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        port_match = re.search(r":(?P<port>\d+)\s+", line)
        if not port_match:
            continue
        process_match = re.search(r'users:\(\("(?P<name>[^"]+)",pid=(?P<pid>\d+)', line)
        listeners[port_match.group("port")] = {
            "line": line.strip(),
            "process": process_match.group("name") if process_match else None,
            "pid": int(process_match.group("pid")) if process_match else None,
        }
    return listeners


def _process_tick_snapshot(proc_root: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8")
            closing = stat_text.rfind(")")
            opening = stat_text.find("(")
            fields = stat_text[closing + 2 :].split()
            ticks = int(fields[11]) + int(fields[12])
            uid = entry.stat().st_uid
            try:
                user = pwd.getpwuid(uid).pw_name
            except KeyError:
                user = str(uid)
            rows[int(entry.name)] = {
                "pid": int(entry.name),
                "comm": stat_text[opening + 1 : closing],
                "user": user,
                "ticks": ticks,
            }
        except (OSError, ValueError, IndexError):
            continue
    return rows


def sample_process_cpu(
    *,
    count: int = DEFAULT_PROCESS_SAMPLE_COUNT,
    interval: float = DEFAULT_PROCESS_SAMPLE_INTERVAL,
    proc_root: Path = Path("/proc"),
    sleep: Callable[[float], None] = time.sleep,
) -> list[list[dict[str, Any]]]:
    if count <= 0 or interval <= 0:
        raise ValueError("process sample count and interval must be positive")
    clock_ticks = float(os.sysconf("SC_CLK_TCK"))
    samples: list[list[dict[str, Any]]] = []
    before = _process_tick_snapshot(proc_root)
    for _ in range(int(count)):
        started = time.monotonic()
        sleep(float(interval))
        elapsed = max(time.monotonic() - started, 1e-9)
        after = _process_tick_snapshot(proc_root)
        rows: list[dict[str, Any]] = []
        for pid, current in after.items():
            previous = before.get(pid)
            if previous is None:
                continue
            delta = max(0, int(current["ticks"]) - int(previous["ticks"]))
            cpu_percent = delta / clock_ticks / elapsed * 100.0
            if cpu_percent <= 0.0:
                continue
            rows.append(
                {
                    "pid": pid,
                    "user": current["user"],
                    "comm": current["comm"],
                    "cpu_percent": round(cpu_percent, 3),
                }
            )
        samples.append(sorted(rows, key=lambda row: float(row["cpu_percent"]), reverse=True)[:40])
        before = after
    return samples


def http_json_snapshot(url: str, timeout: float = 5.0) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=float(timeout)) as response:
            payload = json.loads(response.read())
    except (OSError, HTTPError, URLError, UnicodeError, json.JSONDecodeError) as exc:
        return {"url": url, "available": False, "error": str(exc)}
    return {
        "url": url,
        "available": isinstance(payload, Mapping),
        "payload": dict(payload) if isinstance(payload, Mapping) else payload,
    }


def collect_environment_snapshot(
    *,
    repo_root: Path,
    config: Path | None,
    build_manifest: Path | None,
    official_url: str,
    candidate_url: str,
    process_sample_count: int = DEFAULT_PROCESS_SAMPLE_COUNT,
    process_sample_interval: float = DEFAULT_PROCESS_SAMPLE_INTERVAL,
) -> dict[str, Any]:
    npu_command = command_snapshot(["npu-smi", "info"])
    ss_command = command_snapshot(["ss", "-lntp"])
    listeners = parse_listeners(str(ss_command.get("stdout") or ""))
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "system": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "git": git_evidence(repo_root),
        "config": artifact_evidence(config),
        "build_manifest": manifest_artifact_evidence(build_manifest),
        "cpu_policy": cpu_policy_snapshot(),
        "process_cpu_samples": sample_process_cpu(
            count=process_sample_count,
            interval=process_sample_interval,
        ),
        "listeners": listeners,
        "official_health": http_json_snapshot(official_url.rstrip("/") + "/api/health"),
        "candidate_health": http_json_snapshot(candidate_url.rstrip("/") + "/api/health"),
        "npu_smi": {
            **npu_command,
            "devices": parse_npu_smi(str(npu_command.get("stdout") or "")),
        },
        "memory": command_snapshot(["free", "-b"]),
        "cann": command_snapshot(["atc", "--version"]),
    }


def _health_ready(snapshot: Mapping[str, Any]) -> bool:
    payload = snapshot.get("payload")
    return bool(
        snapshot.get("available")
        and isinstance(payload, Mapping)
        and payload.get("status") == "ready"
        and payload.get("backend") == "ascend_acl"
    )


def evaluate_environment_snapshot(
    snapshot: Mapping[str, Any],
    *,
    candidate_state: str,
    max_npu_temperature_c: int = DEFAULT_NPU_TEMPERATURE_LIMIT,
    max_process_cpu_percent: float = DEFAULT_PROCESS_CPU_LIMIT,
    process_whitelist: Sequence[str] = tuple(DEFAULT_PROCESS_WHITELIST),
    require_temperature_limit: bool = True,
) -> dict[str, Any]:
    if candidate_state not in {"free", "ready"}:
        raise ValueError(f"candidate_state invalid: {candidate_state}")
    listeners = snapshot.get("listeners")
    listeners = listeners if isinstance(listeners, Mapping) else {}
    official_listener = listeners.get("8501")
    candidate_listener = listeners.get("8502")
    allowed_pids = {
        int(row["pid"])
        for row in (official_listener, candidate_listener)
        if isinstance(row, Mapping) and row.get("pid") is not None
    }
    allowed_pids.add(os.getpid())
    allowed_names = set(process_whitelist)
    samples = snapshot.get("process_cpu_samples")
    samples = samples if isinstance(samples, Sequence) else []
    offenders: list[dict[str, Any]] = []
    for sample_index, raw_rows in enumerate(samples, start=1):
        if not isinstance(raw_rows, Sequence):
            continue
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                continue
            if (
                float(raw.get("cpu_percent") or 0.0) > float(max_process_cpu_percent)
                and int(raw.get("pid") or -1) not in allowed_pids
                and str(raw.get("comm") or "") not in allowed_names
            ):
                offenders.append({"sample": sample_index, **dict(raw)})

    npu = snapshot.get("npu_smi")
    devices = npu.get("devices") if isinstance(npu, Mapping) else []
    devices = devices if isinstance(devices, Sequence) else []
    temperatures = [
        int(row["temperature_c"])
        for row in devices
        if isinstance(row, Mapping) and row.get("temperature_c") is not None
    ]
    cpu_policy = snapshot.get("cpu_policy")
    cpu_policy = cpu_policy if isinstance(cpu_policy, Mapping) else {}
    policies = cpu_policy.get("policies")
    policies = policies if isinstance(policies, Sequence) else []
    governor_ok = bool(
        not cpu_policy.get("supported")
        or (
            policies
            and all(
                isinstance(row, Mapping)
                and (
                    "performance" not in (row.get("available_governors") or [])
                    or row.get("governor") == "performance"
                )
                for row in policies
            )
        )
    )
    config = snapshot.get("config")
    manifest = snapshot.get("build_manifest")
    checks = {
        "official_port_8501_listening": isinstance(official_listener, Mapping),
        "official_health_ready": _health_ready(
            snapshot.get("official_health") if isinstance(snapshot.get("official_health"), Mapping) else {}
        ),
        "candidate_port_state": (
            candidate_listener is None
            if candidate_state == "free"
            else isinstance(candidate_listener, Mapping)
        ),
        "candidate_health_ready": (
            True
            if candidate_state == "free"
            else _health_ready(
                snapshot.get("candidate_health")
                if isinstance(snapshot.get("candidate_health"), Mapping)
                else {}
            )
        ),
        "npu_smi_available": bool(isinstance(npu, Mapping) and npu.get("available")),
        "npu_health_ok": bool(devices) and all(
            isinstance(row, Mapping) and row.get("health") == "OK" for row in devices
        ),
        "npu_temperature_at_most_limit": bool(temperatures)
        and (
            not require_temperature_limit
            or max(temperatures) <= int(max_npu_temperature_c)
        ),
        "process_cpu_below_limit": len(offenders) == 0,
        "cpu_governor_consistent": governor_ok,
        "git_clean": bool(
            isinstance(snapshot.get("git"), Mapping)
            and snapshot["git"].get("clean") is True
        ),
        "config_bound": bool(
            isinstance(config, Mapping)
            and config.get("configured")
            and config.get("exists")
            and config.get("sha256")
        ),
        "manifest_artifacts_bound": bool(
            isinstance(manifest, Mapping) and manifest.get("passed") is True
        ),
    }
    return {
        "candidate_state": candidate_state,
        "limits": {
            "max_npu_temperature_c": int(max_npu_temperature_c),
            "temperature_limit_required": bool(require_temperature_limit),
            "max_process_cpu_percent": float(max_process_cpu_percent),
        },
        "offenders": offenders,
        "temperatures_c": temperatures,
        "checks": checks,
        "passed": all(checks.values()),
    }


def compare_environment_snapshots(
    current: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    def nested(source: Mapping[str, Any], *keys: str) -> Any:
        value: Any = source
        for key in keys:
            if not isinstance(value, Mapping):
                return None
            value = value.get(key)
        return value

    checks = {
        "system_machine": nested(current, "system", "machine")
        == nested(reference, "system", "machine"),
        "system_release": nested(current, "system", "release")
        == nested(reference, "system", "release"),
        "python_executable": nested(current, "system", "python_executable")
        == nested(reference, "system", "python_executable"),
        "git_head": nested(current, "git", "head") == nested(reference, "git", "head"),
        "git_origin": nested(current, "git", "origin") == nested(reference, "git", "origin"),
        "config_sha256": nested(current, "config", "sha256")
        == nested(reference, "config", "sha256"),
        "build_manifest_sha256": nested(current, "build_manifest", "sha256")
        == nested(reference, "build_manifest", "sha256"),
        "manifest_identity": nested(current, "build_manifest", "manifest_identity")
        == nested(reference, "build_manifest", "manifest_identity"),
        "cpu_policy": current.get("cpu_policy") == reference.get("cpu_policy"),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _load_reference_snapshot(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"environment reference is not a mapping: {path}")
    snapshot = payload.get("snapshot", payload)
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"environment reference snapshot is missing: {path}")
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect and fail-close an Ascend 310B benchmark environment snapshot."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-url", default="http://127.0.0.1:8501")
    parser.add_argument("--candidate-url", default="http://127.0.0.1:8502")
    parser.add_argument("--candidate-state", choices=("free", "ready"), required=True)
    parser.add_argument("--max-npu-temperature-c", type=int, default=65)
    parser.add_argument("--max-process-cpu-percent", type=float, default=10.0)
    parser.add_argument("--process-samples", type=int, default=3)
    parser.add_argument("--process-sample-interval", type=float, default=1.0)
    parser.add_argument("--temperature-wait-seconds", type=float, default=600.0)
    parser.add_argument("--reference-snapshot", type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"environment snapshot already exists: {args.output}")
    deadline = time.monotonic() + max(0.0, float(args.temperature_wait_seconds))
    attempts: list[dict[str, Any]] = []
    while True:
        snapshot = collect_environment_snapshot(
            repo_root=args.repo_root.resolve(),
            config=args.config.resolve(),
            build_manifest=args.build_manifest.resolve(),
            official_url=str(args.official_url),
            candidate_url=str(args.candidate_url),
            process_sample_count=int(args.process_samples),
            process_sample_interval=float(args.process_sample_interval),
        )
        evaluation = evaluate_environment_snapshot(
            snapshot,
            candidate_state=str(args.candidate_state),
            max_npu_temperature_c=int(args.max_npu_temperature_c),
            max_process_cpu_percent=float(args.max_process_cpu_percent),
        )
        attempts.append(
            {
                "created_at": snapshot["created_at"],
                "temperatures_c": evaluation["temperatures_c"],
                "passed": evaluation["passed"],
            }
        )
        checks = evaluation["checks"]
        temperature_only = not checks["npu_temperature_at_most_limit"] and all(
            value
            for key, value in checks.items()
            if key != "npu_temperature_at_most_limit"
        )
        if evaluation["passed"] or not temperature_only or time.monotonic() >= deadline:
            break
        time.sleep(min(15.0, max(0.0, deadline - time.monotonic())))

    reference_comparison: dict[str, Any] | None = None
    if args.reference_snapshot is not None:
        reference_comparison = compare_environment_snapshots(
            snapshot,
            _load_reference_snapshot(args.reference_snapshot.resolve()),
        )
    passed = bool(
        evaluation["passed"]
        and (reference_comparison is None or reference_comparison["passed"])
    )
    report = {
        "schema_version": 1,
        "kind": "ascend310b_benchmark_environment",
        "snapshot": snapshot,
        "evaluation": evaluation,
        "reference_comparison": reference_comparison,
        "temperature_attempts": attempts,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "evaluation": evaluation,
                "reference_comparison": reference_comparison,
                "passed": passed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
