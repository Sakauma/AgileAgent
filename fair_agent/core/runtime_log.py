from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from fair_agent.core.config import resolve_path


SENSITIVE_KEYS = {"password", "secret", "token", "authorization", "cookie", "api_key"}

STATE_EVENTS = {
    "CREATED": "experiment.created",
    "DATA_AUDITED": "incremental.data_audit.completed",
    "PARTITIONED": "incremental.data_view.generated",
    "BASE_TRAINED": "incremental.base_training.completed",
    "BASE_FROZEN": "incremental.base_model.frozen",
    "INCREMENT_TRAINED": "incremental.training.completed",
    "INCREMENT_FROZEN": "incremental.candidate.frozen",
    "THRESHOLD_CALIBRATED": "incremental.dev_calibration.completed",
    "DIAGNOSED": "incremental.dev_diagnosis.completed",
    "RECOVERY_REQUIRED": "incremental.recovery.selected",
    "LOCK_PREDICTIONS_FROZEN": "incremental.lock.unlabeled_predictions_frozen",
    "LOCK_UNSEALED": "incremental.lock.unsealed",
    "EVALUATED": "incremental.lock_recheck.completed",
    "ACCEPTED": "incremental.candidate.accepted",
    "REJECTED": "incremental.candidate.rejected",
    "REGISTERED": "generation.registered",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_trace_id(prefix: str = "trace") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _safe_value(value: Any, key: str = "") -> Any:
    if key.lower() in SENSITIVE_KEYS:
        return "***"
    if isinstance(value, Mapping):
        return {str(item_key): _safe_value(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class StructuredEventLog:
    """Thread-safe JSONL event log with bounded file rotation and query support."""

    def __init__(self, root: str | Path, max_file_bytes: int, retained_files: int) -> None:
        self.root = resolve_path(root)
        self.max_file_bytes = int(max_file_bytes)
        self.retained_files = int(retained_files)
        self._lock = threading.Lock()

    def _files(self) -> list[Path]:
        if not self.root.exists():
            return []
        return sorted(self.root.glob("agent-*.jsonl"), key=lambda path: path.stat().st_mtime)

    def _active_path(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        base = self.root / f"agent-{stamp}.jsonl"
        if not base.exists() or base.stat().st_size < self.max_file_bytes:
            return base
        index = 1
        while True:
            candidate = self.root / f"agent-{stamp}-{index:02d}.jsonl"
            if not candidate.exists() or candidate.stat().st_size < self.max_file_bytes:
                return candidate
            index += 1

    @contextmanager
    def _process_lock(self):
        lock_path = self.root / ".agent-log.lock"
        with lock_path.open("a+b") as lock_handle:
            try:
                import fcntl

                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            except ImportError:
                pass
            try:
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except ImportError:
                    pass

    def append(
        self,
        event: str,
        *,
        level: str = "info",
        component: str = "agent",
        trace_id: str | None = None,
        message: str = "",
        duration_ms: float | None = None,
        details: Mapping[str, Any] | None = None,
        **identifiers: Any,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "timestamp": utc_now(),
            "level": str(level),
            "component": str(component),
            "event": str(event),
            "trace_id": trace_id or new_trace_id(),
            "message": str(message),
        }
        if duration_ms is not None:
            record["duration_ms"] = round(float(duration_ms), 3)
        record.update({key: _safe_value(value, key) for key, value in identifiers.items() if value is not None})
        if details:
            record["details"] = _safe_value(details)
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with self._process_lock():
                with self._active_path().open("a", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                files = self._files()
                for stale in files[:-self.retained_files]:
                    stale.unlink(missing_ok=True)
        return record

    def query(
        self,
        *,
        limit: int = 200,
        level: str | None = None,
        component: str | None = None,
        trace_id: str | None = None,
        batch_id: str | None = None,
        job_id: str | None = None,
        experiment_id: str | None = None,
        run_id: str | None = None,
        protocol_id: str | None = None,
        generation_id: str | None = None,
    ) -> list[Dict[str, Any]]:
        maximum = max(1, min(int(limit), 2000))
        filters = {
            "level": level,
            "component": component,
            "trace_id": trace_id,
            "batch_id": batch_id,
            "job_id": job_id,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "protocol_id": protocol_id,
            "generation_id": generation_id,
        }
        rows: list[Dict[str, Any]] = []
        with self._lock:
            files = list(reversed(self._files()))
            for path in files:
                try:
                    lines: Iterable[str] = reversed(path.read_text(encoding="utf-8").splitlines())
                except OSError:
                    continue
                for line in lines:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if any(value is not None and str(record.get(key)) != str(value) for key, value in filters.items()):
                        continue
                    rows.append(record)
                    if len(rows) >= maximum:
                        return rows
        return rows


def event_log_from_config(config: Mapping[str, Any] | None = None) -> StructuredEventLog:
    if config is None or not isinstance(config.get("logging"), Mapping):
        from fair_agent.core.config import load_config

        fallback = load_config()
        settings = fallback["logging"]
    else:
        settings = config["logging"]
    return StructuredEventLog(
        settings["root"], int(settings["max_file_bytes"]), int(settings["retained_files"])
    )


def mirror_state_event(
    event_log: StructuredEventLog,
    state: str,
    *,
    status: str,
    experiment_id: str,
    run_id: str,
    protocol_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if state == "PARTITIONED" and status.startswith("execution_"):
        event = {
            "execution_started": "experiment.training_pipeline.started",
            "execution_finished": "experiment.training_pipeline.completed",
            "execution_failed": "experiment.training_pipeline.failed",
        }.get(status, "experiment.training_pipeline.changed")
    else:
        event = STATE_EVENTS.get(state, "experiment.state.changed")
    level = "error" if status in {"failed", "error"} or status.endswith("_failed") or state == "REJECTED" else "info"
    return event_log.append(
        event,
        level=level,
        component="incremental",
        experiment_id=experiment_id,
        run_id=run_id,
        protocol_id=protocol_id,
        status=status,
        details={"state": state, **dict(details or {})},
    )
