from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def append_log(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_command(command: List[str], cwd: Path, log_path: Path, name: str, timeout: Optional[int] = None) -> int:
    started = datetime.now().isoformat(timespec="seconds")
    append_log(log_path, {"event": "start", "name": name, "command": command, "time": started})
    try:
        proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        append_log(log_path, {"event": "timeout", "name": name, "time": datetime.now().isoformat(timespec="seconds"), "timeout": timeout, "stdout_tail": str(exc.stdout or "")[-4000:], "stderr_tail": str(exc.stderr or "")[-4000:]})
        return 124
    except Exception as exc:
        append_log(log_path, {"event": "error", "name": name, "time": datetime.now().isoformat(timespec="seconds"), "error": repr(exc)})
        return 1
    append_log(
        log_path,
        {
            "event": "finish",
            "name": name,
            "returncode": proc.returncode,
            "time": datetime.now().isoformat(timespec="seconds"),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        },
    )
    return proc.returncode


def command_exists(command: Iterable[str]) -> bool:
    try:
        proc = subprocess.run(list(command), text=True, capture_output=True, timeout=20)
        return proc.returncode == 0
    except Exception:
        return False
