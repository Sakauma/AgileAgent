from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"exists": False, "sha256": None, "size_bytes": None}
    return {"exists": True, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def verify_sha256s(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"valid": False, "checked": 0, "errors": ["checksum_file_missing"]}
    errors = []
    checked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"invalid_line:{line}")
            continue
        expected, name = parts[0], parts[1].strip().lstrip("*")
        target = path.parent / name
        checked += 1
        if not target.exists():
            errors.append(f"missing:{name}")
        elif sha256_file(target) != expected:
            errors.append(f"mismatch:{name}")
    return {"valid": checked > 0 and not errors, "checked": checked, "errors": errors}
