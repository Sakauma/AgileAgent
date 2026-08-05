from __future__ import annotations

from typing import Dict, Iterable

from fair_agent.core.config import resolve_path
from fair_agent.core.hashes import hash_if_exists


def fingerprints(paths: Iterable[str]) -> Dict[str, Dict[str, object]]:
    """Return file-state records used by the local blackboard snapshot."""
    return {path: hash_if_exists(resolve_path(path)) for path in paths}
