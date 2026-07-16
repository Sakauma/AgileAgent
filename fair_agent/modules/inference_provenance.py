from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Mapping

from fair_agent.core.config import resolve_path
from fair_agent.modules.incremental_lineage import load_lineage_catalogs


class InferenceSourceRouter:
    """Resolve whether an input belongs to the base or an accepted incremental domain."""

    def __init__(
        self,
        workbench_settings: Mapping[str, Any],
        *,
        enabled: bool,
        unknown_policy: str,
    ) -> None:
        self.settings = dict(workbench_settings)
        self.enabled = bool(enabled)
        self.unknown_policy = str(unknown_policy)
        self._lock = threading.RLock()
        self._signature: tuple[tuple[str, int, int], ...] | None = None
        self._index: Dict[str, Dict[str, Any]] = {}
        self.refresh(force=True)

    def _catalog_paths(self) -> list[Path]:
        lineage = self.settings.get("lineage")
        if not isinstance(lineage, Mapping):
            return []
        paths = []
        base = resolve_path(lineage["base_manifest"])
        if base.is_file():
            paths.append(base)
        accepted = resolve_path(lineage["root"]) / "accepted"
        if accepted.is_dir():
            paths.extend(sorted(accepted.glob("*.json")))
        return paths

    def _current_signature(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in self._catalog_paths()
        )

    def refresh(self, force: bool = False) -> None:
        with self._lock:
            signature = self._current_signature()
            if not force and signature == self._signature:
                return
            index: Dict[str, Dict[str, Any]] = {}
            for catalog in load_lineage_catalogs(self.settings):
                catalog_id = str(catalog.get("catalog_id") or "unknown")
                kind = str(catalog.get("kind") or "")
                default_scope = "incremental" if kind == "accepted_incremental_batch" else "base"
                for row in catalog["files"]:
                    digest = str(row.get("image_sha256") or "")
                    if not digest:
                        continue
                    scope = str(row.get("source_scope") or default_scope)
                    if scope not in {"base", "incremental"}:
                        raise ValueError(f"数据血缘包含非法推理域：{scope}")
                    decision = {
                        "source_scope": scope,
                        "inference_scope": scope,
                        "incremental_protocol": "auto" if scope == "incremental" else None,
                        "known": True,
                        "rejected": False,
                        "reason": "accepted_incremental_lineage" if scope == "incremental" else "frozen_base_lineage",
                        "catalog_id": catalog_id,
                        "generation_id": row.get("generation_id") or catalog.get("generation_id"),
                        "batch_id": row.get("batch_id") or catalog.get("batch_id"),
                        "round_id": row.get("round_id"),
                    }
                    previous = index.get(digest)
                    if previous is not None and previous["source_scope"] != scope:
                        raise ValueError(f"同一图像同时登记为原始域和增量域：{digest[:12]}")
                    index[digest] = decision
            self._index = index
            self._signature = signature

    def resolve(self, image_sha256: str) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "source_scope": "unrestricted",
                "inference_scope": "incremental",
                "incremental_protocol": "auto",
                "known": False,
                "rejected": False,
                "reason": "source_aware_routing_disabled",
                "catalog_id": None,
                "generation_id": None,
                "batch_id": None,
                "round_id": None,
            }
        self.refresh()
        with self._lock:
            known = self._index.get(str(image_sha256))
            if known is not None:
                return dict(known)
        if self.unknown_policy == "incremental_active":
            inference_scope, protocol, rejected = "incremental", "auto", False
        elif self.unknown_policy == "reject":
            inference_scope, protocol, rejected = "none", None, True
        else:
            inference_scope, protocol, rejected = "base", None, False
        return {
            "source_scope": "unknown",
            "inference_scope": inference_scope,
            "incremental_protocol": protocol,
            "known": False,
            "rejected": rejected,
            "reason": f"unknown_source_{self.unknown_policy}",
            "catalog_id": None,
            "generation_id": None,
            "batch_id": None,
            "round_id": None,
        }


def attach_source_decision(result: Dict[str, Any], source: Mapping[str, Any]) -> None:
    agent = result.setdefault("agent", {})
    decision = agent.setdefault("decision", {})
    decision["source_scope"] = source["source_scope"]
    decision["inference_scope"] = source["inference_scope"]
    decision["source_known"] = bool(source["known"])
    decision["source_reason"] = source["reason"]
    decision["source_generation_id"] = source.get("generation_id")
    decision["source_batch_id"] = source.get("batch_id")
