from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.hashes import sha256_file


LINEAGE_SCHEMA_VERSION = 1


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    normalized = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _scan_cache_files(settings: Mapping[str, Any]) -> list[Dict[str, Any]]:
    lineage = settings.get("lineage") if isinstance(settings.get("lineage"), Mapping) else {}
    rows = []
    for raw_root in lineage.get("cache_roots", []):
        root = resolve_path(raw_root)
        if not root.exists():
            continue
        rows.extend(
            {
                "path": rel_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(root.rglob("*")) if path.is_file()
        )
    return rows


def _validate_catalog(payload: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    if int(payload.get("schema_version") or 0) != LINEAGE_SCHEMA_VERSION:
        raise ValueError(f"数据血缘目录版本不受支持：{path}")
    expected = str(payload.get("manifest_sha256") or "")
    actual = _canonical_sha256(payload)
    if expected != actual:
        raise ValueError(f"数据血缘目录哈希不匹配：{path}")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError(f"数据血缘目录缺少逐文件记录：{path}")
    cache_files = payload.get("cache_files", [])
    if not isinstance(cache_files, list):
        raise ValueError(f"数据血缘目录的缓存记录非法：{path}")
    return dict(payload)


def load_lineage_catalogs(settings: Mapping[str, Any]) -> list[Dict[str, Any]]:
    lineage = settings.get("lineage")
    if not isinstance(lineage, Mapping):
        return []
    root = resolve_path(lineage["root"])
    paths: list[Path] = []
    base_manifest = resolve_path(lineage["base_manifest"])
    if base_manifest.is_file():
        paths.append(base_manifest)
    history = root / "accepted"
    if history.is_dir():
        paths.extend(sorted(history.glob("*.json")))
    catalogs = []
    for path in paths:
        catalogs.append(_validate_catalog(json.loads(path.read_text(encoding="utf-8")), path))
    return catalogs


def _split_records(
    split_path: Path,
    *,
    split_name: str,
    source_scope: str,
    round_id: str | None = None,
) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    for raw in split_path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        image = resolve_path(value)
        label = image.with_suffix(".txt")
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"4+2 血缘文件不存在：{image} / {label}")
        row: Dict[str, Any] = {
            "stem": image.stem,
            "image": value,
            "label": str(Path(value).with_suffix(".txt")).replace("\\", "/"),
            "image_sha256": sha256_file(image),
            "label_sha256": sha256_file(label),
            "split": split_name,
            "source_scope": source_scope,
        }
        if round_id is not None:
            row["round_id"] = round_id
        records.append(row)
    return records


def build_base_lineage(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Create the immutable catalog for the learned 4+2 train/dev data."""
    lineage = settings.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("未配置增量数据血缘目录。")
    target = resolve_path(lineage["base_manifest"])
    if target.exists():
        return _validate_catalog(json.loads(target.read_text(encoding="utf-8")), target)
    source = resolve_path(lineage["base_split_manifest"])
    if not source.is_file():
        raise FileNotFoundError(f"4+2 划分清单不存在：{source}")
    split_manifest = json.loads(source.read_text(encoding="utf-8"))
    if split_manifest.get("protocol") != "competition_score_priority_strict_4plus2_partition":
        raise ValueError(f"4+2 划分协议不匹配：{source}")
    lists = split_manifest.get("lists")
    if not isinstance(lists, Mapping):
        raise ValueError(f"4+2 划分清单缺少 lists：{source}")

    def split_file(name: str) -> Path:
        entry = lists.get(name)
        if not isinstance(entry, Mapping) or not entry.get("path"):
            raise ValueError(f"4+2 划分缺少列表：{name}")
        return resolve_path(str(entry["path"]))

    files = _split_records(
        split_file("base_train_plus_dev"),
        split_name="base_train_plus_dev",
        source_scope="base",
    )
    rounds = (split_manifest.get("increment_rounds") or {}).get("rounds") or []
    for round_row in rounds:
        round_id = str(round_row["round_id"])
        for role in ("train", "dev"):
            split_name = f"{round_id}_{role}"
            files.extend(
                _split_records(
                    split_file(split_name),
                    split_name=split_name,
                    source_scope="incremental",
                    round_id=round_id,
                )
            )
    unique_files = {
        (str(item.get("image_sha256")), str(item.get("label_sha256"))): item for item in files
    }
    files = sorted(unique_files.values(), key=lambda item: (str(item.get("split")), str(item.get("stem"))))
    if not files:
        raise ValueError("4+2 划分没有可冻结的 train/dev 文件。")
    payload: Dict[str, Any] = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "catalog_id": "base",
        "kind": "frozen_production_lineage",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": rel_path(source),
        "source_sha256": sha256_file(source),
        "class_ids": sorted(int(value) for value in split_manifest["class_map"]),
        "files": files,
        "cache_files": _scan_cache_files(settings),
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    _atomic_json(target, payload)
    return payload


def ensure_lineage_catalogs(settings: Mapping[str, Any]) -> list[Dict[str, Any]]:
    lineage = settings.get("lineage")
    if not isinstance(lineage, Mapping):
        return []
    catalogs = load_lineage_catalogs(settings)
    if catalogs:
        return catalogs
    if bool(lineage.get("auto_initialize_base")):
        build_base_lineage(settings)
        return load_lineage_catalogs(settings)
    if bool(lineage.get("required")):
        raise ValueError("缺少冻结的基础数据指纹，增量训练已阻止。")
    return []


def audit_incremental_records(
    records: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> Dict[str, Any]:
    catalogs = ensure_lineage_catalogs(settings)
    known_stems = {
        str(row.get("stem") or "").casefold()
        for catalog in catalogs for row in catalog["files"] if row.get("stem")
    }
    known_images = {
        str(row.get("image_sha256"))
        for catalog in catalogs for row in catalog["files"] if row.get("image_sha256")
    }
    known_labels = {
        str(row.get("label_sha256"))
        for catalog in catalogs for row in catalog["files"] if row.get("label_sha256")
    }
    known_caches = {
        str(row.get("sha256"))
        for catalog in catalogs for row in catalog.get("cache_files", []) if row.get("sha256")
    }
    old_images = []
    old_labels = []
    for row in records:
        stem = Path(str(row["image"])).stem.casefold()
        if stem in known_stems or str(row.get("image_sha256")) in known_images:
            old_images.append(str(row["image"]))
        if row.get("label") and str(row.get("label_sha256")) in known_labels:
            old_labels.append(str(row["label"]))

    lineage = settings.get("lineage") if isinstance(settings.get("lineage"), Mapping) else {}
    cache_files = _scan_cache_files(settings)
    old_cache_files = [row for row in cache_files if row["sha256"] in known_caches]
    unverified_cache_files = [row for row in cache_files if row["sha256"] not in known_caches]
    evidence = "current" if catalogs else "missing"
    compliant = (
        evidence == "current"
        and not old_images
        and not old_labels
        and not old_cache_files
        and not unverified_cache_files
    )
    if not lineage or not bool(lineage.get("required")):
        compliant = not old_images and not old_labels and not old_cache_files and not unverified_cache_files
        evidence = "not_required" if not catalogs else "current"
    return {
        "lineage_evidence": evidence,
        "lineage_catalog_count": len(catalogs),
        "lineage_catalog_hashes": [catalog["manifest_sha256"] for catalog in catalogs],
        "old_raw_image_count": len(set(old_images)),
        "old_raw_label_count": len(set(old_labels)),
        "old_cache_count": len(old_cache_files),
        "unverified_cache_count": len(unverified_cache_files),
        "old_raw_images": sorted(set(old_images)),
        "old_raw_labels": sorted(set(old_labels)),
        "old_cache_files": old_cache_files,
        "unverified_cache_files": unverified_cache_files,
        "compliance": "passed" if compliant else "blocked",
    }


def freeze_accepted_batch(
    settings: Mapping[str, Any],
    batch_id: str,
    generation_id: str,
    dataset_fingerprint: str,
    records: Iterable[Mapping[str, Any]],
) -> Path | None:
    lineage = settings.get("lineage")
    if not isinstance(lineage, Mapping):
        return None
    target = resolve_path(lineage["root"]) / "accepted" / f"{generation_id}.json"
    if target.exists():
        raise FileExistsError(f"已存在同名增量血缘记录：{target}")
    files = [
        {
            "stem": Path(str(row["image"])).stem,
            "image_sha256": row["image_sha256"],
            "label_sha256": row.get("label_sha256"),
            "source_scope": "incremental",
            "generation_id": generation_id,
            "batch_id": batch_id,
        }
        for row in records
    ]
    payload: Dict[str, Any] = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "catalog_id": generation_id,
        "kind": "accepted_incremental_batch",
        "batch_id": batch_id,
        "generation_id": generation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": dataset_fingerprint,
        "files": files,
        "cache_files": _scan_cache_files(settings),
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    _atomic_json(target, payload)
    return target
