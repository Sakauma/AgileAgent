from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Sequence

import yaml
from PIL import Image

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.runtime_log import StructuredEventLog, new_trace_id, utc_now
from fair_agent.modules.incremental_lineage import audit_incremental_records


TERMINAL_JOB_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "PROMOTED", "REJECTED", "ACCEPTED", "ROLLED_BACK",
    "ROLLBACK_FAILED",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def _slug(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "-" for character in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:40] or "incremental-batch"


def _class_names_from_yaml(root: Path) -> Dict[int, str]:
    for name in ("data.yaml", "dataset.yaml", "batch.yaml"):
        matches = sorted(root.rglob(name))
        for path in matches:
            try:
                payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError, UnicodeDecodeError):
                continue
            names = payload.get("names")
            if isinstance(names, list):
                return {index: str(value) for index, value in enumerate(names)}
            if isinstance(names, Mapping):
                try:
                    return {int(key): str(value) for key, value in names.items()}
                except (TypeError, ValueError):
                    continue
    return {}


def _parse_override_names(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


class IncrementalBatchStore:
    """Persistent, append-only store for uploaded incremental learning batches."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        event_log: StructuredEventLog,
        active_classes: Mapping[int, str] | Sequence[str],
        known_classes: Mapping[int, str] | None = None,
    ) -> None:
        self.settings = dict(settings)
        self.root = resolve_path(self.settings["root"])
        self.event_log = event_log
        if isinstance(active_classes, Mapping):
            self.active_class_map = {int(class_id): str(name).strip() for class_id, name in active_classes.items()}
        else:
            self.active_class_map = {index: str(name).strip() for index, name in enumerate(active_classes)}
        self.active_classes = {name.lower() for name in self.active_class_map.values()}
        self.known_class_map = {
            int(class_id): str(name).strip()
            for class_id, name in (known_classes or self.active_class_map).items()
        }
        self.active_class_ids_by_name = {
            name.lower(): class_id for class_id, name in self.active_class_map.items()
        }
        self.known_class_ids_by_name = {
            name.casefold(): class_id for class_id, name in self.known_class_map.items()
        }
        self._lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    def _batch_dir(self, batch_id: str) -> Path:
        if not batch_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in batch_id.lower()):
            raise ValueError("增量批次编号无效。")
        path = (self.root / batch_id).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("增量批次路径越界。")
        return path

    def _manifest_path(self, batch_id: str) -> Path:
        return self._batch_dir(batch_id) / "batch_manifest.json"

    def _class_registry_path(self, batch_id: str) -> Path:
        return self._batch_dir(batch_id) / "class_registry.yaml"

    def _write_class_registry(
        self,
        batch_id: str,
        bindings: Sequence[Mapping[str, Any]],
        revision: int,
        reason: str,
    ) -> Dict[str, Any]:
        payload = {
            "schema_version": 1,
            "batch_id": batch_id,
            "revision": int(revision),
            "updated_at": utc_now(),
            "reason": reason,
            "bindings": [dict(item) for item in bindings],
        }
        history = self._batch_dir(batch_id) / "class_registry_history" / f"revision-{revision:04d}.yaml"
        if history.exists():
            raise FileExistsError(f"类别注册表版本已存在：{history.name}")
        _atomic_yaml(history, payload)
        current = self._class_registry_path(batch_id)
        _atomic_yaml(current, payload)
        return {
            "path": "class_registry.yaml",
            "revision": int(revision),
            "sha256": _sha256_file(current),
            "history_path": f"class_registry_history/{history.name}",
            "history_sha256": _sha256_file(history),
        }

    def _load(self, batch_id: str) -> Dict[str, Any]:
        path = self._manifest_path(batch_id)
        if not path.is_file():
            raise KeyError(batch_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def get(self, batch_id: str, include_files: bool = True) -> Dict[str, Any]:
        payload = self._load(batch_id)
        if not include_files:
            payload.pop("files", None)
        return payload

    def list(self) -> list[Dict[str, Any]]:
        rows = []
        for path in sorted(self.root.glob("*/batch_manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            payload.pop("files", None)
            rows.append(payload)
        return rows

    def _reserved_class_registry(self) -> tuple[set[int], Dict[str, int], int]:
        reserved_ids: set[int] = set()
        names_to_ids: Dict[str, int] = {}
        highest_generated_name = len(self.active_class_map)
        for path in self.root.glob("*/batch_manifest.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("status") == "REJECTED":
                continue
            for item in payload.get("audit", {}).get("class_bindings", []):
                try:
                    global_id = int(item["global_class_id"])
                    display_name = str(item["display_name"]).strip()
                except (KeyError, TypeError, ValueError):
                    continue
                reserved_ids.add(global_id)
                if display_name:
                    names_to_ids.setdefault(display_name.casefold(), global_id)
                    suffix = display_name.removeprefix("类别")
                    if display_name.startswith("类别") and suffix.isdigit():
                        highest_generated_name = max(highest_generated_name, int(suffix))
        return reserved_ids, names_to_ids, highest_generated_name

    def _safe_extract(self, archive: Path, destination: Path) -> None:
        maximum_files = int(self.settings["max_extracted_files"])
        maximum_bytes = int(self.settings["max_extracted_bytes"])
        total_bytes = 0
        seen: set[str] = set()
        with zipfile.ZipFile(archive) as handle:
            members = [item for item in handle.infolist() if not item.is_dir()]
            if not members or len(members) > maximum_files:
                raise ValueError(f"压缩包文件数必须位于1到{maximum_files}之间。")
            for member in members:
                raw_name = member.filename.replace("\\", "/")
                pure = PurePosixPath(raw_name)
                if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                    raise ValueError("压缩包包含不安全路径。")
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("压缩包不能包含符号链接。")
                normalized = pure.as_posix().lower()
                if normalized in seen:
                    raise ValueError("压缩包包含重复路径。")
                seen.add(normalized)
                total_bytes += int(member.file_size)
                if total_bytes > maximum_bytes:
                    raise ValueError("压缩包解压后超过容量限制。")
                target = destination.joinpath(*pure.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)

    @staticmethod
    def _label_for(image: Path, labels_by_stem: Mapping[str, list[Path]]) -> Path | None:
        matches = labels_by_stem.get(image.stem.lower(), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            image_parts = [part.lower() for part in image.parts]
            ranked = sorted(matches, key=lambda path: sum(part.lower() in image_parts for part in path.parts), reverse=True)
            if len(ranked) == 1 or ranked[0].parent != ranked[1].parent:
                return ranked[0]
            raise ValueError(f"图像 {image.name} 对应多个同名标签。")
        return None

    def _audit(self, batch_dir: Path, override_names: str | None) -> Dict[str, Any]:
        extracted = batch_dir / "extracted"
        allowed_suffixes = {str(value).lower() for value in self.settings["allowed_image_extensions"]}
        images = sorted(path for path in extracted.rglob("*") if path.is_file() and path.suffix.lower() in allowed_suffixes)
        labels = [path for path in extracted.rglob("*.txt") if path.is_file()]
        labels_by_stem: Dict[str, list[Path]] = {}
        for label in labels:
            labels_by_stem.setdefault(label.stem.lower(), []).append(label)
        if len(images) < int(self.settings["minimum_images"]):
            raise ValueError(f"增量数据至少需要{int(self.settings['minimum_images'])}张图像。")
        if not images:
            raise ValueError("压缩包中没有可识别图像。")
        if len({image.stem.lower() for image in images}) != len(images):
            raise ValueError("数据集中存在重复图像 stem，无法建立稳定映射。")

        records: list[Dict[str, Any]] = []
        class_counts: Counter[int] = Counter()
        object_count = 0
        for image in images:
            label = self._label_for(image, labels_by_stem)
            if label is None:
                raise ValueError(f"缺少与 {image.name} 对应的YOLO标签。")
            try:
                with Image.open(image) as opened:
                    width, height = opened.size
                    opened.verify()
            except Exception as exc:
                raise ValueError(f"图像无法解码：{image.name}") from exc
            if width * height > int(self.settings["max_image_pixels"]):
                raise ValueError(f"图像像素数超过限制：{image.name}")
            classes: list[int] = []
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(
                        f"{label.name}:{line_number} 必须使用5列YOLO标签：class x_center y_center width height。"
                    )
                try:
                    class_id = int(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                except ValueError as exc:
                    raise ValueError(f"{label.name}:{line_number} 含非法数值。") from exc
                x_center, y_center, box_width, box_height = coordinates
                outside = (
                    x_center - box_width / 2 < 0 or x_center + box_width / 2 > 1
                    or y_center - box_height / 2 < 0 or y_center + box_height / 2 > 1
                )
                if (
                    class_id < 0 or any(value < 0.0 or value > 1.0 for value in coordinates)
                    or box_width <= 0 or box_height <= 0 or outside
                ):
                    raise ValueError(f"{label.name}:{line_number} 类别或坐标越界。")
                classes.append(class_id)
                class_counts[class_id] += 1
                object_count += 1
            relative = image.relative_to(extracted).as_posix()
            lower_parts = {part.lower() for part in image.parts}
            if lower_parts & {"lock", "sealed", "holdout"}:
                split_hint = "lock"
            elif lower_parts & {"val", "valid", "dev", "validation"}:
                split_hint = "val"
            else:
                split_hint = "train"
            records.append({
                "index": len(records),
                "image": relative,
                "label": label.relative_to(extracted).as_posix(),
                "width": width,
                "height": height,
                "classes": sorted(set(classes)),
                "object_count": len(classes),
                "split_hint": split_hint,
                "image_sha256": _sha256_file(image),
                "label_sha256": _sha256_file(label),
            })
        all_ids = sorted(class_counts)
        if not all_ids:
            raise ValueError("增量数据集中没有有效目标。")

        override_values = _parse_override_names(override_names)
        yaml_names = _class_names_from_yaml(extracted)
        if override_values:
            if len(override_values) != len(all_ids):
                raise ValueError("手动类别名称数量必须与标签中实际类别数量一致。")
            source_names = {source_id: override_values[index] for index, source_id in enumerate(all_ids)}
            semantic_source = "user_override"
        elif yaml_names:
            source_names = {source_id: yaml_names[source_id] for source_id in all_ids if source_id in yaml_names}
            semantic_source = "dataset_names"
        else:
            source_names = {}
            semantic_source = "generated"

        reserved_ids, reserved_names_to_ids, highest_generated_name = self._reserved_class_registry()
        used_global_ids = set(self.known_class_map) | reserved_ids
        known_class_ids_by_name = dict(reserved_names_to_ids)
        known_class_ids_by_name.update(self.known_class_ids_by_name)
        next_global_id = max(used_global_ids, default=-1) + 1
        generated_index = max(len(used_global_ids), highest_generated_name) + 1
        generated: list[str] = []
        bindings: list[Dict[str, Any]] = []
        class_names: Dict[int, str] = {}
        source_class_map: Dict[int, str] = {}
        source_to_training: Dict[int, int] = {}
        local_to_global: Dict[int, int] = {}
        bound_global_ids: set[int] = set()

        for training_id, source_id in enumerate(all_ids):
            supplied_name = str(source_names.get(source_id) or "").strip()
            known_global_id = known_class_ids_by_name.get(supplied_name.casefold()) if supplied_name else None
            if known_global_id is not None:
                global_id = known_global_id
                display_name = supplied_name
                status = "confirmed"
                binding_source = semantic_source
                is_existing = supplied_name.casefold() in self.active_class_ids_by_name
            else:
                if supplied_name and source_id not in used_global_ids and source_id not in bound_global_ids:
                    global_id = source_id
                else:
                    while next_global_id in used_global_ids or next_global_id in bound_global_ids:
                        next_global_id += 1
                    global_id = next_global_id
                    next_global_id += 1
                if supplied_name:
                    display_name = supplied_name
                    status = "confirmed"
                    binding_source = semantic_source
                else:
                    display_name = f"类别{generated_index}"
                    generated_index += 1
                    generated.append(display_name)
                    status = "provisional"
                    binding_source = "generated"
                is_existing = False
            if global_id in bound_global_ids:
                raise ValueError("增量数据中多个本地类别映射到了同一个全局类别。")
            bound_global_ids.add(global_id)
            source_class_map[source_id] = display_name
            source_to_training[source_id] = training_id
            class_names[training_id] = display_name
            local_to_global[training_id] = global_id
            bindings.append({
                "source_class_id": source_id,
                "training_class_id": training_id,
                "global_class_id": global_id,
                "display_name": display_name,
                "semantic_status": status,
                "semantic_source": binding_source,
                "is_existing_class": is_existing,
            })

        mode = "target_incremental" if bindings and all(item["is_existing_class"] for item in bindings) else "class_incremental"
        compliance = audit_incremental_records(records, self.settings)
        return {
            "image_count": len(records),
            "label_count": sum(1 for row in records if row["label"]),
            "object_count": object_count,
            "class_map": {str(key): value for key, value in class_names.items()},
            "source_class_map": {str(key): value for key, value in source_class_map.items()},
            "source_to_training": {str(key): value for key, value in source_to_training.items()},
            "local_to_global": {str(key): value for key, value in local_to_global.items()},
            "class_bindings": bindings,
            "class_counts": {str(key): class_counts[key] for key in all_ids},
            "label_format": "class_id_bbox",
            "incremental_mode": mode,
            "generated_class_names": generated,
            "requires_class_confirmation": bool(generated),
            "has_provisional_class_names": bool(generated),
            **compliance,
            "files": records,
        }

    def _allocate_batch(self) -> tuple[str, str, Path]:
        trace_id = new_trace_id("batch")
        batch_id = f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        batch_dir = self._batch_dir(batch_id)
        with self._lock:
            batch_dir.mkdir(parents=True, exist_ok=False)
        return trace_id, batch_id, batch_dir

    def create(self, filename: str, data: bytes, display_name: str | None = None, class_names: str | None = None) -> Dict[str, Any]:
        if not filename.lower().endswith(".zip"):
            raise ValueError("增量数据集必须上传ZIP压缩包。")
        if not data or len(data) > int(self.settings["max_archive_bytes"]):
            raise ValueError("增量数据压缩包为空或超过容量限制。")
        trace_id, batch_id, batch_dir = self._allocate_batch()
        archive = batch_dir / "source.zip"
        archive.write_bytes(data)
        return self._finalize_upload(filename, display_name, class_names, trace_id, batch_id, batch_dir, len(data), _sha256_bytes(data))

    def create_stream(self, filename: str, source: Any, display_name: str | None = None, class_names: str | None = None) -> Dict[str, Any]:
        if not filename.lower().endswith(".zip"):
            raise ValueError("增量数据集必须上传ZIP压缩包。")
        trace_id, batch_id, batch_dir = self._allocate_batch()
        archive = batch_dir / "source.zip"
        digest = hashlib.sha256()
        size_bytes = 0
        maximum = int(self.settings["max_archive_bytes"])
        try:
            source.seek(0)
            with archive.open("wb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    if size_bytes > maximum:
                        raise ValueError("增量数据压缩包超过容量限制。")
                    digest.update(chunk)
                    output.write(chunk)
            if not size_bytes:
                raise ValueError("增量数据压缩包为空。")
        except Exception:
            shutil.rmtree(batch_dir, ignore_errors=True)
            raise
        return self._finalize_upload(filename, display_name, class_names, trace_id, batch_id, batch_dir, size_bytes, digest.hexdigest())

    def _finalize_upload(
        self,
        filename: str,
        display_name: str | None,
        class_names: str | None,
        trace_id: str,
        batch_id: str,
        batch_dir: Path,
        size_bytes: int,
        archive_sha256: str,
    ) -> Dict[str, Any]:
        archive = batch_dir / "source.zip"
        self.event_log.append("incremental.upload.saved", component="incremental", trace_id=trace_id, batch_id=batch_id, details={"filename": filename, "size_bytes": size_bytes, "sha256": archive_sha256})
        started = time.perf_counter()
        try:
            self._safe_extract(archive, batch_dir / "extracted")
        except Exception as exc:
            extraction_error = str(exc)
        else:
            extraction_error = None
        with self._lock:
            try:
                if extraction_error:
                    raise ValueError(extraction_error)
                audit = self._audit(batch_dir, class_names)
                status = "AUDITED"
                error = None
            except Exception as exc:
                audit = {}
                status = "REJECTED"
                error = str(exc)
            manifest: Dict[str, Any] = {
                "schema_version": 2,
                "batch_id": batch_id,
                "name": (display_name or Path(filename).stem).strip()[:80],
                "status": status,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "source": {"filename": Path(filename).name, "size_bytes": size_bytes, "sha256": archive_sha256},
                "audit": {key: value for key, value in audit.items() if key != "files"},
                "files": audit.get("files", []),
                "error": error,
                "trace_id": trace_id,
                "training_job_id": None,
            }
            if status == "AUDITED":
                manifest["class_registry"] = self._write_class_registry(
                    batch_id,
                    audit["class_bindings"],
                    revision=1,
                    reason="dataset_audit",
                )
            _atomic_json(self._manifest_path(batch_id), manifest)
        self.event_log.append(
            "incremental.audit.completed" if status == "AUDITED" else "incremental.audit.failed",
            level="info" if status == "AUDITED" else "error",
            component="incremental",
            trace_id=trace_id,
            batch_id=batch_id,
            duration_ms=(time.perf_counter() - started) * 1000,
            message=error or "增量数据审计通过",
            details=manifest["audit"],
        )
        self.event_log.append(
            "incremental.data_audit.completed" if status == "AUDITED" else "incremental.data_audit.failed",
            level="info" if status == "AUDITED" else "error",
            component="incremental",
            trace_id=trace_id,
            batch_id=batch_id,
            message=error or "增量数据审计通过",
            details={
                **manifest["audit"],
                "source_archive_sha256": archive_sha256,
                "manifest": rel_path(self._manifest_path(batch_id)),
            },
        )
        return manifest

    def image_path(self, batch_id: str, index: int) -> Path:
        manifest = self._load(batch_id)
        files = manifest.get("files", [])
        if not 0 <= int(index) < len(files):
            raise IndexError(index)
        path = (self._batch_dir(batch_id) / "extracted" / files[int(index)]["image"]).resolve()
        if (self._batch_dir(batch_id) / "extracted").resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        return path

    def inject(self, batch_id: str) -> Dict[str, Any]:
        with self._lock:
            manifest = self._load(batch_id)
            if manifest["status"] not in {"AUDITED", "INJECTED"}:
                raise ValueError("只有审计通过的批次可以注入。")
            if manifest["status"] == "INJECTED":
                return manifest
            audit = manifest.get("audit", {})
            current_compliance = audit_incremental_records(manifest["files"], self.settings)
            manifest["audit"].update(current_compliance)
            _atomic_json(self._manifest_path(batch_id), manifest)
            if current_compliance.get("compliance") != "passed":
                raise ValueError("增量数据血缘审计未通过，禁止生成训练视图。")
            batch_dir = self._batch_dir(batch_id)
            prepared = batch_dir / "prepared"
            if prepared.exists():
                raise ValueError("训练视图已存在但状态不一致，请检查批次日志。")
            records = list(manifest["files"])
            has_val = any(row["split_hint"] == "val" for row in records)
            has_lock = any(row["split_hint"] == "lock" for row in records)
            split_rows: Dict[str, list[Dict[str, Any]]] = {"train": [], "val": [], "lock": []}
            validation_fraction = float(self.settings["validation_fraction"])
            for row in records:
                split_rows[row["split_hint"]].append(row)

            lock_fraction = self.settings.get("lock_fraction")
            split_seed = int(self.settings.get("split_seed", self.settings.get("training", {}).get("seed", 20260705)))
            if lock_fraction is not None and not has_lock:
                selected = self._stratified_holdout(
                    split_rows["train"], float(lock_fraction), split_seed, "lock"
                )
                selected_ids = {id(row) for row in selected}
                split_rows["train"] = [row for row in split_rows["train"] if id(row) not in selected_ids]
                split_rows["lock"] = selected

            if not has_val:
                selected = self._stratified_holdout(
                    split_rows["train"], validation_fraction, split_seed + 1, "dev"
                )
                selected_ids = {id(row) for row in selected}
                split_rows["train"] = [row for row in split_rows["train"] if id(row) not in selected_ids]
                split_rows["val"] = selected
            elif not split_rows["train"]:
                raise ValueError("增量数据没有可用于训练的样本。")

            required_splits = ["train", "val"] + (["lock"] if lock_fraction is not None or has_lock else [])
            all_classes = {int(value) for row in records for value in row["classes"]}
            for split in required_splits:
                present = {int(value) for row in split_rows[split] for value in row["classes"]}
                missing = sorted(all_classes - present)
                if missing:
                    raise ValueError(f"{split}划分缺少类别{missing}，无法完成逐类训练、校准和独立复核。")
            extracted = batch_dir / "extracted"
            source_to_training = {
                int(key): int(value) for key, value in manifest["audit"]["source_to_training"].items()
            }
            for split, rows in split_rows.items():
                for row in rows:
                    image_source = extracted / row["image"]
                    label_source = extracted / row["label"]
                    target_root = batch_dir / "sealed_lock" if split == "lock" else prepared
                    image_target = target_root / "images" / split / f"{image_source.stem}{image_source.suffix.lower()}"
                    label_target = target_root / "labels" / split / f"{image_source.stem}.txt"
                    image_target.parent.mkdir(parents=True, exist_ok=True)
                    label_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_source, image_target)
                    normalized_lines = []
                    for line in label_source.read_text(encoding="utf-8").splitlines():
                        fields = line.split()
                        if not fields:
                            continue
                        if len(fields) == 4:
                            source_class_id, coordinates = 0, fields
                        else:
                            source_class_id, coordinates = int(fields[0]), fields[1:]
                        normalized_lines.append(
                            f"{source_to_training[source_class_id]} {' '.join(coordinates)}"
                        )
                    label_target.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")
            class_map = {int(key): value for key, value in manifest["audit"]["class_map"].items()}
            dataset_yaml = {
                "path": str(prepared.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": class_map,
            }
            (prepared / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
            assignment = {
                split: [
                    {
                        "stem": Path(str(row["image"])).stem,
                        "image_sha256": row["image_sha256"],
                        "label_sha256": row.get("label_sha256"),
                    }
                    for row in rows
                ]
                for split, rows in split_rows.items()
            }
            dataset_fingerprint = _sha256_bytes(json.dumps(
                {
                    "source_archive_sha256": manifest["source"]["sha256"],
                    "assignment": assignment,
                    "local_to_global": manifest["audit"]["local_to_global"],
                    "lineage_catalog_hashes": manifest["audit"].get("lineage_catalog_hashes", []),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"))
            lock_manifest = {
                "schema_version": 1,
                "batch_id": batch_id,
                "sealed_at": utc_now(),
                "seed": split_seed,
                "fraction": float(lock_fraction) if lock_fraction is not None else None,
                "auto_sealed": not has_lock and lock_fraction is not None,
                "dataset_fingerprint": dataset_fingerprint,
                "local_to_global": manifest["audit"]["local_to_global"],
                "files": assignment["lock"],
                "content_read_by_training": False,
            }
            _atomic_json(batch_dir / "sealed_lock" / "lock_manifest.json", lock_manifest)
            internal_batch = {
                "schema_version": 2,
                "batch_id": batch_id,
                "incremental_mode": manifest["audit"]["incremental_mode"],
                "learning_data_scope": "incremental_dataset_only",
                "old_raw_image_count": int(manifest["audit"]["old_raw_image_count"]),
                "old_raw_label_count": int(manifest["audit"].get("old_raw_label_count", 0)),
                "old_cache_count": int(manifest["audit"].get("old_cache_count", 0)),
                "lineage_evidence": manifest["audit"].get("lineage_evidence"),
                "dataset": "prepared/dataset.yaml",
                "sealed_lock_manifest": rel_path(batch_dir / "sealed_lock" / "lock_manifest.json"),
                "dataset_fingerprint": dataset_fingerprint,
                "class_map": manifest["audit"]["class_map"],
                "source_class_map": manifest["audit"]["source_class_map"],
                "source_to_training": manifest["audit"]["source_to_training"],
                "local_to_global": manifest["audit"]["local_to_global"],
                "class_bindings": manifest["audit"]["class_bindings"],
                "counts": {
                    split: len(rows) for split, rows in split_rows.items()
                    if split != "lock" or lock_fraction is not None or has_lock
                },
                "training_access": {
                    "allowed_splits": ["train", "val"],
                    "lock_content_available": False,
                },
                "source_archive_sha256": manifest["source"]["sha256"],
            }
            (batch_dir / "batch.yaml").write_text(yaml.safe_dump(internal_batch, allow_unicode=True, sort_keys=False), encoding="utf-8")
            manifest["status"] = "INJECTED"
            manifest["updated_at"] = utc_now()
            manifest["injection"] = internal_batch
            _atomic_json(self._manifest_path(batch_id), manifest)
        self.event_log.append("incremental.inject.completed", component="incremental", trace_id=manifest["trace_id"], batch_id=batch_id, details=manifest["injection"])
        batch_yaml = self._batch_dir(batch_id) / "batch.yaml"
        dataset_yaml = self._batch_dir(batch_id) / "prepared" / "dataset.yaml"
        self.event_log.append(
            "incremental.data_view.generated", component="incremental",
            trace_id=manifest["trace_id"], batch_id=batch_id,
            details={
                **manifest["injection"],
                "batch_yaml": rel_path(batch_yaml),
                "batch_yaml_sha256": _sha256_file(batch_yaml),
                "dataset_yaml": rel_path(dataset_yaml),
                "dataset_yaml_sha256": _sha256_file(dataset_yaml),
            },
        )
        return manifest

    @staticmethod
    def _stratified_holdout(
        rows: Sequence[Dict[str, Any]],
        fraction: float,
        seed: int,
        purpose: str,
    ) -> list[Dict[str, Any]]:
        if not rows:
            raise ValueError(f"没有可用于自动封存{purpose}的数据。")
        classes = {int(value) for row in rows for value in row["classes"]}
        counts = Counter(int(value) for row in rows for value in set(row["classes"]))
        too_small = sorted(class_id for class_id in classes if counts[class_id] < 2)
        if too_small:
            raise ValueError(f"类别{too_small}样本不足，无法同时覆盖{purpose}与剩余训练集。")
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['image_sha256']}".encode("utf-8")
            ).hexdigest(),
        )
        target = max(1, int(math.ceil(len(rows) * fraction)), len(classes))
        selected: list[Dict[str, Any]] = []
        selected_ids: set[int] = set()
        remaining = Counter(counts)
        grouped: Dict[tuple[int, ...], list[Dict[str, Any]]] = {}
        order_index = {id(row): index for index, row in enumerate(ordered)}
        for row in ordered:
            grouped.setdefault(tuple(sorted({int(value) for value in row["classes"]})), []).append(row)
        selected_by_group: Counter[tuple[int, ...]] = Counter()
        desired_by_group = {
            key: len(group_rows) * fraction for key, group_rows in grouped.items()
        }

        def can_select(row: Mapping[str, Any]) -> bool:
            return all(remaining[int(class_id)] > 1 for class_id in set(row["classes"]))

        def select(row: Dict[str, Any]) -> None:
            key = tuple(sorted({int(value) for value in row["classes"]}))
            selected.append(row)
            selected_ids.add(id(row))
            selected_by_group[key] += 1
            for value in set(row["classes"]):
                remaining[int(value)] -= 1

        for key in sorted(grouped):
            quota = int(math.floor(desired_by_group[key]))
            for row in grouped[key]:
                if selected_by_group[key] >= quota:
                    break
                if can_select(row):
                    select(row)

        covered = {int(value) for row in selected for value in row["classes"]}
        for class_id in sorted(classes - covered):
            candidates = [
                row for row in ordered
                if id(row) not in selected_ids and class_id in row["classes"] and can_select(row)
            ]
            candidate = max(
                candidates,
                key=lambda row: (
                    desired_by_group[tuple(sorted({int(value) for value in row["classes"]}))]
                    - selected_by_group[tuple(sorted({int(value) for value in row["classes"]}))],
                    -order_index[id(row)],
                ),
                default=None,
            )
            if candidate is None:
                raise ValueError(f"类别{class_id}无法在{purpose}和训练集中同时保留。")
            select(candidate)
        while len(selected) < target:
            candidates = [
                row for row in ordered if id(row) not in selected_ids and can_select(row)
            ]
            if not candidates:
                break
            candidate = max(
                candidates,
                key=lambda row: (
                    desired_by_group[tuple(sorted({int(value) for value in row["classes"]}))]
                    - selected_by_group[tuple(sorted({int(value) for value in row["classes"]}))],
                    -order_index[id(row)],
                ),
            )
            select(candidate)
        return selected

    def rename_classes(self, batch_id: str, names: Mapping[str | int, str]) -> Dict[str, Any]:
        with self._lock:
            manifest = self._load(batch_id)
            if manifest["status"] not in {"AUDITED", "INJECTED", "FAILED", "TRAINED_CANDIDATE"}:
                raise ValueError("当前批次状态不允许修改类别名称。")
            bindings = list(manifest.get("audit", {}).get("class_bindings") or [])
            if not bindings:
                raise ValueError("当前批次没有可重命名的类别绑定。")
            updates = {int(class_id): str(value).strip() for class_id, value in names.items()}
            known_source_ids = {int(item["source_class_id"]) for item in bindings}
            if not updates or not set(updates).issubset(known_source_ids):
                raise ValueError("类别重命名包含未知的源类别ID。")
            if any(not value or len(value) > 80 for value in updates.values()):
                raise ValueError("类别名称不能为空且不能超过80个字符。")

            candidate_names = {
                int(item["source_class_id"]): updates.get(int(item["source_class_id"]), str(item["display_name"]))
                for item in bindings
            }
            normalized = [value.casefold() for value in candidate_names.values()]
            if len(normalized) != len(set(normalized)):
                raise ValueError("同一批次中的类别名称不能重复。")
            for item in bindings:
                source_id = int(item["source_class_id"])
                if source_id not in updates:
                    continue
                active_id = self.active_class_ids_by_name.get(updates[source_id].casefold())
                if active_id is not None and active_id != int(item["global_class_id"]):
                    raise ValueError("重命名不能把新增类别合并到已有类别；请使用带官方类别映射的数据包重新审计。")
                item["display_name"] = updates[source_id]
                item["semantic_status"] = "confirmed"
                item["semantic_source"] = "user_rename"

            audit = manifest["audit"]
            audit["class_bindings"] = bindings
            audit["source_class_map"] = {
                str(item["source_class_id"]): item["display_name"] for item in bindings
            }
            audit["class_map"] = {
                str(item["training_class_id"]): item["display_name"] for item in bindings
            }
            audit["generated_class_names"] = [
                item["display_name"] for item in bindings if item["semantic_status"] == "provisional"
            ]
            audit["requires_class_confirmation"] = bool(audit["generated_class_names"])
            audit["has_provisional_class_names"] = bool(audit["generated_class_names"])
            manifest["updated_at"] = utc_now()
            registry_revision = int(manifest.get("class_registry", {}).get("revision") or 0) + 1
            manifest["class_registry"] = self._write_class_registry(
                batch_id,
                bindings,
                revision=registry_revision,
                reason="user_rename",
            )

            batch_dir = self._batch_dir(batch_id)
            if manifest["status"] in {"INJECTED", "FAILED", "TRAINED_CANDIDATE"}:
                dataset_path = batch_dir / "prepared" / "dataset.yaml"
                batch_path = batch_dir / "batch.yaml"
                if dataset_path.is_file():
                    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}
                    dataset["names"] = {int(key): value for key, value in audit["class_map"].items()}
                    dataset_path.write_text(yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False), encoding="utf-8")
                if batch_path.is_file():
                    batch_payload = yaml.safe_load(batch_path.read_text(encoding="utf-8")) or {}
                    batch_payload["class_map"] = audit["class_map"]
                    batch_payload["source_class_map"] = audit["source_class_map"]
                    batch_payload["class_bindings"] = bindings
                    batch_path.write_text(
                        yaml.safe_dump(batch_payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
                    )
                if manifest.get("injection"):
                    manifest["injection"]["class_map"] = audit["class_map"]
                    manifest["injection"]["source_class_map"] = audit["source_class_map"]
                    manifest["injection"]["class_bindings"] = bindings
            _atomic_json(self._manifest_path(batch_id), manifest)

        self.event_log.append(
            "incremental.classes.renamed",
            component="incremental",
            trace_id=manifest["trace_id"],
            batch_id=batch_id,
            message="增量类别显示名称已更新",
            details={"class_bindings": bindings},
        )
        return manifest

    def update_training(self, batch_id: str, job_id: str, status: str, **details: Any) -> Dict[str, Any]:
        with self._lock:
            manifest = self._load(batch_id)
            manifest["status"] = status
            manifest["updated_at"] = utc_now()
            manifest["training_job_id"] = job_id
            manifest.setdefault("training", {}).update(details)
            _atomic_json(self._manifest_path(batch_id), manifest)
            return manifest


class TrainingJobManager:
    def __init__(
        self,
        store: IncrementalBatchStore,
        settings: Mapping[str, Any],
        event_log: StructuredEventLog,
        config: Mapping[str, Any] | None = None,
        promotion_callback: Any | None = None,
        rollback_callback: Any | None = None,
    ) -> None:
        self.store = store
        self.settings = dict(settings)
        self.event_log = event_log
        self.config = dict(config) if config is not None else None
        self.promotion_callback = promotion_callback
        self.rollback_callback = rollback_callback
        self._processes: Dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()

    def _job_path(self, batch_id: str, job_id: str) -> Path:
        return self.store._batch_dir(batch_id) / "jobs" / f"{job_id}.json"

    def _write_job(self, job: Mapping[str, Any]) -> None:
        _atomic_json(self._job_path(str(job["batch_id"]), str(job["job_id"])), job)

    def get(self, batch_id: str, job_id: str) -> Dict[str, Any]:
        path = self._job_path(batch_id, job_id)
        if not path.is_file():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, batch_id: str | None = None) -> list[Dict[str, Any]]:
        pattern = f"{batch_id}/jobs/*.json" if batch_id else "*/jobs/*.json"
        rows = []
        for path in self.store.root.glob(pattern):
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda row: row["created_at"], reverse=True)

    def _resolve_training_initialization(
        self, manifest: Mapping[str, Any], train: Mapping[str, Any],
    ) -> Dict[str, Any]:
        configured = resolve_path(train["initial_weights"])
        if not configured.is_file():
            raise FileNotFoundError(f"增量训练初始化权重不存在：{configured}")
        initialization = {
            "parent_generation_id": None,
            "parent_model_id": None,
            "initial_weight": rel_path(configured),
            "initial_weight_sha256": _sha256_file(configured),
            "source": "configured_base",
        }
        if self.config is None:
            return initialization
        from fair_agent.modules.generation_management import active_generation_registry
        from fair_agent.modules.model_generations import load_generation_registry

        registry = load_generation_registry(active_generation_registry(self.config))
        parent_id = str(registry["channels"]["production"])
        parent = registry["generations_by_id"][parent_id]
        initialization["parent_generation_id"] = parent_id
        mode = str(manifest.get("audit", {}).get("incremental_mode") or "")
        global_ids = {
            int(item["global_class_id"])
            for item in manifest.get("audit", {}).get("class_bindings", [])
        }
        if mode == "target_incremental":
            owner_ids = {parent["class_owners"].get(class_id) for class_id in global_ids}
            owner_ids.discard(None)
            if len(owner_ids) != 1:
                raise ValueError("一个目标增量批次的类别必须由同一个当前模型拥有，才能确定初始化权重。")
            owner_id = str(next(iter(owner_ids)))
            owner = registry["models_by_id"][owner_id]
            configured = owner["resolved_path"]
            initialization.update({
                "parent_model_id": owner_id,
                "initial_weight": rel_path(configured),
                "initial_weight_sha256": _sha256_file(configured),
                "source": "current_class_owner",
            })
        else:
            base_ids = [
                model_id for model_id in (parent.get("model_members") or parent["class_owners"].values())
                if registry["models_by_id"][str(model_id)]["role"] == "frozen_base"
            ]
            if len(set(base_ids)) == 1:
                initialization["parent_model_id"] = str(base_ids[0])
        return initialization

    def _create_training_snapshot(
        self, batch_id: str, job_id: str, initialization: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        batch_dir = self.store._batch_dir(batch_id)
        manifest = self.store.get(batch_id)
        if initialization is None:
            initialization = self._resolve_training_initialization(
                manifest, dict(self.settings["training"])
            )
        dataset_source = batch_dir / "prepared" / "dataset.yaml"
        registry_source = self.store._class_registry_path(batch_id)
        batch_source = batch_dir / "batch.yaml"
        lock_manifest = batch_dir / "sealed_lock" / "lock_manifest.json"
        if not dataset_source.is_file() or not registry_source.is_file() or not batch_source.is_file():
            raise FileNotFoundError("训练数据视图或类别注册表不存在。")
        snapshot_dir = batch_dir / "jobs" / "snapshots" / job_id
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        dataset_snapshot = snapshot_dir / "dataset.yaml"
        registry_snapshot = snapshot_dir / "class_registry.yaml"
        batch_snapshot = snapshot_dir / "batch.yaml"
        shutil.copy2(dataset_source, dataset_snapshot)
        shutil.copy2(registry_source, registry_snapshot)
        shutil.copy2(batch_source, batch_snapshot)
        injection = manifest.get("injection", {})
        current_compliance = audit_incremental_records(manifest.get("files", []), self.settings)
        if current_compliance.get("compliance") != "passed":
            raise ValueError("训练启动前的数据血缘复核未通过。")
        if any(int(current_compliance.get(key, -1)) != 0 for key in ("old_raw_image_count", "old_raw_label_count", "old_cache_count", "unverified_cache_count")):
            raise ValueError("训练快照检测到旧数据或旧缓存交集。")
        if current_compliance.get("lineage_evidence") not in {"current", "not_required"}:
            raise ValueError("训练快照缺少有效基础数据血缘证据。")
        payload = {
            "schema_version": 2,
            "batch_id": batch_id,
            "job_id": job_id,
            "created_at": utc_now(),
            "dataset": rel_path(dataset_snapshot),
            "dataset_sha256": _sha256_file(dataset_snapshot),
            "class_registry": rel_path(registry_snapshot),
            "class_registry_sha256": _sha256_file(registry_snapshot),
            "class_registry_revision": int(manifest.get("class_registry", {}).get("revision") or 0),
            "batch": rel_path(batch_snapshot),
            "batch_sha256": _sha256_file(batch_snapshot),
            "dataset_fingerprint": injection.get("dataset_fingerprint"),
            "lineage_catalog_hashes": current_compliance.get("lineage_catalog_hashes", []),
            "old_raw_image_count": int(current_compliance["old_raw_image_count"]),
            "old_raw_label_count": int(current_compliance["old_raw_label_count"]),
            "old_cache_count": int(current_compliance["old_cache_count"]),
            "unverified_cache_count": int(current_compliance["unverified_cache_count"]),
            "lock_manifest": rel_path(lock_manifest) if lock_manifest.is_file() else None,
            "lock_manifest_sha256": _sha256_file(lock_manifest) if lock_manifest.is_file() else None,
            "training_access": injection.get("training_access"),
            "initialization": dict(initialization),
        }
        _atomic_json(snapshot_dir / "snapshot_manifest.json", payload)
        payload["snapshot_manifest"] = rel_path(snapshot_dir / "snapshot_manifest.json")
        payload["snapshot_manifest_sha256"] = _sha256_file(snapshot_dir / "snapshot_manifest.json")
        return payload

    def start(self, batch_id: str, wait: bool = False) -> Dict[str, Any]:
        manifest = self.store.get(batch_id)
        if manifest["status"] not in {"INJECTED", "FAILED"}:
            raise ValueError("批次必须先完成注入，且同一批次不能重复并发训练。")
        job_id = f"train-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        trace_id = new_trace_id("train")
        train = dict(self.settings["training"])
        python = str(train.get("python") or sys.executable)
        initialization = self._resolve_training_initialization(manifest, train)
        snapshot = self._create_training_snapshot(batch_id, job_id, initialization)
        command = [
            python, "-m", "fair_agent.modules.incremental_workbench", "train-worker",
            "--batch-dir", str(self.store._batch_dir(batch_id)),
            "--job-id", job_id,
            "--dataset-snapshot", str(resolve_path(snapshot["dataset"])),
            "--class-registry-snapshot", str(resolve_path(snapshot["class_registry"])),
            "--weights", str(resolve_path(initialization["initial_weight"])),
            "--parent-generation-id", str(initialization.get("parent_generation_id") or ""),
            "--parent-model-id", str(initialization.get("parent_model_id") or ""),
            "--initial-weight-sha256", str(initialization["initial_weight_sha256"]),
            "--device", str(train["device"]),
            "--imgsz", str(train["imgsz"]),
            "--batch", str(train["batch"]),
            "--epochs", str(train["epochs"]),
            "--patience", str(train["patience"]),
            "--workers", str(train["workers"]),
            "--optimizer", str(train["optimizer"]),
            "--lr0", str(train["lr0"]),
            "--seed", str(train["seed"]),
            "--deterministic", str(bool(train["deterministic"])).lower(),
            "--amp", str(bool(train["amp"])).lower(),
        ]
        job = {
            "schema_version": 1,
            "job_id": job_id,
            "batch_id": batch_id,
            "trace_id": trace_id,
            "status": "QUEUED",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "command": command,
            "training_snapshot": snapshot,
            "initialization": initialization,
            "log_url": f"/api/incremental/jobs/{job_id}/logs?batch_id={batch_id}",
        }
        self._write_job(job)
        self.store.update_training(batch_id, job_id, "TRAINING", started_at=job["created_at"])
        if wait:
            self._run(job, command)
            return self.get(batch_id, job_id)
        thread = threading.Thread(target=self._run, args=(job, command), daemon=True, name=job_id)
        thread.start()
        return job

    def _run(self, job: Dict[str, Any], command: list[str]) -> None:
        batch_id, job_id, trace_id = job["batch_id"], job["job_id"], job["trace_id"]
        batch_dir = self.store._batch_dir(batch_id)
        log_path = batch_dir / "jobs" / f"{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        job["status"] = "RUNNING"
        job["started_at"] = utc_now()
        self._write_job(job)
        self.event_log.append("incremental.training.started", component="training", trace_id=trace_id, batch_id=batch_id, job_id=job_id, details={"argv": command})
        started = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8") as output:
                process = subprocess.Popen(command, cwd=resolve_path("."), stdout=output, stderr=subprocess.STDOUT, text=True)
                with self._lock:
                    self._processes[job_id] = process
                returncode = process.wait()
            status = "COMPLETED" if returncode == 0 else "FAILED"
        except Exception as exc:
            returncode = -1
            status = "FAILED"
            log_path.write_text(f"训练任务启动失败：{exc}\n", encoding="utf-8")
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
        if job.get("status") == "CANCELLING":
            status = "CANCELLED"
        job.update({"status": status, "finished_at": utc_now(), "returncode": returncode, "duration_ms": round((time.perf_counter() - started) * 1000, 3)})
        self._write_job(job)
        batch_status = "TRAINED_CANDIDATE" if status == "COMPLETED" else status
        self.store.update_training(batch_id, job_id, batch_status, finished_at=job["finished_at"], returncode=returncode, log=rel_path(log_path))
        self.event_log.append("incremental.training.finished", level="info" if status == "COMPLETED" else "error", component="training", trace_id=trace_id, batch_id=batch_id, job_id=job_id, duration_ms=job["duration_ms"], details={"status": status, "returncode": returncode, "log": rel_path(log_path)})
        self.event_log.append(
            "incremental.training.completed" if status == "COMPLETED" else "incremental.training.failed",
            level="info" if status == "COMPLETED" else "error",
            component="incremental",
            trace_id=trace_id,
            batch_id=batch_id,
            job_id=job_id,
            duration_ms=job["duration_ms"],
            details={"status": status, "returncode": returncode, "log": rel_path(log_path)},
        )
        if status == "COMPLETED":
            candidate_path = batch_dir / "training" / job_id / "candidate_manifest.json"
            if candidate_path.is_file():
                candidate_evidence = json.loads(candidate_path.read_text(encoding="utf-8"))
                self.event_log.append(
                    "incremental.training.artifact_frozen", component="training",
                    trace_id=trace_id, batch_id=batch_id, job_id=job_id,
                    details={
                        "candidate_manifest": rel_path(candidate_path),
                        "candidate_manifest_sha256": _sha256_file(candidate_path),
                        "best_weight": candidate_evidence.get("best_weight"),
                        "best_weight_sha256": candidate_evidence.get("best_weight_sha256"),
                        "dataset_snapshot_sha256": candidate_evidence.get("dataset_snapshot_sha256"),
                        "class_registry_snapshot_sha256": candidate_evidence.get("class_registry_snapshot_sha256"),
                    },
                )
            lifecycle_cfg = self.settings.get("lifecycle")
            if self.config is not None and isinstance(lifecycle_cfg, Mapping) and bool(lifecycle_cfg.get("auto_continue")):
                from fair_agent.modules.incremental_lifecycle import IncrementalLifecycle

                job["lifecycle_status"] = "RUNNING"
                job["status"] = "LIFECYCLE_RUNNING"
                self._write_job(job)
                try:
                    lifecycle_result = IncrementalLifecycle(
                        self.store,
                        self.config,
                        self.event_log,
                        self.promotion_callback,
                        self.rollback_callback,
                    ).run(batch_id, job_id)
                    job["lifecycle_status"] = str(lifecycle_result["status"])
                    job["status"] = str(lifecycle_result["status"])
                    job["lifecycle_result"] = lifecycle_result
                    self._write_job(job)
                except Exception as exc:
                    current_status = str(self.store.get(batch_id, include_files=False).get("status"))
                    failure_status = "ROLLBACK_FAILED" if current_status == "ROLLBACK_FAILED" else "FAILED"
                    job["lifecycle_status"] = failure_status
                    job["status"] = failure_status
                    job["lifecycle_error"] = str(exc)
                    self._write_job(job)
                    self.store.update_training(
                        batch_id, job_id, failure_status, lifecycle_error=str(exc),
                        lifecycle_error_type=type(exc).__name__,
                    )
                    self.event_log.append(
                        "incremental.lifecycle.failed", level="error", component="incremental",
                        trace_id=trace_id, batch_id=batch_id, job_id=job_id,
                        message=str(exc), details={"error_type": type(exc).__name__},
                    )
            else:
                self.event_log.append(
                    "incremental.dev_calibration.pending", level="warning", component="incremental",
                    trace_id=trace_id, batch_id=batch_id, job_id=job_id,
                    message="候选权重已生成，等待dev阈值校准。",
                )

    def cancel(self, batch_id: str, job_id: str) -> Dict[str, Any]:
        job = self.get(batch_id, job_id)
        if job["status"] in TERMINAL_JOB_STATES:
            return job
        with self._lock:
            process = self._processes.get(job_id)
            if process is None:
                raise ValueError("训练进程尚未启动或已不在当前服务进程中。")
            job["status"] = "CANCELLING"
            self._write_job(job)
            process.terminate()
        self.event_log.append("incremental.training.cancel_requested", level="warning", component="training", trace_id=job["trace_id"], batch_id=batch_id, job_id=job_id)
        return job

    def read_log(self, batch_id: str, job_id: str, tail_lines: int = 300) -> str:
        self.get(batch_id, job_id)
        path = self.store._batch_dir(batch_id) / "jobs" / f"{job_id}.log"
        if not path.is_file():
            return ""
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, min(tail_lines, 2000)):])


def train_worker(arguments: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--dataset-snapshot", required=True)
    parser.add_argument("--class-registry-snapshot", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--parent-generation-id", default="")
    parser.add_argument("--parent-model-id", default="")
    parser.add_argument("--initial-weight-sha256", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--lr0", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--deterministic", choices=["true", "false"], required=True)
    parser.add_argument("--amp", choices=["true", "false"], required=True)
    args = parser.parse_args(arguments)
    batch_dir = Path(args.batch_dir).resolve()
    dataset = Path(args.dataset_snapshot).resolve()
    class_registry = Path(args.class_registry_snapshot).resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if not class_registry.is_file():
        raise FileNotFoundError(class_registry)
    initial_weight = Path(args.weights).resolve()
    if not initial_weight.is_file() or _sha256_file(initial_weight) != args.initial_weight_sha256:
        raise ValueError("训练初始化权重缺失或启动前哈希不一致。")
    registry_payload = yaml.safe_load(class_registry.read_text(encoding="utf-8")) or {}
    from ultralytics import YOLO

    model = YOLO(args.weights)
    result = model.train(
        data=str(dataset), project=str(batch_dir / "training"), name=args.job_id,
        exist_ok=False, device=args.device, imgsz=args.imgsz, batch=args.batch,
        epochs=args.epochs, patience=args.patience, workers=args.workers,
        optimizer=args.optimizer, lr0=args.lr0, seed=args.seed,
        deterministic=args.deterministic == "true", amp=args.amp == "true",
    )
    save_dir = Path(result.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("训练完成但未生成best.pt")
    initial_weight_sha256_after = _sha256_file(initial_weight)
    if initial_weight_sha256_after != args.initial_weight_sha256:
        raise RuntimeError("训练过程修改了冻结的初始化权重。")
    output = {
        "job_id": args.job_id,
        "completed_at": utc_now(),
        "best_weight": str(best),
        "best_weight_sha256": _sha256_file(best),
        "parent_generation_id": args.parent_generation_id or None,
        "parent_model_id": args.parent_model_id or None,
        "initial_weight": str(initial_weight),
        "initial_weight_sha256": args.initial_weight_sha256,
        "initial_weight_sha256_after": initial_weight_sha256_after,
        "frozen_source_unchanged": True,
        "dataset_snapshot": str(dataset),
        "dataset_snapshot_sha256": _sha256_file(dataset),
        "class_registry_snapshot": str(class_registry),
        "class_registry_snapshot_sha256": _sha256_file(class_registry),
        "class_registry_revision": int(registry_payload.get("revision") or 0),
        "class_bindings": registry_payload.get("bindings", []),
        "status": "candidate_requires_calibration_and_recheck",
    }
    _atomic_json(save_dir / "candidate_manifest.json", output)
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train-worker":
        raise SystemExit(train_worker(sys.argv[2:]))
    raise SystemExit("仅支持train-worker子命令")
