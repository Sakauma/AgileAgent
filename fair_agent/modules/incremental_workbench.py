from __future__ import annotations

import hashlib
import json
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


TERMINAL_JOB_STATES = {"COMPLETED", "FAILED", "CANCELLED"}


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


def _parse_override_names(raw: str | None) -> Dict[int, str]:
    if not raw:
        return {}
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return {index: value for index, value in enumerate(values)}


class IncrementalBatchStore:
    """Persistent, append-only store for uploaded incremental learning batches."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        event_log: StructuredEventLog,
        active_classes: Sequence[str],
    ) -> None:
        self.settings = dict(settings)
        self.root = resolve_path(self.settings["root"])
        self.event_log = event_log
        self.active_classes = {str(name).strip().lower() for name in active_classes}
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
            if label is None and bool(self.settings["require_labels"]):
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
            if label is not None:
                for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError(f"{label.name}:{line_number} 不是5列YOLO标签。")
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
            split_hint = "val" if lower_parts & {"val", "valid", "dev", "validation"} else "train"
            records.append({
                "index": len(records),
                "image": relative,
                "label": label.relative_to(extracted).as_posix() if label else None,
                "width": width,
                "height": height,
                "classes": sorted(set(classes)),
                "object_count": len(classes),
                "split_hint": split_hint,
                "image_sha256": _sha256_file(image),
                "label_sha256": _sha256_file(label) if label else None,
            })
        names = _parse_override_names(override_names) or _class_names_from_yaml(extracted)
        all_ids = sorted(class_counts)
        if not all_ids:
            raise ValueError("增量数据集中没有有效目标。")
        if all_ids != list(range(len(all_ids))):
            raise ValueError("YOLO本地类别ID必须从0开始连续编号。")
        for class_id in all_ids:
            names.setdefault(class_id, f"new_class_{class_id}")
        if set(names) != set(all_ids):
            names = {class_id: names.get(class_id, f"new_class_{class_id}") for class_id in all_ids}
        class_names = {class_id: names[class_id] for class_id in all_ids}
        existing = [name for name in class_names.values() if name.strip().lower() in self.active_classes]
        generated = [name for name in class_names.values() if name.startswith("new_class_")]
        mode = "target_incremental" if class_names and len(existing) == len(class_names) else "class_incremental"
        return {
            "image_count": len(records),
            "label_count": sum(1 for row in records if row["label"]),
            "object_count": object_count,
            "class_map": {str(key): value for key, value in class_names.items()},
            "class_counts": {str(key): class_counts[key] for key in all_ids},
            "incremental_mode": mode,
            "generated_class_names": generated,
            "requires_class_confirmation": bool(generated),
            "old_raw_image_count": 0,
            "compliance": "passed",
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
            audit = self._audit(batch_dir, class_names)
            status = "AUDITED"
            error = None
        except Exception as exc:
            audit = {}
            status = "REJECTED"
            error = str(exc)
        manifest: Dict[str, Any] = {
            "schema_version": 1,
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
            if manifest["audit"].get("requires_class_confirmation"):
                raise ValueError("数据包缺少可确认的类别名称，请上传含names字段的data.yaml或在上传时填写类别名称。")
            if manifest["status"] == "INJECTED":
                return manifest
            batch_dir = self._batch_dir(batch_id)
            prepared = batch_dir / "prepared"
            if prepared.exists():
                raise ValueError("训练视图已存在但状态不一致，请检查批次日志。")
            records = list(manifest["files"])
            has_val = any(row["split_hint"] == "val" for row in records)
            split_rows: Dict[str, list[Dict[str, Any]]] = {"train": [], "val": []}
            validation_fraction = float(self.settings["validation_fraction"])
            for row in records:
                split = row["split_hint"]
                if not has_val:
                    ratio = int(row["image_sha256"][:8], 16) / 0xFFFFFFFF
                    split = "val" if ratio < validation_fraction else "train"
                split_rows[split].append(row)
            if not split_rows["val"]:
                split_rows["val"].append(split_rows["train"].pop())
            if not split_rows["train"]:
                split_rows["train"].append(split_rows["val"].pop(0))
            extracted = batch_dir / "extracted"
            for split, rows in split_rows.items():
                for row in rows:
                    image_source = extracted / row["image"]
                    label_source = extracted / row["label"]
                    image_target = prepared / "images" / split / f"{image_source.stem}{image_source.suffix.lower()}"
                    label_target = prepared / "labels" / split / f"{image_source.stem}.txt"
                    image_target.parent.mkdir(parents=True, exist_ok=True)
                    label_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_source, image_target)
                    shutil.copy2(label_source, label_target)
            class_map = {int(key): value for key, value in manifest["audit"]["class_map"].items()}
            dataset_yaml = {
                "path": str(prepared.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": class_map,
            }
            (prepared / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
            internal_batch = {
                "schema_version": 1,
                "batch_id": batch_id,
                "incremental_mode": manifest["audit"]["incremental_mode"],
                "learning_data_scope": "incremental_dataset_only",
                "old_raw_image_count": 0,
                "dataset": "prepared/dataset.yaml",
                "class_map": manifest["audit"]["class_map"],
                "counts": {"train": len(split_rows["train"]), "val": len(split_rows["val"])},
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
    def __init__(self, store: IncrementalBatchStore, settings: Mapping[str, Any], event_log: StructuredEventLog) -> None:
        self.store = store
        self.settings = dict(settings)
        self.event_log = event_log
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

    def start(self, batch_id: str) -> Dict[str, Any]:
        manifest = self.store.get(batch_id)
        if manifest["status"] not in {"INJECTED", "FAILED"}:
            raise ValueError("批次必须先完成注入，且同一批次不能重复并发训练。")
        job_id = f"train-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        trace_id = new_trace_id("train")
        train = dict(self.settings["training"])
        python = str(train.get("python") or sys.executable)
        command = [
            python, "-m", "fair_agent.modules.incremental_workbench", "train-worker",
            "--batch-dir", str(self.store._batch_dir(batch_id)),
            "--job-id", job_id,
            "--weights", str(resolve_path(train["initial_weights"])),
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
            "log_url": f"/api/incremental/jobs/{job_id}/logs?batch_id={batch_id}",
        }
        self._write_job(job)
        self.store.update_training(batch_id, job_id, "TRAINING", started_at=job["created_at"])
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
    parser.add_argument("--weights", required=True)
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
    dataset = batch_dir / "prepared" / "dataset.yaml"
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
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
    output = {
        "job_id": args.job_id,
        "completed_at": utc_now(),
        "best_weight": str(best),
        "best_weight_sha256": _sha256_file(best),
        "status": "candidate_requires_calibration_and_recheck",
    }
    _atomic_json(save_dir / "candidate_manifest.json", output)
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train-worker":
        raise SystemExit(train_worker(sys.argv[2:]))
    raise SystemExit("仅支持train-worker子命令")
