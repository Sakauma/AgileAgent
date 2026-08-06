from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import yaml
from PIL import Image

from fair_agent.core.config import ROOT, rel_path, resolve_path
from fair_agent.core.runtime_log import StructuredEventLog, event_log_from_config, mirror_state_event
from fair_agent.modules.strict_incremental import image_class_ids, read_split, source_label


STATES = (
    "CREATED",
    "DATA_AUDITED",
    "PARTITIONED",
    "BASE_TRAINED",
    "BASE_FROZEN",
    "INCREMENT_TRAINED",
    "INCREMENT_FROZEN",
    "THRESHOLD_CALIBRATED",
    "LOCK_UNSEALED",
    "EVALUATED",
    "ACCEPTED",
    "REJECTED",
    "REGISTERED",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any], volatile_keys: Iterable[str] = ()) -> str:
    """Hash a JSON payload without wall-clock or self-referential fields."""
    canonical = deepcopy(dict(payload))
    for key in volatile_keys:
        canonical.pop(key, None)
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_experiment_config(path: str | Path) -> Dict[str, Any]:
    resolved = resolve_path(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("增量实验配置顶层必须是映射")
    payload["_config_path"] = str(resolved)
    validate_experiment_config(payload)
    return payload


def _class_map(config: Mapping[str, Any]) -> Dict[int, str]:
    return {int(key): str(value) for key, value in config["dataset"]["class_map"].items()}


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    errors = []
    experiment = config.get("experiment", {})
    if not experiment.get("id"):
        errors.append("experiment.id is required")
    try:
        int(experiment.get("seed"))
    except (TypeError, ValueError):
        errors.append("experiment.seed must be an integer")
    dataset = config.get("dataset", {})
    splits = dataset.get("source_splits", {})
    if set(splits) != {"train", "dev", "lock"}:
        errors.append("dataset.source_splits must contain train/dev/lock")
    try:
        class_map = _class_map(config)
    except (KeyError, TypeError, ValueError):
        class_map = {}
        errors.append("dataset.class_map must map integer IDs to names")
    partition = config.get("partition", {})
    base_ids = {int(value) for value in partition.get("base_class_ids", [])}
    if not base_ids or not base_ids <= set(class_map):
        errors.append("partition.base_class_ids must be a non-empty subset of class_map")
    if partition.get("cooccurrence_policy") != "reject":
        errors.append("partition.cooccurrence_policy must be reject")
    rounds = partition.get("rounds", [])
    if not isinstance(rounds, list) or not rounds:
        errors.append("partition.rounds must be a non-empty list")
    known = set(base_ids)
    round_ids = set()
    for index, round_spec in enumerate(rounds, 1):
        round_id = str(round_spec.get("id") or "")
        new_ids = {int(value) for value in round_spec.get("new_class_ids", [])}
        if not round_id or round_id in round_ids:
            errors.append(f"round {index} has a missing or duplicate id")
        round_ids.add(round_id)
        if not new_ids or not new_ids <= set(class_map) or new_ids & known:
            errors.append(f"round {round_id or index} new_class_ids are invalid or already known")
        if round_spec.get("train_selector") != "contains_only_new_classes":
            errors.append(f"round {round_id or index} train_selector is unsupported")
        if round_spec.get("dev_selector") != "contains_only_new_classes":
            errors.append(f"round {round_id or index} dev_selector is unsupported")
        known.update(new_ids)
    if known != set(class_map):
        errors.append("base classes plus all rounds must cover class_map exactly")
    training = config.get("training", {})
    if not training.get("adapter_config"):
        errors.append("training.adapter_config is required")
    acceptance = config.get("acceptance", {})
    required_acceptance = {
        "min_base_map50",
        "min_new_map50",
        "min_krr",
        "calibration_target_precision",
        "min_lock_precision",
        "max_false_activation_rate",
        "max_base_weight_drift",
    }
    missing_acceptance = sorted(required_acceptance - set(acceptance))
    if missing_acceptance:
        errors.append("acceptance is missing: " + ", ".join(missing_acceptance))
    if errors:
        raise ValueError("Invalid incremental experiment config: " + "; ".join(errors))


def _parse_stem(stem: str) -> tuple[str, str]:
    parts = stem.split("_")
    sensor = parts[0] if parts and parts[0] in {"ir", "sar"} else "unknown"
    scene = parts[3] if len(parts) > 3 and parts[3] in {"air", "forest", "sea", "urban"} else "unknown"
    return sensor, scene


def _file_record(image: Path, split: str) -> Dict[str, Any]:
    label = source_label(image)
    with Image.open(image) as opened:
        width, height = opened.size
    labels = label.read_text(encoding="utf-8").splitlines()
    class_ids = sorted(image_class_ids(image))
    sensor, scene = _parse_stem(image.stem)
    return {
        "split": split,
        "stem": image.stem,
        "image_path": rel_path(image),
        "label_path": rel_path(label),
        "image_bytes": image.stat().st_size,
        "label_bytes": label.stat().st_size,
        "image_sha256": sha256_file(image),
        "label_sha256": sha256_file(label),
        "width": width,
        "height": height,
        "sensor": sensor,
        "scene": scene,
        "class_ids": class_ids,
        "num_objects": len([line for line in labels if line.strip()]),
    }


def dataset_snapshot(config: Mapping[str, Any], include_lock_content: bool = False) -> Dict[str, Any]:
    split_paths = {
        name: resolve_path(path) for name, path in config["dataset"]["source_splits"].items()
    }
    split_images = {name: read_split(path) for name, path in split_paths.items()}
    stems = {name: [image.stem for image in rows] for name, rows in split_images.items()}
    intersections = {
        "train_dev": sorted(set(stems["train"]) & set(stems["dev"])),
        "train_lock": sorted(set(stems["train"]) & set(stems["lock"])),
        "dev_lock": sorted(set(stems["dev"]) & set(stems["lock"])),
    }
    if any(intersections.values()):
        raise ValueError(f"源划分存在重复stem：{intersections}")
    files = []
    for split in ("train", "dev"):
        files.extend(_file_record(image, split) for image in split_images[split])
    lock = {
        "sealed": not include_lock_content,
        "split_sha256": sha256_file(split_paths["lock"]),
        "stems": stems["lock"],
        "files": [],
    }
    if include_lock_content:
        lock["files"] = [_file_record(image, "lock") for image in split_images["lock"]]
    class_map = _class_map(config)
    base_ids = {int(value) for value in config["partition"]["base_class_ids"]}
    round_rows = []
    round_selected_files: list[list[Dict[str, Any]]] = []
    assigned_new = set()
    for round_spec in config["partition"]["rounds"]:
        new_ids = {int(value) for value in round_spec["new_class_ids"]}
        selected = [row for row in files if row["split"] in {"train", "dev"} and set(row["class_ids"]) <= new_ids]
        invalid = [row["stem"] for row in files if set(row["class_ids"]) & new_ids and not set(row["class_ids"]) <= new_ids]
        if invalid:
            raise ValueError(f"新增类图像存在旧类共现：{invalid[:10]}")
        round_rows.append({
            "id": round_spec["id"],
            "new_class_ids": sorted(new_ids),
            "counts": {
                split: sum(row["split"] == split for row in selected) for split in ("train", "dev")
            },
            "object_count": sum(row["num_objects"] for row in selected),
            "object_counts": {
                split: sum(row["num_objects"] for row in selected if row["split"] == split)
                for split in ("train", "dev")
            },
            "sensor_distribution": dict(sorted(Counter(row["sensor"] for row in selected).items())),
            "scene_distribution": dict(sorted(Counter(row["scene"] for row in selected).items())),
            "stems": {split: [row["stem"] for row in selected if row["split"] == split] for split in ("train", "dev")},
            "files": [
                {
                    "split": row["split"],
                    "stem": row["stem"],
                    "image_sha256": row["image_sha256"],
                    "label_sha256": row["label_sha256"],
                }
                for row in selected
            ],
        })
        round_selected_files.append(selected)
        expected = dict(round_spec.get("expected_counts") or {})
        for split in ("train", "dev"):
            if split in expected and round_rows[-1]["counts"][split] != int(expected[split]):
                raise ValueError(
                    f"{round_spec['id']} {split}数量不符："
                    f"expected={expected[split]} actual={round_rows[-1]['counts'][split]}"
                )
        assigned_new.update(new_ids)
    base_files = [row for row in files if set(row["class_ids"]) <= base_ids]
    excluded = [row for row in files if not set(row["class_ids"]) <= base_ids | assigned_new]
    if excluded:
        raise ValueError(f"存在无法归属的数据：{[row['stem'] for row in excluded[:10]]}")
    incremental_files = [row for selected in round_selected_files for row in selected]
    incremental_stems = {row["stem"] for row in incremental_files}
    base_stems = {row["stem"] for row in base_files}
    old_raw_stems = sorted(base_stems & incremental_stems)
    base_image_hashes = {row["image_sha256"] for row in base_files}
    base_label_hashes = {row["label_sha256"] for row in base_files}
    incremental_image_hashes = {row["image_sha256"] for row in incremental_files}
    incremental_label_hashes = {row["label_sha256"] for row in incremental_files}
    old_raw_content_hashes = sorted(base_image_hashes & incremental_image_hashes)
    old_raw_label_hashes = sorted(base_label_hashes & incremental_label_hashes)
    leaked_incremental_images = {
        row["image_path"]
        for row in incremental_files
        if row["stem"] in base_stems or row["image_sha256"] in base_image_hashes
    }
    leaked_incremental_labels = {
        row["label_path"]
        for row in incremental_files
        if row["stem"] in base_stems or row["label_sha256"] in base_label_hashes
    }

    cache_roots = [resolve_path(path) for path in config.get("audit", {}).get("cache_roots", [])]
    cache_files: list[Dict[str, Any]] = []
    for cache_root in cache_roots:
        if not cache_root.exists():
            continue
        for cache_file in sorted(path for path in cache_root.rglob("*") if path.is_file()):
            cache_files.append({
                "path": rel_path(cache_file),
                "bytes": cache_file.stat().st_size,
                "sha256": sha256_file(cache_file),
            })
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "class_map": class_map,
        "source_splits": {
            name: {"path": rel_path(path), "sha256": sha256_file(path), "stems": stems[name]}
            for name, path in split_paths.items()
        },
        "split_intersections": intersections,
        "distribution": {
            "sensor": dict(sorted(Counter(row["sensor"] for row in files).items())),
            "scene": dict(sorted(Counter(row["scene"] for row in files).items())),
            "class_presence": {
                str(class_id): sum(class_id in row["class_ids"] for row in files)
                for class_id in sorted(class_map)
            },
        },
        "files": files,
        "lock": lock,
        "base": {
            "class_ids": sorted(base_ids),
            "counts": {split: sum(row["split"] == split for row in base_files) for split in ("train", "dev")},
            "object_count": sum(row["num_objects"] for row in base_files),
            "object_counts": {
                split: sum(row["num_objects"] for row in base_files if row["split"] == split)
                for split in ("train", "dev")
            },
            "stems": {split: [row["stem"] for row in base_files if row["split"] == split] for split in ("train", "dev")},
            "files": [
                {
                    "split": row["split"],
                    "stem": row["stem"],
                    "image_sha256": row["image_sha256"],
                    "label_sha256": row["label_sha256"],
                }
                for row in base_files
            ],
        },
        "rounds": round_rows,
        "incremental_train_ratio_to_base": {
            row["id"]: (
                row["counts"]["train"] / max(1, sum(item["split"] == "train" for item in base_files))
            )
            for row in round_rows
        },
        "old_raw_stems": old_raw_stems,
        "old_raw_content_hashes": old_raw_content_hashes,
        "old_raw_label_hashes": old_raw_label_hashes,
        "old_raw_image_paths": sorted(leaked_incremental_images),
        "old_raw_label_paths": sorted(leaked_incremental_labels),
        "old_raw_image_count": len(leaked_incremental_images),
        "old_raw_label_count": len(leaked_incremental_labels),
        "cache_audit": {
            "roots": [rel_path(path) for path in cache_roots],
            "files": cache_files,
            "old_feature_cache_count": len(cache_files),
        },
        "lock_content_read": include_lock_content,
    }
    payload["snapshot_sha256"] = _canonical_sha256(payload, {"created_at", "snapshot_sha256"})
    return payload


def environment_snapshot() -> Dict[str, Any]:
    def command(*argv: str) -> str:
        result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=30)
        return result.stdout.strip() if result.returncode == 0 else ""

    torch_info: Dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except ImportError:
        torch_info = {"available": False}
    dirty = command("git", "diff", "--binary")
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "git_dirty_diff_sha256": hashlib.sha256(dirty.encode("utf-8")).hexdigest(),
        "torch": torch_info,
        "pip_freeze": command(sys.executable, "-m", "pip", "freeze").splitlines(),
    }


def artifact_inventory(root: Path) -> list[Dict[str, Any]]:
    rows = []
    excluded_parts = {"images", "labels"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or excluded_parts & set(path.relative_to(root).parts):
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


class ExperimentLedger:
    def __init__(
        self,
        root: Path,
        run_id: str,
        experiment_id: str = "unknown",
        event_log: StructuredEventLog | None = None,
    ) -> None:
        self.root = root / run_id
        if self.root.exists():
            raise FileExistsError(f"拒绝覆盖实验目录：{self.root}")
        self.root.mkdir(parents=True)
        self.events_path = self.root / "events.jsonl"
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.event_log = event_log

    def event(self, state: str, status: str = "completed", **details: Any) -> None:
        if state not in STATES:
            raise ValueError(f"未知实验状态：{state}")
        row = {
            "time": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "state": state,
            "status": status,
            **details,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.event_log is not None:
            protocol_id = details.get("protocol") or details.get("protocol_id")
            mirror_state_event(
                self.event_log,
                state,
                status=status,
                experiment_id=self.experiment_id,
                run_id=self.run_id,
                protocol_id=str(protocol_id) if protocol_id else None,
                details=details,
            )

    def write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


def validate_experiment(path: str | Path) -> Dict[str, Any]:
    config = load_experiment_config(path)
    snapshot = dataset_snapshot(config, include_lock_content=False)
    base_summary = {
        key: snapshot["base"][key]
        for key in ("class_ids", "counts", "object_count", "object_counts")
    }
    round_summaries = [
        {
            key: row[key]
            for key in (
                "id",
                "new_class_ids",
                "counts",
                "object_count",
                "object_counts",
                "sensor_distribution",
                "scene_distribution",
            )
        }
        for row in snapshot["rounds"]
    ]
    return {
        "valid": (
            snapshot["old_raw_image_count"] == 0
            and snapshot["old_raw_label_count"] == 0
            and snapshot["cache_audit"]["old_feature_cache_count"] == 0
        ),
        "experiment_id": config["experiment"]["id"],
        "base": base_summary,
        "rounds": round_summaries,
        "incremental_train_ratio_to_base": snapshot["incremental_train_ratio_to_base"],
        "old_raw_image_count": snapshot["old_raw_image_count"],
        "old_raw_label_count": snapshot["old_raw_label_count"],
        "old_feature_cache_count": snapshot["cache_audit"]["old_feature_cache_count"],
        "lock_sealed": snapshot["lock"]["sealed"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "execution_adapter": {
            "version": 1,
            "supports_current_config": (
                len(config["partition"]["base_class_ids"]) == 3
                and len(config["partition"]["rounds"]) == 1
                and len(config["partition"]["rounds"][0]["new_class_ids"]) == 1
            ),
            "scope": "single_round_3plus1",
        },
    }


def compile_training_adapter(config: Mapping[str, Any], ledger: ExperimentLedger) -> Path:
    adapter_path = resolve_path(config["training"]["adapter_config"])
    adapter = yaml.safe_load(adapter_path.read_text(encoding="utf-8"))
    if not isinstance(adapter, dict):
        raise ValueError("训练适配器配置无效")
    class_map = _class_map(config)
    base_ids = sorted(int(value) for value in config["partition"]["base_class_ids"])
    rounds = list(config["partition"]["rounds"])
    if len(base_ids) != 3 or len(rounds) != 1 or len(rounds[0]["new_class_ids"]) != 1:
        raise ValueError("当前YOLO11s训练适配器只执行单轮3+1；通用配置可描述多轮，但需逐轮生成专家")
    new_id = int(rounds[0]["new_class_ids"][0])
    protocol_id = str(config["training"].get("protocol_id") or rounds[0]["id"])
    expected = dict(rounds[0].get("expected_counts") or {})
    preferred_device = str(adapter.get("runtime", {}).get("preferred_devices", ["0"])[0])
    adapter["seed"] = int(config["experiment"]["seed"])
    adapter["paths"]["source_splits"] = {
        "train": config["dataset"]["source_splits"]["train"],
        "val": config["dataset"]["source_splits"]["dev"],
        "lock": config["dataset"]["source_splits"]["lock"],
    }
    adapter["paths"].update({
        "dataset_root": str(ledger.root / "data_views"),
        "run_root": str(ledger.root / "training"),
        "report_root": str(ledger.root / "evaluation"),
    })
    adapter["runtime"]["parallel"] = False
    adapter["protocols"] = [{
        "id": protocol_id,
        "display_name": f"{class_map[new_id]}严格3+1",
        "base_classes": [class_map[value] for value in base_ids],
        "new_class": class_map[new_id],
        "new_global_id": new_id,
        "base_local_to_global": {local: global_id for local, global_id in enumerate(base_ids)},
        "expected_incremental_counts": expected,
        "preferred_device": preferred_device,
    }]
    acceptance = config["acceptance"]
    adapter["calibration"]["target_precision"] = float(acceptance["calibration_target_precision"])
    adapter["acceptance"].update({
        "min_base_map50": float(acceptance["min_base_map50"]),
        "min_new_map50": float(acceptance["min_new_map50"]),
        "min_krr": float(acceptance["min_krr"]),
        "min_lock_precision": float(acceptance["min_lock_precision"]),
        "max_false_activation_rate": float(acceptance["max_false_activation_rate"]),
        "max_base_weight_drift": float(acceptance["max_base_weight_drift"]),
    })
    adapter["experiment_audit"] = {
        "root": str(ledger.root),
        "events": str(ledger.events_path),
        "generic_config_sha256": sha256_file(Path(config["_config_path"])),
        "experiment_id": ledger.experiment_id,
        "run_id": ledger.run_id,
        "global_logging": (
            {
                "root": str(ledger.event_log.root),
                "max_file_bytes": ledger.event_log.max_file_bytes,
                "retained_files": ledger.event_log.retained_files,
            }
            if ledger.event_log is not None
            else None
        ),
    }
    output = ledger.root / "training_adapter.yaml"
    output.write_text(yaml.safe_dump(adapter, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output


def run_experiment(
    path: str | Path,
    run_id: str | None = None,
    parent_manifest: str | Path | None = None,
    validate_only: bool = False,
) -> Dict[str, Any]:
    config = load_experiment_config(path)
    if validate_only:
        return validate_experiment(path)
    experiment_id = str(config["experiment"]["id"])
    resolved_run_id = run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    root = resolve_path(config["experiment"].get("output_root", f"runs/experiments/{experiment_id}"))
    ledger = ExperimentLedger(
        root,
        resolved_run_id,
        experiment_id=experiment_id,
        event_log=event_log_from_config(),
    )
    ledger.event("CREATED", config=rel_path(Path(config["_config_path"])))
    try:
        portable_config = {key: value for key, value in config.items() if not key.startswith("_")}
        config_snapshot = ledger.root / "experiment.yaml"
        config_snapshot.write_text(
            yaml.safe_dump(portable_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        environment = environment_snapshot()
        ledger.write_json("environment.json", environment)
        snapshot = dataset_snapshot(config, include_lock_content=False)
        ledger.write_json("dataset_snapshot.json", snapshot)
        compliance = {
            "old_raw_image_count": snapshot["old_raw_image_count"],
            "old_raw_label_count": snapshot["old_raw_label_count"],
            "old_feature_cache_count": snapshot["cache_audit"]["old_feature_cache_count"],
        }
        ledger.event("DATA_AUDITED", snapshot_sha256=snapshot["snapshot_sha256"], **compliance)
        if any(compliance.values()):
            raise RuntimeError(f"增量训练访问边界不合规：{compliance}")
        ledger.event(
            "PARTITIONED",
            base_counts=snapshot["base"]["counts"],
            rounds=[{"id": row["id"], "counts": row["counts"]} for row in snapshot["rounds"]],
        )
        adapter = compile_training_adapter(config, ledger)
        argv = [
            sys.executable,
            str(ROOT / "tools" / "70_run_strict_3plus1.py"),
            "--config",
            str(adapter),
            "--run-id",
            resolved_run_id,
        ]
        started = time.monotonic()
        log_path = ledger.root / "training.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                argv,
                cwd=ROOT,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            ledger.event(
                "PARTITIONED",
                status="execution_started",
                argv=argv,
                cwd=str(ROOT),
                pid=process.pid,
            )
            returncode = process.wait()
        duration = time.monotonic() - started
        ledger.event(
            "PARTITIONED",
            status="execution_finished" if returncode == 0 else "execution_failed",
            pid=process.pid,
            returncode=returncode,
            duration_seconds=duration,
        )
        lock_snapshot = None
        if returncode == 0:
            lock_snapshot = dataset_snapshot(config, include_lock_content=True)
            ledger.write_json("dataset_snapshot_lock_unsealed.json", lock_snapshot)
        manifest = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "run_id": resolved_run_id,
            "status": "completed" if returncode == 0 else "failed",
            "config_snapshot": rel_path(config_snapshot),
            "config_sha256": sha256_file(config_snapshot),
            "dataset_snapshot": rel_path(ledger.root / "dataset_snapshot.json"),
            "dataset_snapshot_sha256": snapshot["snapshot_sha256"],
            "lock_snapshot": (
                rel_path(ledger.root / "dataset_snapshot_lock_unsealed.json") if lock_snapshot else None
            ),
            "lock_snapshot_sha256": lock_snapshot.get("snapshot_sha256") if lock_snapshot else None,
            "environment": rel_path(ledger.root / "environment.json"),
            "training_adapter": rel_path(adapter),
            "argv": argv,
            "cwd": str(ROOT),
            "pid": process.pid,
            "returncode": returncode,
            "duration_seconds": duration,
            "compliance": compliance,
            "artifacts": artifact_inventory(ledger.root),
            "parent_manifest": rel_path(resolve_path(parent_manifest)) if parent_manifest else None,
            "parent_manifest_sha256": sha256_file(resolve_path(parent_manifest)) if parent_manifest else None,
        }
        manifest_path = ledger.write_json("run_manifest.json", manifest)
        manifest_hash = sha256_file(manifest_path)
        (ledger.root / "run_manifest.sha256").write_text(
            f"{manifest_hash}  run_manifest.json\n", encoding="ascii"
        )
        manifest["manifest_sha256"] = manifest_hash
        manifest["manifest_hash_file"] = rel_path(ledger.root / "run_manifest.sha256")
        if returncode != 0:
            ledger.event("REJECTED", status="failed", reason="training_pipeline_failed")
        return manifest
    except Exception as exc:
        ledger.event(
            "REJECTED",
            status="failed",
            reason="orchestrator_exception",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def reproduce_experiment(manifest_path: str | Path, run_id: str | None = None) -> Dict[str, Any]:
    resolved_manifest = resolve_path(manifest_path)
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    config_path = resolve_path(manifest["config_snapshot"])
    config = load_experiment_config(config_path)
    current = dataset_snapshot(config, include_lock_content=False)
    if current["snapshot_sha256"] != manifest["dataset_snapshot_sha256"]:
        raise ValueError("源数据指纹与父实验不一致，拒绝声称复现")
    return run_experiment(config_path, run_id=run_id, parent_manifest=resolved_manifest)
