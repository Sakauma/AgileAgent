#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zipfile
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import load_config, rel_path, resolve_path
from fair_agent.core.hashes import sha256_file
from fair_agent.core.runtime_log import StructuredEventLog
from fair_agent.modules.incremental_guardian import assess_incremental_candidate
from fair_agent.modules.incremental_lineage import _canonical_sha256
from fair_agent.modules.incremental_workbench import IncrementalBatchStore, TrainingJobManager
from fair_agent.modules.model_generations import generation_web_settings, load_generation_registry


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _class_ids(path: Path) -> set[int]:
    return {
        int(line.split()[0])
        for line in path.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _sensor(path: Path) -> str:
    return "sar" if path.name.lower().startswith("sar_") else "ir"


def _read_split(path: str | Path) -> list[Path]:
    return [
        resolve_path(line.strip())
        for line in resolve_path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_base_lineage(config: Mapping[str, Any], class_id: int) -> Path:
    target = resolve_path(config["incremental_workbench"]["lineage"]["base_manifest"])
    files = []
    for split_path in config["_experiment"]["dataset"]["base_splits"]:
        for image in _read_split(split_path):
            if class_id in _class_ids(image):
                continue
            label = image.with_suffix(".txt")
            files.append({
                "split": rel_path(resolve_path(split_path)),
                "stem": image.stem,
                "image_sha256": sha256_file(image),
                "label_sha256": sha256_file(label),
            })
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "catalog_id": "isolated_base",
        "kind": "frozen_base_only_lineage",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "class_ids": [0, 1, 3],
        "files": sorted(files, key=lambda row: (row["split"], row["stem"])),
        "cache_files": [],
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    _atomic_json(target, payload)
    return target


def _initialize_registry(source: Path, target: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload = deepcopy(payload)
    payload["models"] = [
        item for item in payload["models"]
        if item["role"] in {"frozen_base", "benchmark_only"}
    ]
    payload["generations"] = [
        item for item in payload["generations"]
        if item["id"] == "base_detection_generation"
    ]
    base = payload["generations"][0]
    base["model_members"] = list(dict.fromkeys(base["class_owners"].values()))
    payload["channels"]["production"] = base["id"]
    payload["channels"]["candidate"] = base["id"]
    _atomic_json(target, payload)
    load_generation_registry(target)


def _effective_config(spec: Mapping[str, Any], run_root: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(resolve_path(spec["agent"]["base_config"]).read_text(encoding="utf-8"))
    raw["runtime"]["local_python"] = str(spec["agent"]["python"])
    raw["runtime"]["default_device"] = str(spec["agent"]["device"])
    raw["routing"]["max_specialists_per_image"] = int(spec["agent"]["max_specialists_per_image"])
    raw["routing"]["parallel_model_execution"] = False
    raw["routing"]["parallel_context_execution"] = False
    raw["routing"]["parallel_context_batch_execution"] = False
    raw["inference"]["backend"] = "ultralytics_cuda"
    raw["inference"]["specialist_imgsz"] = int(spec["training"]["imgsz"])
    raw["inference"]["batch_size"] = 4
    raw["logging"]["root"] = str(run_root / "logs")
    workbench = raw["incremental_workbench"]
    workbench["root"] = str(run_root / "batches")
    workbench["lineage"].update({
        "root": str(run_root / "lineage"),
        "base_manifest": str(run_root / "lineage" / "base_dataset.json"),
        "auto_initialize_base": False,
        "cache_roots": [],
    })
    workbench["lifecycle"]["auto_continue"] = True
    workbench["training"].update({
        "python": str(spec["agent"]["python"]),
        "device": str(spec["agent"]["device"]),
        **dict(spec["training"]),
        "seed": int(spec["experiment"]["seed"]),
    })
    registry_path = run_root / "generations.json"
    raw["generation"].update({
        "registry": str(registry_path),
        "runtime_registry": str(registry_path),
        "report_root": str(run_root / "generation_rechecks"),
        "auto_promote": True,
    })
    raw["web"]["generation_registry"] = str(registry_path)
    effective_path = run_root / "effective_agent_config.yaml"
    effective_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    config = load_config(effective_path)
    config["_experiment"] = deepcopy(spec)
    return config


def _select_rounds(spec: Mapping[str, Any]) -> list[Dict[str, list[Path]]]:
    dataset = spec["dataset"]
    class_id = int(dataset["source_global_class_id"])
    rounds = int(spec["experiment"]["rounds"])
    plans = list(dataset.get("round_plan") or [])
    if len(plans) != rounds:
        raise ValueError("round_plan数量必须与实验轮数一致。")
    rng = random.Random(int(spec["experiment"]["seed"]))
    pools: Dict[str, list[Path]] = {str(sensor): [] for sensor in dataset["sensors"]}
    for image in _read_split(dataset["source_split"]):
        if _class_ids(image) == {class_id} and _sensor(image) in pools:
            pools[_sensor(image)].append(image)
    for sensor, rows in pools.items():
        rows.sort(key=lambda path: path.name)
        rng.shuffle(rows)
        needed = sum(
            int(plan[split].get(sensor, 0))
            for plan in plans for split in ("train", "val", "lock")
        )
        if len(rows) < needed:
            raise ValueError(f"{sensor}纯新增类样本不足：需要{needed}，实际{len(rows)}")
        del rows[needed:]
    assignments = []
    offsets = {sensor: 0 for sensor in pools}
    for plan in plans:
        row: Dict[str, list[Path]] = {"train": [], "val": [], "lock": []}
        for split in ("train", "val", "lock"):
            for sensor, pool in pools.items():
                count = int(plan[split].get(sensor, 0))
                start = offsets[sensor]
                row[split].extend(pool[start:start + count])
                offsets[sensor] += count
        assignments.append(row)
    stems = [image.stem for row in assignments for images in row.values() for image in images]
    if len(stems) != len(set(stems)):
        raise ValueError("多轮小样本划分存在重复stem。")
    return assignments


def _write_archive(
    target: Path, assignment: Mapping[str, Iterable[Path]], class_id: int, class_name: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.yaml", yaml.safe_dump({"names": {0: class_name}}, sort_keys=False))
        for split, images in assignment.items():
            for image in images:
                archive.write(image, f"images/{split}/{image.name}")
                lines = []
                for line in image.with_suffix(".txt").read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if fields and int(fields[0]) == class_id:
                        lines.append("0 " + " ".join(fields[1:]))
                if not lines:
                    raise ValueError(f"增量样本没有类别{class_id}标签：{image}")
                archive.writestr(f"labels/{split}/{image.stem}.txt", "\n".join(lines) + "\n")


def _model_hashes(registry: Mapping[str, Any], generation_id: str) -> Dict[str, str]:
    hashes = {}
    for model_id, model in registry["models_by_id"].items():
        if model["role"] == "benchmark_only":
            continue
        expected = str(model["sha256"])
        actual = sha256_file(model["resolved_path"])
        if actual != expected:
            raise RuntimeError(f"注册权重哈希不一致：{model_id}")
        hashes[str(model_id)] = actual
    return hashes


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    expected_rounds = int(summary["expected_rounds"])
    lines = [
        "# 多批次小样本持续学习验证",
        "",
        f"- 实验编号：`{summary['run_id']}`",
        f"- 最终状态：`{summary['status']}`",
        f"- 批次数：{len(summary['rounds'])}/{expected_rounds}",
        "",
        "| 轮次 | 模式 | train/dev/lock | New-mAP50 | KRR | 组合mAP50 | 专家数 | 推理ms | 总耗时s | 状态 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rounds"]:
        metrics = row.get("metrics", {})
        counts = row.get("split_counts", {})
        lines.append(
            f"| {row['round']} | {row['incremental_mode']} | "
            f"{counts.get('train', 0)}/{counts.get('val', 0)}/{counts.get('lock', 0)} | "
            f"{metrics.get('new_map50', 0):.4f} | {metrics.get('krr', 0):.4f} | "
            f"{metrics.get('combined_map50', 0):.4f} | {metrics.get('specialist_count', 0)} | "
            f"{metrics.get('mean_inference_ms', 0):.2f} | {row['elapsed_seconds']:.1f} | {row['status']} |"
        )
    lines.extend(["", "完整数据指纹、阈值、权重哈希和逐图预测见同目录JSON与各轮manifest。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def reassess_existing_run(spec_path: Path, run_id: str) -> Dict[str, Any]:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    run_root = resolve_path(spec["experiment"]["output_root"]) / run_id
    report_root = resolve_path(spec["experiment"]["report_root"]) / run_id
    summary_path = report_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = load_config(spec["agent"]["base_config"])
    rounds = []
    for row in summary["rounds"]:
        batch_manifest_path = run_root / "batches" / str(row["batch_id"]) / "batch_manifest.json"
        if not batch_manifest_path.is_file():
            raise FileNotFoundError(batch_manifest_path)
        batch = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
        assessment = assess_incremental_candidate(
            row["metrics"],
            batch.get("audit") or {},
            config["gates"],
            config["incremental_guardian"],
        )
        rounds.append({
            "round": int(row["round"]),
            "historical_status": str(row["status"]),
            "official_full_score_passed": bool(assessment["accepted"]),
            "reclassified_status": (
                "FULL_SCORE_PASSED_WITH_WARNINGS"
                if assessment["accepted"] and assessment["warnings"]
                else "FULL_SCORE_PASSED" if assessment["accepted"] else "OFFICIAL_GATE_REJECTED"
            ),
            "metrics": dict(row["metrics"]),
            "guardian_assessment": assessment,
            "historical_promotion_preserved": str(row["status"]) == "PROMOTED",
        })
    output = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_summary": rel_path(summary_path),
        "source_summary_sha256": sha256_file(summary_path),
        "note": "只按当前官方满分档与告警口径重新判分，不改写历史代际链或production。",
        "rounds": rounds,
        "official_full_score_round_count": sum(
            row["official_full_score_passed"] for row in rounds
        ),
    }
    output_path = report_root / "guardian_reassessment.json"
    _atomic_json(output_path, output)
    report_lines = [
        "# 增量学习守护器重新判分",
        "",
        "本报告不修改历史代际链，仅纠正组合 mAP50 被误设为硬门禁的判分口径。",
        "",
        "| 轮次 | New-mAP50 | KRR | 累计组合mAP50 | 官方满分档 | 内部告警 | 历史执行状态 |",
        "|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rounds:
        values = row["metrics"]
        warnings = "、".join(row["guardian_assessment"]["warnings"]) or "无"
        report_lines.append(
            f"| {row['round']} | {values['new_map50']:.4f} | {values['krr']:.4f} | "
            f"{values['combined_map50']:.4f} | "
            f"{'通过' if row['official_full_score_passed'] else '未通过'} | "
            f"{warnings} | {row['historical_status']} |"
        )
    report_lines.extend([
        "",
        "历史状态为 REJECTED 的轮次没有被追溯晋升；需要形成连续四轮production链时，应使用新守护器重新执行实验。",
        "",
    ])
    (report_root / "guardian_reassessment.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return output


def run(spec_path: Path, run_id: str | None = None, resume: bool = False) -> Dict[str, Any]:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = resolve_path(spec["experiment"]["output_root"]) / run_id
    report_root = resolve_path(spec["experiment"]["report_root"]) / run_id
    assignments = _select_rounds(spec)
    if resume:
        if not run_root.is_dir() or not report_root.is_dir():
            raise FileNotFoundError("待续跑的实验目录不存在。")
        config = load_config(run_root / "effective_agent_config.yaml")
        config["_experiment"] = deepcopy(spec)
        registry_path = resolve_path(config["generation"]["registry"])
        base_lineage = resolve_path(config["incremental_workbench"]["lineage"]["base_manifest"])
        summary: Dict[str, Any] = json.loads(
            (report_root / "summary.json").read_text(encoding="utf-8")
        )
        summary["status"] = "RUNNING"
        start_round = len(summary["rounds"]) + 1
        previous_hashes = dict(summary["rounds"][-1]["model_hashes"]) if summary["rounds"] else {}
    else:
        run_root.mkdir(parents=True, exist_ok=False)
        report_root.mkdir(parents=True, exist_ok=False)
        registry_path = run_root / "generations.json"
        _initialize_registry(resolve_path(spec["agent"]["base_registry"]), registry_path)
        config = _effective_config(spec, run_root)
        registry_path = resolve_path(config["generation"]["registry"])
        base_lineage = _build_base_lineage(config, int(spec["dataset"]["source_global_class_id"]))
        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "expected_rounds": int(spec["experiment"]["rounds"]),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": rel_path(spec_path),
            "effective_config": rel_path(resolve_path(run_root / "effective_agent_config.yaml")),
            "registry": rel_path(registry_path),
            "base_lineage": rel_path(base_lineage),
            "rounds": [],
            "status": "RUNNING",
        }
        start_round = 1
        previous_hashes = {}
    log_cfg = config["logging"]
    event_log = StructuredEventLog(log_cfg["root"], int(log_cfg["max_file_bytes"]), int(log_cfg["retained_files"]))
    _atomic_json(report_root / "summary.json", summary)
    for index, assignment in enumerate(assignments[start_round - 1:], start=start_round):
        started = time.perf_counter()
        archive = run_root / "archives" / f"round-{index:02d}.zip"
        _write_archive(
            archive, assignment, int(spec["dataset"]["source_global_class_id"]),
            str(spec["dataset"]["class_name"]),
        )
        registry = load_generation_registry(registry_path)
        settings = generation_web_settings(registry)
        active = {class_id: settings["class_names"][class_id] for class_id in settings["active_class_ids"]}
        store = IncrementalBatchStore(
            config["incremental_workbench"], event_log, active, settings["class_names"]
        )
        manifest = store.create(
            archive.name, archive.read_bytes(), f"小样本增量第{index}轮", str(spec["dataset"]["class_name"])
        )
        if manifest["status"] != "AUDITED":
            raise RuntimeError(f"第{index}轮数据审计失败：{manifest.get('error')}")
        expected_mode = "class_incremental" if index == 1 else "target_incremental"
        if manifest["audit"]["incremental_mode"] != expected_mode:
            raise RuntimeError(
                f"第{index}轮模式错误：期望{expected_mode}，实际{manifest['audit']['incremental_mode']}"
            )
        manifest = store.inject(manifest["batch_id"])
        manager = TrainingJobManager(store, config["incremental_workbench"], event_log, config)
        job = manager.start(manifest["batch_id"], wait=True)
        lifecycle = job.get("lifecycle_result") or {}
        registry = load_generation_registry(registry_path)
        production = str(registry["channels"]["production"])
        hashes = _model_hashes(registry, production)
        drifted = {
            model_id: {"before": expected, "after": hashes.get(model_id)}
            for model_id, expected in previous_hashes.items()
            if hashes.get(model_id) != expected
        }
        recheck = lifecycle.get("recheck") or {}
        row = {
            "round": index,
            "batch_id": manifest["batch_id"],
            "job_id": job["job_id"],
            "status": str(job["status"]),
            "incremental_mode": expected_mode,
            "sensor_counts": dict(Counter(_sensor(path) for paths in assignment.values() for path in paths)),
            "split_counts": {split: len(paths) for split, paths in assignment.items()},
            "production_generation": production,
            "model_hashes": hashes,
            "frozen_model_drift": drifted,
            "metrics": recheck.get("metrics", {}),
            "gates": recheck.get("gates", {}),
            "recheck_manifest": recheck.get("manifest"),
            "elapsed_seconds": time.perf_counter() - started,
        }
        summary["rounds"].append(row)
        _atomic_json(report_root / "summary.json", summary)
        if drifted:
            summary["status"] = "FAILED_FROZEN_MODEL_DRIFT"
            break
        previous_hashes = hashes
        if job["status"] != "PROMOTED" and not (
            job["status"] == "REJECTED"
            and bool(spec["experiment"].get("continue_after_rejection"))
        ):
            summary["status"] = f"STOPPED_{job['status']}"
            break
    else:
        summary["status"] = (
            "PASSED"
            if all(row["status"] == "PROMOTED" for row in summary["rounds"])
            else "COMPLETED_WITH_REJECTIONS"
        )
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    expected_rounds = int(spec["experiment"]["rounds"])
    summary["expected_rounds"] = expected_rounds
    summary["all_rounds_promoted"] = (
        len(summary["rounds"]) == expected_rounds and summary["status"] == "PASSED"
    )
    summary["all_rounds_processed"] = len(summary["rounds"]) == expected_rounds
    _atomic_json(report_root / "summary.json", summary)
    _write_report(report_root / "report.md", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="执行多批次小样本持续学习隔离验证。")
    parser.add_argument("--config", default="configs/incremental/multibatch_small_sample.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reassess", action="store_true", help="按当前守护器口径重新判分已有run，不启动训练。")
    args = parser.parse_args()
    if args.reassess:
        if not args.run_id:
            parser.error("--reassess必须同时提供--run-id")
        summary = reassess_existing_run(resolve_path(args.config), args.run_id)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    summary = run(resolve_path(args.config), args.run_id, resume=args.resume)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in {"PASSED", "COMPLETED_WITH_REJECTIONS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
