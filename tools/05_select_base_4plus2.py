#!/usr/bin/env python3
"""Wait for Base seed queues, re-evaluate their best weights, and promote one."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_TAGS = ("yolo11s", "yolo11m", "yolo26s", "yolo26m")


def parse_csv(value: str) -> list[str]:
    rows = [item.strip() for item in value.split(",") if item.strip()]
    if not rows:
        raise argparse.ArgumentTypeError("列表不能为空")
    if len(rows) != len(set(rows)):
        raise argparse.ArgumentTypeError("列表不能包含重复值")
    return rows


def parse_seeds(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def candidate_paths(project: Path, tags: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "model_tag": tag,
            "seed": seed,
            "run_dir": project / f"{tag}_seed{seed}",
            "weight": project / f"{tag}_seed{seed}" / "weights" / "best.pt",
            "results": project / f"{tag}_seed{seed}" / "results.csv",
        }
        for tag in tags
        for seed in seeds
    ]


def queue_state(
    project: Path, tags: list[str]
) -> tuple[set[tuple[str, int]], list[str]]:
    completed: set[tuple[str, int]] = set()
    failures: list[str] = []
    for tag in tags:
        path = project / f"{tag}_queue_summary.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("runs", []):
            status = row.get("status")
            seed = int(row.get("seed"))
            if status in {"complete", "already_complete"}:
                completed.add((tag, seed))
            elif status == "failed":
                failures.append(
                    f"{tag}/seed={seed}: {row.get('error', 'unknown error')}"
                )
    return completed, failures


def wait_for_candidates(
    project: Path,
    tags: list[str],
    seeds: list[int],
    poll_seconds: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    candidates = candidate_paths(project, tags, seeds)
    started = time.monotonic()
    last_ready = -1
    while True:
        completed, failures = queue_state(project, tags)
        if failures:
            raise RuntimeError("Base 训练队列失败：" + "; ".join(failures))
        ready = [
            row
            for row in candidates
            if (row["model_tag"], row["seed"]) in completed
            and row["weight"].is_file()
            and row["results"].is_file()
        ]
        if len(ready) != last_ready:
            print(
                json.dumps(
                    {
                        "event": "base_candidate_wait",
                        "ready": len(ready),
                        "expected": len(candidates),
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            last_ready = len(ready)
        if len(ready) == len(candidates):
            return candidates
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(
                f"等待 Base 候选超时：ready={len(ready)} expected={len(candidates)}"
            )
        time.sleep(poll_seconds)


def validate_candidate(
    candidate: dict[str, Any],
    dataset_yaml: Path,
    output_dir: Path,
    device: str,
    imgsz: int,
    batch: int,
    workers: int,
) -> dict[str, Any]:
    from ultralytics import YOLO

    name = f"{candidate['model_tag']}_seed{candidate['seed']}"
    print(
        json.dumps(
            {
                "event": "base_candidate_validation_start",
                "candidate": name,
                "weight": candidate["weight"].as_posix(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    result = YOLO(str(candidate["weight"])).val(
        data=str(dataset_yaml),
        split="val",
        device=device,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        conf=0.001,
        iou=0.70,
        max_det=300,
        rect=True,
        augment=False,
        plots=False,
        save_json=False,
        project=str(output_dir / "validation"),
        name=name,
        exist_ok=True,
        verbose=True,
    )
    box = result.box
    ap50 = [float(value) for value in box.ap50]
    row = {
        "model_tag": candidate["model_tag"],
        "seed": candidate["seed"],
        "weight": candidate["weight"].as_posix(),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class_ap50": {
            name: ap50[index]
            for index, name in result.names.items()
            if index < len(ap50)
        },
        "speed_ms": {key: float(value) for key, value in result.speed.items()},
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def write_markdown(path: Path, rows: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    lines = [
        "# 4+2 Base 模型复评排名",
        "",
        "本步骤属于 base_learning，不计入 incremental_learning，且选模本身不更新检测器权重。",
        "",
        "选择口径：固定 Base dev 的 mAP50 主排序，mAP50-95 次排序；未读取 Base lock。",
        "",
        "| 排名 | 模型 | seed | mAP50 | mAP50-95 | Precision | Recall |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | {row['model_tag']} | {row['seed']} | "
            f"{row['map50']:.6f} | {row['map50_95']:.6f} | "
            f"{row['precision']:.6f} | {row['recall']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"最终选择：`{selected['model_tag']}`，seed `{selected['seed']}`，"
            f"Base dev mAP50 `{selected['map50']:.6f}`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="等待并复评 4+2 Base 多种子候选。")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--tags", type=parse_csv, default=list(DEFAULT_TAGS))
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("3407,20260821,8675309"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=604800)
    args = parser.parse_args()

    project = args.project.expanduser().resolve()
    dataset_yaml = args.dataset_yaml.expanduser().resolve()
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"数据配置不存在：{dataset_yaml}")

    candidates = wait_for_candidates(
        project,
        args.tags,
        args.seeds,
        args.poll_seconds,
        args.timeout_seconds,
    )
    output_dir = project / "selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "phase": "base_learning",
        "counted_as_incremental_learning": False,
        "detector_weights_updated": False,
        "component": "base_detector_candidate_selection",
        "selection_primary": "Base dev mAP50",
        "selection_secondary": "Base dev mAP50-95",
        "lock_used": False,
        "candidates": [],
    }
    progress_path = output_dir / "base_selection.json"
    atomic_json(progress_path, payload)

    for candidate in candidates:
        row = validate_candidate(
            candidate,
            dataset_yaml,
            output_dir,
            args.device,
            args.imgsz,
            args.batch,
            args.workers,
        )
        payload["candidates"].append(row)
        atomic_json(progress_path, payload)

    ranked = sorted(
        payload["candidates"],
        key=lambda row: (row["map50"], row["map50_95"], row["recall"]),
        reverse=True,
    )
    selected = ranked[0]
    selected_dir = output_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_weight = selected_dir / "best_base.pt"
    shutil.copy2(selected["weight"], selected_weight)
    source_run = Path(selected["weight"]).parents[1]
    for filename in ("args.yaml", "results.csv"):
        source = source_run / filename
        if source.is_file():
            shutil.copy2(source, selected_dir / filename)

    payload["ranking"] = ranked
    payload["selected"] = {
        **selected,
        "promoted_weight": selected_weight.as_posix(),
    }
    atomic_json(progress_path, payload)
    write_markdown(output_dir / "base_selection.md", ranked, selected)
    print(json.dumps(payload["selected"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
