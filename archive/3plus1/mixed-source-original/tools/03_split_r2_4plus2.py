#!/usr/bin/env python3
"""Archive the legacy 3+1 split and create the fixed score-priority 4+2 split.

This command only writes split lists and manifests. It never copies source images,
rewrites labels, materializes YOLO datasets, or starts model training.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.incremental_round_registry import (  # noqa: E402
    DEFAULT_ROUND_REGISTRY,
    load_incremental_round_registry,
)


SPLITS_ROOT = ROOT / "splits"
OUTPUT_ROOT = SPLITS_ROOT / "strict_4plus2"
ARCHIVE_ROOT = SPLITS_ROOT / "archive" / "2026-08-21_strict_3plus1"
R1_ROOT = ROOT / "datasets_r1_base_train"
R2_ROOT = ROOT / "datasets_r2_inc_train"

BASE_TARGET_COUNTS = {"train": 600, "dev": 75, "lock": 75}
INCREMENT_TARGET_COUNTS = {"train": 112, "dev": 14, "lock": 14}
SEED_SEARCH_START = 20260821
SEED_SEARCH_COUNT = 10000
BASE_SELECTED_SEED = 20262898
INCREMENT_SELECTED_SEED = 20269064
FROZEN_GROUP_RANDOM_SEEDS = {
    "base": {
        "ir|air": 3562872155675198326,
        "ir|forest": 12530660688887189505,
        "ir|sea": 10888832000615582986,
        "ir|urban": 7940934951811070468,
        "sar|forest": 15911180380568349388,
        "sar|sea": 6935513029721223624,
        "sar|urban": 919857528413993921,
    },
    "increment": {
        "ir|forest": 10495797039138642283,
        "ir|sea": 4022962815375780357,
        "ir|urban": 16843744284724356417,
        "sar|forest": 17124843293024128029,
        "sar|sea": 7824169049612020,
        "sar|urban": 1264619966119254864,
    },
}

LEGACY_TOP_LEVEL_NAMES = [
    "README.md",
    "manifest.json",
    "pool_train.txt",
    "pool_train_ir.txt",
    "pool_train_sar.txt",
    "pool_dev.txt",
    "pool_dev_ir.txt",
    "pool_dev_sar.txt",
    "mixed_test.txt",
    "mixed_test_ir.txt",
    "mixed_test_sar.txt",
]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_lines(path: Path, values: Iterable[str]) -> None:
    rows = list(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return result.stdout.strip()


def archive_legacy_split() -> Dict[str, Any]:
    """Create a dated snapshot while keeping compatibility files in place."""
    source_paths: List[Path] = []
    for name in LEGACY_TOP_LEVEL_NAMES:
        path = SPLITS_ROOT / name
        if not path.is_file():
            raise FileNotFoundError("Legacy split file is missing: {}".format(path))
        source_paths.append(path)
    legacy_protocol = SPLITS_ROOT / "strict_3plus1"
    if not legacy_protocol.is_dir():
        raise FileNotFoundError("Legacy strict_3plus1 directory is missing")
    source_paths.extend(sorted(path for path in legacy_protocol.rglob("*") if path.is_file()))

    snapshot_root = ARCHIVE_ROOT / "snapshot"
    manifest_path = ARCHIVE_ROOT / "ARCHIVE_MANIFEST.json"
    if ARCHIVE_ROOT.exists():
        if not manifest_path.is_file():
            raise FileExistsError("Archive exists without a manifest: {}".format(ARCHIVE_ROOT))
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative in existing.get("files", []):
            archived = snapshot_root / relative
            if not archived.is_file():
                raise RuntimeError("Existing legacy archive is incomplete: {}".format(archived))
        return existing

    entries: List[str] = []
    for source in source_paths:
        relative = source.relative_to(SPLITS_ROOT)
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))
        if target.stat().st_size != source.stat().st_size:
            raise RuntimeError("Archive copy size mismatch: {}".format(relative.as_posix()))
        entries.append(relative.as_posix())

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "archive_id": "2026-08-21_strict_3plus1",
        "source_protocol": "full_coverage_strict_3plus1_dataset_partition",
        "source_git_commit": git_head(),
        "archived_on": "2026-08-21",
        "copy_policy": "dated_snapshot_with_compatibility_copy_retained",
        "file_count": len(entries),
        "files": entries,
    }
    write_json(manifest_path, manifest)
    (ARCHIVE_ROOT / "README.md").write_text(
        "# 3+1 数据划分归档\n\n"
        "该目录是 2026-08-21 开始 4+2 工作前的完整划分快照。\n"
        "`snapshot/` 保留原始目录结构，`ARCHIVE_MANIFEST.json` 记录归档文件清单。\n"
        "为了让现有 3+1 配置和测试在 4+2 代码迁移前仍可运行，原路径保留为兼容副本。\n",
        encoding="utf-8",
    )
    return manifest


def read_classes(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scan_dataset(dataset_root: Path, alias: str, expected_classes: Sequence[str]) -> List[Dict[str, Any]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError("Dataset root does not exist: {}".format(dataset_root))
    classes = read_classes(dataset_root / "classes.txt")
    if classes != list(expected_classes):
        raise ValueError("Unexpected classes in {}: {}".format(dataset_root, classes))

    rows: List[Dict[str, Any]] = []
    images = sorted(dataset_root.glob("*.png"))
    labels = sorted(path for path in dataset_root.glob("*.txt") if path.name != "classes.txt")
    if {path.stem for path in images} != {path.stem for path in labels}:
        raise ValueError("Image/label stems are not identical in {}".format(dataset_root))

    for image in images:
        parts = image.stem.split("_")
        if len(parts) != 5:
            raise ValueError("Unexpected image name: {}".format(image.name))
        sensor, round_token, phase_token, scene, frame_token = parts
        if sensor not in {"ir", "sar"} or scene not in {"air", "forest", "sea", "urban"}:
            raise ValueError("Unexpected sensor or scene in {}".format(image.name))
        frame = int(frame_token)
        label = dataset_root / "{}.txt".format(image.stem)
        object_counts: Counter = Counter()
        for line_number, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            columns = raw.split()
            if len(columns) != 5:
                raise ValueError("{}:{} has invalid YOLO columns".format(label, line_number))
            class_id = int(columns[0])
            if class_id < 0 or class_id >= len(classes):
                raise ValueError("{}:{} has invalid class {}".format(label, line_number, class_id))
            coordinates = [float(value) for value in columns[1:]]
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                raise ValueError("{}:{} has out-of-range coordinates".format(label, line_number))
            object_counts[class_id] += 1
        rows.append(
            {
                "image_path": "{}/{}".format(alias, image.name),
                "label_path": "{}/{}".format(alias, label.name),
                "sensor": sensor,
                "scene": scene,
                "sequence": "{}|{}".format(sensor, scene),
                "dataset_round": "{}_{}".format(round_token, phase_token),
                "frame": frame,
                "class_ids": set(object_counts),
                "object_counts": object_counts,
            }
        )
    return rows


def hamilton_quotas(group_sizes: Mapping[str, int], target_total: int) -> Dict[str, int]:
    source_total = sum(group_sizes.values())
    raw = {key: group_sizes[key] * target_total / source_total for key in group_sizes}
    quotas = {key: int(raw[key]) for key in group_sizes}
    remaining = target_total - sum(quotas.values())
    order = sorted(group_sizes, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def build_group_quotas(
    rows: Sequence[Mapping[str, Any]], target_counts: Mapping[str, int]
) -> Dict[str, Dict[str, int]]:
    sizes = Counter(str(row["sequence"]) for row in rows)
    train = hamilton_quotas(sizes, target_counts["train"])
    remaining_sizes = {key: sizes[key] - train[key] for key in sizes}
    dev = hamilton_quotas(remaining_sizes, target_counts["dev"])
    quotas = {
        key: {
            "train": train[key],
            "dev": dev[key],
            "lock": sizes[key] - train[key] - dev[key],
        }
        for key in sorted(sizes)
    }
    for key, values in quotas.items():
        if min(values.values()) < 1:
            raise ValueError("Each sequence must appear in every split: {} {}".format(key, values))
    return quotas


def assign_frozen_split(
    rows: Sequence[Dict[str, Any]],
    quotas: Mapping[str, Mapping[str, int]],
    group_random_seeds: Mapping[str, int],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence"]].append(row)
    result: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": [], "lock": []}
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda row: int(row["frame"]))
        random.Random(group_random_seeds[key]).shuffle(candidates)
        train_end = quotas[key]["train"]
        dev_end = train_end + quotas[key]["dev"]
        result["train"].extend(candidates[:train_end])
        result["dev"].extend(candidates[train_end:dev_end])
        result["lock"].extend(candidates[dev_end:])
    for values in result.values():
        values.sort(key=lambda row: row["image_path"])
    return result


def object_counts(rows: Sequence[Mapping[str, Any]]) -> Counter:
    total: Counter = Counter()
    for row in rows:
        total.update(row["object_counts"])
    return total


def class_image_counts(rows: Sequence[Mapping[str, Any]]) -> Counter:
    total: Counter = Counter()
    for row in rows:
        total.update(row["class_ids"])
    return total


def neighbor_metrics(split: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    train_frames: Dict[str, set] = defaultdict(set)
    for row in split["train"]:
        train_frames[str(row["sequence"])].add(int(row["frame"]))
    by_split: Dict[str, Any] = {}
    for name in ("dev", "lock"):
        adjacent = 0
        sandwiched = 0
        within_two = 0
        minimum_distances: List[int] = []
        for row in split[name]:
            frames = train_frames[str(row["sequence"])]
            distance = min(abs(int(row["frame"]) - frame) for frame in frames)
            minimum_distances.append(distance)
            if distance <= 1:
                adjacent += 1
            if distance <= 2:
                within_two += 1
            frame = int(row["frame"])
            if frame - 1 in frames and frame + 1 in frames:
                sandwiched += 1
        count = len(split[name])
        by_split[name] = {
            "count": count,
            "adjacent_train_frame_count": adjacent,
            "adjacent_train_frame_ratio": round(adjacent / count, 6),
            "sandwiched_by_train_frames_count": sandwiched,
            "within_two_train_frames_count": within_two,
            "maximum_nearest_train_distance": max(minimum_distances),
        }
    return by_split


def candidate_score(
    split: Mapping[str, Sequence[Mapping[str, Any]]],
    require_new_only_in_eval: bool,
) -> Tuple[int, ...]:
    neighbors = neighbor_metrics(split)
    only_new_coverage = 1
    if require_new_only_in_eval:
        only_new_coverage = int(
            all(any(row["class_ids"] == {5} for row in split[name]) for name in ("dev", "lock"))
        )
    dev_objects = object_counts(split["dev"])
    lock_objects = object_counts(split["lock"])
    class_balance = sum(abs(dev_objects[key] - lock_objects[key]) for key in set(dev_objects) | set(lock_objects))
    return (
        only_new_coverage,
        min(
            neighbors["dev"]["adjacent_train_frame_count"],
            neighbors["lock"]["adjacent_train_frame_count"],
        ),
        neighbors["dev"]["adjacent_train_frame_count"]
        + neighbors["lock"]["adjacent_train_frame_count"],
        neighbors["dev"]["sandwiched_by_train_frames_count"]
        + neighbors["lock"]["sandwiched_by_train_frames_count"],
        -class_balance,
    )


def reproduce_score_priority_split(
    rows: Sequence[Dict[str, Any]],
    target_counts: Mapping[str, int],
    selected_seed: int,
    group_random_seeds: Mapping[str, int],
    require_new_only_in_eval: bool = False,
) -> Tuple[Dict[str, List[Dict[str, Any]]], int, Tuple[int, ...], Dict[str, Dict[str, int]]]:
    quotas = build_group_quotas(rows, target_counts)
    split = assign_frozen_split(rows, quotas, group_random_seeds)
    score = candidate_score(split, require_new_only_in_eval)
    if {name: len(values) for name, values in split.items()} != dict(target_counts):
        raise RuntimeError("Frozen split counts do not match targets")
    return split, selected_seed, score, quotas


def distribution(split: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, rows in split.items():
        result[name] = {
            "images": len(rows),
            "sensor": dict(sorted(Counter(row["sensor"] for row in rows).items())),
            "scene": dict(sorted(Counter(row["scene"] for row in rows).items())),
            "sensor_scene": dict(sorted(Counter(row["sequence"] for row in rows).items())),
            "class_images": {
                str(key): value for key, value in sorted(class_image_counts(rows).items())
            },
            "class_objects": {
                str(key): value for key, value in sorted(object_counts(rows).items())
            },
        }
    return result


def write_split_lists(
    base: Mapping[str, Sequence[Mapping[str, Any]]],
    increment: Mapping[str, Sequence[Mapping[str, Any]]],
    rounds: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    rows_by_name: Dict[str, List[str]] = {}
    for name in ("train", "dev", "lock"):
        rows_by_name["base_{}".format(name)] = [row["image_path"] for row in base[name]]
        rows_by_name["increment_{}".format(name)] = [row["image_path"] for row in increment[name]]
        assigned: List[str] = []
        for round_spec in rounds:
            round_id = str(round_spec["round_id"])
            new_ids = set(int(value) for value in round_spec["new_class_ids"])
            selected = [
                row["image_path"]
                for row in increment[name]
                if set(row["class_ids"]) & new_ids
            ]
            rows_by_name["{}_{}".format(round_id, name)] = selected
            assigned.extend(selected)
        if sorted(assigned) != sorted(rows_by_name["increment_{}".format(name)]):
            raise ValueError("Increment {} rows are not uniquely owned by rounds".format(name))
    rows_by_name["base_train_plus_dev"] = sorted(
        rows_by_name["base_train"] + rows_by_name["base_dev"]
    )
    rows_by_name["increment_train_plus_dev"] = sorted(
        rows_by_name["increment_train"] + rows_by_name["increment_dev"]
    )
    rows_by_name["base_all"] = sorted(
        rows_by_name["base_train"] + rows_by_name["base_dev"] + rows_by_name["base_lock"]
    )
    rows_by_name["increment_all"] = sorted(
        rows_by_name["increment_train"]
        + rows_by_name["increment_dev"]
        + rows_by_name["increment_lock"]
    )
    rows_by_name["mixed_dev"] = sorted(rows_by_name["base_dev"] + rows_by_name["increment_dev"])
    rows_by_name["mixed_lock"] = sorted(rows_by_name["base_lock"] + rows_by_name["increment_lock"])

    result: Dict[str, Dict[str, Any]] = {}
    for name, rows in sorted(rows_by_name.items()):
        path = OUTPUT_ROOT / "{}.txt".format(name)
        write_lines(path, rows)
        result[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "count": len(rows),
        }
    return result


def create_readme(manifest: Mapping[str, Any]) -> None:
    base_counts = manifest["base_dataset"]["counts"]
    inc_counts = manifest["increment_dataset"]["counts"]
    round_rows = "\n".join(
        "| {round_id} {names} | {train} | {dev} | {lock} | — | {total} |".format(
            round_id=row["round_id"],
            names="/".join(
                manifest["class_map"][str(class_id)]
                for class_id in row["new_global_class_ids"]
            ),
            train=row["counts"]["train"],
            dev=row["counts"]["dev"],
            lock=row["counts"]["lock"],
            total=sum(row["counts"].values()),
        )
        for row in manifest["increment_rounds"]["rounds"]
    )
    text = """# 正式 4+2 比赛优先数据划分

该目录是 `4 旧类 + 2 新类` 正式工作的固定数据清单。划分以比赛得分为优先，
按 `sensor | scene` 分层后在帧级随机分配，不做时间隔离，并显式优选评估帧旁边存在训练帧的随机种子。

| 数据 | train | dev | lock | train+dev | all |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 四类 Base | {base_train} | {base_dev} | {base_lock} | {base_refit} | {base_all} |
| R2 二类增量 | {inc_train} | {inc_dev} | {inc_lock} | {inc_refit} | {inc_all} |
{round_rows}

- 首轮开发：使用 `*_train.txt` 训练、`*_dev.txt` 选参、`*_lock.txt` 冻结评分。
- 比赛复训：超参和阈值冻结后，可使用 `*_train_plus_dev.txt` 重训，lock 仍保留。
- `*_all.txt` 仅用于明确放弃本地 lock 独立性后的全量复训。
- R2 总清单按类别注册表预先固化为至少两轮不同新增类别；每轮训练只投影该轮类别。
- 该工具只生成划分，不启动训练。

具体随机种子、各层配额、类别/传感器/场景分布和相邻帧覆盖率见 `manifest.json`。
""".format(
        base_train=base_counts["train"],
        base_dev=base_counts["dev"],
        base_lock=base_counts["lock"],
        base_refit=base_counts["train"] + base_counts["dev"],
        base_all=sum(base_counts.values()),
        inc_train=inc_counts["train"],
        inc_dev=inc_counts["dev"],
        inc_lock=inc_counts["lock"],
        inc_refit=inc_counts["train"] + inc_counts["dev"],
        inc_all=sum(inc_counts.values()),
        round_rows=round_rows,
    )
    (OUTPUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="归档 3+1 划分并生成比赛优先的固定 4+2 划分。")
    parser.add_argument(
        "--round-registry", type=Path, default=ROOT / DEFAULT_ROUND_REGISTRY
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只验证已生成的划分与归档是否完整。",
    )
    args = parser.parse_args()
    round_registry = load_incremental_round_registry(args.round_registry)
    class_names = [
        round_registry["class_names"][class_id]
        for class_id in sorted(round_registry["class_names"])
    ]
    base_class_ids = list(round_registry["base"]["class_ids"])
    rounds = list(round_registry["rounds"])
    increment_class_ids = [
        class_id for row in rounds for class_id in row["new_class_ids"]
    ]

    if args.verify_only:
        archive = json.loads((ARCHIVE_ROOT / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
        for relative in archive["files"]:
            path = ARCHIVE_ROOT / "snapshot" / relative
            if not path.is_file():
                raise RuntimeError("Archive verification failed: {}".format(path))
        manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest["lists"].values():
            path = ROOT / item["path"]
            rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
            if len(rows) != item["count"]:
                raise RuntimeError("Split verification failed: {}".format(path))
        print("archive_files={} split_lists={} status=verified".format(archive["file_count"], len(manifest["lists"])))
        return 0

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError("Refusing to overwrite fixed 4+2 split: {}".format(OUTPUT_ROOT))

    archive = archive_legacy_split()
    base_rows = scan_dataset(
        R1_ROOT,
        "datasets_r1_base_train",
        [class_names[class_id] for class_id in base_class_ids],
    )
    increment_rows = scan_dataset(R2_ROOT, "datasets_r2_inc_train", class_names)
    if len(base_rows) != 750 or len(increment_rows) != 140:
        raise ValueError("Unexpected source counts: R1={} R2={}".format(len(base_rows), len(increment_rows)))

    base_split, base_seed, base_score, base_quotas = reproduce_score_priority_split(
        base_rows,
        BASE_TARGET_COUNTS,
        BASE_SELECTED_SEED,
        FROZEN_GROUP_RANDOM_SEEDS["base"],
    )
    increment_split, increment_seed, increment_score, increment_quotas = reproduce_score_priority_split(
        increment_rows,
        INCREMENT_TARGET_COUNTS,
        INCREMENT_SELECTED_SEED,
        FROZEN_GROUP_RANDOM_SEEDS["increment"],
        require_new_only_in_eval=True,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    lists = write_split_lists(base_split, increment_split, rounds)

    incremental_owners = {
        str(row["specialist"]["model_id"]): {
            "round_id": row["round_id"],
            "global_class_ids": row["new_class_ids"],
            "local_to_global": {
                str(key): value
                for key, value in row["specialist"]["local_to_global"].items()
            },
        }
        for row in rounds
    }
    increment_rounds = {
        "registry": Path(round_registry["path"]).relative_to(ROOT).as_posix(),
        "materializer": "tools/11_prepare_incremental_round_splits.py",
        "created_before_round_training": True,
        "source_lists_unchanged": True,
        "rounds_are_pairwise_disjoint": True,
        "rounds_cover_increment_source_lists": True,
        "rounds": [
            {
                "round_id": row["round_id"],
                "round_index": row["round_index"],
                "parent_generation_id": row["parent_generation_id"],
                "generation_id": row["generation_id"],
                "new_global_class_ids": row["new_class_ids"],
                "counts": {
                    role: lists["{}_{}".format(row["round_id"], role)]["count"]
                    for role in ("train", "dev", "lock")
                },
            }
            for row in rounds
        ],
    }

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": "competition_score_priority_strict_4plus2_partition",
        "created_on": "2026-08-21",
        "source_git_commit": git_head(),
        "class_map": {str(index): name for index, name in enumerate(class_names)},
        "owners": {
            "frozen_base_model": {"global_class_ids": base_class_ids},
            **incremental_owners,
        },
        "allocation_policy": {
            "objective": "competition_score_priority",
            "ratios": {"train": 0.8, "dev": 0.1, "lock": 0.1},
            "method": "sensor_scene_stratified_frame_level_random",
            "temporal_isolation": False,
            "temporal_buffer_count": 0,
            "adjacent_frame_leakage_intentional": True,
            "seed_selection": {
                "range_start_inclusive": SEED_SEARCH_START,
                "range_end_exclusive": SEED_SEARCH_START + SEED_SEARCH_COUNT,
                "lexicographic_objectives": [
                    "R2 new-only armored_vehicle coverage in both dev and lock",
                    "maximize minimum dev/lock count with an adjacent train frame",
                    "maximize total dev/lock count with an adjacent train frame",
                    "maximize evaluation frames sandwiched by train frames",
                    "minimize dev/lock object-count imbalance",
                ],
            },
        },
        "base_dataset": {
            "source": "datasets_r1_base_train",
            "source_images": len(base_rows),
            "global_class_ids": base_class_ids,
            "counts": dict(BASE_TARGET_COUNTS),
            "selected_seed": base_seed,
            "selection_score": list(base_score),
            "sequence_quotas": base_quotas,
            "distribution": distribution(base_split),
            "temporal_leakage": neighbor_metrics(base_split),
        },
        "increment_dataset": {
            "source": "datasets_r2_inc_train",
            "source_images": len(increment_rows),
            "source_objects": sum(sum(row["object_counts"].values()) for row in increment_rows),
            "new_global_class_ids": increment_class_ids,
            "old_labels_retained_for_audit": True,
            "new_head_label_projection_required": True,
            "counts": dict(INCREMENT_TARGET_COUNTS),
            "selected_seed": increment_seed,
            "selection_score": list(increment_score),
            "sequence_quotas": increment_quotas,
            "distribution": distribution(increment_split),
            "temporal_leakage": neighbor_metrics(increment_split),
        },
        "increment_rounds": increment_rounds,
        "lists": lists,
        "release_refit_policy": {
            "development": "train only; dev for selection; lock opened after prediction freeze",
            "competition_refit": "train_plus_dev after hyperparameters and thresholds are frozen",
            "full_data_retrain_only": "all may be used only after explicitly giving up local lock independence",
        },
        "legacy_archive": {
            "path": ARCHIVE_ROOT.relative_to(ROOT).as_posix(),
            "source_protocol": archive["source_protocol"],
            "file_count": archive["file_count"],
        },
        "training_started": False,
    }
    write_json(OUTPUT_ROOT / "manifest.json", manifest)
    create_readme(manifest)
    active = {
        "schema_version": 1,
        "active_protocol": "strict_4plus2",
        "manifest": "splits/strict_4plus2/manifest.json",
        "legacy_compatibility_protocol": "splits/strict_3plus1/manifest.json",
        "updated_on": "2026-08-21",
    }
    write_json(SPLITS_ROOT / "active.json", active)
    print(
        "base={}/{}/{} increment={}/{}/{} base_seed={} increment_seed={} training_started=false".format(
            len(base_split["train"]),
            len(base_split["dev"]),
            len(base_split["lock"]),
            len(increment_split["train"]),
            len(increment_split["dev"]),
            len(increment_split["lock"]),
            base_seed,
            increment_seed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
