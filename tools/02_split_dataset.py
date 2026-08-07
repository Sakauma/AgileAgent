#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from math import floor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fair_agent.dataset_utils import (
    REPORTS_DIR,
    count_class_presence,
    count_values,
    read_classes,
    read_metadata,
)


SEED = 20260705
TARGET_COUNTS = {"train": 560, "dev_val": 95, "lock_val": 95}

ACTIVE_SPLITS_DIR = ROOT / "splits"
LEGACY_SPLITS_DIR = ROOT / "archive" / "splits_legacy_random_560_95_95"
TEMPORAL_TRAIN_RATIO = 0.70
TEMPORAL_DEV_RATIO = 0.15
TEMPORAL_LOCK_RATIO = 0.15
EMBARGO_FRAME_DISTANCE = 4
DEFAULT_SIMULATED_INCREMENT_CLASS = "warship"


# ---------------------------------------------------------------------------
# Legacy random split. Kept so the published v1 manifests remain reproducible.
# ---------------------------------------------------------------------------


def group_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    detailed = defaultdict(list)
    for row in rows:
        key = f"{row['sensor']}|{row['scene']}|{row['classes_present']}"
        detailed[key].append(row)

    grouped = defaultdict(list)
    for key, group in detailed.items():
        sensor, scene, _classes_present = key.split("|", 2)
        final_key = key if len(group) >= 3 else f"{sensor}|{scene}"
        grouped[final_key].extend(group)
    return dict(grouped)


def allocate_exact(group_sizes: Dict[str, int], target: int, total: int) -> Dict[str, int]:
    raw = []
    for key, size in group_sizes.items():
        quota = size * target / total
        base = floor(quota)
        raw.append((key, base, quota - base))
    allocation = {key: base for key, base, _rem in raw}
    remaining = target - sum(allocation.values())
    for key, _base, _rem in sorted(raw, key=lambda item: (-item[2], item[0]))[:remaining]:
        allocation[key] += 1
    return allocation


def split_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    rng = random.Random(SEED)
    groups = group_rows(rows)
    for group in groups.values():
        rng.shuffle(group)

    total = len(rows)
    group_sizes = {key: len(group) for key, group in groups.items()}
    train_alloc = allocate_exact(group_sizes, TARGET_COUNTS["train"], total)
    remaining_sizes = {key: group_sizes[key] - train_alloc[key] for key in groups}
    dev_alloc = allocate_exact(
        remaining_sizes,
        TARGET_COUNTS["dev_val"],
        total - TARGET_COUNTS["train"],
    )

    splits = {"train": [], "dev_val": [], "lock_val": []}
    for key in sorted(groups):
        group = groups[key]
        train_count = train_alloc[key]
        dev_count = dev_alloc[key]
        splits["train"].extend(group[:train_count])
        splits["dev_val"].extend(group[train_count : train_count + dev_count])
        splits["lock_val"].extend(group[train_count + dev_count :])

    for values in splits.values():
        values.sort(key=lambda row: row["image_path"])
    return splits


def write_split(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(row["image_path"] for row in rows) + "\n", encoding="utf-8")


def format_counter(counter: Counter) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}: {value}" for key, value in sorted(counter.items()))


def split_report(splits: Dict[str, List[Dict[str, str]]]) -> str:
    all_paths = []
    lines = [
        "# 数据划分报告",
        "",
        f"随机种子：`{SEED}`",
        "",
        "| 划分 | 图片数 | 传感器 | 场景 | 包含类别 |",
        "|---|---:|---|---|---|",
    ]
    for split_name in ["train", "dev_val", "lock_val"]:
        values = splits[split_name]
        all_paths.extend(row["image_path"] for row in values)
        lines.append(
            f"| {split_name} | {len(values)} | {format_counter(count_values(values, 'sensor'))} | "
            f"{format_counter(count_values(values, 'scene'))} | "
            f"{format_counter(count_class_presence(values))} |"
        )

    duplicates = len(all_paths) - len(set(all_paths))
    lines.extend(
        [
            "",
            "## 完整性检查",
            "",
            f"- 总图片数：{len(all_paths)}",
            f"- 重复图片数：{duplicates}",
            f"- 互斥划分：{'是' if duplicates == 0 else '否'}",
            "",
            "## 说明",
            "",
            "该随机逐帧划分只用于复现历史实验；当前 3+1 性能测试使用活动 splits。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_legacy_splits(
    rows: Sequence[Dict[str, str]], output_dir: Path = LEGACY_SPLITS_DIR
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空历史划分目录：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_rows(rows)
    for split_name, values in splits.items():
        write_split(output_dir / f"{split_name}.txt", values)
    for sensor in ["ir", "sar"]:
        for split_name in ["train", "dev_val", "lock_val"]:
            write_split(
                output_dir / f"{split_name}_{sensor}.txt",
                [row for row in splits[split_name] if row["sensor"] == sensor],
            )
    (output_dir / "split_report.md").write_text(split_report(splits), encoding="utf-8")
    print(" ".join(f"{name}={len(value)}" for name, value in splits.items()))


# ---------------------------------------------------------------------------
# Full-coverage source pools and one fixed strict 3+1 simulation.
# ---------------------------------------------------------------------------


def class_ids(row: Mapping[str, str]) -> set[int]:
    return {int(value) for value in row.get("class_ids", "").split(";") if value != ""}


def sequence_key(row: Mapping[str, str]) -> str:
    return f"{row['sensor']}|{row['dataset_round']}|{row['scene']}"


def frame_id(row: Mapping[str, str]) -> int:
    try:
        return int(row["image_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"无法解析连续帧编号：{row.get('image_path', '<unknown>')}") from exc


def _trim_later_split(
    earlier: Sequence[Dict[str, str]],
    later: List[Dict[str, str]],
    embargo: List[Dict[str, str]],
    minimum_distance: int,
) -> None:
    if not earlier or not later:
        raise ValueError("每个时间序列必须同时覆盖训练、开发和测试。")
    last_earlier = frame_id(earlier[-1])
    while later and frame_id(later[0]) - last_earlier <= minimum_distance:
        embargo.append(later.pop(0))
    if not later:
        raise ValueError("连续帧隔离带耗尽了后续划分。")


def full_coverage_source_pools(
    rows: Sequence[Dict[str, str]],
    historical_embargo_frame_distance: int = EMBARGO_FRAME_DISTANCE,
) -> tuple[Dict[str, List[Dict[str, str]]], List[Dict[str, Any]]]:
    """Preserve the existing dev/test members and reclaim every embargo image.

    The previous active split removed four frames on each side of a temporal
    boundary.  Temporal isolation is not part of the competition protocol, so
    those images now belong to the training pool.  The old boundary calculation
    remains here only to make the preserved dev/test membership reproducible.
    """
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[sequence_key(row)].append(row)

    pools: Dict[str, List[Dict[str, str]]] = {
        "pool_train": [],
        "pool_dev": [],
        "mixed_test": [],
        "embargo": [],
    }
    details: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        ordered = sorted(group, key=frame_id)
        if len(ordered) < 12:
            raise ValueError(f"时间序列 {key} 只有 {len(ordered)} 张，无法可靠三分。")
        if len({frame_id(row) for row in ordered}) != len(ordered):
            raise ValueError(f"时间序列 {key} 存在重复帧编号。")

        dev_count = max(1, round(len(ordered) * TEMPORAL_DEV_RATIO))
        test_count = max(1, round(len(ordered) * TEMPORAL_LOCK_RATIO))
        train_count = len(ordered) - dev_count - test_count
        train_rows = list(ordered[:train_count])
        dev_rows = list(ordered[train_count : train_count + dev_count])
        test_rows = list(ordered[train_count + dev_count :])
        embargo_rows: List[Dict[str, str]] = []
        _trim_later_split(
            train_rows,
            dev_rows,
            embargo_rows,
            historical_embargo_frame_distance,
        )
        _trim_later_split(
            dev_rows,
            test_rows,
            embargo_rows,
            historical_embargo_frame_distance,
        )

        historical_frame_bounds = {
            "train_last": frame_id(train_rows[-1]),
            "dev_first": frame_id(dev_rows[0]),
            "dev_last": frame_id(dev_rows[-1]),
            "test_first": frame_id(test_rows[0]),
        }
        reclaimed_rows = list(embargo_rows)
        train_rows.extend(reclaimed_rows)

        pools["pool_train"].extend(train_rows)
        pools["pool_dev"].extend(dev_rows)
        pools["mixed_test"].extend(test_rows)
        details.append(
            {
                "sequence": key,
                "source_count": len(ordered),
                "counts": {
                    "pool_train": len(train_rows),
                    "pool_dev": len(dev_rows),
                    "mixed_test": len(test_rows),
                    "embargo": 0,
                },
                "reclaimed_to_pool_train": len(reclaimed_rows),
                "historical_frame_bounds": historical_frame_bounds,
            }
        )

    for values in pools.values():
        values.sort(key=lambda row: row["image_path"])
    _validate_full_coverage_pools(rows, pools)
    return pools, details


def _validate_full_coverage_pools(
    source_rows: Sequence[Dict[str, str]],
    pools: Mapping[str, Sequence[Dict[str, str]]],
) -> None:
    expected = {row["image_path"] for row in source_rows}
    all_paths = [row["image_path"] for values in pools.values() for row in values]
    if len(all_paths) != len(set(all_paths)) or set(all_paths) != expected:
        raise RuntimeError("活动源池没有互斥且完整覆盖 750 张图像。")
    if pools.get("embargo"):
        raise RuntimeError("750 张全量划分不允许保留未使用的 embargo 图像。")


def _class_image_counts(
    rows: Sequence[Mapping[str, str]], values: Iterable[int]
) -> Dict[str, int]:
    wanted = set(values)
    counts = Counter()
    for row in rows:
        for value in class_ids(row) & wanted:
            counts[str(value)] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_strict_3plus1_protocol(
    pools: Mapping[str, Sequence[Dict[str, str]]],
    class_names: Sequence[str],
    increment_class_name: str,
    output_dir: Path,
) -> Dict[str, Any]:
    ids_by_name = {name.casefold(): index for index, name in enumerate(class_names)}
    try:
        new_id = ids_by_name[increment_class_name.casefold()]
    except KeyError as exc:
        raise ValueError(
            f"模拟新增类别 {increment_class_name!r} 不在 classes.txt：{list(class_names)}"
        ) from exc
    all_ids = set(range(len(class_names)))
    if len(all_ids) != 4:
        raise ValueError(f"严格 3+1 模拟要求当前数据恰好四类，实际为 {len(all_ids)} 类。")
    old_ids = all_ids - {new_id}

    def partition_detection(
        rows: Sequence[Dict[str, str]], split_name: str
    ) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        base_rows: List[Dict[str, str]] = []
        increment_rows: List[Dict[str, str]] = []
        for row in rows:
            present = class_ids(row)
            if new_id in present:
                if present != {new_id}:
                    raise ValueError(
                        f"{split_name} 中 {row['image_path']} 同时含基础类与模拟新增类；"
                        "无法满足增量数据只包含新增类别。"
                    )
                increment_rows.append(row)
            else:
                if not present or not present <= old_ids:
                    raise ValueError(f"{split_name} 中存在不属于三类基础集合的标签：{row}")
                base_rows.append(row)
        return base_rows, increment_rows

    base_train, increment_train = partition_detection(pools["pool_train"], "pool_train")
    base_dev, increment_dev = partition_detection(pools["pool_dev"], "pool_dev")
    mixed_test = list(pools["mixed_test"])
    old_test = [row for row in mixed_test if class_ids(row) & old_ids]
    new_test = [row for row in mixed_test if new_id in class_ids(row)]
    if {row["image_path"] for row in old_test + new_test} != {
        row["image_path"] for row in mixed_test
    }:
        raise RuntimeError("混合测试集没有被旧类图与新增类图完整覆盖。")

    lists = {
        "base_train": base_train,
        "base_dev": base_dev,
        "increment_train": increment_train,
        "increment_dev": increment_dev,
        "mixed_test": mixed_test,
        # 场景识别是独立的已知场景任务，可使用所有类别来源的图像，
        # 但训练代码只能读取场景/传感器标签，不能读取目标类别标签。
        "scene_train": list(pools["pool_train"]),
        "scene_dev": list(pools["pool_dev"]),
        "scene_test": mixed_test,
    }
    for name, values in lists.items():
        write_split(output_dir / f"{name}.txt", values)

    same_image = {
        row["image_path"] for row in old_test
    } & {row["image_path"] for row in new_test}
    manifest = {
        "schema_version": 2,
        "protocol": "strict_3plus1_class_incremental_simulation",
        "simulation_only": True,
        "base_class_ids": sorted(old_ids),
        "base_class_names": [class_names[value] for value in sorted(old_ids)],
        "increment_class_id": new_id,
        "increment_class_name": class_names[new_id],
        "counts": {name: len(values) for name, values in lists.items()},
        "mixed_test_composition": {
            "old_class_images": len(old_test),
            "new_class_images": len(new_test),
            "membership_lists_published": False,
        },
        "paths": {
            name: (output_dir / f"{name}.txt").relative_to(ROOT).as_posix()
            for name in lists
        },
        "detection_contract": {
            "base_training_classes": sorted(old_ids),
            "base_training_contains_increment_class": False,
            "increment_training_classes": [new_id],
            "increment_training_contains_base_classes": False,
            "test_inference_scope": "complete_mixed_test_for_parent_and_candidate",
            "same_image_old_new_required": False,
            "same_image_old_new_count": len(same_image),
        },
        "scene_contract": {
            "task": "known_scene_and_sensor_classification",
            "known_scenes": sorted({row["scene"] for values in pools.values() for row in values}),
            "uses_all_source_classes": True,
            "target_class_labels_access": False,
            "shared_detector_features_allowed": False,
            "scene_to_target_class_hard_binding_allowed": False,
        },
        "evaluation": {
            "base_test_map": "所有 owner 完成完整 mixed_test 无标签推理并冻结预测后，按不含 increment_class_id 的图像子集及 base_class_ids 计分。",
            "old_map_before": "父代在完整 mixed_test 上推理后按 base_class_ids 计分。",
            "old_map_after": "增量后候选代在同一完整 mixed_test 上按 base_class_ids 计分。",
            "new_map": "候选代在同一完整 mixed_test 上按 increment_class_id 计分。",
            "krr": "old_map_after / old_map_before",
            "score_gates": {
                "base_test_map50": 0.80,
                "new_map50": 0.60,
                "krr": 0.95,
            },
            "full_map50_role": "diagnostic_only",
            "unlabeled_inference_before_scoring": True,
            "label_aware_routing_allowed": False,
            "scene_hard_routing_allowed": False,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def active_readme(
    pools: Mapping[str, Sequence[Dict[str, str]]],
    protocol: Mapping[str, Any],
) -> str:
    counts = protocol["counts"]
    return "\n".join(
        [
            "# 严格 3+1 类别增量数据划分",
            "",
            "该目录只描述一套固定的 3+1 模拟实验，不做交叉验证，也不轮换新增类别。",
            "当前实例暂时把 `warship` 作为模拟新增类别；正式官方增量数据到达后，当前四类全部属于基础类。",
            "",
            "## 检测模型数据边界",
            "",
            "| 阶段 | 清单 | 图片数 | 可见目标类别 |",
            "|---|---|---:|---|",
            f"| 三类基础训练 | `strict_3plus1/base_train.txt` | {counts['base_train']} | {', '.join(protocol['base_class_names'])} |",
            f"| 三类基础验证 | `strict_3plus1/base_dev.txt` | {counts['base_dev']} | {', '.join(protocol['base_class_names'])} |",
            f"| 单类增量训练 | `strict_3plus1/increment_train.txt` | {counts['increment_train']} | {protocol['increment_class_name']} |",
            f"| 单类增量验证 | `strict_3plus1/increment_dev.txt` | {counts['increment_dev']} | {protocol['increment_class_name']} |",
            f"| 最终混合测试 | `strict_3plus1/mixed_test.txt` | {counts['mixed_test']} | 全部四类 |",
            "",
            f"混合测试集由 {protocol['mixed_test_composition']['old_class_images']} 张旧类图和 "
            f"{protocol['mixed_test_composition']['new_class_images']} 张新增类图组成；不要求同一张图同时含旧类和新类。",
            "活动目录不发布旧/新增类别成员清单，单张测试图身份在预测冻结前保持未知。",
            "冻结基础检测器和增量专家都必须先对完整混合测试集的每张图执行无标签推理并冻结预测，再解封标签评分。",
            "正式门槛固定为基础测试代理 mAP50 >= 0.80、New-mAP50 >= 0.60、KRR >= 0.95；base_dev 只用于选权重，四类总体 mAP50 只作诊断。",
            "不得依据测试标签、文件名、数据集身份或场景类别决定是否运行某个类别 owner。",
            "",
            "`pool_train.txt` 与 `pool_dev.txt` 只是生成上述模型专用清单的源池，不能直接作为三类基础检测器的训练数据。",
            "",
            "## 已知场景识别",
            "",
            f"场景模型使用 `scene_train/dev/test.txt`（{counts['scene_train']}/{counts['scene_dev']}/{counts['scene_test']}），"
            "覆盖 air、forest、sea、urban 全部已知场景。场景训练只能读取场景/传感器标签，不得读取目标类别标签、"
            "共享检测器特征或建立场景到目标类别的硬绑定。",
            "",
            "## 750 张全量覆盖",
            "",
            f"源池为 {len(pools['pool_train'])}/{len(pools['pool_dev'])}/{len(pools['mixed_test'])}，"
            "三者互斥且恰好覆盖全部 750 张图。上一版的 51 张边界隔离图已全部并入训练源池，"
            "活动划分不再强制连续帧边界间距；3+1 类别隔离和测试标签封存约束保持不变。"
            "上一版严格时序划分已归档到 `archive/splits_strict_temporal_3plus1_405_117/`，"
            "旧随机逐帧划分已归档到 `archive/splits_legacy_random_560_95_95/`。",
            "",
            "可用其他可独立拆分的类别重新生成模板实例：",
            "",
            "```bash",
            "python tools/02_split_dataset.py --protocol strict-3plus1 --increment-class warship --output-dir reports/splits_check",
            "```",
            "",
        ]
    )


def write_strict_3plus1_splits(
    rows: Sequence[Dict[str, str]],
    output_dir: Path = ACTIVE_SPLITS_DIR,
    increment_class_name: str = DEFAULT_SIMULATED_INCREMENT_CLASS,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空活动划分目录：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = read_classes()
    all_ids = set(range(len(class_names)))
    observed_ids = {value for row in rows for value in class_ids(row)}
    if observed_ids != all_ids:
        raise ValueError(
            f"metadata 类别 ID 与 classes.txt 不一致：metadata={sorted(observed_ids)} "
            f"classes={sorted(all_ids)}"
        )

    pools, sequence_details = full_coverage_source_pools(rows)
    for name, values in pools.items():
        write_split(output_dir / f"{name}.txt", values)
    for sensor in ("ir", "sar"):
        for name in ("pool_train", "pool_dev", "mixed_test"):
            write_split(
                output_dir / f"{name}_{sensor}.txt",
                [row for row in pools[name] if row["sensor"] == sensor],
            )

    protocol_dir = output_dir / "strict_3plus1"
    protocol = write_strict_3plus1_protocol(
        pools,
        class_names,
        increment_class_name,
        protocol_dir,
    )
    manifest = {
        "schema_version": 2,
        "protocol": "full_coverage_strict_3plus1_dataset_partition",
        "seed": SEED,
        "source_image_count": len(rows),
        "class_map": {str(index): name for index, name in enumerate(class_names)},
        "simulated_increment_class": protocol["increment_class_name"],
        "allocation_policy": {
            "all_source_images_used": True,
            "reclaimed_previous_embargo_to_pool_train": True,
            "reclaimed_image_count": sum(
                int(item["reclaimed_to_pool_train"]) for item in sequence_details
            ),
            "dev_and_test_membership_preserved": True,
            "temporal_gap_constraint": None,
        },
        "counts": {name: len(values) for name, values in pools.items()},
        "class_image_counts": {
            name: _class_image_counts(values, all_ids) for name, values in pools.items()
        },
        "sensor_counts": {
            name: dict(sorted(count_values(values, "sensor").items()))
            for name, values in pools.items()
        },
        "sequence_details": sequence_details,
        "strict_3plus1_manifest": (protocol_dir / "manifest.json").relative_to(ROOT).as_posix(),
    }
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(active_readme(pools, protocol), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="生成固定基础数据划分与严格 3+1 模拟清单。")
    parser.add_argument(
        "--protocol",
        choices=("strict-3plus1", "legacy"),
        default="strict-3plus1",
        help="strict-3plus1 生成唯一活动划分；legacy 只复现已归档的旧随机划分。",
    )
    parser.add_argument(
        "--increment-class",
        default=DEFAULT_SIMULATED_INCREMENT_CLASS,
        help="strict-3plus1 中临时作为新增类别的当前数据类别名称。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="输出目录；相对路径按仓库根目录解析。",
    )
    args = parser.parse_args()

    rows = read_metadata(REPORTS_DIR / "metadata.csv")
    if len(rows) != sum(TARGET_COUNTS.values()):
        print(f"Expected {sum(TARGET_COUNTS.values())} metadata rows, got {len(rows)}")
        return 1

    if args.protocol == "legacy":
        if args.increment_class != DEFAULT_SIMULATED_INCREMENT_CLASS:
            parser.error("--increment-class 不适用于 legacy")
        output_dir = args.output_dir or LEGACY_SPLITS_DIR
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        write_legacy_splits(rows, output_dir)
        return 0

    output_dir = args.output_dir or ACTIVE_SPLITS_DIR
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    manifest = write_strict_3plus1_splits(rows, output_dir, args.increment_class)
    strict = json.loads((output_dir / "strict_3plus1" / "manifest.json").read_text(encoding="utf-8"))
    print(
        " ".join(
            [
                *(f"{name}={count}" for name, count in manifest["counts"].items()),
                f"base_train={strict['counts']['base_train']}",
                f"increment_train={strict['counts']['increment_train']}",
                f"increment_class={strict['increment_class_name']}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
