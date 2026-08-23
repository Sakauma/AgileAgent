#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fair_agent.dataset_utils import (
    REPORTS_DIR,
    count_values,
    read_classes,
    read_metadata,
)


SEED = 20260705
EXPECTED_IMAGE_COUNT = 750

ACTIVE_SPLITS_DIR = ROOT / "splits"
DEV_WINDOW_RATIO = 0.15
TEST_WINDOW_RATIO = 0.15
EVALUATION_BOUNDARY_DISTANCE = 4
DEFAULT_SIMULATED_INCREMENT_CLASS = "warship"


def write_split(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(row["image_path"] for row in rows) + "\n", encoding="utf-8")


def class_ids(row: Mapping[str, str]) -> set[int]:
    return {int(value) for value in row.get("class_ids", "").split(";") if value != ""}


def sequence_key(row: Mapping[str, str]) -> str:
    return f"{row['sensor']}|{row['dataset_round']}|{row['scene']}"


def frame_id(row: Mapping[str, str]) -> int:
    try:
        return int(row["image_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"无法解析连续帧编号：{row.get('image_path', '<unknown>')}") from exc


def _assign_boundary_rows_to_training(
    earlier: Sequence[Dict[str, str]],
    later: List[Dict[str, str]],
    training: List[Dict[str, str]],
    minimum_distance: int,
) -> None:
    if not earlier or not later:
        raise ValueError("每个时间序列必须同时覆盖训练、开发和测试。")
    last_earlier = frame_id(earlier[-1])
    while later and frame_id(later[0]) - last_earlier <= minimum_distance:
        training.append(later.pop(0))
    if not later:
        raise ValueError("评估窗口没有剩余样本。")


def full_coverage_source_pools(
    rows: Sequence[Dict[str, str]],
    evaluation_boundary_distance: int = EVALUATION_BOUNDARY_DISTANCE,
) -> tuple[Dict[str, List[Dict[str, str]]], List[Dict[str, Any]]]:
    """Build deterministic sequence-aware train, development, and test pools."""
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[sequence_key(row)].append(row)

    pools: Dict[str, List[Dict[str, str]]] = {
        "pool_train": [],
        "pool_dev": [],
        "mixed_test": [],
    }
    details: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        ordered = sorted(group, key=frame_id)
        if len(ordered) < 12:
            raise ValueError(f"时间序列 {key} 只有 {len(ordered)} 张，无法可靠三分。")
        if len({frame_id(row) for row in ordered}) != len(ordered):
            raise ValueError(f"时间序列 {key} 存在重复帧编号。")

        dev_count = max(1, round(len(ordered) * DEV_WINDOW_RATIO))
        test_count = max(1, round(len(ordered) * TEST_WINDOW_RATIO))
        train_count = len(ordered) - dev_count - test_count
        train_rows = list(ordered[:train_count])
        dev_rows = list(ordered[train_count : train_count + dev_count])
        test_rows = list(ordered[train_count + dev_count :])
        boundary_training_rows: List[Dict[str, str]] = []
        _assign_boundary_rows_to_training(
            train_rows,
            dev_rows,
            boundary_training_rows,
            evaluation_boundary_distance,
        )
        _assign_boundary_rows_to_training(
            dev_rows,
            test_rows,
            boundary_training_rows,
            evaluation_boundary_distance,
        )

        evaluation_frame_bounds = {
            "train_last": frame_id(train_rows[-1]),
            "dev_first": frame_id(dev_rows[0]),
            "dev_last": frame_id(dev_rows[-1]),
            "test_first": frame_id(test_rows[0]),
        }
        train_rows.extend(boundary_training_rows)

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
                },
                "boundary_training_count": len(boundary_training_rows),
                "evaluation_frame_bounds": evaluation_frame_bounds,
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
    base_test = [row for row in mixed_test if class_ids(row) and new_id not in class_ids(row)]
    new_test = [row for row in mixed_test if new_id in class_ids(row)]
    if {row["image_path"] for row in base_test + new_test} != {
        row["image_path"] for row in mixed_test
    }:
        raise RuntimeError("混合测试集没有被旧类图与新增类图完整覆盖。")
    if {row["image_path"] for row in base_test} & {row["image_path"] for row in new_test}:
        raise ValueError("基础测试集不得包含模拟新增类别。")

    lists = {
        "base_train": base_train,
        "base_dev": base_dev,
        "increment_train": increment_train,
        "increment_dev": increment_dev,
        "base_test": base_test,
        "mixed_test": mixed_test,
        # 场景识别是独立的已知场景任务，可使用所有类别来源的图像，
        # 但训练代码只能读取场景/传感器标签，不能读取目标类别标签。
        "scene_train": list(pools["pool_train"]),
        "scene_dev": list(pools["pool_dev"]),
        "scene_test": mixed_test,
    }
    for name, values in lists.items():
        write_split(output_dir / f"{name}.txt", values)

    same_image = {row["image_path"] for row in base_test} & {
        row["image_path"] for row in new_test
    }
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
            "old_class_images": len(base_test),
            "new_class_images": len(new_test),
            "base_test_list_published_for_scoring_only": True,
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
            "base_test_map": "所有 owner 完成完整 mixed_test 无标签推理并冻结预测后，读取预先固定的 base_test 清单并按 base_class_ids 计分。",
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
            "# 固定 3+1 类别增量数据划分",
            "",
            f"本目录固定一套覆盖 750 张基础数据的 3+1 模拟实验。当前实例以 `{protocol['increment_class_name']}` 为新增类别，",
            f"基础类别为 {', '.join(f'`{name}`' for name in protocol['base_class_names'])}。",
            "",
            "## 源池分配",
            "",
            "图像按 `sensor | dataset_round | scene` 组成序列，并按帧号排序。每个序列末段形成测试窗口，",
            "其前一段形成开发窗口；评估窗口边界 4 帧范围内的样本归入训练源池。最终源池规模为：",
            "",
            "| 源池 | 图片数 | IR | SAR | 用途 |",
            "| --- | ---: | ---: | ---: | --- |",
            f"| `pool_train.txt` | {len(pools['pool_train'])} | "
            f"{sum(row['sensor'] == 'ir' for row in pools['pool_train'])} | "
            f"{sum(row['sensor'] == 'sar' for row in pools['pool_train'])} | 检测与场景训练清单的来源 |",
            f"| `pool_dev.txt` | {len(pools['pool_dev'])} | "
            f"{sum(row['sensor'] == 'ir' for row in pools['pool_dev'])} | "
            f"{sum(row['sensor'] == 'sar' for row in pools['pool_dev'])} | 检测与场景开发清单的来源 |",
            f"| `mixed_test.txt` | {len(pools['mixed_test'])} | "
            f"{sum(row['sensor'] == 'ir' for row in pools['mixed_test'])} | "
            f"{sum(row['sensor'] == 'sar' for row in pools['mixed_test'])} | 固定混合测试集 |",
            "",
            "三个源池互斥并完整覆盖 750 张图。`manifest.json` 记录分配规则、逐序列窗口边界、类别分布和传感器分布。",
            "",
            "## 检测模型清单",
            "",
            "| 阶段 | 清单 | 图片数 | 可见目标类别 |",
            "| --- | --- | ---: | --- |",
            f"| 三类基础训练 | `strict_3plus1/base_train.txt` | {counts['base_train']} | {', '.join(protocol['base_class_names'])} |",
            f"| 三类基础验证 | `strict_3plus1/base_dev.txt` | {counts['base_dev']} | {', '.join(protocol['base_class_names'])} |",
            f"| 单类增量训练 | `strict_3plus1/increment_train.txt` | {counts['increment_train']} | {protocol['increment_class_name']} |",
            f"| 单类增量验证 | `strict_3plus1/increment_dev.txt` | {counts['increment_dev']} | {protocol['increment_class_name']} |",
            f"| 基础测试 | `strict_3plus1/base_test.txt` | {counts['base_test']} | {', '.join(protocol['base_class_names'])} |",
            f"| 最终混合测试 | `strict_3plus1/mixed_test.txt` | {counts['mixed_test']} | 全部四类 |",
            "",
            f"混合测试集由 {protocol['mixed_test_composition']['old_class_images']} 张旧类图和 "
            f"{protocol['mixed_test_composition']['new_class_images']} 张新增类图组成。基础检测器与增量检测器先对完整 "
            f"{counts['mixed_test']} 张混合测试集执行无标签推理并冻结预测，评分器随后读取固定标签与评分清单：",
            "",
            "| 指标 | 评分范围 | 发布门槛 |",
            "| --- | --- | ---: |",
            "| 基础 mAP50 | `base_test.txt` 中的基础类别 | `0.80` |",
            "| New-mAP50 | 完整 `mixed_test.txt` 中的新增类别 | `0.60` |",
            "| KRR | 完整 `mixed_test.txt` 中增量前后的基础类别 mAP50 比值 | `0.95` |",
            "",
            "`base_train.txt`、`base_dev.txt`、`increment_train.txt` 和 `increment_dev.txt` 是检测训练入口。",
            "`pool_train.txt` 与 `pool_dev.txt` 生成这些类别隔离清单。在线路由使用当前 production 代际、",
            "无标签图像内容和场景软证据；评分器在预测冻结后读取评分标签与评分子集。",
            "",
            "## 已知场景识别",
            "",
            f"场景模型使用 `strict_3plus1/scene_train.txt`、`scene_dev.txt` 和 `scene_test.txt`，规模为 "
            f"{counts['scene_train']}/{counts['scene_dev']}/{counts['scene_test']}，覆盖 air、forest、sea、urban 四个已知场景。",
            "场景模型训练输入由图像、传感器标签和场景标签组成。",
            "",
            "## 重新生成",
            "",
            "使用可独立拆分的四类数据和 metadata 生成同一协议：",
            "",
            "```bash",
            "python tools/02_split_dataset.py \\",
            f"  --increment-class {protocol['increment_class_name']} \\",
            "  --output-dir reports/splits_check",
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
            "sequence_ordered_evaluation": True,
            "evaluation_boundary_distance": EVALUATION_BOUNDARY_DISTANCE,
            "boundary_images_assigned_to_training": True,
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
    if len(rows) != EXPECTED_IMAGE_COUNT:
        print(f"Expected {EXPECTED_IMAGE_COUNT} metadata rows, got {len(rows)}")
        return 1

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
