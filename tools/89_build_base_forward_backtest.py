#!/usr/bin/env python3
"""Build expanding-window base backtests from an existing blocked CV manifest.

The tuning window trains only on earlier temporal blocks and validates on the
next block.  A later block is kept out of tuning and becomes a post-selection
validation window.  No mixed-test path or label is read.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import read_split, sha256_file


FORBIDDEN_MARKERS = ("mixed_test", "base_test", "lock")


def reject_test_reference(path: Path, role: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"前向回测 {role} 不得引用 test/lock：{path}")


def sequence_position(path: Path) -> tuple[str, int]:
    prefix, separator, frame = path.stem.rpartition("_")
    if not separator or not frame.isdigit():
        raise ValueError(f"图像名缺少数值帧号：{path.name}")
    return prefix, int(frame)


def write_split(path: Path, images: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(str(image.resolve()) for image in images) + "\n",
        encoding="utf-8",
    )


def load_blocks(manifest_path: Path) -> tuple[dict[int, list[Path]], Mapping[str, Any]]:
    manifest_path = manifest_path.resolve()
    reject_test_reference(manifest_path, "source manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("selection_scope") != "base_train_and_dev_only"
        or bool(manifest.get("lock_data_access", True))
        or manifest.get("strategy") != "per_sequence_contiguous_block_kfold"
    ):
        raise ValueError("source manifest 不是无测试连续块交叉验证清单")

    blocks: dict[int, list[Path]] = {}
    expected_fold_ids = list(range(int(manifest["fold_count"])))
    for row in manifest["folds"]:
        fold_id = int(row["fold"])
        split_path = Path(str(row["val_split"])).resolve()
        reject_test_reference(split_path, f"fold_{fold_id} validation")
        if sha256_file(split_path) != str(row["val_split_sha256"]):
            raise ValueError(f"fold_{fold_id} validation 清单哈希不一致")
        images = read_split(split_path)
        if len(images) != int(row["val_count"]):
            raise ValueError(f"fold_{fold_id} validation 数量不一致")
        blocks[fold_id] = images
    if sorted(blocks) != expected_fold_ids:
        raise ValueError("source manifest folds 必须从0开始连续编号")

    stems = [image.stem for fold_id in expected_fold_ids for image in blocks[fold_id]]
    if len(stems) != len(set(stems)) or len(stems) != int(
        manifest["combined_non_test_count"]
    ):
        raise ValueError("source manifest 的连续块没有唯一完整覆盖非测试数据")
    return blocks, manifest


def build_window(
    name: str,
    blocks: Mapping[int, Sequence[Path]],
    validation_fold: int,
    output_root: Path,
) -> dict[str, Any]:
    training_folds = list(range(validation_fold))
    if validation_fold not in blocks or not training_folds:
        raise ValueError(f"{name} validation fold 必须至少有一个更早训练折")
    training = [
        image for fold_id in training_folds for image in blocks[int(fold_id)]
    ]
    validation = list(blocks[validation_fold])
    training_stems = {image.stem for image in training}
    validation_stems = {image.stem for image in validation}
    if not training or not validation or training_stems & validation_stems:
        raise ValueError(f"{name} 前向 train/validation 必须非空且互斥")

    grouped_train: dict[str, list[int]] = defaultdict(list)
    grouped_validation: dict[str, list[int]] = defaultdict(list)
    for image in training:
        sequence, frame = sequence_position(image)
        grouped_train[sequence].append(frame)
    for image in validation:
        sequence, frame = sequence_position(image)
        grouped_validation[sequence].append(frame)
    if set(grouped_train) != set(grouped_validation):
        raise ValueError(f"{name} train/validation 序列集合不一致")
    sequence_audit = {}
    for sequence in sorted(grouped_train):
        train_frames = grouped_train[sequence]
        validation_frames = grouped_validation[sequence]
        if max(train_frames) >= min(validation_frames):
            raise ValueError(f"{name} 序列 {sequence} 使用了验证时刻之后的训练帧")
        sequence_audit[sequence] = {
            "train_count": len(train_frames),
            "train_first_frame": min(train_frames),
            "train_last_frame": max(train_frames),
            "validation_count": len(validation_frames),
            "validation_first_frame": min(validation_frames),
            "validation_last_frame": max(validation_frames),
        }

    window_root = output_root / name
    train_path = window_root / "train_source.txt"
    validation_path = window_root / "val_source.txt"
    write_split(train_path, sorted(training, key=sequence_position))
    write_split(validation_path, sorted(validation, key=sequence_position))
    return {
        "name": name,
        "training_folds": training_folds,
        "validation_fold": validation_fold,
        "train_count": len(training),
        "val_count": len(validation),
        "train_split": str(train_path),
        "train_split_sha256": sha256_file(train_path),
        "val_split": str(validation_path),
        "val_split_sha256": sha256_file(validation_path),
        "train_val_overlap": [],
        "strictly_forward_in_every_sequence": True,
        "sequence_audit": sequence_audit,
    }


def build_forward_backtest(
    source_manifest: Path,
    output_root: Path,
    tuning_validation_fold: int,
    post_validation_fold: int,
) -> dict[str, Any]:
    source_manifest = source_manifest.resolve()
    output_root = output_root.resolve()
    reject_test_reference(output_root, "output")
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有前向回测：{output_root}")
    blocks, source = load_blocks(source_manifest)
    if post_validation_fold != tuning_validation_fold + 1:
        raise ValueError("后置验证折必须紧跟调参验证折")
    if post_validation_fold != max(blocks):
        raise ValueError("后置验证必须使用最后一个连续时间块")

    output_root.mkdir(parents=True)
    try:
        tuning = build_window(
            "tuning", blocks, int(tuning_validation_fold), output_root
        )
        post_validation = build_window(
            "post_validation", blocks, int(post_validation_fold), output_root
        )
        manifest = {
            "schema_version": 1,
            "selection_scope": "base_train_and_dev_forward_only",
            "lock_data_access": False,
            "strategy": "expanding_window_temporal_backtest",
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
            "combined_non_test_count": int(source["combined_non_test_count"]),
            "tuning": tuning,
            "post_validation": post_validation,
            "post_validation_labels_must_remain_closed_during_tuning": True,
        }
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuning-validation-fold", type=int, default=3)
    parser.add_argument("--post-validation-fold", type=int, default=4)
    args = parser.parse_args()
    report = build_forward_backtest(
        args.source_manifest,
        args.output,
        int(args.tuning_validation_fold),
        int(args.post_validation_fold),
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
