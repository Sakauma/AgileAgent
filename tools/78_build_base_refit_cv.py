#!/usr/bin/env python3
"""Build blocked cross-validation sources from non-test base train/dev images.

Every sequence is sorted by frame number and divided into contiguous blocks.
Fold ``i`` validates on block ``i`` from every sequence and trains on all other
blocks.  This lets the final refit use late known frames without ever reading
the mixed-test list or labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import (
    read_split,
    read_yolo_labels,
    sha256_file,
    source_label,
)


FORBIDDEN_MARKERS = ("mixed_test", "base_test", "lock")
BASE_GLOBAL_CLASS_IDS = {0, 1, 3}


def reject_test_reference(path: Path, role: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"基础 refit {role} 不得引用 test/lock：{path}")


def sequence_position(path: Path) -> tuple[str, int]:
    prefix, separator, frame = path.stem.rpartition("_")
    if not separator or not frame.isdigit():
        raise ValueError(f"图像名缺少数值帧号：{path.name}")
    return prefix, int(frame)


def contiguous_blocks(values: Sequence[Path], folds: int) -> list[list[Path]]:
    if folds < 2:
        raise ValueError("folds 必须至少为 2")
    if len(values) < folds:
        raise ValueError(f"序列样本数 {len(values)} 小于 folds={folds}")
    quotient, remainder = divmod(len(values), folds)
    output: list[list[Path]] = []
    offset = 0
    for index in range(folds):
        size = quotient + (1 if index < remainder else 0)
        output.append(list(values[offset : offset + size]))
        offset += size
    if offset != len(values) or any(not block for block in output):
        raise AssertionError("连续块划分不完整")
    return output


def write_split(path: Path, images: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(str(image.resolve()) for image in images) + "\n",
        encoding="utf-8",
    )


def verify_base_only(images: Sequence[Path]) -> dict[str, Any]:
    class_counts: dict[int, int] = defaultdict(int)
    for image in images:
        labels = read_yolo_labels(source_label(image))
        if not labels:
            raise ValueError(f"基础 refit 图像没有标注：{image}")
        classes = {int(row[0]) for row in labels}
        invalid = sorted(classes - BASE_GLOBAL_CLASS_IDS)
        if invalid:
            raise ValueError(f"基础 refit 出现非基础类别：{image.name} -> {invalid}")
        for class_id, *_ in labels:
            class_counts[int(class_id)] += 1
    return {
        "allowed_global_class_ids": sorted(BASE_GLOBAL_CLASS_IDS),
        "instance_counts": {str(key): value for key, value in sorted(class_counts.items())},
    }


def build_refit_folds(
    base_train_split: Path,
    base_dev_split: Path,
    output_root: Path,
    folds: int,
) -> dict[str, Any]:
    base_train_split = base_train_split.resolve()
    base_dev_split = base_dev_split.resolve()
    output_root = output_root.resolve()
    reject_test_reference(base_train_split, "train split")
    reject_test_reference(base_dev_split, "dev split")
    reject_test_reference(output_root, "output")
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有 refit folds：{output_root}")

    base_train = read_split(base_train_split)
    base_dev = read_split(base_dev_split)
    train_stems = {image.stem for image in base_train}
    dev_stems = {image.stem for image in base_dev}
    overlap = sorted(train_stems & dev_stems)
    combined = list(base_train) + list(base_dev)
    combined_stems = [image.stem for image in combined]
    if not base_train or not base_dev or overlap or len(combined_stems) != len(set(combined_stems)):
        raise ValueError(
            "base train/dev 必须非空、stem 唯一且互斥："
            f"train={len(base_train)} dev={len(base_dev)} overlap={overlap}"
        )
    label_audit = verify_base_only(combined)

    grouped: dict[str, list[Path]] = defaultdict(list)
    for image in combined:
        grouped[sequence_position(image)[0]].append(image)
    sequence_blocks = {
        sequence: contiguous_blocks(
            sorted(images, key=lambda image: sequence_position(image)[1]), folds
        )
        for sequence, images in sorted(grouped.items())
    }
    ordered = sorted(combined, key=sequence_position)
    all_stems = set(combined_stems)
    validation_coverage: set[str] = set()
    fold_audits = []
    output_root.mkdir(parents=True)
    for fold_index in range(folds):
        validation = [
            image
            for sequence in sorted(sequence_blocks)
            for image in sequence_blocks[sequence][fold_index]
        ]
        validation_stems = {image.stem for image in validation}
        training = [image for image in ordered if image.stem not in validation_stems]
        training_stems = {image.stem for image in training}
        if training_stems & validation_stems or training_stems | validation_stems != all_stems:
            raise AssertionError(f"fold {fold_index} 未形成完整互斥划分")
        if validation_coverage & validation_stems:
            raise AssertionError(f"fold {fold_index} 验证集重复")
        validation_coverage.update(validation_stems)

        fold_root = output_root / f"fold_{fold_index}"
        train_path = fold_root / "train_source.txt"
        val_path = fold_root / "val_source.txt"
        write_split(train_path, training)
        write_split(val_path, validation)
        per_sequence = {}
        for sequence in sorted(sequence_blocks):
            block = sequence_blocks[sequence][fold_index]
            frames = [sequence_position(image)[1] for image in block]
            per_sequence[sequence] = {
                "count": len(block),
                "first_frame": min(frames),
                "last_frame": max(frames),
            }
        fold_audits.append(
            {
                "fold": fold_index,
                "train_count": len(training),
                "val_count": len(validation),
                "train_split": str(train_path),
                "train_split_sha256": sha256_file(train_path),
                "val_split": str(val_path),
                "val_split_sha256": sha256_file(val_path),
                "train_val_overlap": [],
                "validation_sequences": per_sequence,
            }
        )

    if validation_coverage != all_stems:
        missing = sorted(all_stems - validation_coverage)
        raise AssertionError(f"交叉验证未覆盖全部非测试图像：{missing}")
    manifest = {
        "schema_version": 1,
        "selection_scope": "base_train_and_dev_only",
        "lock_data_access": False,
        "strategy": "per_sequence_contiguous_block_kfold",
        "fold_count": folds,
        "base_train_split": str(base_train_split),
        "base_train_split_sha256": sha256_file(base_train_split),
        "base_train_count": len(base_train),
        "base_dev_split": str(base_dev_split),
        "base_dev_split_sha256": sha256_file(base_dev_split),
        "base_dev_count": len(base_dev),
        "combined_non_test_count": len(combined),
        "combined_unique_stems": len(all_stems),
        "validation_coverage_count": len(validation_coverage),
        "labels": label_audit,
        "folds": fold_audits,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_sparse_refit(
    base_train_split: Path,
    base_dev_split: Path,
    output_root: Path,
    validation_stride: int,
    validation_offset: int,
) -> dict[str, Any]:
    """Build a high-coverage final refit split with a sparse checkpoint holdout.

    This split is not an independent performance estimate.  Its validation
    frames only select ``best.pt`` after blocked OOF validation has fixed the
    architecture and hyperparameters.
    """
    base_train_split = base_train_split.resolve()
    base_dev_split = base_dev_split.resolve()
    output_root = output_root.resolve()
    reject_test_reference(base_train_split, "train split")
    reject_test_reference(base_dev_split, "dev split")
    reject_test_reference(output_root, "output")
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有 final refit：{output_root}")
    if validation_stride < 3 or not 0 < validation_offset < validation_stride - 1:
        raise ValueError("sparse validation 要求 stride>=3 且 offset 避开序列端点")

    base_train = read_split(base_train_split)
    base_dev = read_split(base_dev_split)
    train_stems = {image.stem for image in base_train}
    dev_stems = {image.stem for image in base_dev}
    overlap = sorted(train_stems & dev_stems)
    combined = list(base_train) + list(base_dev)
    combined_stems = [image.stem for image in combined]
    if not base_train or not base_dev or overlap or len(combined_stems) != len(set(combined_stems)):
        raise ValueError(
            "base train/dev 必须非空、stem 唯一且互斥："
            f"train={len(base_train)} dev={len(base_dev)} overlap={overlap}"
        )
    label_audit = verify_base_only(combined)

    grouped: dict[str, list[Path]] = defaultdict(list)
    for image in combined:
        grouped[sequence_position(image)[0]].append(image)
    validation = []
    sequence_audit = {}
    for sequence, images in sorted(grouped.items()):
        ordered = sorted(images, key=lambda image: sequence_position(image)[1])
        selected = [
            image
            for index, image in enumerate(ordered)
            if index % validation_stride == validation_offset
        ]
        if not selected or ordered[0] in selected or ordered[-1] in selected:
            raise ValueError(f"稀疏验证未安全覆盖序列：{sequence}")
        validation.extend(selected)
        sequence_audit[sequence] = {
            "source_count": len(ordered),
            "validation_count": len(selected),
            "validation_frames": [sequence_position(image)[1] for image in selected],
            "first_frame_kept_for_training": sequence_position(ordered[0])[1],
            "last_frame_kept_for_training": sequence_position(ordered[-1])[1],
        }
    validation_stems = {image.stem for image in validation}
    ordered_combined = sorted(combined, key=sequence_position)
    training = [image for image in ordered_combined if image.stem not in validation_stems]
    training_stems = {image.stem for image in training}
    if training_stems & validation_stems or len(training) + len(validation) != len(combined):
        raise AssertionError("final refit 未形成完整互斥划分")

    output_root.mkdir(parents=True)
    train_path = output_root / "train_source.txt"
    val_path = output_root / "val_source.txt"
    write_split(train_path, training)
    write_split(val_path, validation)
    manifest = {
        "schema_version": 1,
        "selection_scope": "base_train_and_dev_only",
        "lock_data_access": False,
        "strategy": "per_sequence_sparse_checkpoint_holdout",
        "performance_evidence": False,
        "purpose": "final_best_checkpoint_selection_after_oof_policy_freeze",
        "base_train_split": str(base_train_split),
        "base_train_split_sha256": sha256_file(base_train_split),
        "base_train_count": len(base_train),
        "base_dev_split": str(base_dev_split),
        "base_dev_split_sha256": sha256_file(base_dev_split),
        "base_dev_count": len(base_dev),
        "combined_non_test_count": len(combined),
        "validation_stride": validation_stride,
        "validation_offset": validation_offset,
        "train_count": len(training),
        "val_count": len(validation),
        "train_split": str(train_path),
        "train_split_sha256": sha256_file(train_path),
        "val_split": str(val_path),
        "val_split_sha256": sha256_file(val_path),
        "train_val_overlap": [],
        "sequence_audit": sequence_audit,
        "labels": label_audit,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--mode", choices=("blocked_cv", "sparse_refit"), default="blocked_cv"
    )
    parser.add_argument("--validation-stride", type=int, default=10)
    parser.add_argument("--validation-offset", type=int, default=5)
    args = parser.parse_args()
    if args.mode == "blocked_cv":
        manifest = build_refit_folds(
            args.base_train,
            args.base_dev,
            args.output,
            int(args.folds),
        )
    else:
        manifest = build_sparse_refit(
            args.base_train,
            args.base_dev,
            args.output,
            int(args.validation_stride),
            int(args.validation_offset),
        )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
