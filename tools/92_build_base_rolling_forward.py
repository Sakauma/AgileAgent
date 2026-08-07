#!/usr/bin/env python3
"""Build rolling-origin base-development windows without opening the last block.

The selected origins use only frames strictly earlier than their validation
block.  The final contiguous block is recorded either as sealed or as
explicitly reused regression data and is not materialized by this tool.  This
makes the protocol reusable for any base-class vocabulary; no semantic class
name is used to construct a window.
"""

from __future__ import annotations

import argparse
import json
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import sha256_file


FORWARD = runpy.run_path(str(ROOT / "tools" / "89_build_base_forward_backtest.py"))
FORBIDDEN_MARKERS = ("mixed_test", "base_test", "lock")


def reject_test_reference(path: Path, role: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"滚动前向开发 {role} 不得引用 test/lock：{path}")


def normalize_validation_folds(values: Sequence[int]) -> list[int]:
    folds = [int(value) for value in values]
    if not folds or folds != sorted(set(folds)):
        raise ValueError("validation folds 必须非空、升序且不重复")
    if folds[0] < 1:
        raise ValueError("每个滚动窗口必须至少有一个更早训练块")
    return folds


def build_rolling_forward(
    source_manifest: Path,
    output_root: Path,
    validation_folds: Sequence[int],
    sealed_fold: int,
    regression_already_opened: bool = False,
) -> dict[str, Any]:
    source_manifest = source_manifest.resolve()
    output_root = output_root.resolve()
    reject_test_reference(source_manifest, "source manifest")
    reject_test_reference(output_root, "output")
    folds = normalize_validation_folds(validation_folds)
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有滚动前向开发集：{output_root}")

    blocks, source = FORWARD["load_blocks"](source_manifest)
    sealed_fold = int(sealed_fold)
    if sealed_fold != max(blocks):
        raise ValueError("sealed fold 必须是最后一个连续时间块")
    if max(folds) >= sealed_fold:
        raise ValueError("滚动选择窗口必须严格早于 sealed fold")

    output_root.mkdir(parents=True)
    try:
        windows: dict[str, Mapping[str, Any]] = {}
        for fold_id in folds:
            if fold_id not in blocks:
                raise ValueError(f"source manifest 缺少 fold {fold_id}")
            name = f"origin_{fold_id}"
            windows[name] = FORWARD["build_window"](
                name,
                blocks,
                fold_id,
                output_root,
            )

        manifest = {
            "schema_version": 1,
            "selection_scope": "base_train_and_dev_rolling_forward_only",
            "lock_data_access": False,
            "strategy": "multi_origin_expanding_window_temporal_backtest",
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
            "combined_non_test_count": int(source["combined_non_test_count"]),
            "selection_windows": windows,
            "regression_window": {
                "status": (
                    "reused_not_independent"
                    if regression_already_opened
                    else "sealed"
                ),
                "validation_fold": sealed_fold,
                "image_count": len(blocks[sealed_fold]),
                "labels_opened": bool(regression_already_opened),
                "must_not_participate_in_candidate_selection": True,
                "independent_evidence": False,
            },
        }
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(
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
    parser.add_argument(
        "--validation-fold",
        type=int,
        action="append",
        required=True,
        help="可重复；候选只在这些滚动起点上选择。",
    )
    parser.add_argument("--sealed-fold", type=int, default=4)
    parser.add_argument(
        "--regression-already-opened",
        action="store_true",
        help="明确记录最后一折此前已解封，只允许作开发回归。",
    )
    args = parser.parse_args()
    report = build_rolling_forward(
        args.source_manifest,
        args.output,
        args.validation_fold,
        int(args.sealed_fold),
        bool(args.regression_already_opened),
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
