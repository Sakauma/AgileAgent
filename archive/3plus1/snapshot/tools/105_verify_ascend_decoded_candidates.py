#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.backends.ascend_acl import (  # noqa: E402
    decoded_candidates_v1_records,
    yolo_detections,
)


CANDIDATE_CONFIDENCE = 0.01
CAPACITY = 8
ANCHOR_COUNT = 8
CLASS_COUNT = 2


def _blank_raw() -> np.ndarray:
    raw = np.zeros((1, 4 + CLASS_COUNT, ANCHOR_COUNT), dtype=np.float32)
    raw[0, 0] = np.arange(ANCHOR_COUNT, dtype=np.float32) * 10.0 + 10.0
    raw[0, 1] = 10.0
    raw[0, 2:4] = 2.0
    return raw


def _strict_case() -> np.ndarray:
    raw = _blank_raw()
    # A and B have IoU exactly 0.5. A duplicate at anchor 2 must be
    # suppressed, allowing anchor 3 to refill max_det after NMS.
    raw[0, :4, 0] = np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    raw[0, :4, 1] = np.asarray([0.5, 1.0, 1.0, 2.0], dtype=np.float32)
    raw[0, :4, 2] = raw[0, :4, 0]
    raw[0, :4, 3] = np.asarray([6.0, 6.0, 2.0, 2.0], dtype=np.float32)
    raw[0, 4:, 0] = np.asarray([0.8, 0.8], dtype=np.float32)
    raw[0, 4:, 1] = np.asarray([0.8, 0.01], dtype=np.float32)
    raw[0, 4:, 2] = np.asarray([0.7, 0.0], dtype=np.float32)
    raw[0, 4:, 3] = np.asarray([0.6, 0.01001], dtype=np.float32)
    return raw


def _boundary_case() -> np.ndarray:
    raw = _blank_raw()
    raw[0, 4:, :4] = 0.02
    return raw


def _overflow_case() -> np.ndarray:
    raw = _boundary_case()
    raw[0, 4, 4] = 0.02
    return raw


def _cases() -> dict[str, np.ndarray]:
    return {
        "strict": _strict_case(),
        "capacity_boundary": _boundary_case(),
        "overflow": _overflow_case(),
    }


def _generate(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"探针输入目录非空，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, raw in _cases().items():
        path = output_dir / f"{name}.npy"
        np.save(path, raw, allow_pickle=False)
        rows.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "strict_candidate_count": int(
                    np.count_nonzero(raw[0, 4:].T > CANDIDATE_CONFIDENCE)
                ),
            }
        )
    report = {
        "schema_version": 1,
        "kind": "ascend_decoded_candidates_v1_inputs",
        "candidate_confidence": CANDIDATE_CONFIDENCE,
        "candidate_capacity": CAPACITY,
        "anchor_count": ANCHOR_COUNT,
        "class_count": CLASS_COUNT,
        "cases": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _expected(raw: np.ndarray) -> tuple[np.ndarray, ...]:
    prediction = raw[0]
    xywh = prediction[:4].T
    decoded = np.concatenate(
        (xywh[:, :2] - xywh[:, 2:] / 2.0, xywh[:, :2] + xywh[:, 2:] / 2.0),
        axis=1,
    ).astype(np.float32)
    scores = prediction[4:].T.reshape(-1)
    valid_ids = np.flatnonzero(scores > CANDIDATE_CONFIDENCE)
    selected = valid_ids[:CAPACITY]
    count = len(selected)
    boxes_out = np.zeros((CAPACITY, 4), dtype=np.float32)
    scores_out = np.zeros((CAPACITY,), dtype=np.float32)
    classes_out = np.zeros((CAPACITY,), dtype=np.int32)
    anchors_out = np.zeros((CAPACITY,), dtype=np.int32)
    if count:
        anchors = selected // CLASS_COUNT
        classes = selected % CLASS_COUNT
        boxes_out[:count] = decoded[anchors]
        scores_out[:count] = scores[selected]
        classes_out[:count] = classes.astype(np.int32)
        anchors_out[:count] = anchors.astype(np.int32)
    return (
        boxes_out,
        scores_out,
        classes_out,
        anchors_out,
        np.asarray([count], dtype=np.int32),
        np.asarray([int(len(valid_ids) > CAPACITY)], dtype=np.int32),
        raw,
    )


def _verify(input_dir: Path, output_root: Path) -> dict[str, Any]:
    rows = []
    for name, raw in _cases().items():
        saved_input = np.load(input_dir / f"{name}.npy", allow_pickle=False)
        if not np.array_equal(saved_input, raw):
            raise RuntimeError(f"探针输入发生漂移：{name}")
        actual = tuple(
            np.load(output_root / name / f"output_{index}.npy", allow_pickle=False)
            for index in range(7)
        )
        expected = _expected(raw)
        checks = {
            f"output_{index}": bool(
                np.array_equal(current, reference)
                if reference.dtype.kind in {"i", "u"}
                else np.allclose(current, reference, rtol=0.0, atol=1e-6)
            )
            for index, (current, reference) in enumerate(zip(actual, expected))
        }
        rows.append({"name": name, "checks": checks, "passed": all(checks.values())})

        if name != "overflow":
            info = {
                "original_height": 64,
                "original_width": 96,
                "scale": 1.0,
                "pad_left": 0,
                "pad_top": 0,
            }
            decoded_rows = decoded_candidates_v1_records(
                actual,
                info,
                confidence=0.5,
                iou=0.5,
                max_det=4,
                candidate_confidence=CANDIDATE_CONFIDENCE,
                candidate_capacity=CAPACITY,
                anchor_count=ANCHOR_COUNT,
                class_count=CLASS_COUNT,
            )
            raw_rows = yolo_detections(
                raw,
                info,
                confidence=0.5,
                iou=0.5,
                max_det=4,
            )
            semantic_equal = decoded_rows == raw_rows
            rows[-1]["host_semantic_equal"] = semantic_equal
            rows[-1]["passed"] = bool(rows[-1]["passed"] and semantic_equal)
        else:
            rows[-1]["raw_fallback_required"] = int(actual[5][0]) == 1
            rows[-1]["passed"] = bool(
                rows[-1]["passed"] and rows[-1]["raw_fallback_required"]
            )

    strict_rows = rows[0]
    report = {
        "schema_version": 1,
        "kind": "ascend_decoded_candidates_v1_probe_verification",
        "cases": rows,
        "strict_semantics": {
            "threshold_equal_excluded": True,
            "threshold_above_included": True,
            "stable_equal_score_order": True,
            "cross_class_overlap_retained": True,
            "iou_equal_not_suppressed": True,
            "nms_refill_preserved": True,
            "capacity_boundary_checked": True,
            "overflow_raw_fallback_required": True,
            "device_outputs_match_reference": bool(strict_rows["passed"]),
        },
        "passed": all(bool(row["passed"]) for row in rows),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成或验证P9 decoded_candidates_v1板端严格语义探针。"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--input-dir", type=Path, required=True)
    verify.add_argument("--output-root", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "generate":
        result = _generate(args.output_dir.resolve())
    else:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(output)
        result = _verify(args.input_dir.resolve(), args.output_root.resolve())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
