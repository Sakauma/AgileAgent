#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fair_agent.modules.ascend_preflight import LetterboxInfo, _postprocess_yolo, _softmax


def raw_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"输出shape不一致：{reference.shape} != {candidate.shape}")
    expected = reference.astype(np.float64)
    actual = candidate.astype(np.float64)
    difference = actual - expected
    denominator = max(float(np.linalg.norm(expected.ravel())), 1e-12)
    expected_max = max(float(np.max(np.abs(expected))), 1.0)
    cosine_denominator = max(
        float(np.linalg.norm(expected.ravel()) * np.linalg.norm(actual.ravel())), 1e-12
    )
    return {
        "shape": list(reference.shape),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / denominator),
        "normalized_max_abs": float(np.max(np.abs(difference)) / expected_max),
        "cosine": float(np.dot(expected.ravel(), actual.ravel()) / cosine_denominator),
    }


def detector_rows(
    raw: np.ndarray,
    preprocessing: dict[str, Any],
    *,
    confidence: float,
    iou: float,
    max_det: int,
) -> list[dict[str, Any]]:
    info = LetterboxInfo(
        original_height=int(preprocessing["original_height"]),
        original_width=int(preprocessing["original_width"]),
        input_height=int(preprocessing["input_height"]),
        input_width=int(preprocessing["input_width"]),
        scale=float(preprocessing["scale"]),
        pad_left=int(preprocessing["pad_left"]),
        pad_top=int(preprocessing["pad_top"]),
        pad_right=int(preprocessing["pad_right"]),
        pad_bottom=int(preprocessing["pad_bottom"]),
    )
    return _postprocess_yolo(
        raw,
        info,
        confidence=confidence,
        iou=iou,
        max_det=max_det,
        candidate_prefilter=True,
    )


def match_detections(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    def sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: (
                int(row["local_class_id"]),
                -float(row["confidence"]),
                *(round(float(value), 5) for value in row["xyxy"]),
            ),
        )

    expected = sorted_rows(reference)
    actual = sorted_rows(candidate)
    if len(expected) != len(actual):
        return {
            "reference_count": len(expected),
            "candidate_count": len(actual),
            "class_ids_equal": False,
            "max_box_abs": None,
            "max_confidence_abs": None,
            "passed": False,
        }
    class_ids_equal = all(
        int(first["local_class_id"]) == int(second["local_class_id"])
        for first, second in zip(expected, actual)
    )
    if not expected:
        max_box_abs = max_confidence_abs = 0.0
    else:
        max_box_abs = max(
            abs(float(left) - float(right))
            for first, second in zip(expected, actual)
            for left, right in zip(first["xyxy"], second["xyxy"])
        )
        max_confidence_abs = max(
            abs(float(first["confidence"]) - float(second["confidence"]))
            for first, second in zip(expected, actual)
        )
    return {
        "reference_count": len(expected),
        "candidate_count": len(actual),
        "class_ids_equal": class_ids_equal,
        "max_box_abs": float(max_box_abs),
        "max_confidence_abs": float(max_confidence_abs),
        "passed": bool(class_ids_equal and max_box_abs <= 1.0 and max_confidence_abs <= 0.02),
    }


def context_record(sensor: np.ndarray, scene: np.ndarray) -> dict[str, Any]:
    sensor_probability = _softmax(sensor)[0]
    scene_probability = _softmax(scene)[0]
    return {
        "sensor_id": int(sensor_probability.argmax()),
        "sensor_probability": sensor_probability.tolist(),
        "scene_id": int(scene_probability.argmax()),
        "scene_probability": scene_probability.tolist(),
    }


def board_output(
    validation_root: Path, model_id: str, output_index: int
) -> np.ndarray:
    aliases = {
        "base_detector": ("base_detector", "base_case_00"),
        "incremental_detector": ("incremental_detector", "incremental_case_00"),
        "scene_sensor_net": ("scene_sensor_net", "scene_case_00"),
    }
    for directory in aliases[model_id]:
        path = validation_root / directory / f"output_{output_index}.npy"
        if path.is_file():
            return np.load(path, allow_pickle=False)
    searched = ", ".join(str(validation_root / item) for item in aliases[model_id])
    raise FileNotFoundError(f"未找到{model_id}输出，已检查：{searched}")


def main() -> int:
    parser = argparse.ArgumentParser(description="比较310B OM输出与无标签golden bundle。")
    parser.add_argument("--golden-case", type=Path, required=True)
    parser.add_argument("--board-validation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    case_id = args.case_id or args.golden_case.name
    case = next(item for item in manifest["cases"] if item["case_id"] == case_id)

    models: dict[str, Any] = {}
    for model_id in ("base_detector", "incremental_detector"):
        expected = np.load(args.golden_case / f"{model_id}_output_0.npy", allow_pickle=False)
        actual = board_output(args.board_validation, model_id, 0)
        expected_rows = detector_rows(
            expected,
            case["models"][model_id]["preprocessing"],
            confidence=args.confidence,
            iou=args.iou,
            max_det=args.max_det,
        )
        actual_rows = detector_rows(
            actual,
            case["models"][model_id]["preprocessing"],
            confidence=args.confidence,
            iou=args.iou,
            max_det=args.max_det,
        )
        models[model_id] = {
            "raw": raw_difference(expected, actual),
            "detections": match_detections(expected_rows, actual_rows),
        }

    expected_sensor = np.load(
        args.golden_case / "scene_sensor_net_output_0.npy", allow_pickle=False
    )
    expected_scene = np.load(
        args.golden_case / "scene_sensor_net_output_1.npy", allow_pickle=False
    )
    actual_sensor = board_output(args.board_validation, "scene_sensor_net", 0)
    actual_scene = board_output(args.board_validation, "scene_sensor_net", 1)
    expected_context = context_record(expected_sensor, expected_scene)
    actual_context = context_record(actual_sensor, actual_scene)
    context = {
        "outputs": [
            raw_difference(expected_sensor, actual_sensor),
            raw_difference(expected_scene, actual_scene),
        ],
        "reference": expected_context,
        "candidate": actual_context,
        "sensor_equal": expected_context["sensor_id"] == actual_context["sensor_id"],
        "scene_equal": expected_context["scene_id"] == actual_context["scene_id"],
    }
    context["passed"] = bool(context["sensor_equal"] and context["scene_equal"])

    report = {
        "schema_version": 1,
        "case_id": case_id,
        "confidence": args.confidence,
        "iou": args.iou,
        "max_det": args.max_det,
        "models": models,
        "context": context,
        "passed": bool(
            all(item["detections"]["passed"] for item in models.values())
            and context["passed"]
        ),
    }
    output = args.output or (args.board_validation / "alignment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
