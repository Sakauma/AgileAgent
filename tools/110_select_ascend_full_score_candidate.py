#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
ACCURACY_GATE_KEYS = {
    "base_map50": "base_map50_min",
    "new_map50": "new_map50_min",
    "krr": "krr_min",
}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label}必须是JSON object：{path}")
    return payload


def _resolve(index_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else index_path.parent / path


def _read_method(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != "ascend310b_full_score_method"
    ):
        raise ValueError(f"满分方法配置非法：{path}")
    return payload


def _score_contract(method: Mapping[str, Any]) -> dict[str, Any]:
    competition = method.get("competition") or {}
    raw_accuracy = competition.get("accuracy_gates") or {}
    performance = competition.get("performance_gate") or {}
    benchmark = method.get("benchmark") or {}
    accuracy_limits = {
        metric: float(raw_accuracy[key]) for metric, key in ACCURACY_GATE_KEYS.items()
    }
    contract = {
        "accuracy_limits": accuracy_limits,
        "batch_image_count": int(performance["batch_image_count"]),
        "batch_rounds": int(performance["batch_rounds"]),
        "median_fps_min": float(performance["median_fps_min"]),
    }
    if (
        int(benchmark["batch_probe_size"]) != contract["batch_image_count"]
        or int(benchmark["batch_rounds"]) != contract["batch_rounds"]
        or float(benchmark["target_batch_fps"]) != contract["median_fps_min"]
    ):
        raise ValueError("方法配置中的benchmark与competition performance_gate不一致")
    return contract


def _read_benchmark(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    report = _read_json(path, "benchmark报告")
    if report.get("schema_version") != 5:
        raise ValueError(f"benchmark schema_version必须为5：{path}")
    protocol = report.get("protocol") or {}
    competition = report.get("competition") or {}
    rounds = competition.get("batch_rounds")
    if (
        protocol.get("batch_probe_size") != contract["batch_image_count"]
        or protocol.get("batch_rounds") != contract["batch_rounds"]
        or float(protocol.get("target_batch_fps", 0.0)) != contract["median_fps_min"]
        or competition.get("batch_image_count") != contract["batch_image_count"]
        or not isinstance(rounds, list)
        or len(rounds) != contract["batch_rounds"]
    ):
        raise ValueError(f"benchmark不符合满分方法中的计分协议：{path}")
    if any(not isinstance(row, Mapping) or "fps" not in row for row in rounds):
        raise ValueError(f"benchmark batch_rounds字段非法：{path}")
    round_fps = [float(row["fps"]) for row in rounds]
    return {
        "path": str(path.resolve()),
        "median_fps": float(competition["batch_fps"]),
        "passed": float(competition["batch_fps"]) >= float(contract["median_fps_min"]),
        "round_fps": round_fps,
    }


def evaluate_candidate(
    index_path: Path,
    row: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_id = str(row.get("id") or "").strip()
    if not candidate_id:
        raise ValueError("候选缺少id")
    prerequisites = row.get("prerequisites")
    if not isinstance(prerequisites, Mapping):
        raise ValueError(f"候选{candidate_id}缺少prerequisites")
    required_prerequisites = (
        "incremental_data_isolation",
        "asset_hashes_verified",
    )
    prerequisite_failures = [
        name for name in required_prerequisites if prerequisites.get(name) is not True
    ]

    score_path = _resolve(index_path, str(row.get("score") or ""))
    score = _read_json(score_path, "score报告")
    if score.get("schema_version") != 2:
        raise ValueError(f"score schema_version必须为2：{score_path}")
    if score.get("unlabeled_predictions_frozen_before_labels") is not True:
        prerequisite_failures.append("predictions_frozen_before_labels")
    metrics = score.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"score报告缺少metrics：{score_path}")
    accuracy_limits = dict(contract["accuracy_limits"])
    missing_metrics = sorted(set(accuracy_limits) - set(metrics))
    if missing_metrics:
        raise ValueError(
            f"score报告缺少计分指标：{score_path}:{','.join(missing_metrics)}"
        )
    accuracy_failures = [
        name
        for name, minimum in accuracy_limits.items()
        if float(metrics.get(name, float("-inf"))) < minimum
    ]
    headroom = min(
        float(metrics[name]) - minimum for name, minimum in accuracy_limits.items()
    )

    primary_path = _resolve(index_path, str(row.get("benchmark") or ""))
    primary = _read_benchmark(primary_path, contract)
    repeat_values = row.get("repeat_benchmarks", [])
    if not isinstance(repeat_values, list):
        raise ValueError(f"候选{candidate_id}.repeat_benchmarks必须是列表")
    repeats = [
        _read_benchmark(_resolve(index_path, str(value)), contract)
        for value in repeat_values
    ]
    all_round_fps = [
        value for report in (primary, *repeats) for value in report["round_fps"]
    ]
    fps_spread = max(all_round_fps) - min(all_round_fps)
    accuracy_passed = not accuracy_failures
    validity_passed = not prerequisite_failures
    full_score = validity_passed and accuracy_passed and primary["passed"]
    diagnostics = list(score.get("diagnostic_warnings") or [])
    if any(not report["passed"] for report in repeats):
        diagnostics.append("repeat_benchmark_below_target_fps")

    return {
        "id": candidate_id,
        "full_score": full_score,
        "accuracy_passed": accuracy_passed,
        "performance_passed": primary["passed"],
        "validity_passed": validity_passed,
        "accuracy_failures": accuracy_failures,
        "prerequisite_failures": prerequisite_failures,
        "metrics": {name: float(metrics[name]) for name in accuracy_limits},
        "minimum_accuracy_headroom": headroom,
        "batch_median_fps": primary["median_fps"],
        "batch_fps_spread": fps_spread,
        "benchmark": primary,
        "repeat_benchmarks": repeats,
        "diagnostic_warnings": sorted(set(str(item) for item in diagnostics)),
        "source": dict(row),
    }


def select_candidates(
    index_path: Path,
    payload: Mapping[str, Any],
    method: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("候选索引schema_version必须为1")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("候选索引至少包含一个候选")
    resolved_method = (
        dict(method)
        if method is not None
        else _read_method(ROOT / "configs/ascend310b/full_score_method.yaml")
    )
    contract = _score_contract(resolved_method)
    candidates = [
        evaluate_candidate(index_path, row, contract) for row in raw_candidates
    ]
    full_score = [row for row in candidates if row["full_score"]]
    if full_score:
        ranked = sorted(
            full_score,
            key=lambda row: (
                -row["minimum_accuracy_headroom"],
                row["batch_fps_spread"],
                -row["batch_median_fps"],
                row["id"],
            ),
        )
        status = "full_score_winner"
        winner = ranked[0]
    else:
        accuracy_pass = [
            row
            for row in candidates
            if row["validity_passed"] and row["accuracy_passed"]
        ]
        ranked = sorted(
            accuracy_pass,
            key=lambda row: (
                -row["batch_median_fps"],
                -row["minimum_accuracy_headroom"],
                row["batch_fps_spread"],
                row["id"],
            ),
        )
        status = "intermediate_only" if ranked else "no_eligible_candidate"
        winner = ranked[0] if ranked else None

    selected_id = winner["id"] if winner is not None else None
    ordered = sorted(
        candidates,
        key=lambda row: (
            row["id"] != selected_id,
            not row["full_score"],
            not row["accuracy_passed"],
            -row["batch_median_fps"],
            row["id"],
        ),
    )
    return {
        "schema_version": 1,
        "status": status,
        "selected_candidate": selected_id,
        "full_score_achieved": status == "full_score_winner",
        "selection_policy": [
            "competition_accuracy_gates",
            "competition_batch_fps_gate",
            *list(resolved_method["threshold_search"]["selection_order"]),
        ],
        "score_contract": contract,
        "candidates": ordered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="选择Ascend310B四项满分候选。")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--method-config",
        type=Path,
        default=ROOT / "configs/ascend310b/full_score_method.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"选择报告已存在，拒绝覆盖：{args.output}")
    index_path = args.candidates.resolve()
    payload = _read_json(index_path, "候选索引")
    method = _read_method(args.method_config.resolve())
    result = select_candidates(index_path, payload, method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["full_score_achieved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
