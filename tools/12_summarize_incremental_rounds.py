#!/usr/bin/env python3
"""Validate and summarize the complete sequential class-incremental lineage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.hashes import sha256_file  # noqa: E402
from fair_agent.modules.incremental_round_registry import (  # noqa: E402
    DEFAULT_ROUND_REGISTRY,
    load_incremental_round_registry,
    rounds_through,
)
from fair_agent.modules.model_generations import load_generation_registry  # noqa: E402


def parse_metrics(value: str) -> tuple[str, Path]:
    round_id, separator, path = value.partition("=")
    if not separator or not round_id.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "--metrics 必须使用 ROUND_ID=/path/to/metrics.json"
        )
    return round_id.strip(), Path(path.strip())


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def validate_round_metrics(
    round_spec: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    lineage = dict(payload.get("lineage") or {})
    expected = {
        "round_id": round_spec["round_id"],
        "round_index": round_spec["round_index"],
        "parent_generation_id": round_spec["parent_generation_id"],
        "generation_id": round_spec["generation_id"],
        "old_class_ids": round_spec["old_class_ids"],
        "new_class_ids": round_spec["new_class_ids"],
        "learned_class_ids": round_spec["learned_class_ids"],
    }
    if any(lineage.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{round_spec['round_id']} 的父子代际或类别集合不一致")
    score_gates = dict(payload.get("score_gates") or {})
    if (
        payload.get("phase") != "joint_evaluation"
        or payload.get("counted_as_incremental_learning") is not False
        or payload.get("detector_weights_updated") is not False
        or payload.get("model_selection_allowed") is not False
        or payload.get("incremental_learning_data_scope")
        != "incremental_dataset_only"
        or int(payload.get("old_raw_image_count", -1)) != 0
        or int(payload.get("old_raw_label_count", -1)) != 0
        or payload.get("predictions_frozen_before_lock_labels") is not True
        or payload.get("accepted") is not True
        or payload.get("competition_accepted") is not True
        or set(score_gates) != {"base_map50", "new_map50", "krr"}
        or not all(value is True for value in score_gates.values())
    ):
        raise ValueError(f"{round_spec['round_id']} 的评测或数据合规声明不完整")
    metrics = dict(payload.get("round_metrics") or {})
    for key in ("new_map50", "krr", "full_map50"):
        value = float(metrics[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{round_spec['round_id']} 的 {key} 越界")
        metrics[key] = value
    return {
        "round_id": round_spec["round_id"],
        "round_index": round_spec["round_index"],
        "parent_generation_id": round_spec["parent_generation_id"],
        "generation_id": round_spec["generation_id"],
        "old_class_ids": round_spec["old_class_ids"],
        "new_class_ids": round_spec["new_class_ids"],
        "learned_class_ids": round_spec["learned_class_ids"],
        "metrics": metrics,
        "score_gates": score_gates,
        "competition_accepted": bool(payload.get("competition_accepted")),
        "metrics_source": resolved.as_posix(),
    }


def validate_registered_round(
    registry: Mapping[str, Any],
    generation_registry: Mapping[str, Any],
    round_spec: Mapping[str, Any],
    metrics_row: Mapping[str, Any],
) -> dict[str, Any]:
    model_id = str(round_spec["specialist"]["model_id"])
    generation_id = str(round_spec["generation_id"])
    try:
        model = generation_registry["models_by_id"][model_id]
        generation = generation_registry["generations_by_id"][generation_id]
    except KeyError as exc:
        raise ValueError(
            f"{round_spec['round_id']} 尚未登记模型或代际"
        ) from exc
    active_rounds = rounds_through(registry, str(round_spec["round_id"]))
    expected_owners = {
        int(class_id): str(registry["base"]["model_id"])
        for class_id in registry["base"]["class_ids"]
    }
    for row in active_rounds:
        for class_id in row["new_class_ids"]:
            expected_owners[int(class_id)] = str(row["specialist"]["model_id"])
    expected_members = [
        str(registry["base"]["model_id"]),
        *(str(row["specialist"]["model_id"]) for row in active_rounds),
    ]
    evidence = dict(model.get("training_evidence") or {})
    evaluation_source = Path(str(evidence.get("evaluation_source") or ""))
    if not evaluation_source.is_absolute():
        evaluation_source = ROOT / evaluation_source
    metrics_source = Path(str(metrics_row["metrics_source"]))
    if not metrics_source.is_absolute():
        metrics_source = ROOT / metrics_source
    if (
        model.get("role") != "class_incremental_expert"
        or model.get("incremental_mode") != "class_incremental"
        or model["owns_classes"] != set(round_spec["new_class_ids"])
        or model["local_to_global"]
        != round_spec["specialist"]["local_to_global"]
        or model.get("hash_valid") is not True
        or evidence.get("phase") != "incremental_learning"
        or evidence.get("round_id") != round_spec["round_id"]
        or evidence.get("parent_generation_id")
        != round_spec["parent_generation_id"]
        or evidence.get("training_data_scope") != "incremental_dataset_only"
        or evidence.get("validation_data_scope") != "incremental_dataset_only"
        or int(evidence.get("old_raw_image_count", -1)) != 0
        or evidence.get("base_detector_weights_frozen") is not True
        or evidence.get("historical_incremental_expert_weights_frozen")
        is not True
        or generation.get("parent") != round_spec["parent_generation_id"]
        or set(int(value) for value in generation["classes"])
        != set(round_spec["learned_class_ids"])
        or generation["old_class_ids"] != set(round_spec["old_class_ids"])
        or generation["new_class_ids"] != set(round_spec["new_class_ids"])
        or generation["class_owners"] != expected_owners
        or generation["model_members"] != expected_members
        or generation.get("status") not in {"registered_candidate", "active"}
        or not evaluation_source.is_file()
        or sha256_file(evaluation_source) != sha256_file(metrics_source)
    ):
        raise ValueError(f"{round_spec['round_id']} 的代际登记证据不完整")
    registered_metrics = dict(generation.get("metrics") or {})
    if any(
        abs(float(registered_metrics.get(key, -1.0)) - float(metrics_row["metrics"][key]))
        > 1e-12
        for key in ("new_map50", "krr", "full_map50")
    ):
        raise ValueError(f"{round_spec['round_id']} 的登记指标与评测证据不一致")
    return {
        "model_id": model_id,
        "model_path": str(model["path"]),
        "generation_id": generation_id,
        "generation_status": str(generation["status"]),
        "evaluation_source": portable_path(evaluation_source),
    }


def markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# 顺序类别增量学习证据",
        "",
        "Scene-SensorNet 与场景门控属于 system_calibration，不计入下表的增量学习训练。",
        "",
        "| 轮次 | 父代 | 子代 | 本轮新类 | 累计类别 | New-mAP50 | KRR | Full-mAP50 | 门禁 |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rounds"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['round_id']} | {row['parent_generation_id']} | "
            f"{row['generation_id']} | {row['new_class_ids']} | "
            f"{row['learned_class_ids']} | {metrics['new_map50']:.6f} | "
            f"{metrics['krr']:.6f} | {metrics['full_map50']:.6f} | "
            f"{'通过' if row['competition_accepted'] else '未通过'} |"
        )
    lines.extend(
        [
            "",
            "每轮 Increment train/dev 只用于该轮新类专家；Base 与历史专家权重冻结。",
            "每轮预测先冻结，再读取截至该轮全部已学类别的 lock 标签进行联合评分。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="汇总至少两轮不同新增类别的 New-mAP50/KRR/Full-mAP50 与父子代际。"
    )
    parser.add_argument(
        "--round-registry", type=Path, default=ROOT / DEFAULT_ROUND_REGISTRY
    )
    parser.add_argument(
        "--metrics", type=parse_metrics, action="append", required=True
    )
    parser.add_argument(
        "--generation-registry",
        type=Path,
        default=ROOT / "models" / "generations.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    registry = load_incremental_round_registry(args.round_registry)
    if len(args.metrics) != len({round_id for round_id, _path in args.metrics}):
        raise ValueError("--metrics 不能重复提供同一轮次")
    metrics_paths = dict(args.metrics)
    expected = {str(row["round_id"]) for row in registry["rounds"]}
    if set(metrics_paths) != expected:
        raise ValueError(
            f"必须提供全部轮次指标：expected={sorted(expected)} "
            f"actual={sorted(metrics_paths)}"
        )
    rounds = [
        validate_round_metrics(row, metrics_paths[str(row["round_id"])])
        for row in registry["rounds"]
    ]
    generation_registry_path = args.generation_registry.expanduser().resolve()
    generation_registry = load_generation_registry(generation_registry_path)
    registrations = [
        validate_registered_round(registry, generation_registry, spec, metrics)
        for spec, metrics in zip(registry["rounds"], rounds)
    ]
    for metrics, registration in zip(rounds, registrations):
        metrics["metrics_source"] = registration["evaluation_source"]
    final_generation_id = str(registry["rounds"][-1]["generation_id"])
    if str(generation_registry["channels"]["candidate"]) != final_generation_id:
        raise ValueError("candidate 频道必须指向最后一轮严格增量代际")
    distinct_new_classes = {
        class_id for row in rounds for class_id in row["new_class_ids"]
    }
    if len(rounds) < 2 or len(distinct_new_classes) < 2:
        raise ValueError("必须至少包含两轮不同新增类别，目标增量更新不能替代类别增量")
    output = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": registry["protocol_id"],
        "round_registry": portable_path(Path(registry["path"])),
        "evidence_status": "complete",
        "sequential_class_incremental_verified": True,
        "distinct_round_count": len(rounds),
        "distinct_new_class_count": len(distinct_new_classes),
        "scene_sensor_is_incremental_learner": False,
        "generation_registry": portable_path(generation_registry_path),
        "registered_lineage_verified": True,
        "rounds": rounds,
        "registrations": registrations,
        "all_rounds_competition_accepted": all(
            row["competition_accepted"] for row in rounds
        ),
    }
    output_dir = args.output_dir.expanduser().resolve()
    atomic_json(output_dir / "round_evidence.json", output)
    (output_dir / "round_evidence.md").write_text(
        markdown_report(output), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["all_rounds_competition_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
