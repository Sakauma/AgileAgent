#!/usr/bin/env python3
"""Select one base-agent policy using tuning folds, then open validation folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.strict_incremental import sha256_file


FORBIDDEN_MARKERS = ("mixed_test", "base_test", "lock")


def named_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate 必须使用 NAME=REPORT")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("candidate 必须使用 NAME=REPORT")
    return name, Path(path)


def reject_test_reference(path: Path, role: str) -> None:
    lowered = str(path).replace("\\", "/").lower()
    if any(marker in lowered for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"{role} 不得引用 test/lock：{path}")


def load_candidate(
    name: str, path: Path, max_degraded_tuning_folds: int
) -> dict[str, Any]:
    path = path.resolve()
    reject_test_reference(path, f"candidate {name}")
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = dict(report.get("selected_policy", {}))
    tuning = dict(report.get("tuning", {}))
    if (
        report.get("selection_scope") != "base_train_and_dev_oof_tune_validate"
        or bool(report.get("lock_data_access", True))
        or int(report.get("focus_class_id", 0)) != 0
        or int(selected.get("degraded_tuning_fold_count", -1))
        > max_degraded_tuning_folds
        or float(selected.get("tuning_delta_vs_generic", 0.0)) <= 0.0
        or list(selected.get("secondaries", report.get("secondaries", [])))
        != list(report.get("secondaries", []))
        or abs(float(selected.get("tuning_map50", -1.0)) - float(tuning.get("fused_map50", -2.0)))
        > 1e-12
    ):
        raise ValueError(f"candidate {name} 不满足无泄露 OOF 调参约束")
    tuning_folds = {int(value) for value in report.get("tuning_folds", [])}
    validation_folds = {int(value) for value in report.get("validation_folds", [])}
    if not tuning_folds or not validation_folds or tuning_folds & validation_folds:
        raise ValueError(f"candidate {name} 的 tuning/validation folds 无效")
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest_sha256": str(report.get("manifest_sha256", "")),
        "tuning_folds": sorted(tuning_folds),
        "validation_folds": sorted(validation_folds),
        "selected_policy": selected,
        "tuning": tuning,
        "validation": dict(report.get("validation", {})),
        "all_oof_diagnostic": dict(report.get("all_oof_diagnostic", {})),
        "source_report": report,
    }


def rank_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["tuning"]["fused_map50"]),
            int(row["selected_policy"]["degraded_tuning_fold_count"]),
            -float(row["selected_policy"]["minimum_tuning_fold_map50"]),
            -float(row["selected_policy"]["worst_tuning_fold_delta"]),
            len(row["selected_policy"].get("secondaries", [])),
            str(row["name"]),
        ),
    )


def select_policy(
    candidates: Sequence[tuple[str, Path]],
    max_degraded_tuning_folds: int,
    min_validation_map50: float,
    min_all_oof_map50: float,
) -> dict[str, Any]:
    if len(candidates) < 2 or len({name for name, _path in candidates}) != len(candidates):
        raise ValueError("至少需要两个名称唯一的 OOF policy candidates")
    if max_degraded_tuning_folds < 0:
        raise ValueError("max-degraded-tuning-folds 不得为负")
    loaded = [
        load_candidate(name, path, max_degraded_tuning_folds)
        for name, path in candidates
    ]
    manifest_hashes = {str(row["manifest_sha256"]) for row in loaded}
    tuning_folds = {tuple(row["tuning_folds"]) for row in loaded}
    validation_folds = {tuple(row["validation_folds"]) for row in loaded}
    if "" in manifest_hashes or len(manifest_hashes) != 1 or len(tuning_folds) != 1 or len(validation_folds) != 1:
        raise ValueError("OOF candidates 必须使用相同 manifest 与 tuning/validation 边界")

    ranking = rank_candidates(loaded)
    selected = dict(ranking[0])
    validation = dict(selected["validation"])
    diagnostic = dict(selected["all_oof_diagnostic"])
    gates = {
        "validation_map50": float(validation.get("fused_map50", 0.0))
        >= float(min_validation_map50),
        "validation_improves_primary": float(validation.get("delta_map50", 0.0)) > 0.0,
        "all_oof_map50": float(diagnostic.get("fused_map50", 0.0))
        >= float(min_all_oof_map50),
    }
    if not all(gates.values()):
        raise RuntimeError(f"调参折选中的 policy 未通过后置验证：{gates}")

    return {
        "schema_version": 1,
        "selection_scope": "base_train_and_dev_oof_candidate_selection",
        "lock_data_access": False,
        "selection_basis": "tuning_folds_only",
        "tuning_folds": selected["tuning_folds"],
        "validation_folds_opened_after_selection": selected["validation_folds"],
        "manifest_sha256": next(iter(manifest_hashes)),
        "candidate_ranking": [
            {
                "rank": index,
                "name": row["name"],
                "path": row["path"],
                "sha256": row["sha256"],
                "tuning_map50": float(row["tuning"]["fused_map50"]),
                "minimum_tuning_fold_map50": float(
                    row["selected_policy"]["minimum_tuning_fold_map50"]
                ),
                "worst_tuning_fold_delta": float(
                    row["selected_policy"]["worst_tuning_fold_delta"]
                ),
                "degraded_tuning_fold_count": int(
                    row["selected_policy"]["degraded_tuning_fold_count"]
                ),
                "secondaries": list(row["selected_policy"].get("secondaries", [])),
            }
            for index, row in enumerate(ranking, start=1)
        ],
        "selected": {
            "name": selected["name"],
            "source_report": selected["path"],
            "source_report_sha256": selected["sha256"],
            "policy": selected["selected_policy"],
            "tuning": selected["tuning"],
        },
        "post_selection_validation": validation,
        "all_oof_diagnostic": diagnostic,
        "thresholds": {
            "max_degraded_tuning_folds": int(max_degraded_tuning_folds),
            "min_validation_map50": float(min_validation_map50),
            "min_all_oof_map50": float(min_all_oof_map50),
        },
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=named_report, action="append", required=True)
    parser.add_argument("--max-degraded-tuning-folds", type=int, default=0)
    parser.add_argument("--min-validation-map50", type=float, default=0.85)
    parser.add_argument("--min-all-oof-map50", type=float, default=0.85)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    reject_test_reference(output, "selection output")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有 OOF policy selection：{output}")
    report = select_policy(
        args.candidate,
        int(args.max_degraded_tuning_folds),
        float(args.min_validation_map50),
        float(args.min_all_oof_map50),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
