#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.config import validate_config  # noqa: E402
from fair_agent.modules.ascend_release import (  # noqa: E402
    FULL_SCORE_ACCURACY_GATES,
    FULL_SCORE_BATCH_FPS_MIN,
    FULL_SCORE_BATCH_IMAGE_COUNT,
    FULL_SCORE_BATCH_ROUNDS,
    verify_ascend_artifacts,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label}必须是JSON object：{path}")
    return payload


def read_yaml(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label}必须是YAML mapping：{path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def checked_entry(entry: Any, label: str) -> tuple[Path, str]:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{label}缺少文件条目")
    path = Path(str(entry.get("path") or ""))
    digest = str(entry.get("sha256") or "")
    if not path.is_file():
        raise FileNotFoundError(f"{label}文件不存在：{path}")
    if len(digest) != 64 or sha256_file(path) != digest:
        raise ValueError(f"{label} SHA256不一致：{path}")
    return path, digest


def copy_entry(entry: Any, destination: Path, label: str) -> dict[str, str]:
    source, digest = checked_entry(entry, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != digest:
        raise RuntimeError(f"{label}复制后SHA256不一致：{destination}")
    return {"path": str(destination.resolve()), "sha256": digest}


def artifact_by_role(manifest: Mapping[str, Any], role: str) -> tuple[str, dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("build manifest缺少artifacts")
    rows = [
        (str(name), dict(row))
        for name, row in artifacts.items()
        if isinstance(row, Mapping) and row.get("role") == role
    ]
    if len(rows) != 1:
        raise ValueError(f"build manifest要求恰好一个{role}，实际{len(rows)}")
    return rows[0]


def validate_method(method: Mapping[str, Any]) -> None:
    if method.get("kind") != "ascend310b_full_score_method":
        raise ValueError("满分方法配置kind非法")
    target = method.get("target") or {}
    if target.get("formal_port") != 8501 or target.get("candidate_port") != 8502:
        raise ValueError("满分方法必须保留8501公共入口和8502候选端口")
    accuracy = (method.get("competition") or {}).get("accuracy_gates") or {}
    performance = (method.get("competition") or {}).get("performance_gate") or {}
    expected_accuracy = {
        "base_map50_min": FULL_SCORE_ACCURACY_GATES["base_map50"],
        "new_map50_min": FULL_SCORE_ACCURACY_GATES["new_map50"],
        "krr_min": FULL_SCORE_ACCURACY_GATES["krr"],
    }
    if any(float(accuracy.get(name, -1.0)) != value for name, value in expected_accuracy.items()):
        raise ValueError("满分方法精度门禁与正式发布器不一致")
    if (
        int(performance.get("batch_image_count", 0)) != FULL_SCORE_BATCH_IMAGE_COUNT
        or int(performance.get("batch_rounds", 0)) != FULL_SCORE_BATCH_ROUNDS
        or float(performance.get("median_fps_min", -1.0))
        != FULL_SCORE_BATCH_FPS_MIN
    ):
        raise ValueError("满分方法性能门禁与正式发布器不一致")


def validate_score(report: Mapping[str, Any]) -> dict[str, float]:
    if report.get("schema_version") != 2:
        raise ValueError("accuracy report必须是score schema v2")
    if report.get("unlabeled_predictions_frozen_before_labels") is not True:
        raise ValueError("accuracy report未证明先冻结预测再读取标签")
    metrics = report.get("metrics")
    gates = report.get("competition_gates")
    if not isinstance(metrics, Mapping) or not isinstance(gates, Mapping):
        raise ValueError("accuracy report缺少metrics/competition_gates")
    values = {name: float(metrics[name]) for name in FULL_SCORE_ACCURACY_GATES}
    failures = [
        name
        for name, minimum in FULL_SCORE_ACCURACY_GATES.items()
        if values[name] < minimum or gates.get(name) is not True
    ]
    if failures or report.get("score_passed", report.get("passed")) is not True:
        raise ValueError("accuracy report未达到满分门禁：" + ",".join(failures))
    return values


def validate_benchmark(report: Mapping[str, Any], label: str) -> float:
    if report.get("schema_version") != 5:
        raise ValueError(f"{label}必须是benchmark schema v5")
    protocol = report.get("protocol") or {}
    competition = report.get("competition") or {}
    rounds = competition.get("batch_rounds")
    if (
        int(protocol.get("batch_probe_size", 0)) != FULL_SCORE_BATCH_IMAGE_COUNT
        or int(protocol.get("batch_rounds", 0)) != FULL_SCORE_BATCH_ROUNDS
        or float(protocol.get("target_batch_fps", -1.0))
        != FULL_SCORE_BATCH_FPS_MIN
        or int(competition.get("batch_image_count", 0))
        != FULL_SCORE_BATCH_IMAGE_COUNT
        or not isinstance(rounds, list)
        or len(rounds) != FULL_SCORE_BATCH_ROUNDS
    ):
        raise ValueError(f"{label}的20图三轮协议非法")
    fps = float(competition.get("batch_fps", -1.0))
    if fps < FULL_SCORE_BATCH_FPS_MIN or competition.get("batch_fps_passed") is not True:
        raise ValueError(f"{label}未达到30 FPS满分门禁")
    gates = report.get("gates")
    if isinstance(gates, Mapping) and any(value is not True for value in gates.values()):
        raise ValueError(f"{label}包含失败的请求/样本门禁")
    return fps


def validate_training_report(report: Mapping[str, Any]) -> dict[str, bool]:
    audit = report.get("dataset_audit") or {}
    shared_drift = float(
        report.get("shared_max_drift", (report.get("initialization") or {}).get("shared_max_drift", -1.0))
    )
    isolation = (
        int(audit.get("old_raw_image_count", -1)) == 0
        and int(audit.get("old_raw_label_count", -1)) == 0
        and int(audit.get("old_feature_cache_count", -1)) == 0
        and audit.get("original_data_modified") is False
    )
    if report.get("kind") != "shared_backbone_dual_head_training":
        raise ValueError("training report kind非法")
    if not isolation or shared_drift != 0.0:
        raise ValueError("training report未通过增量数据隔离或Base零漂移前置条件")
    return {
        "incremental_data_isolation": True,
        "shared_max_drift_zero": True,
    }


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def materialize_release(
    *,
    candidate_config_path: Path,
    method_config_path: Path,
    score_path: Path,
    benchmark_path: Path,
    repeat_benchmark_paths: list[Path],
    release_root: Path,
    internal_port: int,
) -> dict[str, Any]:
    candidate = read_yaml(candidate_config_path, "候选配置")
    method = read_yaml(method_config_path, "满分方法配置")
    score = read_json(score_path, "accuracy report")
    benchmark = read_json(benchmark_path, "performance report")
    repeats = [read_json(path, f"repeat benchmark {path}") for path in repeat_benchmark_paths]
    validate_method(method)
    metrics = validate_score(score)
    primary_fps = validate_benchmark(benchmark, "performance report")
    repeat_fps = [
        validate_benchmark(report, f"repeat benchmark {path}")
        for path, report in zip(repeat_benchmark_paths, repeats)
    ]
    if internal_port in {8501, 8502} or not 1 <= internal_port <= 65535:
        raise ValueError("正式主实例内部端口必须避开8501/8502且位于1..65535")

    ascend = candidate.get("ascend_backend") or {}
    if (
        candidate.get("runtime", {}).get("server_port") != 8502
        or ascend.get("validation_candidate") is not True
        or ascend.get("validated") is not False
        or ascend.get("model_layout") != "shared_backbone_dual_head_v1"
        or ascend.get("context_mode") != "fixed_neutral_v1"
    ):
        raise ValueError("输入必须是通过8502隔离评分的共享双头候选配置")
    preflight = verify_ascend_artifacts(ascend, require_validation=False)
    if preflight["status"] != "passed":
        raise ValueError("候选资产校验失败：" + ";".join(preflight["errors"]))

    source_manifest_path = Path(str(ascend["build_manifest"])).resolve()
    source_manifest = read_json(source_manifest_path, "source build manifest")
    training_path, _ = checked_entry(source_manifest.get("training_report"), "training_report")
    training_report = read_json(training_path, "training report")
    validity = validate_training_report(training_report)

    for name in ("om", "provenance", "validation", "configs", "reports"):
        target = release_root / name
        if target.exists():
            raise FileExistsError(f"release目标已存在，拒绝覆盖：{target}")
    for name in ("om", "provenance", "validation", "configs", "reports"):
        (release_root / name).mkdir(parents=True, exist_ok=False)

    candidate_copy = release_root / "provenance/candidate-8502.yaml"
    method_copy = release_root / "provenance/full_score_method.yaml"
    source_manifest_copy = release_root / "provenance/source-build-manifest.json"
    shutil.copy2(candidate_config_path, candidate_copy)
    shutil.copy2(method_config_path, method_copy)
    shutil.copy2(source_manifest_path, source_manifest_copy)

    release_manifest = copy.deepcopy(source_manifest)
    release_manifest.update(
        {
            "release_kind": "ascend310b_full_score_primary_v1",
            "release_git_sha": git_head(),
            "source_build_manifest": {
                "path": str(source_manifest_copy.resolve()),
                "sha256": sha256_file(source_manifest_copy),
            },
            "method_config": {
                "path": str(method_copy.resolve()),
                "sha256": sha256_file(method_copy),
            },
        }
    )
    for top_name in ("training_report", "export_manifest"):
        entry = source_manifest.get(top_name)
        source, _ = checked_entry(entry, top_name)
        destination = release_root / f"provenance/{top_name}{source.suffix}"
        release_manifest[top_name] = copy_entry(entry, destination, top_name)

    copied_artifacts: dict[str, dict[str, Any]] = {}
    for role, om_name in (
        ("dual_detector", "shared_backbone_dual_head.om"),
        ("context", "scene_sensor_net.om"),
    ):
        model_id, artifact = artifact_by_role(source_manifest, role)
        copied = copy.deepcopy(artifact)
        for field in ("source_weight", "onnx", "aipp", "atc_log"):
            source, _ = checked_entry(artifact.get(field), f"{role}.{field}")
            destination = release_root / f"provenance/{role}-{field}{source.suffix}"
            copied[field] = copy_entry(artifact.get(field), destination, f"{role}.{field}")
        copied["om"] = copy_entry(
            artifact.get("om"), release_root / f"om/{om_name}", f"{role}.om"
        )
        copied_artifacts[model_id] = copied
    release_manifest["artifacts"] = copied_artifacts

    manifest_path = release_root / "provenance/release-build-manifest.json"
    write_json(manifest_path, release_manifest)
    manifest_digest = sha256_file(manifest_path)

    accuracy_copy = release_root / "validation/score.json"
    performance_copy = release_root / "validation/benchmark.json"
    shutil.copy2(score_path, accuracy_copy)
    shutil.copy2(benchmark_path, performance_copy)
    repeat_entries = []
    for index, path in enumerate(repeat_benchmark_paths, start=1):
        destination = release_root / f"validation/benchmark-repeat-{index}.json"
        shutil.copy2(path, destination)
        repeat_entries.append(
            {
                "path": str(destination.resolve()),
                "sha256": sha256_file(destination),
                "passed": True,
                "batch_fps": repeat_fps[index - 1],
            }
        )

    generation_source = Path(str(candidate.get("generation", {}).get("registry") or ""))
    if not generation_source.is_file():
        raise FileNotFoundError(f"候选generation registry不存在：{generation_source}")
    generation_copy = release_root / "provenance/generations.json"
    shutil.copy2(generation_source, generation_copy)

    validity.update(
        {
            "asset_hashes_verified": True,
            "predictions_frozen_before_labels": True,
        }
    )
    validation_summary = {
        "schema_version": 1,
        "kind": "ascend310b_full_score_release_validation",
        "build_manifest_sha256": manifest_digest,
        "method_config": {
            "path": str(method_copy.resolve()),
            "sha256": sha256_file(method_copy),
        },
        "source_candidate": {
            "path": str(candidate_copy.resolve()),
            "sha256": sha256_file(candidate_copy),
        },
        "accuracy": {
            "path": str(accuracy_copy.resolve()),
            "sha256": sha256_file(accuracy_copy),
            "passed": True,
            "metrics": metrics,
        },
        "performance": {
            "path": str(performance_copy.resolve()),
            "sha256": sha256_file(performance_copy),
            "passed": True,
            "batch_fps": primary_fps,
        },
        "repeat_performance": repeat_entries,
        "validity": validity,
        "diagnostics_block_release": False,
        "passed": True,
    }
    validation_path = release_root / "validation/validation-summary.json"
    write_json(validation_path, validation_summary)

    production = copy.deepcopy(candidate)
    production["runtime"]["server_port"] = int(internal_port)
    production["generation"]["registry"] = str(generation_copy.resolve())
    production["performance"]["report_root"] = str((release_root / "reports").resolve())
    production_ascend = production["ascend_backend"]
    production_ascend.update(
        {
            "validated": True,
            "validation_candidate": False,
            "validation_report": str(validation_path.resolve()),
            "validation_report_sha256": sha256_file(validation_path),
            "build_manifest": str(manifest_path.resolve()),
            "build_manifest_sha256": manifest_digest,
        }
    )
    dual_entry = next(
        row for row in copied_artifacts.values() if row.get("role") == "dual_detector"
    )
    context_entry = next(
        row for row in copied_artifacts.values() if row.get("role") == "context"
    )
    detector = next(iter(production_ascend["models"].values()))
    detector.update(dual_entry["om"])
    production_ascend["context_model"] = dict(context_entry["om"])
    validate_config(production)
    verification = verify_ascend_artifacts(production_ascend, require_validation=True)
    if verification["status"] != "passed":
        raise RuntimeError("正式release校验失败：" + ";".join(verification["errors"]))

    config_path = release_root / "configs/agent_pipeline_ascend310b.yaml"
    config_path.write_text(
        yaml.safe_dump(production, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "kind": "ascend310b_full_score_primary_release",
        "release_root": str(release_root.resolve()),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "public_port": 8501,
        "internal_port": internal_port,
        "candidate_port_reserved": 8502,
        "metrics": metrics,
        "batch_fps": primary_fps,
        "repeat_batch_fps": repeat_fps,
        "build_manifest_sha256": manifest_digest,
        "validation_report_sha256": sha256_file(validation_path),
        "verification": verification,
    }
    write_json(release_root / "release.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把四项满分的8502共享双头候选物化为可验证的正式主线release。"
    )
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument(
        "--method-config",
        type=Path,
        default=ROOT / "configs/ascend310b/full_score_method.yaml",
    )
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--repeat-benchmark", type=Path, action="append", default=[])
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--internal-port", type=int, default=18501)
    args = parser.parse_args()
    result = materialize_release(
        candidate_config_path=args.candidate_config.resolve(),
        method_config_path=args.method_config.resolve(),
        score_path=args.score.resolve(),
        benchmark_path=args.benchmark.resolve(),
        repeat_benchmark_paths=[path.resolve() for path in args.repeat_benchmark],
        release_root=args.release_root.resolve(),
        internal_port=args.internal_port,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
