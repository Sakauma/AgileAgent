from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "20260823-4plus2-yolo26-content-gate-v2"
PACKAGE = ROOT / "models/ascend310b/full-score" / RELEASE_ID
FIXED_RELEASE = Path("/home/HwHiAiUser/agileagent/releases") / RELEASE_ID


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_rows() -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in (PACKAGE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        rows[relative] = digest
    return rows


def release_local_path(path: str) -> Path:
    absolute = Path(path)
    relative = absolute.relative_to(FIXED_RELEASE)
    return PACKAGE / relative


def test_packaged_release_has_complete_verified_checksum_inventory() -> None:
    rows = checksum_rows()
    packaged = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }

    assert set(rows) == packaged
    assert len(rows) == 31
    for relative, expected in rows.items():
        assert sha256_file(PACKAGE / relative) == expected


def test_packaged_release_config_and_manifests_are_self_contained() -> None:
    config = yaml.safe_load(
        (PACKAGE / "configs/agent_pipeline_ascend310b.yaml").read_text(
            encoding="utf-8"
        )
    )
    ascend = config["ascend_backend"]
    manifest_path = release_local_path(ascend["build_manifest"])
    validation_path = release_local_path(ascend["validation_report"])

    assert config["runtime"]["server_port"] == 18501
    assert config["inference"]["confidence_default"] == 0.10
    assert ascend["validated"] is True
    assert ascend["validation_candidate"] is False
    assert ascend["model_layout"] == "independent_yolo26_e2e_v1"
    assert ascend["context_mode"] == "model"
    assert ascend["schedule_mode"] == "unified_enqueue"
    assert len(ascend["models"]) == 2
    assert sha256_file(manifest_path) == ascend["build_manifest_sha256"]
    assert sha256_file(validation_path) == ascend["validation_report_sha256"]

    for detector in ascend["models"].values():
        assert detector["output_contract"] == "yolo26_e2e_v1"
        assert detector["max_det"] == 300
        assert sha256_file(release_local_path(detector["path"])) == detector["sha256"]
    assert (
        sha256_file(release_local_path(ascend["context_model"]["path"]))
        == ascend["context_model"]["sha256"]
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_layout"] == "independent_yolo26_e2e_v1"
    assert set(manifest["artifacts"]) == {
        "base_detector",
        "incremental_detector",
        "scene_sensor_net",
    }
    for artifact in manifest["artifacts"].values():
        for name in ("source_weight", "onnx", "aipp", "om", "atc_log"):
            entry = artifact[name]
            local = release_local_path(entry["path"])
            assert local.is_file()
            assert sha256_file(local) == entry["sha256"]

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["passed"] is True
    for name in ("method_config", "source_candidate", "accuracy", "performance"):
        entry = validation[name]
        local = release_local_path(entry["path"])
        assert local.is_file()
        assert sha256_file(local) == entry["sha256"]
    for entry in validation["repeat_performance"]:
        local = release_local_path(entry["path"])
        assert local.is_file()
        assert sha256_file(local) == entry["sha256"]


def test_model_manifest_promotes_packaged_ascend_release() -> None:
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    releases = manifest["ascend_releases"]
    release = next(row for row in releases if row["status"] == "production")

    assert release["id"] == RELEASE_ID
    assert release["ready_without_training"] is True
    assert release["target"] == "Ascend310B1"
    assert release["model_layout"] == "independent_yolo26_e2e_v1"
    assert release["context_mode"] == "model"
    assert release["metrics"]["base_map50"] >= 0.80
    assert release["metrics"]["new_map50"] >= 0.60
    assert release["metrics"]["krr"] >= 0.95
    assert min(release["metrics"]["post_promotion_batch_median_fps"]) >= 30.0
    for row in release["models"].values():
        model_path = ROOT / row["path"]
        assert model_path.is_file()
        assert sha256_file(model_path) == row["sha256"]

    historical = next(row for row in releases if row["id"] == "20260816-full-score-1493b04")
    assert historical["status"] == "historical"


def test_functional_registry_references_current_model_manifest() -> None:
    registry = yaml.safe_load(
        (ROOT / "configs/functional_models.yaml").read_text(encoding="utf-8")
    )
    expected = sha256_file(ROOT / "models/manifest.json")
    references = [
        row["evidence"]["sha256"]
        for row in registry["models"]
        if row.get("evidence", {}).get("path") == "models/manifest.json"
    ]

    assert references
    assert set(references) == {expected}


def test_materializer_is_zero_training_and_preserves_ports() -> None:
    script = (
        ROOT / "scripts/materialize_ascend310b_full_score_release.sh"
    ).read_text(encoding="utf-8")

    assert "RELEASE_PARENT=/home/HwHiAiUser/agileagent/releases" in script
    assert f"RELEASE_ID={RELEASE_ID}" in script
    assert 'RELEASE_ROOT="${RELEASE_PARENT}/${RELEASE_ID}"' in script
    assert "sha256sum -c SHA256SUMS" in script
    assert "tools/95_verify_ascend_release.py" in script
    assert "build_ascend_yolo26_e2e_oms.sh" not in script
    assert "AGILE_AGENT_ASCEND_PORT" not in script
    assert "systemctl" not in script


def test_ascend_start_and_stop_default_to_packaged_production_release() -> None:
    expected = f"/home/HwHiAiUser/agileagent/releases/{RELEASE_ID}"
    start = (ROOT / "scripts/start_agent_ascend310b.sh").read_text(encoding="utf-8")
    stop = (ROOT / "scripts/stop_agent_ascend310b.sh").read_text(encoding="utf-8")

    assert f"AGILE_AGENT_ASCEND_RELEASE:-{expected}" in start
    assert f"AGILE_AGENT_ASCEND_RELEASE:-{expected}" in stop


def test_post_promotion_evidence_remains_full_score() -> None:
    score = json.loads(
        (PACKAGE / "validation/score.json").read_text(encoding="utf-8")
    )
    assert score["passed"] is True
    assert score["metrics"]["base_map50"] >= 0.80
    assert score["metrics"]["new_map50"] >= 0.60
    assert score["metrics"]["krr"] >= 0.95

    for name in (
        "benchmark-post-promotion.json",
        "benchmark-post-promotion-repeat.json",
    ):
        benchmark = json.loads(
            (PACKAGE / "validation" / name).read_text(encoding="utf-8")
        )
        assert benchmark["competition"]["batch_fps_passed"] is True
        assert benchmark["competition"]["batch_fps"] >= 30.0
