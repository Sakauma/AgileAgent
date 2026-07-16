from __future__ import annotations

import json
from pathlib import Path

from fair_agent.modules.incremental_lineage import _canonical_sha256
from fair_agent.modules.inference_provenance import InferenceSourceRouter, attach_source_decision
from fair_agent.modules import generation_management


def write_catalog(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["manifest_sha256"] = _canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def settings(tmp_path: Path) -> dict:
    return {
        "lineage": {
            "root": str(tmp_path / "lineage"),
            "base_manifest": str(tmp_path / "lineage" / "base.json"),
        }
    }


def test_router_separates_base_incremental_and_unknown_sources(tmp_path: Path) -> None:
    config = settings(tmp_path)
    write_catalog(Path(config["lineage"]["base_manifest"]), {
        "schema_version": 1,
        "catalog_id": "production-history",
        "kind": "frozen_production_lineage",
        "files": [
            {"image_sha256": "a" * 64, "source_scope": "base"},
            {"image_sha256": "b" * 64, "source_scope": "incremental", "round_id": "round-01"},
        ],
        "cache_files": [],
    })
    write_catalog(tmp_path / "lineage" / "accepted" / "generation-2.json", {
        "schema_version": 1,
        "catalog_id": "generation-2",
        "kind": "accepted_incremental_batch",
        "generation_id": "generation-2",
        "batch_id": "batch-2",
        "files": [{"image_sha256": "c" * 64}],
        "cache_files": [],
    })

    router = InferenceSourceRouter(config, enabled=True, unknown_policy="base_only")

    assert router.resolve("a" * 64)["incremental_protocol"] is None
    assert router.resolve("a" * 64)["source_scope"] == "base"
    assert router.resolve("b" * 64)["incremental_protocol"] == "auto"
    accepted = router.resolve("c" * 64)
    assert accepted["source_scope"] == "incremental"
    assert accepted["generation_id"] == "generation-2"
    unknown = router.resolve("d" * 64)
    assert unknown["source_scope"] == "unknown"
    assert unknown["inference_scope"] == "base"
    assert unknown["incremental_protocol"] is None


def test_router_refreshes_when_accepted_catalog_is_added(tmp_path: Path) -> None:
    config = settings(tmp_path)
    router = InferenceSourceRouter(config, enabled=True, unknown_policy="base_only")
    assert router.resolve("e" * 64)["source_scope"] == "unknown"

    write_catalog(tmp_path / "lineage" / "accepted" / "generation-3.json", {
        "schema_version": 1,
        "catalog_id": "generation-3",
        "kind": "accepted_incremental_batch",
        "generation_id": "generation-3",
        "files": [{"image_sha256": "e" * 64}],
        "cache_files": [],
    })

    assert router.resolve("e" * 64)["source_scope"] == "incremental"


def test_source_decision_is_added_without_removing_model_decision() -> None:
    result = {"agent": {"decision": {"base_detection_count": 2}}}
    attach_source_decision(result, {
        "source_scope": "base",
        "inference_scope": "base",
        "known": True,
        "reason": "frozen_base_lineage",
        "generation_id": None,
        "batch_id": None,
    })
    assert result["agent"]["decision"]["base_detection_count"] == 2
    assert result["agent"]["decision"]["source_scope"] == "base"


def test_generation_recheck_never_runs_candidate_on_base_domain(monkeypatch) -> None:
    calls = []

    def fake_engine(_config, _registry, generation_id):
        return generation_id

    def fake_run_engine(generation_id, image_paths, _config):
        names = [path.name for path in image_paths]
        calls.append((generation_id, names))
        return [{"generation_id": generation_id, "filename": name} for name in names]

    monkeypatch.setattr(generation_management, "_engine", fake_engine)
    monkeypatch.setattr(generation_management, "_run_engine", fake_run_engine)

    results = generation_management._run_source_scoped_recheck(
        {},
        {},
        "generation-parent",
        "generation-candidate",
        "generation-root",
        [Path("base-a.png"), Path("base-b.png")],
        [Path("incremental-old.png")],
        [Path("incremental-current.png")],
    )

    assert calls == [
        ("generation-root", ["base-a.png", "base-b.png"]),
        ("generation-root", ["base-a.png", "base-b.png"]),
        ("generation-parent", ["incremental-old.png"]),
        ("generation-candidate", ["incremental-old.png"]),
        ("generation-candidate", ["incremental-current.png"]),
    ]
    assert results["base_before"] == results["base_after"]
    assert all(
        row["generation_id"] != "generation-candidate"
        for row in results["base_after"]
    )
