from __future__ import annotations

from pathlib import Path

import yaml

from fair_agent.modules import generation_management


def test_online_inference_has_no_source_identity_router() -> None:
    assert not Path("fair_agent/modules/inference_provenance.py").exists()
    config = yaml.safe_load(Path("configs/agent_pipeline.yaml").read_text(encoding="utf-8"))
    assert "source_aware" not in config["routing"]
    assert "unknown_source_policy" not in config["routing"]


def test_production_recheck_runs_candidate_on_every_after_image(monkeypatch) -> None:
    calls = []

    def fake_engine(_config, _registry, generation_id):
        return generation_id

    def fake_run_engine(generation_id, image_paths, _config):
        names = [path.name for path in image_paths]
        calls.append((generation_id, names))
        return [{"generation_id": generation_id, "filename": name} for name in names]

    monkeypatch.setattr(generation_management, "_engine", fake_engine)
    monkeypatch.setattr(generation_management, "_run_engine", fake_run_engine)

    results = generation_management._run_production_recheck(
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
        ("generation-candidate", ["base-a.png", "base-b.png"]),
        ("generation-parent", ["incremental-old.png"]),
        ("generation-candidate", ["incremental-old.png"]),
        ("generation-candidate", ["incremental-current.png"]),
    ]
    assert all(
        row["generation_id"] == "generation-candidate"
        for row in results["base_after"]
    )
