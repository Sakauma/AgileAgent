from __future__ import annotations

import copy
import json
import runpy
import struct
from pathlib import Path

import pytest

from fair_agent.core.hashes import sha256_file
from fair_agent.modules.ascend_benchmark_guard import (
    compare_environment_snapshots,
    cpu_policy_snapshot,
    evaluate_environment_snapshot,
    manifest_artifact_evidence,
    parse_listeners,
    parse_npu_smi,
)


def _snapshot() -> dict:
    health = {
        "available": True,
        "payload": {"status": "ready", "backend": "ascend_acl"},
    }
    return {
        "system": {
            "machine": "aarch64",
            "release": "5.10.0+",
            "python_executable": "/usr/local/miniconda3/envs/agileagent/bin/python",
        },
        "git": {
            "head": "a" * 40,
            "origin": "https://github.com/Sakauma/AgileAgent.git",
            "clean": True,
        },
        "config": {"configured": True, "exists": True, "sha256": "b" * 64},
        "build_manifest": {
            "configured": True,
            "exists": True,
            "sha256": "c" * 64,
            "passed": True,
            "manifest_identity": {
                "git_sha": "a" * 40,
                "soc_version": "Ascend310B1",
                "cann_version": "7.0.RC1",
                "precision": "mixed_float16",
            },
        },
        "cpu_policy": {"supported": False, "state": "unsupported", "policies": []},
        "process_cpu_samples": [[], [], []],
        "listeners": {
            "8501": {"pid": 101, "process": "python"},
            "8502": {"pid": 102, "process": "python"},
        },
        "official_health": health,
        "candidate_health": health,
        "npu_smi": {
            "available": True,
            "devices": [
                {
                    "npu": 0,
                    "name": "310B1",
                    "health": "OK",
                    "power_w": 9.8,
                    "temperature_c": 60,
                }
            ],
        },
    }


def test_parse_npu_smi_extracts_health_and_temperature() -> None:
    output = """
| 0       310B1                 | OK              | 9.8          60                81    / 81            |
| 0       0                     | NA              | 0            5224 / 11577                            |
"""
    assert parse_npu_smi(output) == [
        {
            "npu": 0,
            "name": "310B1",
            "health": "OK",
            "power_w": 9.8,
            "temperature_c": 60,
        }
    ]


def test_cpu_policy_snapshot_records_unsupported_or_performance(tmp_path: Path) -> None:
    assert cpu_policy_snapshot(tmp_path) == {
        "supported": False,
        "state": "unsupported",
        "policies": [],
    }
    policy = tmp_path / "policy0"
    policy.mkdir()
    (policy / "scaling_governor").write_text("performance\n", encoding="utf-8")
    (policy / "scaling_available_governors").write_text(
        "powersave performance\n", encoding="utf-8"
    )
    (policy / "scaling_driver").write_text("test_driver\n", encoding="utf-8")
    result = cpu_policy_snapshot(tmp_path)
    assert result["supported"] is True
    assert result["policies"] == [
        {
            "policy": "policy0",
            "governor": "performance",
            "available_governors": ["powersave", "performance"],
            "driver": "test_driver",
        }
    ]


def test_parse_listeners_binds_candidate_pid() -> None:
    output = (
        'LISTEN 0 2048 127.0.0.1:8501 0.0.0.0:* users:(("python",pid=101,fd=6))\n'
        'LISTEN 0 2048 127.0.0.1:8502 0.0.0.0:* users:(("python",pid=102,fd=7))\n'
    )
    assert parse_listeners(output)["8502"]["pid"] == 102


def test_environment_guard_accepts_unsupported_governor_and_rejects_noise() -> None:
    snapshot = _snapshot()
    result = evaluate_environment_snapshot(snapshot, candidate_state="ready")
    assert result["passed"] is True
    assert result["checks"]["cpu_governor_consistent"] is True

    snapshot["process_cpu_samples"][1].append(
        {"pid": 999, "comm": "xscreensaver", "user": "HwHiAiUser", "cpu_percent": 38.0}
    )
    result = evaluate_environment_snapshot(snapshot, candidate_state="ready")
    assert result["passed"] is False
    assert result["checks"]["process_cpu_below_limit"] is False
    assert result["offenders"][0]["comm"] == "xscreensaver"


def test_environment_guard_records_but_does_not_gate_post_run_temperature() -> None:
    snapshot = _snapshot()
    snapshot["npu_smi"]["devices"][0]["temperature_c"] = 67

    before = evaluate_environment_snapshot(snapshot, candidate_state="ready")
    after = evaluate_environment_snapshot(
        snapshot,
        candidate_state="ready",
        require_temperature_limit=False,
    )

    assert before["passed"] is False
    assert before["checks"]["npu_temperature_at_most_limit"] is False
    assert after["passed"] is True
    assert after["temperatures_c"] == [67]
    assert after["limits"]["temperature_limit_required"] is False


def test_environment_guard_requires_free_candidate_port_before_start() -> None:
    snapshot = _snapshot()
    result = evaluate_environment_snapshot(snapshot, candidate_state="free")
    assert result["checks"]["candidate_port_state"] is False
    del snapshot["listeners"]["8502"]
    snapshot["candidate_health"] = {"available": False, "error": "connection refused"}
    assert evaluate_environment_snapshot(snapshot, candidate_state="free")["passed"] is True


def test_environment_comparison_detects_governor_drift() -> None:
    reference = _snapshot()
    current = copy.deepcopy(reference)
    assert compare_environment_snapshots(current, reference)["passed"] is True
    current["cpu_policy"] = {
        "supported": True,
        "state": "configured",
        "policies": [{"policy": "policy0", "governor": "powersave"}],
    }
    result = compare_environment_snapshots(current, reference)
    assert result["passed"] is False
    assert result["checks"]["cpu_policy"] is False


def test_manifest_evidence_verifies_onnx_aipp_and_om_hashes(tmp_path: Path) -> None:
    files = {}
    for name in ("onnx", "aipp", "om"):
        path = tmp_path / f"base.{name}"
        path.write_bytes(name.encode())
        files[name] = {"path": str(path), "sha256": sha256_file(path)}
    manifest = tmp_path / "build-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "git_sha": "a" * 40,
                "soc_version": "Ascend310B1",
                "cann_version": "7.0.RC1",
                "precision": "mixed_float16",
                "artifacts": {"base_detector": files},
            }
        ),
        encoding="utf-8",
    )
    result = manifest_artifact_evidence(manifest)
    assert result["passed"] is True
    Path(files["om"]["path"]).write_bytes(b"drift")
    assert manifest_artifact_evidence(manifest)["passed"] is False


def test_p8_relative_stability_uses_two_percent_boundary() -> None:
    script = runpy.run_path("tools/97_benchmark_ascend_api.py")
    within = script["relative_difference_within"]
    assert within(102.0, 100.0, 0.02) is True
    assert within(97.9, 100.0, 0.02) is False


@pytest.mark.parametrize(
    ("color_type", "expected"),
    [(0, "grayscale"), (2, "rgb"), (6, "rgba")],
)
def test_benchmark_png_contract_accepts_supported_dvpp_formats(
    tmp_path: Path,
    color_type: int,
    expected: str,
) -> None:
    script = runpy.run_path("tools/97_benchmark_ascend_api.py")
    path = tmp_path / f"color-{color_type}.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">IIBBBBB", 640, 512, 8, color_type, 0, 0, 0)
        + b"\x00\x00\x00\x00"
    )

    assert script["validate_png"](path)["color_type"] == expected


def test_score_batch_multipart_uses_files_field_and_strict_fps_gate(
    tmp_path: Path,
) -> None:
    script = runpy.run_path("tools/97_benchmark_ascend_api.py")
    first = tmp_path / "one.png"
    second = tmp_path / "two.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    body = script["batch_multipart_body"](
        [first, second], 0.5, "ScoreBoundary"
    )
    assert body.count(b'name="files"') == 2
    assert b'name="file";' not in body
    assert b'name="confidence"' in body
    assert body.endswith(b"--ScoreBoundary--\r\n")
    assert 20 * 1000.0 / 666.6666666667 < 30.0
    assert 20 * 1000.0 / 666.6666666666 >= 30.0
