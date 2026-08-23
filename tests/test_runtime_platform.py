from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fair_agent.core.config import (
    ASCEND_CONFIG,
    DEFAULT_CONFIG,
    configured_python,
    detect_host_architecture,
    load_config,
    runtime_config_path,
    select_runtime_config,
)
from fair_agent.modules import configuration


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("x86_64", "x86"),
        ("AMD64", "x86"),
        ("aarch64", "arm"),
        ("ARM64", "arm"),
        ("armv7l", "arm"),
    ],
)
def test_host_architecture_aliases_are_normalized(
    machine: str, expected: str
) -> None:
    assert detect_host_architecture(machine) == expected


def test_auto_config_selects_cuda_for_x86_and_ascend_for_arm(monkeypatch) -> None:
    monkeypatch.delenv("AGILE_AGENT_CONFIG", raising=False)
    monkeypatch.delenv("AGILE_AGENT_ASCEND_RELEASE", raising=False)
    assert runtime_config_path("auto", machine="x86_64") == DEFAULT_CONFIG
    assert runtime_config_path("auto", machine="aarch64") == ASCEND_CONFIG
    x86 = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    arm = yaml.safe_load(ASCEND_CONFIG.read_text(encoding="utf-8"))
    assert x86["inference"]["backend"] == "ultralytics_cuda"
    assert arm["inference"]["backend"] == "ascend_acl"


def test_explicit_config_and_environment_override_auto_selection(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "custom.yaml"
    explicit_selection = select_runtime_config(
        explicit, machine="aarch64", environ={"AGILE_AGENT_CONFIG": "ignored.yaml"}
    )
    assert explicit_selection["config_path"] == explicit
    assert explicit_selection["selection"] == "explicit"

    environment_selection = select_runtime_config(
        "auto",
        machine="x86_64",
        environ={"AGILE_AGENT_CONFIG": str(explicit)},
    )
    assert environment_selection["config_path"] == explicit
    assert environment_selection["selection"] == "environment"


def test_arm_release_root_supplies_its_own_runtime_config(tmp_path: Path) -> None:
    selection = select_runtime_config(
        "auto",
        machine="arm64",
        environ={"AGILE_AGENT_ASCEND_RELEASE": str(tmp_path)},
    )
    assert selection["config_path"] == (
        tmp_path / "configs" / "agent_pipeline_ascend310b.yaml"
    )
    assert selection["selection"] == "ascend_release"


def test_loaded_x86_config_records_automatic_runtime_metadata(monkeypatch) -> None:
    monkeypatch.delenv("AGILE_AGENT_CONFIG", raising=False)
    monkeypatch.delenv("AGILE_AGENT_ASCEND_RELEASE", raising=False)
    monkeypatch.setattr("fair_agent.core.config.platform.machine", lambda: "AMD64")
    config = load_config("auto")
    assert config["_config_path"] == "configs/agent_pipeline.yaml"
    assert config["_runtime_platform"] == {
        "machine": "AMD64",
        "architecture": "x86",
        "selection": "architecture",
        "automatic": True,
        "config_path": "configs/agent_pipeline.yaml",
        "backend": "ultralytics_cuda",
        "device_family": "x86_cuda",
        "model_format": "pt",
        "architecture_match": True,
    }


def test_configured_python_honors_startup_environment(monkeypatch, tmp_path: Path) -> None:
    selected = tmp_path / "bin" / "python"
    monkeypatch.setenv("AGILE_AGENT_PYTHON", str(selected))
    assert configured_python({"runtime": {"local_python": "/configured/python"}}) == selected


def test_persistent_auto_config_writes_the_selected_yaml(
    monkeypatch, tmp_path: Path
) -> None:
    selected = tmp_path / "selected.yaml"
    selected.write_text("runtime:\n  server_port: 8501\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(configuration, "runtime_config_path", lambda _path: selected)

    def fake_write(path, data, operation):
        calls.append((path, data, operation))
        return path

    monkeypatch.setattr(configuration, "write_config", fake_write)
    assert (
        configuration.set_persistent_value(
            "auto", "runtime.server_port", "8502"
        )
        == selected
    )
    assert configuration.unset_persistent_value(
        "auto", "runtime.server_port"
    ) == selected
    assert [call[0] for call in calls] == [selected, selected]
    assert calls[0][1]["runtime"]["server_port"] == 8502
    assert "server_port" not in calls[1][1]["runtime"]
