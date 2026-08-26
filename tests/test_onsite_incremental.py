from __future__ import annotations

import io
import json
import shutil
import subprocess
import zipfile
from copy import deepcopy
from pathlib import Path

from PIL import Image
from starlette.testclient import TestClient

from fair_agent.cli import build_parser
from fair_agent.core.config import load_config
from fair_agent.core.runtime_log import StructuredEventLog
from fair_agent.modules.onsite_incremental import (
    AscendDeploymentExecutor,
    LoopbackGenerationController,
    OnsiteIncrementalWorkflow,
    RUNTIME_CONTROL_CAPABILITY,
    inspect_onsite_bundle,
)
from fair_agent.web.app import create_app


def _bundle(path: Path, *, with_names: bool = True) -> Path:
    image = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(image, format="PNG")
    with zipfile.ZipFile(path, "w") as archive:
        if with_names:
            archive.writestr("classes.yaml", "names:\n  0: onsite_vehicle\n")
        for index in range(6):
            archive.writestr(f"images/train/new_{index}.png", image.getvalue())
            archive.writestr(
                f"labels/train/new_{index}.txt", "0 0.5 0.5 0.2 0.2\n"
            )
    return path


def _full_names_bundle(path: Path) -> Path:
    image = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(image, format="PNG")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "data.yaml",
            "names:\n"
            "  0: soldier\n"
            "  1: small_aircraft\n"
            "  2: warship\n"
            "  3: tank\n"
            "  4: patrol_boat\n"
            "  5: armored_vehicle\n"
            "  6: onsite_vehicle\n",
        )
        for index in range(6):
            archive.writestr(f"images/train/new_{index}.png", image.getvalue())
            archive.writestr(
                f"labels/train/new_{index}.txt", "6 0.5 0.5 0.2 0.2\n"
            )
    return path


def _flat_classes_txt_bundle(path: Path) -> Path:
    image = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(image, format="PNG")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "classes.txt",
            "soldier\nsmall_aircraft\nwarship\ntank\n"
            "patrol_boat\narmored_vehicle\nonsite_vehicle\n",
        )
        for index in range(6):
            archive.writestr(f"flat_{index}.png", image.getvalue())
            archive.writestr(
                f"flat_{index}.txt", "6 0.5 0.5 0.2 0.2\n"
            )
    return path


def _config(tmp_path: Path) -> dict:
    config = deepcopy(load_config())
    registry = tmp_path / "generations.json"
    shutil.copy2("models/generations.json", registry)
    config["generation"]["registry"] = str(registry)
    config["generation"]["runtime_registry"] = str(registry)
    config["generation"]["auto_promote"] = False
    config["incremental_workbench"]["root"] = str(tmp_path / "batches")
    config["incremental_workbench"]["training"]["python"] = "python"
    config["incremental_workbench"]["training"]["device"] = "0"
    config["performance"]["target_api_fps"] = 1.0
    config["performance"]["batch_probe_size"] = 2
    config["performance"]["benchmark_rounds"] = 1
    config["logging"]["root"] = str(tmp_path / "logs")
    return config


class _Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.batch_dir = root / "batch-1"
        self.batch_dir.mkdir(parents=True)
        lock = self.batch_dir / "sealed_lock" / "images" / "lock"
        lock.mkdir(parents=True)
        Image.new("RGB", (32, 24), "white").save(lock / "new.png")
        self.updated = []

    def _batch_dir(self, _batch_id: str) -> Path:
        return self.batch_dir

    def create(self, _filename, _data, _name=None, _class_names=None):
        return {
            "batch_id": "batch-1",
            "status": "AUDITED",
            "files": [{"image": "images/train/new.png", "label": "labels/train/new.txt"}],
            "audit": {
                "incremental_mode": "class_incremental",
                "requires_class_confirmation": False,
                "class_bindings": [{
                    "source_class_id": 0,
                    "training_class_id": 0,
                    "global_class_id": 6,
                    "display_name": "onsite_vehicle",
                    "semantic_status": "confirmed",
                    "is_existing_class": False,
                }],
                "old_raw_image_count": 0,
            },
        }

    def inject(self, _batch_id: str):
        return {
            **self.create("", b""),
            "status": "INJECTED",
            "injection": {
                "counts": {"train": 4, "val": 1, "lock": 1},
                "dataset_fingerprint": "dataset-v1",
            },
        }

    def update_training(self, _batch_id, _job_id, status, **details):
        self.updated.append((status, details))
        return {"status": status}


class _Manager:
    def __init__(self) -> None:
        self.job = {
            "job_id": "job-1",
            "status": "ACCEPTED",
            "lifecycle_result": {
                "status": "ACCEPTED",
                "generation": {
                    "generation_id": "candidate-g1",
                    "parent_generation_id": "incremental_detection_generation_4plus2",
                },
                "recheck": {
                    "accepted": True,
                    "manifest": "reports/recheck.json",
                    "metrics": {"new_map50": 0.8, "krr": 1.0},
                    "gates": {"base_map50": True, "new_map50": True, "krr": True},
                },
            },
        }

    def start(self, _batch_id: str, wait: bool = False):
        assert wait is False
        return {"job_id": "job-1", "status": "QUEUED"}

    def get(self, _batch_id: str, _job_id: str):
        return self.job


class _InterruptedManager(_Manager):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = []

    def get(self, _batch_id: str, _job_id: str):
        raise KeyboardInterrupt

    def cancel(self, batch_id: str, job_id: str):
        self.cancelled.append((batch_id, job_id))
        return {"status": "CANCELLING"}


class _Engine:
    def __init__(self) -> None:
        self.closed = False

    def predict_batch(self, rows, _confidence, _protocol):
        return [{"detection_count": 1} for _row in rows]

    def close(self):
        self.closed = True


class _RuntimeManager:
    def __init__(self, config: dict, parent_id: str) -> None:
        self.config = config
        self.generation_id = parent_id
        self.promotions = []
        self.rollbacks = []

    def settings(self):
        return {"generation_id": self.generation_id}

    def promote(self, candidate_id, manifest_path, shadow_engine, smoke):
        self.promotions.append(
            (candidate_id, manifest_path, shadow_engine, dict(smoke))
        )
        previous = self.generation_id
        self.generation_id = candidate_id
        return {
            "previous": previous,
            "production": candidate_id,
            "runtime_swap": "atomic",
        }

    def rollback(self, target_id):
        self.rollbacks.append(target_id)
        previous = self.generation_id
        self.generation_id = target_id
        return {
            "previous": previous,
            "production": target_id,
            "runtime_swap": "atomic_rollback",
        }


def test_bundle_contract_is_read_only_and_prefers_classes_yaml(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "onsite.zip")
    before = bundle.stat().st_mtime_ns
    result = inspect_onsite_bundle(bundle)
    assert result["image_count"] == 6
    assert result["label_count"] == 6
    assert result["declared_classes"] == {"0": "onsite_vehicle"}
    assert bundle.stat().st_mtime_ns == before


def test_plan_filters_full_dataset_names_to_classes_present_in_labels(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    bundle = _full_names_bundle(tmp_path / "full-names.zip")
    workflow = OnsiteIncrementalWorkflow(
        config,
        None,
        None,
        None,
        run_root=tmp_path / "runs",
        capability_probe=lambda *_args: {"ready": True, "device_count": 1},
    )
    inspected = inspect_onsite_bundle(bundle)
    result = workflow.plan(bundle, target="x86")
    assert inspected["source_class_ids"] == [6]
    assert inspected["declared_classes"] == {"6": "onsite_vehicle"}
    assert result["declared_new_class_count"] == 1
    assert result["predicted_new_class_ids"] == [6]
    assert result["predicted_final_class_count"] == 7


def test_flat_dataset_and_classes_txt_match_current_competition_format(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    bundle = _flat_classes_txt_bundle(tmp_path / "flat.zip")
    workflow = OnsiteIncrementalWorkflow(
        config,
        None,
        None,
        None,
        run_root=tmp_path / "runs",
        capability_probe=lambda *_args: {"ready": True, "device_count": 1},
    )
    inspected = inspect_onsite_bundle(bundle)
    result = workflow.plan(bundle, target="x86")
    assert inspected["image_count"] == 6
    assert inspected["label_count"] == 6
    assert inspected["declared_classes"] == {"6": "onsite_vehicle"}
    assert result["ready"] is True
    assert result["predicted_new_class_ids"] == [6]


def test_plan_allows_explicitly_requested_provisional_class_names(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    workflow = OnsiteIncrementalWorkflow(
        config,
        None,
        None,
        None,
        run_root=tmp_path / "runs",
        capability_probe=lambda *_args: {"ready": True, "device_count": 1},
    )
    bundle = _bundle(tmp_path / "unnamed.zip", with_names=False)
    blocked = workflow.plan(bundle, target="x86")
    allowed = workflow.plan(
        bundle, target="x86", allow_provisional_names=True
    )
    assert blocked["ready"] is False
    assert allowed["ready"] is True


def test_onsite_plan_requires_cuda_host_and_complete_ascend_spec(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workflow = OnsiteIncrementalWorkflow(
        config,
        _Store(tmp_path / "store"),
        _Manager(),
        StructuredEventLog(tmp_path / "logs", 1_000_000, 2),
        run_root=tmp_path / "runs",
        capability_probe=lambda *_args: {"ready": True, "device_count": 1},
    )
    bundle = _bundle(tmp_path / "onsite.zip")
    x86 = workflow.plan(bundle, target="x86")
    assert x86["ready"] is True
    assert x86["active_class_count_before"] == 6
    assert x86["predicted_new_class_ids"] == [6]
    ascend = workflow.plan(bundle, target="ascend310b")
    assert ascend["ready"] is False
    assert any("deployment spec" in value for value in ascend["blocking_reasons"])


def test_plan_does_not_initialize_mutable_runtime_registry(tmp_path: Path) -> None:
    config = deepcopy(load_config())
    runtime = tmp_path / "runtime" / "generations.json"
    config["generation"]["registry"] = "models/generations.json"
    config["generation"]["runtime_registry"] = str(runtime)
    workflow = OnsiteIncrementalWorkflow(
        config,
        None,
        None,
        None,
        run_root=tmp_path / "runs",
        capability_probe=lambda *_args: {"ready": True, "device_count": 1},
    )
    result = workflow.plan(_bundle(tmp_path / "onsite.zip"), target="x86")
    assert result["ready"] is True
    assert not runtime.exists()
    assert not (tmp_path / "runs").exists()


def test_onsite_candidate_is_promoted_only_after_lock_and_fps_gates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = _Store(tmp_path / "store")
    engine = _Engine()
    promoted = []
    frozen = tmp_path / "accepted-lineage.json"
    frozen.write_text("{}", encoding="utf-8")
    workflow = OnsiteIncrementalWorkflow(
        config,
        store,
        _Manager(),
        StructuredEventLog(tmp_path / "logs", 1_000_000, 2),
        run_root=tmp_path / "runs",
        capability_probe=lambda *_args: {"ready": True, "device_count": 1},
        shadow_loader=lambda *_args: (engine, {"smoke_images": 1}),
        promoter=lambda _config, candidate, manifest: promoted.append(
            (candidate, manifest)
        )
        or {"production": candidate},
        rollback=lambda *_args: {"production": "parent"},
        lineage_freezer=lambda *_args: frozen,
    )
    result = workflow.run(
        _bundle(tmp_path / "onsite.zip"),
        target="x86",
        fps_probe_size=2,
        fps_warmup_rounds=0,
        fps_rounds=1,
    )
    assert result["status"] == "PROMOTED"
    assert result["final_class_count"] == 7
    assert promoted == [("candidate-g1", "reports/recheck.json")]
    assert engine.closed is True
    assert store.updated[-1][0] == "PROMOTED"
    statuses = [row["status"] for row in result["history"]]
    assert statuses.index("CANDIDATE_ACCEPTED") < statuses.index("FPS_GATE_PASSED")
    assert statuses.index("FPS_GATE_PASSED") < statuses.index("PROMOTED")


def test_keyboard_interrupt_cancels_training_and_keeps_parent_production(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manager = _InterruptedManager()
    promoted = []
    workflow = OnsiteIncrementalWorkflow(
        config,
        _Store(tmp_path / "store"),
        manager,
        StructuredEventLog(tmp_path / "logs", 1_000_000, 2),
        run_root=tmp_path / "runs",
        capability_probe=lambda *_args: {"ready": True, "device_count": 1},
        promoter=lambda *_args: promoted.append(True) or {},
    )
    result = workflow.run(_bundle(tmp_path / "onsite.zip"), target="x86")
    assert result["status"] == "CANCELLED"
    assert manager.cancelled == [("batch-1", "job-1")]
    assert result["rollback"]["training"]["status"] == "CANCELLING"
    assert promoted == []


def test_ascend_executor_requires_accuracy_and_complete_fps_reports(tmp_path: Path) -> None:
    accuracy = tmp_path / "score.json"
    fps = tmp_path / "benchmark.json"
    accuracy.write_text(json.dumps({"score_passed": True}), encoding="utf-8")
    fps.write_text(
        json.dumps({"competition": {"batch_fps": 31.2, "batch_fps_passed": True}}),
        encoding="utf-8",
    )
    stages = []
    for stage_id in ("export", "candidate_deploy", "accuracy_gate", "fps_gate", "promote"):
        row = {"id": stage_id, "command": ["tool", stage_id]}
        if stage_id == "accuracy_gate":
            row["report"] = str(accuracy)
            row["require"] = {"score_passed": True}
        if stage_id == "fps_gate":
            row["report"] = str(fps)
            row["require"] = {"competition.batch_fps_passed": True}
        stages.append(row)
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    executor = AscendDeploymentExecutor(
        {
            "schema_version": 1,
            "target": "ascend310b",
            "stages": stages,
            "rollback": {"command": ["tool", "rollback"]},
        },
        tmp_path / "run",
        {"target_fps": 30.0},
        runner=runner,
    )
    result = executor.execute()
    assert result["status"] == "PROMOTED"
    assert [row[1] for row in calls] == [
        "export",
        "candidate_deploy",
        "accuracy_gate",
        "fps_gate",
        "promote",
    ]
    rollback = executor.rollback()
    assert rollback["id"] == "rollback"


def test_loopback_runtime_endpoint_promotes_the_loaded_service_atomically(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    parent_id = "incremental_detection_generation_4plus2"
    runtime = _RuntimeManager(config, parent_id)
    shadow = _Engine()
    monkeypatch.setattr(
        "fair_agent.modules.generation_management.shadow_load_generation",
        lambda _config, candidate: (
            shadow,
            {"generation_id": candidate, "smoke_images": 1},
        ),
    )
    app = create_app(engine_provider=lambda: _Engine(), config=config)
    app.state.runtime_manager = runtime
    with TestClient(app) as client:
        rejected = client.post(
            "/api/runtime/generation",
            json={"action": "rollback", "target_id": parent_id},
        )
        response = client.post(
            "/api/runtime/generation",
            headers={"X-Agile-Agent-Control": RUNTIME_CONTROL_CAPABILITY},
            json={
                "action": "promote",
                "candidate_id": "candidate-g1",
                "manifest_path": "reports/recheck.json",
                "expected_parent_id": parent_id,
            },
        )
    assert rejected.status_code == 403
    assert response.status_code == 200
    assert response.json()["result"]["runtime_swap"] == "atomic"
    assert runtime.generation_id == "candidate-g1"
    assert runtime.promotions[0][2] is shadow


def test_loopback_controller_marks_registry_only_activation_when_service_is_offline(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    config["runtime"]["server_port"] = 65530
    monkeypatch.setattr(
        "fair_agent.modules.onsite_incremental.promote_generation",
        lambda _config, candidate, _manifest: {"production": candidate},
    )
    controller = LoopbackGenerationController(config, probe_timeout=0.01)
    result = controller.promote(config, "candidate-g1", "reports/recheck.json")
    assert result["production"] == "candidate-g1"
    assert result["runtime_swap"] == "next_process_start"
    assert result["running_service_detected"] is False


def test_cli_exposes_single_onsite_command() -> None:
    args = build_parser().parse_args(
        [
            "incremental",
            "onsite",
            "--bundle",
            "new.zip",
            "--class-names",
            "new_vehicle",
            "--plan-only",
        ]
    )
    assert args.incremental_action == "onsite"
    assert args.func.__name__ == "cmd_onsite_incremental"
