from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "42_predict_submission.py"
SPEC = importlib.util.spec_from_file_location("submission_predict", SCRIPT)
assert SPEC and SPEC.loader
submission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submission)


def test_output_must_be_new_child_of_allowed_root(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    with pytest.raises(ValueError):
        submission.prepare_output_dir(root, root)
    with pytest.raises(ValueError):
        submission.prepare_output_dir(tmp_path / "outside", root)
    target = root / "run_001"
    submission.prepare_output_dir(target, root)
    with pytest.raises(FileExistsError):
        submission.prepare_output_dir(target, root)


def test_duplicate_stems_are_rejected(tmp_path: Path) -> None:
    images = [tmp_path / "a" / "same.png", tmp_path / "b" / "same.jpg"]
    with pytest.raises(ValueError, match="Duplicate image stem"):
        submission.validate_unique_stems(images)


def test_prediction_result_count_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "images"
    source.mkdir()
    (source / "one.png").write_bytes(b"not-read-by-fake-model")
    model = tmp_path / "best.pt"
    model.write_bytes(b"weight")
    fake = types.ModuleType("ultralytics")

    class FakeYOLO:
        def __init__(self, _path: str) -> None:
            pass

        def predict(self, **_kwargs):
            return []

    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    config = {
        "model": str(model), "expected_sha256": submission.sha256_file(model),
        "source": {"path": str(source), "image_exts": [".png"]},
        "predict": {"imgsz": 640, "conf": 0.001, "iou": 0.7, "max_det": 300},
        "output": {"root": str(tmp_path / "runs"), "package_name": "test"},
        "names": {0: "soldier"},
    }
    with pytest.raises(RuntimeError, match="result count mismatch"):
        submission.run_prediction(config, None, None, None)
