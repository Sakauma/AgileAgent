from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_all_images_4plus2",
    ROOT / "tools/14_evaluate_all_images_4plus2.py",
)
assert SPEC is not None and SPEC.loader is not None
evaluate_all_images = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_all_images)


class FakeTensor:
    def __init__(self, values: list) -> None:
        self.values = values

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list:
        return self.values


class FakeBoxes:
    def __init__(self, local_id: int) -> None:
        self.xyxy = FakeTensor([[1.0, 2.0, 3.0, 4.0]])
        self.conf = FakeTensor([0.75])
        self.cls = FakeTensor([local_id])

    def __len__(self) -> int:
        return 1


def test_chunked_prediction_binds_results_to_explicit_input_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class FakeModel:
        def __init__(self, _weight: str) -> None:
            pass

        def predict(self, *, source: list[str], **_kwargs: object) -> list:
            calls.append(source)
            return [
                SimpleNamespace(
                    path=f"image{index}.jpg",
                    boxes=FakeBoxes(0),
                    speed={"inference": 2.0},
                )
                for index, _path in enumerate(source)
            ]

    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=FakeModel),
    )
    images = [tmp_path / f"source_{index}.png" for index in range(5)]

    records, speed = evaluate_all_images.predict_records_chunked(
        tmp_path / "model.pt",
        images,
        {0: 5},
        device="cpu",
        imgsz=1280,
        batch=2,
        confidence=0.01,
        source_name="incremental_model",
    )

    assert [len(chunk) for chunk in calls] == [2, 2, 1]
    assert [row["image_id"] for row in records] == [
        "source_0",
        "source_1",
        "source_2",
        "source_3",
        "source_4",
    ]
    assert {row["class_id"] for row in records} == {5}
    assert speed == {"inference": pytest.approx(2.0)}


def test_lock_reproduction_delta_keeps_official_result_separate() -> None:
    official = {
        "base_map50": 0.84,
        "new_map50": 0.75,
        "krr": 0.99,
        "full_map50": 0.800605,
        "overall": {"tp": 335, "fp": 108},
    }
    reproduction = {
        "base_map50": 0.83,
        "new_map50": 0.75,
        "krr": 1.00,
        "full_map50": 0.800599,
        "overall": {"tp": 335, "fp": 108},
    }

    delta = evaluate_all_images.lock_reproduction_delta(
        official, reproduction
    )

    assert delta["full_map50"] == pytest.approx(-0.000006)
    assert delta["tp"] == 0
    assert delta["fp"] == 0
