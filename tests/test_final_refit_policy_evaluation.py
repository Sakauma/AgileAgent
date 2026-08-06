from __future__ import annotations

import inspect
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = runpy.run_path(str(ROOT / "tools" / "87_evaluate_final_refit_policy.py"))


def test_checkpoint_predictions_are_frozen_before_labels_open() -> None:
    source = inspect.getsource(MODULE["main"])

    freeze = source.index('report_path.parent / "freeze_manifest.json"')
    label_open = source.index("targets = yolo_ground_truth(images)")

    assert freeze < label_open
