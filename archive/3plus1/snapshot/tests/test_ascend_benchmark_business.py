from __future__ import annotations

import importlib.util
import json
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools/106_compare_ascend_benchmark_business.py"


def _module():
    spec = importlib.util.spec_from_file_location("compare_ascend_business", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_business_comparison_accepts_more_reference_rounds() -> None:
    module = _module()
    report = {
        "requests": [
            {
                "image": "a.png",
                "business_sha256": "a" * 64,
                "detection_count": 2,
            },
            {
                "image": "a.png",
                "business_sha256": "a" * 64,
                "detection_count": 2,
            },
        ]
    }

    rows = module._business_by_image(report, label="reference")

    assert rows == {
        "a.png": {
            "business_sha256": "a" * 64,
            "detection_count": 2,
            "samples": 2,
        }
    }


def test_business_report_loader_rejects_unstable_reference(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "image": "a.png",
                        "business_sha256": "a" * 64,
                        "detection_count": 2,
                    },
                    {
                        "image": "a.png",
                        "business_sha256": "b" * 64,
                        "detection_count": 2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = module._load(path)
    try:
        module._business_by_image(loaded, label="reference")
    except RuntimeError as exc:
        assert "跨轮不稳定" in str(exc)
    else:
        raise AssertionError("unstable business hashes were accepted")
