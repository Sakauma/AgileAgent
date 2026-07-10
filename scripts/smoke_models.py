#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.hashes import verify_sha256s


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and load released YOLO weights on x86 CPU.")
    parser.add_argument("--load-only", action="store_true", help="Load each model without synthetic inference.")
    args = parser.parse_args()

    checksum_path = ROOT / "models" / "SHA256SUMS.txt"
    checksums = verify_sha256s(checksum_path)
    if not checksums["valid"]:
        raise SystemExit(f"Model checksum verification failed: {checksums['errors']}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is missing. Install with: pip install -e '.[inference]'") from exc

    model_paths = sorted((ROOT / "models").rglob("*.pt"))
    image = Image.new("RGB", (64, 64), color=(0, 0, 0))
    results = []
    for path in model_paths:
        started = time.perf_counter()
        model = YOLO(str(path))
        prediction_count = None
        if not args.load_only:
            prediction = model.predict(source=image, imgsz=64, device="cpu", verbose=False)
            prediction_count = len(prediction)
            if prediction_count != 1:
                raise RuntimeError(f"Unexpected result count for {path}: {prediction_count}")
        results.append({
            "model": path.relative_to(ROOT).as_posix(),
            "loaded": True,
            "synthetic_results": prediction_count,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        })

    print(json.dumps({"checksums": checksums, "models": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
