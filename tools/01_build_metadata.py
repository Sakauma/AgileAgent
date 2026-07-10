#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fair_agent.dataset_utils import REPORTS_DIR, scan_dataset, write_metadata


def main() -> int:
    summary = scan_dataset()
    if summary["errors"]:
        print("Dataset validation errors exist; metadata was not written.")
        for error in summary["errors"][:20]:
            print(error)
        return 1

    output_path = REPORTS_DIR / "metadata.csv"
    write_metadata(summary["rows"], output_path)
    print(f"metadata_rows={len(summary['rows'])} output={output_path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
