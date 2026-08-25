#!/usr/bin/env python3
"""Create a local OPP view without unreadable custom-vendor files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    if output.exists() or manifest.exists():
        raise FileExistsError("OPP overlay or manifest already exists")
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.mkdir(parents=True)
    linked: list[str] = []
    skipped: list[str] = []
    for entry in sorted(source.iterdir(), key=lambda path: path.name):
        if entry.name == "vendors":
            skipped.append(str(entry))
            continue
        os.symlink(entry, output / entry.name, target_is_directory=entry.is_dir())
        linked.append(str(entry))
    payload = {
        "schema_version": 1,
        "source": str(source),
        "output": str(output),
        "cann_modified": False,
        "linked_entries": linked,
        "skipped_entries": skipped,
        "reason": "exclude board-image root-only custom vendor entries",
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
