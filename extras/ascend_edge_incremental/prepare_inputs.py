#!/usr/bin/env python3
"""Materialize a label-free image view from registry-owned split manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .protocol import SCOPES, load_protocol


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按增量轮次注册表创建不可覆盖的板端 probe 图像视图。"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    protocol = load_protocol(args.registry, repo_root)
    images = protocol.image_paths(args.scope)
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    if output.exists() or manifest.exists():
        raise FileExistsError("probe view or manifest already exists; refusing overwrite")
    output.mkdir(parents=True)
    for image in images:
        os.symlink(image, output / image.name)
    payload = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "scope": args.scope,
        "labels_read": False,
        "registry": str(protocol.registry_path),
        "split_manifests": [
            str(path) for path in protocol.split_manifests(args.scope)
        ],
        "image_count": len(images),
        "output": str(output),
        "symlink_only": True,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
