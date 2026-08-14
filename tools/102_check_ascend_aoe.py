#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_snapshot(command: list[str]) -> Dict[str, Any]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="检查目标AOE能否保持P0的precision_mode_v2契约，不执行调优。"
    )
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"AOE兼容报告已存在，拒绝覆盖：{args.output}")
    manifest_path = args.build_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("soc_version") != "Ascend310B1":
        raise RuntimeError("构建清单SoC不是Ascend310B1。")
    if manifest.get("precision") != "mixed_float16":
        raise RuntimeError("构建清单precision不是mixed_float16。")
    aoe_value = shutil.which("aoe")
    if aoe_value is None:
        raise RuntimeError("未找到aoe。")
    aoe = Path(aoe_value).resolve()
    help_snapshot = command_snapshot([str(aoe), "-h"])
    precision_probe = command_snapshot(
        [str(aoe), "--precision_mode_v2=mixed_float16", "-h"]
    )
    help_text = help_snapshot["stdout"] + "\n" + help_snapshot["stderr"]
    probe_text = precision_probe["stdout"] + "\n" + precision_probe["stderr"]
    precision_v2_supported = not (
        precision_probe["returncode"] != 0
        or "not supported" in probe_text
        or "unrecognized option" in probe_text
    )

    artifacts = {}
    for name, row in manifest.get("artifacts", {}).items():
        onnx = Path(row["onnx"]["path"])
        aipp = Path(row["aipp"]["path"])
        for path in (onnx, aipp):
            if not path.is_file():
                raise FileNotFoundError(path)
        artifacts[str(name)] = {
            "onnx": {"path": str(onnx), "sha256": sha256(onnx)},
            "aipp": {"path": str(aipp), "sha256": sha256(aipp)},
            "input_contract": row["input_contract"],
        }

    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "requested_contract": {
            "soc_version": manifest["soc_version"],
            "cann_version": manifest["cann_version"],
            "precision_mode_v2": manifest["precision"],
            "framework": 5,
            "artifacts": artifacts,
        },
        "aoe": {
            "path": str(aoe),
            "sha256": sha256(aoe),
            "supports_insert_op_conf": "--insert_op_conf" in help_text,
            "supports_job_type": "--job_type" in help_text,
            "supports_precision_mode_v2_mixed_float16": precision_v2_supported,
            "help": help_snapshot,
            "precision_mode_v2_probe": precision_probe,
        },
        "decision": {
            "tuning_attempted": False,
            "candidate_om_created": False,
            "allowed": precision_v2_supported,
            "reason": (
                "目标AOE支持与P0相同的precision_mode_v2契约。"
                if precision_v2_supported
                else "目标AOE明确拒绝precision_mode_v2=mixed_float16；禁止用allow_mix_precision近似替代。"
            ),
        },
        "inputs": {
            "build_manifest": str(manifest_path),
            "build_manifest_sha256": sha256(manifest_path),
        },
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
