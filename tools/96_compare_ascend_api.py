#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.hashes import sha256_file
from fair_agent.modules.ascend_alignment import compare_api_records, read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按类别与IoU稳定匹配，比较两份Ascend API 89图JSONL。"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-box-abs", type=float, default=1.0)
    parser.add_argument("--max-confidence-abs", type=float, default=0.02)
    args = parser.parse_args()

    report = compare_api_records(
        read_jsonl(args.reference),
        read_jsonl(args.candidate),
        max_box_abs=args.max_box_abs,
        max_confidence_abs=args.max_confidence_abs,
    )
    report["inputs"] = {
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256_file(args.reference),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256_file(args.candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"对齐报告已存在，拒绝覆盖：{args.output}")
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
