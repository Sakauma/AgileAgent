#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.hashes import sha256_file  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark报告不是JSON对象：{path}")
    requests = payload.get("requests")
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes)):
        raise ValueError(f"benchmark报告缺少requests：{path}")
    return payload


def _business_by_image(
    report: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in report["requests"]:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label}请求记录非法")
        image = str(row.get("image") or "")
        digest = str(row.get("business_sha256") or "")
        if not image or len(digest) != 64:
            raise ValueError(f"{label}请求缺少image/business_sha256")
        grouped.setdefault(image, []).append(row)
    result = {}
    for image, rows in grouped.items():
        hashes = {str(row["business_sha256"]) for row in rows}
        detection_counts = {int(row["detection_count"]) for row in rows}
        if len(hashes) != 1 or len(detection_counts) != 1:
            raise RuntimeError(f"{label}同图业务结果跨轮不稳定：{image}")
        result[image] = {
            "business_sha256": next(iter(hashes)),
            "detection_count": next(iter(detection_counts)),
            "samples": len(rows),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按89图比较Ascend benchmark排除耗时字段后的业务JSON SHA256。"
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=89)
    args = parser.parse_args()
    reference_path = args.reference.resolve()
    candidate_path = args.candidate.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    reference = _business_by_image(_load(reference_path), label="reference")
    candidate = _business_by_image(_load(candidate_path), label="candidate")
    all_images = sorted(set(reference) | set(candidate))
    mismatches = [
        {
            "image": image,
            "reference": reference.get(image),
            "candidate": candidate.get(image),
        }
        for image in all_images
        if reference.get(image) != candidate.get(image)
        and (
            image not in reference
            or image not in candidate
            or reference[image]["business_sha256"]
            != candidate[image]["business_sha256"]
            or reference[image]["detection_count"]
            != candidate[image]["detection_count"]
        )
    ]
    gates = {
        "reference_image_count": len(reference) == int(args.expected_images),
        "candidate_image_count": len(candidate) == int(args.expected_images),
        "image_sets_equal": set(reference) == set(candidate),
        "business_json_zero_difference": len(mismatches) == 0,
    }
    report = {
        "schema_version": 1,
        "kind": "ascend_benchmark_business_comparison",
        "inputs": {
            "reference": str(reference_path),
            "reference_sha256": sha256_file(reference_path),
            "candidate": str(candidate_path),
            "candidate_sha256": sha256_file(candidate_path),
        },
        "reference_samples": sum(row["samples"] for row in reference.values()),
        "candidate_samples": sum(row["samples"] for row in candidate.values()),
        "compared_images": len(set(reference) & set(candidate)),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
