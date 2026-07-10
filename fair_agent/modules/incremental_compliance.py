from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


NEW_MAP50_THRESHOLD = 0.60
KRR_THRESHOLD = 0.95


def normalized_stems(paths: Iterable[str | Path]) -> list[str]:
    stems = [Path(value).stem for value in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate image stems are not allowed in an incremental split")
    return sorted(stems)


def verify_new_images_only(training_images: Sequence[str | Path], allowed_new_images: Sequence[str | Path]) -> Dict[str, Any]:
    training = normalized_stems(training_images)
    allowed = normalized_stems(allowed_new_images)
    training_set = set(training)
    allowed_set = set(allowed)
    unexpected = sorted(training_set - allowed_set)
    missing = sorted(allowed_set - training_set)
    return {
        "compliant": not unexpected and not missing,
        "training_image_count": len(training),
        "allowed_new_image_count": len(allowed),
        "unexpected_stems": unexpected,
        "missing_stems": missing,
        "old_raw_image_count": len(unexpected),
    }


def evaluate_incremental_metrics(
    new_map50: float,
    krr: float,
    compliance: Mapping[str, Any] | bool,
    new_threshold: float = NEW_MAP50_THRESHOLD,
    krr_threshold: float = KRR_THRESHOLD,
) -> Dict[str, Any]:
    compliant = bool(compliance if isinstance(compliance, bool) else compliance.get("compliant", False))
    new_pass = float(new_map50) >= float(new_threshold)
    krr_pass = float(krr) >= float(krr_threshold)
    return {
        "compliant": compliant,
        "new_map50_pass": new_pass,
        "krr_pass": krr_pass,
        "passed": compliant and new_pass and krr_pass,
        "new_map50_threshold": float(new_threshold),
        "krr_threshold": float(krr_threshold),
    }
