from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


NEW_MAP50_THRESHOLD = 0.60
KRR_THRESHOLD = 0.95


def normalized_stems(paths: Iterable[str | Path]) -> list[str]:
    stems = [Path(value).stem for value in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate image stems are not allowed in an incremental split")
    return sorted(stems)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_new_images_only(
    training_images: Sequence[str | Path],
    allowed_new_images: Sequence[str | Path],
    verify_content: bool = False,
) -> Dict[str, Any]:
    training = normalized_stems(training_images)
    allowed = normalized_stems(allowed_new_images)
    training_set = set(training)
    allowed_set = set(allowed)
    unexpected = sorted(training_set - allowed_set)
    missing = sorted(allowed_set - training_set)
    content_mismatches: list[str] = []
    content_check_errors: list[str] = []
    if verify_content:
        training_by_stem = {Path(value).stem: Path(value) for value in training_images}
        allowed_by_stem = {Path(value).stem: Path(value) for value in allowed_new_images}
        for stem in sorted(training_set & allowed_set):
            try:
                if _file_sha256(training_by_stem[stem]) != _file_sha256(allowed_by_stem[stem]):
                    content_mismatches.append(stem)
            except OSError:
                content_check_errors.append(stem)
    return {
        "compliant": not unexpected and not missing and not content_mismatches and not content_check_errors,
        "training_image_count": len(training),
        "allowed_new_image_count": len(allowed),
        "unexpected_stems": unexpected,
        "missing_stems": missing,
        "content_check_performed": verify_content,
        "content_mismatches": content_mismatches,
        "content_check_errors": content_check_errors,
        "old_raw_image_count": len(unexpected) + len(content_mismatches) + len(content_check_errors),
    }


def verify_class_incremental_learning_scope(
    training_images: Sequence[str | Path],
    validation_images: Sequence[str | Path],
    allowed_training_images: Sequence[str | Path],
    allowed_validation_images: Sequence[str | Path],
    base_classes: Sequence[str],
    new_classes: Sequence[str],
    verify_content: bool = False,
) -> Dict[str, Any]:
    train = verify_new_images_only(training_images, allowed_training_images, verify_content=verify_content)
    val = verify_new_images_only(validation_images, allowed_validation_images, verify_content=verify_content)
    train_stems = set(normalized_stems(training_images))
    val_stems = set(normalized_stems(validation_images))
    overlap = sorted(train_stems & val_stems)
    base = [str(value) for value in base_classes]
    new = [str(value) for value in new_classes]
    class_partition_valid = bool(base and new) and not (set(base) & set(new)) and len(base) == len(set(base)) and len(new) == len(set(new))
    old_raw_count = int(train["old_raw_image_count"]) + int(val["old_raw_image_count"])
    compliant = bool(train["compliant"] and val["compliant"] and not overlap and class_partition_valid)
    return {
        "task_type": "class_incremental_object_detection",
        "learning_data_scope": "incremental_dataset_only",
        "training": train,
        "validation": val,
        "train_validation_overlap": overlap,
        "base_classes": base,
        "new_classes": new,
        "class_partition_valid": class_partition_valid,
        "old_raw_image_count": old_raw_count,
        "learning_scope_verified": compliant,
        "compliant": compliant,
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
