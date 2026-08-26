"""Materialize the fixed 4->4+2 onsite demo contract from an input directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


IMAGE_SUFFIXES = {".png"}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _label_path(image: Path) -> Path:
    adjacent = image.with_suffix(".txt")
    if adjacent.is_file():
        return adjacent
    sibling = image.parent.parent / "labels" / f"{image.stem}.txt"
    if sibling.is_file():
        return sibling
    if image.parent.parent.name == "images":
        canonical = (
            image.parent.parent.parent
            / "labels"
            / image.parent.name
            / f"{image.stem}.txt"
        )
        if canonical.is_file():
            return canonical
    raise FileNotFoundError(f"missing YOLO label for {image}")


def _label_classes(image: Path) -> set[int]:
    values: set[int] = set()
    label = _label_path(image)
    for line_number, raw in enumerate(
        label.read_text(encoding="utf-8").splitlines(), 1
    ):
        columns = raw.split()
        if not columns:
            continue
        if len(columns) != 5:
            raise ValueError(f"invalid YOLO row: {label}:{line_number}")
        class_id = int(float(columns[0]))
        coordinates = [float(value) for value in columns[1:]]
        if class_id < 0 or any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"invalid YOLO value: {label}:{line_number}")
        values.add(class_id)
    return values


def resolve_dataset_directory(path: Path, expected_name: str) -> Path:
    root = path.expanduser().resolve()
    candidate = root / expected_name
    if candidate.is_dir():
        return candidate
    if root.is_dir():
        return root
    raise FileNotFoundError(root)


def index_images(root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        previous = images.get(path.stem)
        if previous is not None:
            raise ValueError(
                f"duplicate image stem in demo input: {path.stem} ({previous}, {path})"
            )
        _label_path(path)
        images[path.stem] = path.resolve()
    if not images:
        raise ValueError(f"no PNG images found below {root}")
    return images


def _manifest_entries(repo_root: Path, manifest_value: str) -> list[str]:
    manifest = (repo_root / manifest_value).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    values = [
        value.strip()
        for value in manifest.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if not values:
        raise ValueError(f"empty reference split: {manifest}")
    return values


def _resolve_reference_images(
    repo_root: Path,
    values: Sequence[str],
    fallback: Mapping[str, Path] | None,
    label: str,
) -> list[Path]:
    resolved: list[Path] = []
    for value in values:
        original = (repo_root / value).resolve()
        image = original if original.is_file() else (fallback or {}).get(Path(value).stem)
        if image is None or not image.is_file():
            raise FileNotFoundError(
                f"{label} image is unavailable: {value}; preinstall its evaluation data "
                "or pass --base-data"
            )
        _label_path(image)
        resolved.append(image.resolve())
    return resolved


def _write_manifest(path: Path, images: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{image.as_posix()}\n" for image in images), encoding="utf-8"
    )


def materialize_demo_contract(
    *,
    repo_root: Path,
    reference_registry: Path,
    incremental_data: Path,
    contract_root: Path,
    base_data: Path | None = None,
) -> dict[str, Any]:
    """Create run-scoped manifests while keeping training Increment-only."""

    repo = repo_root.expanduser().resolve()
    contract = contract_root.expanduser().resolve()
    try:
        contract.relative_to(repo)
    except ValueError as exc:
        raise ValueError("demo contract root must stay inside the repository") from exc
    if contract.exists():
        raise FileExistsError(contract)
    registry_path = reference_registry.expanduser().resolve()
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry = _mapping(payload, "registry")
    if registry.get("protocol_id") != "strict_4plus2_sequential_class_incremental":
        raise ValueError("the offline demo only accepts the registered 4->4+2 protocol")

    increment_root = resolve_dataset_directory(
        incremental_data, "datasets_r2_inc_train"
    )
    increment_index = index_images(increment_root)
    base_index = None
    if base_data is not None:
        base_root = resolve_dataset_directory(base_data, "datasets_r1_base_train")
        base_index = index_images(base_root)

    split_root = contract / "splits"
    base = _mapping(registry.get("base"), "base")
    base_splits = _mapping(base.get("splits"), "base.splits")
    base_counts: dict[str, int] = {}
    for split_name in ("train", "dev", "lock"):
        values = _manifest_entries(repo, str(base_splits[split_name]))
        images = _resolve_reference_images(
            repo, values, base_index, f"Base {split_name}"
        )
        destination = split_root / f"base_{split_name}.txt"
        _write_manifest(destination, images)
        base_splits[split_name] = destination.relative_to(repo).as_posix()
        base_counts[split_name] = len(images)
    base["splits"] = base_splits
    registry["base"] = base

    round_counts: dict[str, dict[str, int]] = {}
    used_increment_stems: set[str] = set()
    rounds = registry.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 2:
        raise ValueError("the 4->4+2 demo requires exactly two registered rounds")
    for raw_round in rounds:
        round_row = _mapping(raw_round, "round")
        round_id = str(round_row["round_id"])
        class_ids = [int(value) for value in round_row.get("new_class_ids") or ()]
        if len(class_ids) != 1:
            raise ValueError(f"{round_id} must introduce exactly one class")
        class_id = class_ids[0]
        raw_splits = _mapping(round_row.get("splits"), f"{round_id}.splits")
        counts: dict[str, int] = {}
        for split_name in ("train", "dev", "lock"):
            values = _manifest_entries(repo, str(raw_splits[split_name]))
            missing = [value for value in values if Path(value).stem not in increment_index]
            if missing:
                raise FileNotFoundError(
                    f"incremental input is missing {len(missing)} registered images; "
                    f"first={missing[0]}"
                )
            images = [increment_index[Path(value).stem] for value in values]
            wrong_class = [
                image.name for image in images if class_id not in _label_classes(image)
            ]
            if wrong_class:
                raise ValueError(
                    f"{round_id}/{split_name} contains images without class {class_id}: "
                    f"{wrong_class[:3]}"
                )
            destination = split_root / f"{round_id}_{split_name}.txt"
            _write_manifest(destination, images)
            raw_splits[split_name] = destination.relative_to(repo).as_posix()
            counts[split_name] = len(images)
            for image in images:
                if image.stem in used_increment_stems:
                    raise ValueError(
                        f"incremental image crosses rounds or splits: {image.stem}"
                    )
                used_increment_stems.add(image.stem)
        round_row["splits"] = raw_splits
        round_counts[round_id] = counts
        raw_round.clear()
        raw_round.update(round_row)

    registry_output = contract / "incremental_round_registry_4plus2.yaml"
    audit_output = contract / "input_audit.json"
    contract.mkdir(parents=True, exist_ok=True)
    registry_output.write_text(
        yaml.safe_dump(registry, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    audit = {
        "schema_version": 1,
        "mode": "offline_ascend310b_4_to_4plus2_demo",
        "protocol_id": registry["protocol_id"],
        "incremental_input": str(increment_root),
        "incremental_images_discovered": len(increment_index),
        "incremental_images_registered": len(used_increment_stems),
        "base_images_used_for_training": 0,
        "old_raw_image_count": 0,
        "rounds": round_counts,
        "base_evaluation": base_counts,
        "network_required": False,
        "registry": str(registry_output),
        "passed": True,
    }
    audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit
