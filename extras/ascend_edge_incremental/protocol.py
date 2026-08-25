"""Registry-driven protocol and filesystem guards for edge adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCOPES = ("training", "selection", "lock", "all")


@dataclass(frozen=True)
class RoundSpec:
    """One registered single-class incremental round."""

    round_id: str
    round_index: int
    generation_id: str
    parent_generation_id: str
    class_id: int
    class_name: str
    train_split: str
    dev_split: str
    lock_split: str


@dataclass(frozen=True)
class EdgeProtocol:
    """Validated subset of the project incremental-round registry."""

    protocol_id: str
    registry_path: Path
    repo_root: Path
    base_class_ids: tuple[int, ...]
    class_names: Mapping[int, str]
    base_splits: Mapping[str, str]
    rounds: tuple[RoundSpec, ...]

    @property
    def new_class_ids(self) -> tuple[int, ...]:
        return tuple(item.class_id for item in self.rounds)

    @property
    def all_class_ids(self) -> tuple[int, ...]:
        return (*self.base_class_ids, *self.new_class_ids)

    def split_manifests(self, scope: str) -> tuple[Path, ...]:
        """Return the registered manifests opened by one workflow scope."""

        if scope not in SCOPES:
            raise ValueError(f"unknown edge incremental scope: {scope}")
        values: list[str] = []
        if scope == "training":
            for item in self.rounds:
                values.extend((item.train_split, item.dev_split))
        elif scope == "selection":
            values.append(self.base_splits["dev"])
            values.extend(item.dev_split for item in self.rounds)
        elif scope == "lock":
            values.append(self.base_splits["lock"])
            values.extend(item.lock_split for item in self.rounds)
        else:
            values.extend(self.base_splits[name] for name in ("train", "dev", "lock"))
            for item in self.rounds:
                values.extend((item.train_split, item.dev_split, item.lock_split))
        return tuple(_resolve_repo_file(self.repo_root, value) for value in values)

    def base_manifests(self, scope: str) -> tuple[Path, ...]:
        if scope == "selection":
            names = ("dev",)
        elif scope == "lock":
            names = ("lock",)
        elif scope == "all":
            names = ("train", "dev", "lock")
        else:
            raise ValueError(f"scope has no Base evaluation subset: {scope}")
        return tuple(
            _resolve_repo_file(self.repo_root, self.base_splits[name])
            for name in names
        )

    def image_paths(self, scope: str) -> tuple[Path, ...]:
        return read_split_manifests(self.repo_root, self.split_manifests(scope))

    def base_image_ids(self, scope: str) -> set[str]:
        paths = read_split_manifests(self.repo_root, self.base_manifests(scope))
        return {path.stem for path in paths}


def _resolve_repo_file(repo_root: Path, value: str | Path) -> Path:
    raw = Path(value)
    resolved = raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"registered path escapes repository: {value}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _single_class(value: Any, round_id: str) -> int:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{round_id}.new_class_ids must be a sequence")
    class_ids = tuple(int(item) for item in value)
    if len(class_ids) != 1:
        raise ValueError(
            f"{round_id} must register exactly one class for the 8-parameter adapter"
        )
    return class_ids[0]


def load_protocol(registry_path: Path, repo_root: Path) -> EdgeProtocol:
    """Load and strictly validate the edge-training contract from YAML."""

    root = repo_root.expanduser().resolve()
    registry = _resolve_repo_file(root, registry_path)
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    body = _mapping(payload, "registry")
    if int(body.get("schema_version", 0)) != 1:
        raise ValueError("unsupported incremental registry schema")

    raw_classes = _mapping(body.get("classes"), "classes")
    class_names: dict[int, str] = {}
    introduced: dict[int, str] = {}
    for raw_id, raw_entry in raw_classes.items():
        class_id = int(raw_id)
        entry = _mapping(raw_entry, f"classes.{class_id}")
        name = str(entry.get("name") or "").strip()
        phase = str(entry.get("introduced_in") or "").strip()
        if not name or not phase or class_id in class_names:
            raise ValueError(f"invalid class registry entry: {raw_id}")
        class_names[class_id] = name
        introduced[class_id] = phase

    base = _mapping(body.get("base"), "base")
    base_ids = tuple(int(value) for value in base.get("class_ids") or ())
    if not base_ids or len(base_ids) != len(set(base_ids)):
        raise ValueError("base.class_ids must contain unique class IDs")
    if any(introduced.get(class_id) != "base" for class_id in base_ids):
        raise ValueError("base classes do not match classes.introduced_in")
    base_splits = {
        str(key): str(value)
        for key, value in _mapping(base.get("splits"), "base.splits").items()
    }
    if not {"train", "dev", "lock"}.issubset(base_splits):
        raise ValueError("base splits must define train/dev/lock")

    raw_rounds = body.get("rounds")
    if not isinstance(raw_rounds, Sequence) or isinstance(raw_rounds, (str, bytes)):
        raise ValueError("rounds must be a sequence")
    rounds: list[RoundSpec] = []
    learned = list(base_ids)
    expected_parent = str(base.get("generation_id") or "")
    for expected_index, raw_round in enumerate(raw_rounds, 1):
        row = _mapping(raw_round, f"rounds[{expected_index - 1}]")
        round_id = str(row.get("round_id") or "").strip()
        round_index = int(row.get("round_index", 0))
        generation_id = str(row.get("generation_id") or "").strip()
        parent_id = str(row.get("parent_generation_id") or "").strip()
        class_id = _single_class(row.get("new_class_ids"), round_id)
        if not round_id or round_index != expected_index:
            raise ValueError("incremental rounds must have contiguous ordered indexes")
        if class_id in learned or introduced.get(class_id) != round_id:
            raise ValueError(f"{round_id} does not introduce a unique registered class")
        if parent_id != expected_parent or not generation_id:
            raise ValueError(f"{round_id} has an invalid parent generation")
        if str(row.get("learning_data_scope")) != "incremental_dataset_only":
            raise ValueError(f"{round_id} is not Increment-only")
        if str(row.get("validation_data_scope")) != "incremental_dataset_only":
            raise ValueError(f"{round_id} dev scope is not Increment-only")
        if row.get("base_detector_weights_frozen") is not True:
            raise ValueError(f"{round_id} does not freeze Base weights")
        if row.get("old_expert_weights_frozen") is not True:
            raise ValueError(f"{round_id} does not freeze old expert weights")
        if int(row.get("old_raw_image_count", -1)) != 0:
            raise ValueError(f"{round_id} replays historical raw images")
        if str(row.get("label_projection")) != "current_round_classes_only":
            raise ValueError(f"{round_id} does not project current-round labels only")
        learned.append(class_id)
        if tuple(int(value) for value in row.get("learned_class_ids") or ()) != tuple(learned):
            raise ValueError(f"{round_id}.learned_class_ids breaks lineage")
        specialist = _mapping(row.get("specialist"), f"{round_id}.specialist")
        mapping = {
            int(local): int(global_id)
            for local, global_id in _mapping(
                specialist.get("local_to_global"),
                f"{round_id}.specialist.local_to_global",
            ).items()
        }
        if set(mapping.values()) != {class_id}:
            raise ValueError(f"{round_id} specialist owner does not match new class")
        splits = _mapping(row.get("splits"), f"{round_id}.splits")
        if not {"train", "dev", "lock"}.issubset(splits):
            raise ValueError(f"{round_id} splits must define train/dev/lock")
        rounds.append(
            RoundSpec(
                round_id=round_id,
                round_index=round_index,
                generation_id=generation_id,
                parent_generation_id=parent_id,
                class_id=class_id,
                class_name=class_names[class_id],
                train_split=str(splits["train"]),
                dev_split=str(splits["dev"]),
                lock_split=str(splits["lock"]),
            )
        )
        expected_parent = generation_id
    if not rounds:
        raise ValueError("edge adaptation requires at least one incremental round")

    protocol = EdgeProtocol(
        protocol_id=str(body.get("protocol_id") or ""),
        registry_path=registry,
        repo_root=root,
        base_class_ids=base_ids,
        class_names=class_names,
        base_splits=base_splits,
        rounds=tuple(rounds),
    )
    if not protocol.protocol_id:
        raise ValueError("registry protocol_id is required")
    return protocol


def read_split_manifests(repo_root: Path, manifests: Sequence[Path]) -> tuple[Path, ...]:
    """Read manifests and deduplicate identical image entries in stable order."""

    images: dict[str, Path] = {}
    for manifest in manifests:
        for line_number, raw in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(), 1
        ):
            value = raw.strip()
            if not value:
                continue
            raw_path = Path(value)
            image = (
                raw_path.resolve()
                if raw_path.is_absolute()
                else (repo_root / raw_path).resolve()
            )
            if not image.is_file() or image.suffix.lower() != ".png":
                raise FileNotFoundError(f"{manifest}:{line_number}: {image}")
            previous = images.get(image.stem)
            if previous is not None:
                raise ValueError(
                    f"duplicate image stem in registered scope: {image.stem} "
                    f"({previous}, {image})"
                )
            images[image.stem] = image
    if not images:
        raise ValueError("registered split scope contains no images")
    return tuple(images.values())


def ensure_isolated_output(output_root: Path, repo_root: Path) -> Path:
    """Refuse overwrite and any output inside versioned production assets."""

    output = output_root.expanduser().resolve()
    root = repo_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite edge output: {output}")
    protected = (
        root / "models",
        root / "configs",
        root / "splits",
        root / ".git",
    )
    for path in protected:
        try:
            output.relative_to(path.resolve())
        except ValueError:
            continue
        raise ValueError(f"edge output may not modify protected project assets: {output}")
    return output
