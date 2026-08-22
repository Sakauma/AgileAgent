#!/usr/bin/env python3
"""Materialize immutable per-round lists before any lock prediction is run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.incremental_round_registry import (  # noqa: E402
    DEFAULT_ROUND_REGISTRY,
    load_incremental_round_registry,
)


def resolve_reference(data_root: Path, reference: str) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else data_root / path


def source_label(image: Path) -> Path:
    label = image.with_suffix(".txt")
    if not label.is_file():
        raise FileNotFoundError(f"增量原始标签不存在：{label}")
    return label


def label_class_ids(image: Path) -> set[int]:
    class_ids: set[int] = set()
    for line_number, raw in enumerate(
        source_label(image).read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw.strip()
        if not value:
            continue
        parts = value.split()
        if len(parts) != 5:
            raise ValueError(
                f"{source_label(image)}:{line_number} 不是五列 YOLO 标签"
            )
        class_ids.add(int(parts[0]))
    return class_ids


def read_source_list(data_root: Path, reference: str) -> list[str]:
    path = resolve_reference(data_root, reference)
    if not path.is_file():
        raise FileNotFoundError(f"增量总清单不存在：{path}")
    rows = [raw.strip() for raw in path.read_text(encoding="utf-8").splitlines()]
    rows = [value for value in rows if value]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"增量总清单为空或包含重复项：{path}")
    return rows


def write_immutable(path: Path, rows: list[str]) -> None:
    content = "\n".join(rows) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"拒绝覆盖内容不同的轮次清单：{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def materialize(
    data_root: Path, registry: Mapping[str, Any]
) -> dict[str, dict[str, int]]:
    rounds = list(registry["rounds"])
    output: dict[str, dict[str, int]] = {str(row["round_id"]): {} for row in rounds}
    for split_role in ("train", "dev", "lock"):
        source_references = {
            str(row["source_splits"][split_role]) for row in rounds
        }
        if len(source_references) != 1:
            raise ValueError(f"{split_role} 必须共享一个待分轮的 Increment 总清单")
        source_rows = read_source_list(data_root, source_references.pop())
        selected = {str(row["round_id"]): [] for row in rounds}
        for reference in source_rows:
            image = resolve_reference(data_root, reference).resolve()
            if not image.is_file():
                raise FileNotFoundError(f"增量图像不存在：{image}")
            present = label_class_ids(image)
            owners = [
                str(row["round_id"])
                for row in rounds
                if present & set(row["new_class_ids"])
            ]
            if len(owners) != 1:
                raise ValueError(
                    f"{reference} 必须只属于一个新增类别轮次，实际：{owners}"
                )
            selected[owners[0]].append(reference)
        if sum(len(rows) for rows in selected.values()) != len(source_rows):
            raise RuntimeError(f"{split_role} 分轮后未完整覆盖 Increment 总清单")
        for round_spec in rounds:
            round_id = str(round_spec["round_id"])
            rows = selected[round_id]
            if not rows:
                raise ValueError(f"{round_id} 的 {split_role} 清单为空")
            target = resolve_reference(
                data_root, str(round_spec["splits"][split_role])
            )
            write_immutable(target, rows)
            output[round_id][split_role] = len(rows)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="依据类别注册表预生成逐轮 Increment train/dev/lock 清单。"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--round-registry", type=Path, default=ROOT / DEFAULT_ROUND_REGISTRY
    )
    args = parser.parse_args()
    registry = load_incremental_round_registry(args.round_registry)
    counts = materialize(args.data_root.expanduser().resolve(), registry)
    for round_id, roles in counts.items():
        print(
            f"{round_id}: train={roles['train']} dev={roles['dev']} "
            f"lock={roles['lock']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
