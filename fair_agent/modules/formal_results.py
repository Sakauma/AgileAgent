from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FORMAL_RESULT_COLUMNS = (
    "class_id",
    "x_center",
    "y_center",
    "width",
    "height",
    "confidence",
)


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def formal_prediction_lines(result: Mapping[str, Any]) -> list[str]:
    """Convert one final fused result to the official six-column text format."""

    image_width = float(result.get("image_width") or 0.0)
    image_height = float(result.get("image_height") or 0.0)
    if image_width <= 0.0 or image_height <= 0.0:
        raise ValueError("正式结果缺少有效图像尺寸。")

    lines: list[str] = []
    for detection in result.get("detections") or ():
        if not isinstance(detection, Mapping):
            raise ValueError("正式结果检测项必须是mapping。")
        class_id = int(detection["class_id"])
        if class_id < 0:
            raise ValueError("正式结果类别编号不能为负数。")
        confidence = float(detection["confidence"])
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("正式结果置信度必须位于[0, 1]。")
        coordinates = [float(value) for value in detection["xyxy"]]
        if len(coordinates) != 4 or not all(
            math.isfinite(value) for value in coordinates
        ):
            raise ValueError("正式结果检测框必须是四个有限xyxy坐标。")
        x1, y1, x2, y2 = coordinates
        if x2 < x1 or y2 < y1:
            raise ValueError("正式结果检测框的xyxy顺序非法。")

        x1 = _bounded(x1, 0.0, image_width)
        x2 = _bounded(x2, 0.0, image_width)
        y1 = _bounded(y1, 0.0, image_height)
        y2 = _bounded(y2, 0.0, image_height)
        center_x = ((x1 + x2) / 2.0) / image_width
        center_y = ((y1 + y2) / 2.0) / image_height
        box_width = (x2 - x1) / image_width
        box_height = (y2 - y1) / image_height
        lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f} {confidence:.6f}"
        )
    return lines


def write_formal_prediction_files(
    output_dir: Path,
    results: Sequence[Mapping[str, Any]],
    filenames: Iterable[str],
) -> list[Path]:
    """Write one same-stem TXT per input, including empty detection files."""

    names = [str(value) for value in filenames]
    if len(results) != len(names):
        raise ValueError("正式结果数量与输入文件数量不一致。")
    stems: list[str] = []
    for name in names:
        leaf = Path(name).name
        if leaf != name or not Path(leaf).stem:
            raise ValueError(f"正式结果输入文件名非法：{name}")
        stems.append(Path(leaf).stem)
    if len(set(stems)) != len(stems):
        raise ValueError("正式结果输入文件stem重复。")

    output_dir.mkdir(parents=True, exist_ok=False)
    written: list[Path] = []
    for stem, result in zip(stems, results):
        lines = formal_prediction_lines(result)
        target = output_dir / f"{stem}.txt"
        target.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        written.append(target)
    return written


def validate_formal_prediction_files(
    paths: Iterable[Path],
    expected_count: int,
) -> bool:
    rows = list(paths)
    if len(rows) != expected_count or any(not path.is_file() for path in rows):
        return False
    for path in rows:
        for raw in path.read_text(encoding="utf-8").splitlines():
            parts = raw.split()
            if len(parts) != len(FORMAL_RESULT_COLUMNS):
                return False
            try:
                class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:5]]
                confidence = float(parts[5])
            except ValueError:
                return False
            if (
                class_id < 0
                or not all(0.0 <= value <= 1.0 for value in coordinates)
                or not 0.0 <= confidence <= 1.0
                or "." not in parts[5]
            ):
                return False
    return True
