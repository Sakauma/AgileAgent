from __future__ import annotations

import os
import re
import shutil
import sys
import unicodedata
from typing import Any, Iterable, Mapping, Sequence, TextIO


ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ANSI_STYLES = {
    "bold": "1",
    "dim": "2",
    "cyan": "36",
    "green": "32",
    "yellow": "33",
    "red": "31",
    "blue": "34",
    "magenta": "35",
    "white": "37",
}


def strip_ansi(value: Any) -> str:
    return ANSI_PATTERN.sub("", str(value))


def supports_color(
    stream: TextIO | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = os.environ if environ is None else environ
    if "NO_COLOR" in env or str(env.get("TERM", "")).lower() == "dumb":
        return False
    if str(env.get("FORCE_COLOR", "")).lower() not in {"", "0", "false", "no"}:
        return True
    target = sys.stdout if stream is None else stream
    return bool(getattr(target, "isatty", lambda: False)())


def style_text(value: Any, *styles: str, enabled: bool = True) -> str:
    text = str(value)
    if not enabled or not styles:
        return text
    codes = [ANSI_STYLES[name] for name in styles if name in ANSI_STYLES]
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def display_width(value: Any) -> int:
    text = strip_ansi(value)
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in text
    )


def truncate_cell(value: Any, width: int) -> str:
    text = str(value)
    if display_width(text) <= width:
        return text
    # Truncating styled text byte-by-byte can split an ANSI escape sequence.
    # A truncated cell therefore falls back to its clean visible text.
    text = strip_ansi(text)
    if width <= 1:
        return "…"[:width]
    kept = []
    used = 0
    for character in text:
        character_width = (
            2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        )
        if used + character_width > width - 1:
            break
        kept.append(character)
        used += character_width
    return "".join(kept) + "…"


def pad_cell(value: Any, width: int) -> str:
    text = truncate_cell(value, width)
    return text + " " * max(0, width - display_width(text))


def terminal_width(default: int = 88) -> int:
    detected = shutil.get_terminal_size((default, 24)).columns
    return max(64, min(120, detected))


def table(
    headers: Iterable[Any],
    rows: Iterable[Iterable[Any]],
    *,
    max_widths: Sequence[int] | None = None,
) -> str:
    header_cells = [str(value) for value in headers]
    body = [[str(value) for value in row] for row in rows]
    widths = [display_width(value) for value in header_cells]
    for row in body:
        if len(row) != len(widths):
            raise ValueError("终端表格行列数不一致")
        for index, value in enumerate(row):
            widths[index] = max(widths[index], display_width(value))
    if max_widths is not None:
        if len(max_widths) != len(widths):
            raise ValueError("终端表格宽度数量与列数不一致")
        widths = [min(width, int(max_widths[index])) for index, width in enumerate(widths)]
    header = "  ".join(pad_cell(value, widths[index]) for index, value in enumerate(header_cells))
    separator = "  ".join("─" * width for width in widths)
    lines = [header, separator]
    lines.extend(
        "  ".join(pad_cell(value, widths[index]) for index, value in enumerate(row))
        for row in body
    )
    return "\n".join(lines)


def panel(
    title: str,
    lines: Iterable[Any],
    *,
    width: int | None = None,
    rounded: bool = True,
) -> str:
    target_width = width or terminal_width()
    inner_width = target_width - 2
    title_text = truncate_cell(title, max(1, inner_width - 4))
    title_prefix = f"─ {title_text} "
    top_left, top_right, bottom_left, bottom_right = (
        ("╭", "╮", "╰", "╯") if rounded else ("┌", "┐", "└", "┘")
    )
    fill_width = max(0, inner_width - display_width(title_prefix))
    top = top_left + title_prefix + "─" * fill_width + top_right
    body = ["│" + pad_cell(str(line), inner_width) + "│" for line in lines]
    bottom = bottom_left + "─" * inner_width + bottom_right
    return "\n".join([top, *body, bottom])
