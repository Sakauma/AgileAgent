from __future__ import annotations

from fair_agent.ui.terminal import (
    display_width,
    panel,
    strip_ansi,
    style_text,
    supports_color,
)


class FakeStream:
    def __init__(self, tty: bool) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


def test_color_support_respects_terminal_and_environment() -> None:
    assert supports_color(FakeStream(True), environ={"TERM": "xterm-256color"})
    assert not supports_color(
        FakeStream(True),
        environ={"TERM": "xterm-256color", "NO_COLOR": "1"},
    )
    assert not supports_color(FakeStream(True), environ={"TERM": "dumb"})
    assert supports_color(FakeStream(False), environ={"FORCE_COLOR": "1"})


def test_styled_text_keeps_visible_width_and_panel_alignment() -> None:
    styled = style_text("灵动 Agent", "bold", "cyan")
    assert strip_ansi(styled) == "灵动 Agent"
    assert display_width(styled) == display_width("灵动 Agent")

    rendered = panel(styled, [f"  {styled}", "  视觉识别终端"], width=40)
    assert all(display_width(line) == 40 for line in rendered.splitlines())
    assert rendered.startswith("╭")
    assert rendered.endswith("╯")
