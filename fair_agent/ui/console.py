from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

from fair_agent.core.config import configured_python, load_config, resolve_path
from fair_agent.modules.operator_view import build_operator_snapshot, render_console


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


MENU = "[1] 总览  [2] 模型  [3] 数据  [4] 增量  [5] 部署  [r] 刷新  [d] 决策  [p] Dry-run  [x] 执行  [q] 退出"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    headers = [str(value) for value in headers]
    values = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in values:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator = "  ".join("-" * width for width in widths)
    body = ["  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in values]
    return "\n".join([line, separator, *body])


def render_page(page: str, state: Dict[str, Any], decision: Dict[str, Any]) -> str:
    snapshot = build_operator_snapshot(state, decision)
    if page == "overview":
        return render_console(snapshot)
    if page == "models":
        return "功能模型\n" + _table(
            ["模型", "功能", "状态", "x86", "310B"],
            [
                [item.get("name"), item.get("function"), item.get("status"), item.get("x86_gpu"), item.get("ascend_310b")]
                for item in snapshot.get("models", [])
            ],
        )
    if page == "data":
        dataset = snapshot.get("dataset", {})
        audit = state.get("data_audit", {})
        return "\n".join(
            [
                "数据概况",
                f"图像={dataset.get('image_count')}  目标={dataset.get('object_count')}  标签={audit.get('total_labels')}",
                f"传感器={dataset.get('sensor', {})}",
                f"场景={dataset.get('scene', {})}",
                f"类别出现图像={dataset.get('class_presence_images', {})}",
            ]
        )
    if page == "incremental":
        incremental = state.get("incremental_learning", {})
        rows = [
            [item.get("protocol"), item.get("incremental_mode"), f"{float(item.get('new_map50') or 0):.5f}", f"{float(item.get('krr') or 0):.5f}", item.get("passed")]
            for item in incremental.get("protocols", [])
        ]
        return "\n".join(
            [
                "增量目标检测",
                f"主模式={incremental.get('primary_mode')}  支持={incremental.get('supported_modes')}  合规={incremental.get('compliance_verified')}",
                _table(["协议", "模式", "New-mAP50", "KRR", "通过"], rows),
            ]
        )
    deployment = snapshot.get("deployment", {})
    submission = snapshot.get("submission", {})
    return "\n".join(
        [
            "部署与提交",
            f"x86 NVIDIA GPU={deployment.get('x86_nvidia_gpu')}",
            f"Ascend 310B={deployment.get('ascend_310b')}",
            "310B 门禁=" + " -> ".join(deployment.get("ascend_gates", [])),
            f"官方测试集={submission.get('official_test_ready')}  提交格式={submission.get('official_format_confirmed')}",
        ]
    )


class ConsoleFrontend:
    def __init__(
        self,
        config_path: str = "configs/agent_pipeline.yaml",
        input_fn: InputFn = input,
        output_fn: OutputFn = print,
        clear_screen: bool | None = None,
    ) -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        self.python = str(configured_python(self.config))
        self.input = input_fn
        self.output = output_fn
        self.page = "overview"
        self.message = ""
        self.clear_screen = sys.stdout.isatty() if clear_screen is None else clear_screen

    def _paths(self) -> tuple[Path, Path]:
        blackboard = self.config.get("blackboard", {})
        state_path = resolve_path(blackboard.get("output_dir", "reports/agent_blackboard")) / blackboard.get("state_json", "blackboard_state.json")
        decision_path = resolve_path(self.config.get("decision", {}).get("outputs", {}).get("decision_json", "reports/agent_blackboard/agent_decision.json"))
        return state_path, decision_path

    def _run_cli(self, arguments: list[str], timeout: int = 600) -> bool:
        result = subprocess.run(
            [self.python, "-m", "fair_agent.cli", "--config", self.config_path, *arguments],
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        output = (result.stdout or "") + (result.stderr or "")
        self.message = ("完成：" if result.returncode == 0 else f"失败({result.returncode})：") + output.strip().replace("\n", " | ")
        return result.returncode == 0

    def _state(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        state_path, decision_path = self._paths()
        if not state_path.exists():
            self._run_cli(["refresh"])
        if not decision_path.exists():
            self._run_cli(["decide"])
        return _load_json(state_path), _load_json(decision_path)

    def _draw(self) -> None:
        state, decision = self._state()
        if self.clear_screen:
            self.output("\033[2J\033[H")
        self.output(render_page(self.page, state, decision))
        self.output("\n" + MENU)
        if self.message:
            self.output("\n" + self.message)
            self.message = ""

    def _decision(self) -> None:
        sensor = self.input("传感器 [sar/ir，默认 sar]：").strip() or "sar"
        scene = self.input("场景 [all/air/forest/sea/urban，默认 all]：").strip() or "all"
        class_focus = self.input("关注类别 [soldier/small_aircraft/warship/tank，默认 soldier]：").strip() or "soldier"
        self._run_cli(["decide", "--sensor", sensor, "--scene", scene, "--class-focus", class_focus])
        self.page = "overview"

    def run(self) -> int:
        pages = {"1": "overview", "2": "models", "3": "data", "4": "incremental", "5": "deployment"}
        while True:
            self._draw()
            try:
                command = self.input("agile-agent> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.output("\n终端工作台已退出。")
                return 0
            if command in {"q", "quit", "exit"}:
                self.output("终端工作台已退出。")
                return 0
            if command in pages:
                self.page = pages[command]
            elif command == "r":
                if self._run_cli(["refresh"]):
                    self._run_cli(["decide"])
            elif command == "d":
                self._decision()
            elif command == "p":
                self._run_cli(["pipeline", "--mode", "dryrun"])
            elif command == "x":
                confirmation = self.input("输入 EXECUTE 确认仅执行低风险动作：").strip()
                self.message = "已取消执行。" if confirmation != "EXECUTE" else ""
                if confirmation == "EXECUTE":
                    self._run_cli(["pipeline", "--mode", "execute"])
            else:
                self.message = "未知命令。请选择菜单中的数字或字母。"


def run_console_frontend(config_path: str = "configs/agent_pipeline.yaml") -> int:
    return ConsoleFrontend(config_path=config_path).run()
