from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict

from fair_agent.core.config import AUTO_CONFIG, configured_python, load_config, resolve_path
from fair_agent.modules.cli_detection import (
    CLASS_LABELS,
    DEFAULT_RESULT_ROOT,
    list_recent_detection_runs,
)
from fair_agent.modules.operator_view import build_operator_snapshot
from fair_agent.ui.terminal import panel, style_text, supports_color, table


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


HEALTH_LABELS = {
    "ready_with_external_gates": "识别就绪",
    "attention": "需要检查",
}


def _styled(value: Any, *styles: str, color: bool) -> str:
    return style_text(value, *styles, enabled=color)


def _keycap(key: str, label: str, *, color: bool) -> str:
    return f"{_styled(f'[{key}]', 'bold', 'cyan', color=color)} {label}"


def render_menu(page: str = "home", *, color: bool = False) -> str:
    home = "" if page in {"home", "overview"} else f"    {_keycap('0', '返回首页', color=color)}"
    return "\n".join(
        [
            f"  {_styled('识别', 'dim', color=color)}    "
            f"{_keycap('1', '单图识别', color=color)}    "
            f"{_keycap('2', '批量识别', color=color)}",
            f"  {_styled('查看', 'dim', color=color)}    "
            f"{_keycap('3', '最近结果', color=color)}    "
            f"{_keycap('4', '运行状态', color=color)}    "
            f"{_keycap('5', '模型信息', color=color)}",
            f"  {_styled('系统', 'dim', color=color)}    "
            f"{_keycap('r', '刷新状态', color=color)}    "
            f"{_keycap('h', '使用帮助', color=color)}    "
            f"{_keycap('q', '退出', color=color)}{home}",
        ]
    )


MENU = render_menu()
BACKEND_LABELS = {
    "ultralytics_cuda": "CUDA",
    "tensorrt_engine": "TensorRT",
    "tensorrt_native": "TensorRT Native",
    "ascend_acl": "Ascend ACL",
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _display_classes(state: Dict[str, Any]) -> str:
    class_names = {
        0: "soldier",
        1: "small_aircraft",
        2: "warship",
        3: "tank",
        4: "patrol_boat",
        5: "armored_vehicle",
    }
    class_ids = state.get("model_generation", {}).get("classes", [])
    labels = [
        CLASS_LABELS.get(class_names.get(int(class_id), ""), str(class_id))
        for class_id in class_ids
    ]
    return " / ".join(labels) if labels else "等待 production 代际"


def render_page(
    page: str,
    state: Dict[str, Any],
    decision: Dict[str, Any],
    *,
    color: bool = False,
) -> str:
    snapshot = build_operator_snapshot(state, decision)
    runtime = snapshot.get("runtime", {})
    health = HEALTH_LABELS.get(str(snapshot.get("health")), str(snapshot.get("health")))
    backend = BACKEND_LABELS.get(str(runtime.get("backend")), str(runtime.get("backend")))
    healthy = str(snapshot.get("health")) == "ready_with_external_gates"
    status_color = "green" if healthy else "yellow"
    status = (
        f"{_styled('●', 'bold', status_color, color=color)} "
        f"{_styled(health, 'bold', status_color, color=color)}"
    )
    runtime_summary = (
        f"{runtime.get('architecture')} · {runtime.get('machine')} · "
        f"{backend} · {runtime.get('model_format')}"
    )
    production = state.get("model_generation", {}).get("production")
    def accent(value: Any) -> str:
        return _styled(value, "bold", "cyan", color=color)

    def muted(value: Any) -> str:
        return _styled(value, "dim", color=color)
    if page in {"home", "overview"}:
        return "\n\n".join(
            [
                panel(
                    f"{accent('◆')} {_styled('灵动 Agent', 'bold', color=color)} · 视觉识别终端",
                    [
                        f"  {status}    {muted(runtime_summary)}",
                        "",
                        f"  {muted('PRODUCTION')}  {production}",
                        f"  {muted('识别类别')}    {_display_classes(state)}",
                        f"  {muted('结果目录')}    {resolve_path(DEFAULT_RESULT_ROOT)}",
                    ],
                ),
                panel(
                    f"{accent('›')} 快速开始",
                    [
                        f"  {accent('1')}  单图识别    {muted('识别一张 IR / SAR 图像')}",
                        f"  {accent('2')}  批量识别    {muted('扫描目录，可选递归子目录')}",
                        "",
                        f"  {muted('自动保存')}  标注图 · JSON · CSV · 预测 TXT · 摘要",
                    ],
                ),
            ]
        )
    if page == "status":
        blockers = [
            item
            for item in snapshot.get("blockers", [])
            if not item.get("external")
        ]
        blocker_lines = [
            f"需要处理：{item.get('label')}"
            for item in blockers
        ] or ["当前没有影响本机识别的问题。"]
        return panel(
            f"{accent('◆')} 运行状态",
            [
                f"  {status}",
                "",
                f"  {muted('证据模式')}  {snapshot.get('evidence_mode')}",
                f"  {muted('运行平台')}  "
                f"{runtime.get('architecture')} · {runtime.get('device_family')}",
                f"  {muted('推理后端')}  {backend}",
                f"  {muted('运行配置')}  {runtime.get('config')}",
                f"  {muted('更新时间')}  {snapshot.get('generated_at')}",
                "",
                *[f"  {line}" for line in blocker_lines],
            ],
        )
    if page == "models":
        rows = [
            [
                item.get("name"),
                item.get("function"),
                _styled(item.get("status"), "green", color=color),
                _styled("就绪", "green", color=color) if item.get("x86_gpu") else muted("—"),
                _styled("就绪", "green", color=color) if item.get("ascend_310b") else muted("—"),
            ]
            for item in snapshot.get("models", [])
        ]
        return "\n\n".join(
            [
                panel(
                    f"{accent('◆')} 当前正式模型",
                    [
                        f"  {muted('检测器')}     {snapshot.get('detector', {}).get('name')}",
                        f"  {muted('PRODUCTION')} {production}",
                        "",
                        f"  {muted('模型与阈值随当前 production 代际自动加载。')}",
                    ],
                ),
                table(
                    [accent("模型"), accent("功能"), accent("状态"), accent("x86"), accent("310B")],
                    rows,
                    max_widths=[24, 29, 12, 8, 8],
                ),
            ]
        )
    if page == "help":
        return panel(
            f"{accent('?')} CLI 使用帮助",
            [
                f"  {accent('1')}  输入单张图像路径并识别",
                f"  {accent('2')}  输入目录，可选择递归识别子目录",
                f"  {accent('3')}  查看最近保存的识别运行目录",
                f"  {accent('4/5')}  查看运行状态与当前正式模型",
                "",
                f"  {muted('命令模式')}  agile-agent detect --source IMAGE_OR_DIR",
                f"  {muted('完整文档')}  docs/CLI.md",
            ],
        )
    return render_page("home", state, decision, color=color)


def render_recent_runs(*, color: bool = False) -> str:
    def accent(value: Any) -> str:
        return _styled(value, "bold", "cyan", color=color)

    def muted(value: Any) -> str:
        return _styled(value, "dim", color=color)

    runs = list_recent_detection_runs()
    if not runs:
        return panel(
            f"{accent('◆')} 最近识别结果",
            [f"  {muted('尚无已保存结果。首次识别后会写入')} {resolve_path(DEFAULT_RESULT_ROOT)}"],
        )
    return "\n\n".join(
        [
            panel(
                f"{accent('◆')} 最近识别结果",
                [f"  {muted('识别产物保存在本机，可按运行目录整体复制。')}"],
            ),
            table(
                [accent("时间"), accent("图像"), accent("目标"), accent("结果目录")],
                [
                    [
                        row.get("created_at"),
                        row.get("image_count"),
                        row.get("detection_count"),
                        row.get("path"),
                    ]
                    for row in runs
                ],
                max_widths=[19, 6, 6, 68],
            ),
        ]
    )


class ConsoleFrontend:
    def __init__(
        self,
        config_path: str = AUTO_CONFIG,
        input_fn: InputFn = input,
        output_fn: OutputFn = print,
        clear_screen: bool | None = None,
        color: bool | None = None,
    ) -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        self.python = str(configured_python(self.config))
        self.input = input_fn
        self.output = output_fn
        self.page = "home"
        self.message = ""
        self.clear_screen = sys.stdout.isatty() if clear_screen is None else clear_screen
        self.color = supports_color(sys.stdout) if color is None else color

    def _prompt(self, label: str = "灵动") -> str:
        brand = _styled(label, "bold", "cyan", color=self.color)
        arrow = _styled("›", "bold", "cyan", color=self.color)
        return f"{brand} {arrow} "

    def _notice(self, message: str) -> str:
        lowered = message.lower()
        if any(word in lowered for word in ("失败", "错误", "未知", "超时")):
            glyph, tone = "×", "red"
        elif "完成" in lowered or "就绪" in lowered:
            glyph, tone = "✓", "green"
        elif "取消" in lowered:
            glyph, tone = "–", "yellow"
        else:
            glyph, tone = "◆", "cyan"
        return f"  {_styled(glyph, 'bold', tone, color=self.color)} {message}"

    def _paths(self) -> tuple[Path, Path]:
        blackboard = self.config.get("blackboard", {})
        state_path = (
            resolve_path(blackboard.get("output_dir", "reports/agent_blackboard"))
            / blackboard.get("state_json", "blackboard_state.json")
        )
        decision_path = resolve_path(
            self.config.get("decision", {})
            .get("outputs", {})
            .get("decision_json", "reports/agent_blackboard/agent_decision.json")
        )
        return state_path, decision_path

    def _call_cli(
        self,
        arguments: list[str],
        *,
        timeout: int = 600,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.python, "-m", "fair_agent.cli", "--config", self.config_path, *arguments],
            text=True,
            capture_output=True,
            timeout=timeout,
        )

    def _run_cli(self, arguments: list[str], timeout: int = 600) -> bool:
        result = self._call_cli(arguments, timeout=timeout)
        output = ((result.stderr or "") + (result.stdout or "")).strip()
        prefix = "完成" if result.returncode == 0 else f"失败({result.returncode})"
        tail = " | ".join(output.splitlines()[-4:])
        self.message = f"{prefix}：{tail}" if tail else prefix
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
        content = (
            render_recent_runs(color=self.color)
            if self.page == "recent"
            else render_page(self.page, state, decision, color=self.color)
        )
        self.output(content)
        self.output("\n" + render_menu(self.page, color=self.color))
        if self.message:
            self.output("\n" + self._notice(self.message))
            self.message = ""

    @staticmethod
    def _clean_path(value: str) -> str:
        text = value.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            return text[1:-1]
        return text

    def _detect(self, *, batch: bool) -> None:
        try:
            prompt = "图像目录" if batch else "图像路径"
            hint = _styled("输入 q 取消", "dim", color=self.color)
            source = self._clean_path(
                self.input(f"{self._prompt(prompt)}{hint}：")
            )
            if not source or source.lower() in {"q", "quit", "cancel"}:
                self.message = "已取消识别。"
                return
            recursive = False
            if batch:
                answer = self.input(
                    f"{self._prompt('递归子目录')}[y/N]："
                ).strip().lower()
                recursive = answer in {"y", "yes", "是"}
            output = self._clean_path(
                self.input(
                    f"{self._prompt('结果目录')}"
                    f"{_styled('直接回车自动创建', 'dim', color=self.color)}："
                )
            )
        except (EOFError, KeyboardInterrupt):
            self.message = "已取消识别。"
            return
        arguments = ["detect", "--source", source]
        if recursive:
            arguments.append("--recursive")
        if output:
            arguments.extend(["--output", output])
        self.output("\n" + self._notice("正在识别并保存结果，请稍候……"))
        try:
            result = self._call_cli(arguments, timeout=3600)
        except subprocess.TimeoutExpired:
            self.message = "识别超时，请检查服务状态或输入规模。"
            return
        command_output = ((result.stderr or "") + (result.stdout or "")).strip()
        self.output("\n" + (command_output or "识别命令没有返回内容。"))
        self.message = "识别完成。" if result.returncode == 0 else f"识别失败，退出码 {result.returncode}。"
        try:
            self.input(
                "\n"
                + _styled("按 Enter 返回主界面", "dim", color=self.color)
                + "……"
            )
        except (EOFError, KeyboardInterrupt):
            pass
        self.page = "home"

    def run(self) -> int:
        pages = {
            "0": "home",
            "3": "recent",
            "4": "status",
            "5": "models",
            "h": "help",
            "help": "help",
        }
        while True:
            self._draw()
            try:
                command = self.input(self._prompt()).strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.output("\n" + self._notice("视觉识别终端已退出。"))
                return 0
            if command in {"q", "quit", "exit"}:
                self.output(self._notice("视觉识别终端已退出。"))
                return 0
            if command == "1":
                self._detect(batch=False)
            elif command == "2":
                self._detect(batch=True)
            elif command in pages:
                self.page = pages[command]
            elif command == "r":
                if self._run_cli(["refresh"]):
                    self._run_cli(["decide"])
                self.page = "home"
            else:
                self.message = "未知命令。请选择菜单中的数字或字母。"


def run_console_frontend(config_path: str = AUTO_CONFIG) -> int:
    return ConsoleFrontend(config_path=config_path).run()
