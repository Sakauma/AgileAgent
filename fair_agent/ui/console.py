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
from fair_agent.ui.terminal import panel, table


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


MENU = """[1] 单图识别    [2] 批量识别    [3] 最近结果
[4] 运行状态    [5] 模型信息    [0] 返回首页
[r] 刷新状态    [h] 使用帮助    [q] 退出"""
HEALTH_LABELS = {
    "ready_with_external_gates": "识别就绪",
    "attention": "需要检查",
}
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


def render_page(page: str, state: Dict[str, Any], decision: Dict[str, Any]) -> str:
    snapshot = build_operator_snapshot(state, decision)
    runtime = snapshot.get("runtime", {})
    health = HEALTH_LABELS.get(str(snapshot.get("health")), str(snapshot.get("health")))
    backend = BACKEND_LABELS.get(str(runtime.get("backend")), str(runtime.get("backend")))
    if page in {"home", "overview"}:
        return "\n\n".join(
            [
                panel(
                    "灵动 Agent · SSH 视觉识别终端",
                    [
                        f"状态：{health}    Production：{state.get('model_generation', {}).get('production')}",
                        f"平台：{runtime.get('architecture')} / {runtime.get('machine')}    后端：{backend}    模型：{runtime.get('model_format')}",
                        f"类别：{_display_classes(state)}",
                        f"结果：{resolve_path(DEFAULT_RESULT_ROOT)}",
                    ],
                ),
                panel(
                    "开始识别",
                    [
                        "输入 1 识别一张图像，输入 2 扫描整个目录。",
                        "CLI 自动使用正式模型、Scene-SensorNet、冻结阈值和场景门控。",
                        "每次识别自动保存标注图、JSON、CSV、预测 TXT 与摘要。",
                    ],
                ),
            ]
        )
    if page == "status":
        blockers = snapshot.get("blockers", [])
        blocker_lines = [
            f"{'外部门禁' if item.get('external') else '内部问题'}：{item.get('label')}"
            for item in blockers
        ] or ["当前没有内部运行阻塞。"]
        return panel(
            "运行状态",
            [
                f"健康：{health}    证据：{snapshot.get('evidence_mode')}",
                f"平台：{runtime.get('architecture')}    设备：{runtime.get('device_family')}",
                f"后端：{backend}    配置：{runtime.get('config')}",
                f"更新时间：{snapshot.get('generated_at')}",
                "",
                *blocker_lines,
            ],
        )
    if page == "models":
        rows = [
            [
                item.get("name"),
                item.get("function"),
                item.get("status"),
                "就绪" if item.get("x86_gpu") else "—",
                "就绪" if item.get("ascend_310b") else "—",
            ]
            for item in snapshot.get("models", [])
        ]
        return "\n\n".join(
            [
                panel(
                    "当前正式模型",
                    [
                        f"检测器：{snapshot.get('detector', {}).get('name')}",
                        f"Production：{state.get('model_generation', {}).get('production')}",
                        "模型与阈值由 production 代际自动选择，CLI 不提供人工切换。",
                    ],
                ),
                table(
                    ["模型", "功能", "状态", "x86", "310B"],
                    rows,
                    max_widths=[24, 29, 12, 8, 8],
                ),
            ]
        )
    if page == "help":
        return panel(
            "CLI 使用帮助",
            [
                "1：输入单张图像路径并识别。",
                "2：输入目录，可选择是否递归识别子目录。",
                "3：查看最近保存的识别运行目录。",
                "4/5：查看运行状态与当前正式模型，不修改生产配置。",
                "命令模式：agile-agent detect --source IMAGE_OR_DIR",
                "完整文档：docs/CLI.md",
            ],
        )
    return render_page("home", state, decision)


def render_recent_runs() -> str:
    runs = list_recent_detection_runs()
    if not runs:
        return panel(
            "最近识别结果",
            [f"尚无已保存结果。首次识别后会写入 {resolve_path(DEFAULT_RESULT_ROOT)}。"],
        )
    return "\n\n".join(
        [
            panel(
                "最近识别结果",
                ["结果保存在板端本地，可使用 scp 将整个运行目录复制到 SSH 客户端。"],
            ),
            table(
                ["时间", "图像", "目标", "结果目录"],
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
    ) -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        self.python = str(configured_python(self.config))
        self.input = input_fn
        self.output = output_fn
        self.page = "home"
        self.message = ""
        self.clear_screen = sys.stdout.isatty() if clear_screen is None else clear_screen

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
        content = render_recent_runs() if self.page == "recent" else render_page(self.page, state, decision)
        self.output(content)
        self.output("\n" + MENU)
        if self.message:
            self.output("\n" + self.message)
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
            source = self._clean_path(self.input(f"{prompt}（输入 q 取消）："))
            if not source or source.lower() in {"q", "quit", "cancel"}:
                self.message = "已取消识别。"
                return
            recursive = False
            if batch:
                answer = self.input("递归扫描子目录？[y/N]：").strip().lower()
                recursive = answer in {"y", "yes", "是"}
            output = self._clean_path(
                self.input("结果目录（直接回车自动创建）：")
            )
        except (EOFError, KeyboardInterrupt):
            self.message = "已取消识别。"
            return
        arguments = ["detect", "--source", source]
        if recursive:
            arguments.append("--recursive")
        if output:
            arguments.extend(["--output", output])
        self.output("\n正在调用正式识别链路，请稍候……")
        try:
            result = self._call_cli(arguments, timeout=3600)
        except subprocess.TimeoutExpired:
            self.message = "识别超时，请检查服务状态或输入规模。"
            return
        command_output = ((result.stderr or "") + (result.stdout or "")).strip()
        self.output("\n" + (command_output or "识别命令没有返回内容。"))
        self.message = "识别完成。" if result.returncode == 0 else f"识别失败，退出码 {result.returncode}。"
        try:
            self.input("\n按 Enter 返回主界面……")
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
                command = self.input("agile-agent> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.output("\n终端工作台已退出。")
                return 0
            if command in {"q", "quit", "exit"}:
                self.output("终端工作台已退出。")
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
