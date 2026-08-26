from __future__ import annotations

import ipaddress
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import yaml
from PIL import Image

from fair_agent.core.config import ROOT, rel_path, resolve_path, runtime_platform_info
from fair_agent.core.runtime_log import StructuredEventLog, utc_now
from fair_agent.modules.generation_management import (
    active_generation_registry,
    promote_generation,
    rollback_generation,
    shadow_load_generation,
)
from fair_agent.modules.incremental_lineage import freeze_accepted_batch
from fair_agent.modules.model_generations import load_generation_registry


FAILED_STATES = {
    "FAILED",
    "REJECTED",
    "ROLLED_BACK",
    "ROLLBACK_FAILED",
    "CANCELLED",
}
TRAINING_TERMINAL_STATES = FAILED_STATES | {"ACCEPTED", "PROMOTED"}
ASCEND_STAGE_ORDER = (
    "export",
    "candidate_deploy",
    "accuracy_gate",
    "fps_gate",
    "promote",
)
RUNTIME_CONTROL_CAPABILITY = "onsite_generation_v1"
RUNTIME_CONTROL_HEADER = "X-Agile-Agent-Control"


def _loopback_base_url(config: Mapping[str, Any]) -> str:
    runtime = dict(config.get("runtime") or {})
    host = str(runtime.get("server_host") or "127.0.0.1").strip()
    probe_host = host.strip("[]")
    try:
        is_loopback = ipaddress.ip_address(probe_host).is_loopback
    except ValueError:
        is_loopback = probe_host.casefold() == "localhost"
    if not is_loopback:
        raise ValueError("现场运行时代际控制只允许连接本机回环服务。")
    rendered_host = f"[{probe_host}]" if ":" in probe_host else probe_host
    return f"http://{rendered_host}:{int(runtime.get('server_port') or 8501)}"


class LoopbackGenerationController:
    """Atomically swap a running local service, or update the next-start registry."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        probe_timeout: float = 0.8,
        transition_timeout: float = 300.0,
    ) -> None:
        self.config = dict(config)
        self.base_url = _loopback_base_url(config)
        self.probe_timeout = float(probe_timeout)
        self.transition_timeout = float(transition_timeout)
        # Never inherit HTTP(S)_PROXY for a privileged loopback transition.
        self._opener = build_opener(ProxyHandler({}))
        self.health = self._read_health()
        self.used_running_service = False

    def _request_json(
        self,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        timeout: float,
    ) -> Dict[str, Any]:
        body = (
            json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        headers = {
            "Accept": "application/json",
            "User-Agent": "AgileAgent-Onsite/1",
        }
        if body is not None:
            headers.update(
                {
                    "Content-Type": "application/json",
                    RUNTIME_CONTROL_HEADER: RUNTIME_CONTROL_CAPABILITY,
                }
            )
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        with self._opener.open(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        if not isinstance(parsed, Mapping):
            raise RuntimeError("本机正式服务返回了非法运行时控制响应。")
        return dict(parsed)

    def _read_health(self) -> Dict[str, Any] | None:
        try:
            payload = self._request_json(
                "/api/health", timeout=self.probe_timeout
            )
        except (
            HTTPError,
            URLError,
            OSError,
            TimeoutError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None
        return payload if payload.get("status") == "ready" else None

    @property
    def service_running(self) -> bool:
        return self.health is not None

    @property
    def service_compatible(self) -> bool:
        return bool(
            self.health
            and self.health.get("runtime_generation_control")
            == RUNTIME_CONTROL_CAPABILITY
        )

    def require_compatible_service(self) -> None:
        if self.service_running and not self.service_compatible:
            raise RuntimeError(
                "检测到正在运行的正式服务，但它不支持现场原子换代；"
                "请先用当前工程重启服务，再执行一键增量。"
            )

    def _transition(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            response = self._request_json(
                "/api/runtime/generation",
                payload=payload,
                timeout=self.transition_timeout,
            )
        except HTTPError as exc:
            message = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"本机正式服务拒绝代际切换（HTTP {exc.code}）：{message}"
            ) from exc
        except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
            # A response can be lost after an atomic swap.  Re-probe before
            # declaring failure so a completed promotion is not rolled back by
            # an ambiguous transport error.
            health = self._read_health()
            expected = str(payload.get("candidate_id") or payload.get("target_id") or "")
            if health and str(health.get("generation_id") or "") == expected:
                self.health = health
                self.used_running_service = True
                return {
                    "production": expected,
                    "runtime_swap": "atomic_recovered_after_transport_error",
                    "transport_error": str(exc),
                }
            raise RuntimeError(f"本机正式服务代际切换通信失败：{exc}") from exc
        if response.get("ok") is not True or not isinstance(response.get("result"), Mapping):
            raise RuntimeError(str(response.get("error") or "本机正式服务代际切换失败。"))
        self.health = {
            **dict(self.health or {}),
            "generation_id": response["result"].get("production"),
        }
        self.used_running_service = True
        return dict(response["result"])

    def promote(
        self,
        config: Mapping[str, Any],
        candidate_id: str,
        manifest_path: str,
    ) -> Dict[str, Any]:
        if not self.service_running:
            result = promote_generation(config, candidate_id, manifest_path)
            return {
                **result,
                "runtime_swap": "next_process_start",
                "running_service_detected": False,
            }
        self.require_compatible_service()
        registry = load_generation_registry(active_generation_registry(config))
        generation = registry["generations_by_id"].get(str(candidate_id))
        if generation is None:
            raise ValueError("本机正式服务待切换候选代际不存在。")
        parent_id = str(generation.get("parent") or "")
        running_id = str((self.health or {}).get("generation_id") or "")
        if running_id != parent_id:
            raise RuntimeError(
                "本机正式服务代际与候选父代不一致："
                f"运行中 {running_id or 'unknown'}，候选父代 {parent_id}。"
            )
        return self._transition(
            {
                "action": "promote",
                "candidate_id": str(candidate_id),
                "manifest_path": str(manifest_path),
                "expected_parent_id": parent_id,
            }
        )

    def rollback(
        self, config: Mapping[str, Any], target_id: str
    ) -> Dict[str, Any]:
        if not self.used_running_service:
            result = rollback_generation(config, target_id)
            return {**result, "runtime_swap": "next_process_start"}
        return self._transition(
            {
                "action": "rollback",
                "target_id": str(target_id),
            }
        )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_run_id(value: str) -> str:
    cleaned = "".join(
        character.lower()
        if character.isalnum() or character in {"-", "_"}
        else "-"
        for character in value.strip()
    ).strip("-_")
    if not cleaned:
        raise ValueError("现场增量 run_id 不能为空。")
    return cleaned[:96]


def _resolve_run_root(value: str | Path | None) -> Path:
    return resolve_path(value or "runs/onsite_incremental").resolve()


def inspect_onsite_bundle(path: str | Path) -> Dict[str, Any]:
    """Read the portable data contract without extracting or hashing the archive."""

    archive_path = Path(path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"现场增量数据包不存在：{archive_path}")
    if archive_path.suffix.lower() != ".zip":
        raise ValueError("现场增量数据包必须是 ZIP。")
    if not zipfile.is_zipfile(archive_path):
        raise ValueError("现场增量数据包不是有效 ZIP。")
    names: Dict[int, str] = {}
    label_class_ids: set[int] = set()
    image_count = 0
    label_count = 0
    image_stems: set[str] = set()
    candidate_labels: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    allowed_images = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if not members:
            raise ValueError("现场增量数据包为空。")
        for member in members:
            pure = PurePosixPath(member.filename.replace("\\", "/"))
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"现场增量数据包包含不安全路径：{member.filename}")
            suffix = pure.suffix.lower()
            if suffix in allowed_images:
                image_count += 1
                image_stems.add(pure.stem.casefold())
            elif suffix == ".txt" and pure.name.casefold() != "classes.txt":
                candidate_labels.append((member, pure))
        for member, pure in candidate_labels:
            path_parts = {part.casefold() for part in pure.parts}
            if "labels" not in path_parts and pure.stem.casefold() not in image_stems:
                continue
            label_count += 1
            try:
                label_text = archive.read(member).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"现场标签不是 UTF-8：{member.filename}"
                ) from exc
            for line_number, line in enumerate(label_text.splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(
                        f"{member.filename}:{line_number} 不是五列 YOLO 检测标签。"
                    )
                try:
                    class_id = int(fields[0])
                except ValueError as exc:
                    raise ValueError(
                        f"{member.filename}:{line_number} 类别 ID 非法。"
                    ) from exc
                if class_id < 0:
                    raise ValueError(
                        f"{member.filename}:{line_number} 类别 ID 不能为负数。"
                    )
                label_class_ids.add(class_id)
        yaml_members = [
            member
            for member in members
            if PurePosixPath(member.filename).name.lower()
            in {"classes.yaml", "data.yaml", "dataset.yaml", "batch.yaml"}
        ]
        for member in sorted(
            yaml_members,
            key=lambda row: (
                PurePosixPath(row.filename).name.lower() != "classes.yaml",
                len(PurePosixPath(row.filename).parts),
            ),
        ):
            try:
                payload = yaml.safe_load(archive.read(member).decode("utf-8")) or {}
            except (UnicodeDecodeError, yaml.YAMLError):
                continue
            raw_names = payload.get("names") if isinstance(payload, Mapping) else None
            if isinstance(raw_names, list):
                names = {index: str(value).strip() for index, value in enumerate(raw_names)}
            elif isinstance(raw_names, Mapping):
                try:
                    names = {
                        int(key): str(value).strip() for key, value in raw_names.items()
                    }
                except (TypeError, ValueError):
                    names = {}
            if names:
                break
        if not names:
            text_members = [
                member
                for member in members
                if PurePosixPath(member.filename).name.casefold() == "classes.txt"
            ]
            for member in sorted(
                text_members, key=lambda row: len(PurePosixPath(row.filename).parts)
            ):
                try:
                    values = [
                        value.strip()
                        for value in archive.read(member).decode("utf-8").splitlines()
                        if value.strip()
                    ]
                except UnicodeDecodeError:
                    continue
                if values:
                    names = {index: value for index, value in enumerate(values)}
                    break
    if image_count == 0 or label_count == 0:
        raise ValueError("现场增量数据包必须同时包含图像和 YOLO 标签。")
    if not label_class_ids:
        raise ValueError("现场增量数据包的标签中没有新增类别目标。")
    used_names = {
        class_id: names[class_id]
        for class_id in sorted(label_class_ids)
        if class_id in names and names[class_id]
    }
    return {
        "path": str(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "image_count": image_count,
        "label_count": label_count,
        "source_class_ids": sorted(label_class_ids),
        "declared_classes": {
            str(key): value for key, value in sorted(used_names.items())
        },
        "has_declared_classes": len(used_names) == len(label_class_ids),
    }


def probe_cuda_training(
    python: str | Path | None,
    device: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Dict[str, Any]:
    """Probe the exact interpreter that the training worker will use."""

    executable = str(python or sys.executable)
    code = (
        "import json, torch\n"
        "import ultralytics\n"
        "print(json.dumps({"
        "'torch': str(torch.__version__),"
        "'ultralytics': str(ultralytics.__version__),"
        "'cuda_available': bool(torch.cuda.is_available()),"
        "'device_count': int(torch.cuda.device_count()),"
        "'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]"
        "}))"
    )
    try:
        completed = runner(
            [executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "ready": False,
            "python": executable,
            "device": str(device),
            "error": str(exc),
        }
    if completed.returncode != 0:
        return {
            "ready": False,
            "python": executable,
            "device": str(device),
            "error": (completed.stderr or completed.stdout).strip(),
        }
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "python": executable,
            "device": str(device),
            "error": f"CUDA 探测输出非法：{exc}",
        }
    requested = str(device).strip()
    valid_device = requested.isdigit() and int(requested) < int(
        payload.get("device_count", 0)
    )
    return {
        **payload,
        "ready": bool(payload.get("cuda_available")) and valid_device,
        "python": executable,
        "device": requested,
        "error": None
        if bool(payload.get("cuda_available")) and valid_device
        else f"训练 GPU {requested} 不可用。",
    }


def load_ascend_deployment_spec(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Ascend 现场部署编排不存在：{resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or int(payload.get("schema_version") or 0) != 1:
        raise ValueError("Ascend 现场部署编排版本不受支持。")
    if payload.get("target") != "ascend310b":
        raise ValueError("现场部署编排 target 必须是 ascend310b。")
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list):
        raise ValueError("Ascend 现场部署编排缺少 stages。")
    stages = []
    for item in raw_stages:
        if not isinstance(item, Mapping):
            raise ValueError("Ascend 现场部署 stage 必须是映射。")
        stage_id = str(item.get("id") or "").strip()
        command = item.get("command")
        if stage_id not in ASCEND_STAGE_ORDER or not isinstance(command, list) or not command:
            raise ValueError(f"Ascend 现场部署 stage 非法：{stage_id or 'unknown'}")
        if not all(isinstance(value, (str, int, float)) for value in command):
            raise ValueError(f"Ascend stage {stage_id} 的 command 必须是参数列表。")
        stages.append({**dict(item), "id": stage_id, "command": [str(value) for value in command]})
    if [item["id"] for item in stages] != list(ASCEND_STAGE_ORDER):
        raise ValueError(
            "Ascend stages 必须严格按 export → candidate_deploy → "
            "accuracy_gate → fps_gate → promote 排列。"
        )
    rollback = payload.get("rollback")
    if not isinstance(rollback, Mapping) or not isinstance(rollback.get("command"), list):
        raise ValueError("Ascend 现场部署编排必须提供 rollback.command。")
    if not rollback["command"]:
        raise ValueError("Ascend rollback.command 不能为空。")
    return {
        **dict(payload),
        "path": str(resolved),
        "stages": stages,
        "rollback": {**dict(rollback), "command": [str(value) for value in rollback["command"]]},
    }


def _render_token(value: str, context: Mapping[str, Any]) -> str:
    expanded = os.path.expandvars(value)
    try:
        return expanded.format_map({key: str(item) for key, item in context.items()})
    except KeyError as exc:
        raise ValueError(f"现场部署命令引用未知变量：{exc.args[0]}") from exc


def _nested_value(payload: Mapping[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"验收报告缺少字段：{dotted}")
        value = value[part]
    return value


class AscendDeploymentExecutor:
    """Run an explicit board candidate chain without invoking a shell."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        run_dir: Path,
        context: Mapping[str, Any],
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.spec = dict(spec)
        self.run_dir = run_dir
        self.context = dict(context)
        self.runner = runner
        self.completed: list[str] = []

    def _command(self, raw: Sequence[str]) -> list[str]:
        return [_render_token(str(value), self.context) for value in raw]

    def _run(self, stage_id: str, raw: Sequence[str]) -> Dict[str, Any]:
        command = self._command(raw)
        log_path = self.run_dir / "deployment" / f"{stage_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            completed = self.runner(
                command,
                cwd=str(ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            log_path.write_text(str(exc) + "\n", encoding="utf-8")
            raise RuntimeError(f"Ascend 阶段 {stage_id} 无法启动：{exc}") from exc
        output = (completed.stdout or "") + (completed.stderr or "")
        log_path.write_text(output, encoding="utf-8")
        result = {
            "id": stage_id,
            "command": command,
            "returncode": int(completed.returncode),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "log": rel_path(log_path),
        }
        if completed.returncode != 0:
            raise RuntimeError(
                f"Ascend 阶段 {stage_id} 失败，退出码 {completed.returncode}；"
                f"日志：{log_path}"
            )
        self.completed.append(stage_id)
        return result

    def _verify_report(self, stage: Mapping[str, Any]) -> Dict[str, Any] | None:
        raw_report = stage.get("report")
        if not raw_report:
            if stage["id"] in {"accuracy_gate", "fps_gate"}:
                raise ValueError(f"Ascend {stage['id']} 必须登记 JSON report。")
            return None
        report_path = Path(_render_token(str(raw_report), self.context)).expanduser().resolve()
        if not report_path.is_file():
            raise FileNotFoundError(f"Ascend {stage['id']} 报告不存在：{report_path}")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Ascend {stage['id']} 报告必须是 JSON 对象。")
        required = stage.get("require") or {}
        if not isinstance(required, Mapping):
            raise ValueError(f"Ascend {stage['id']}.require 必须是映射。")
        failures = [
            f"{key}={_nested_value(payload, str(key))!r}，要求 {expected!r}"
            for key, expected in required.items()
            if _nested_value(payload, str(key)) != expected
        ]
        if failures:
            raise RuntimeError(
                f"Ascend {stage['id']} 报告门禁未通过：" + "; ".join(failures)
            )
        if stage["id"] == "accuracy_gate" and not bool(
            payload.get("score_passed", payload.get("passed", False))
        ):
            raise RuntimeError("Ascend accuracy gate 未通过。")
        if stage["id"] == "fps_gate":
            competition = payload.get("competition") or {}
            target = float(self.context["target_fps"])
            if (
                competition.get("batch_fps_passed") is not True
                or float(competition.get("batch_fps", -1.0)) < target
            ):
                raise RuntimeError(f"Ascend FPS gate 未达到 {target:g} FPS。")
        return {"path": str(report_path), "payload": dict(payload)}

    def execute(self) -> Dict[str, Any]:
        rows = []
        reports: Dict[str, Any] = {}
        for stage in self.spec["stages"]:
            rows.append(self._run(str(stage["id"]), stage["command"]))
            report = self._verify_report(stage)
            if report is not None:
                reports[str(stage["id"])] = report
        return {"status": "PROMOTED", "stages": rows, "reports": reports}

    def rollback(self) -> Dict[str, Any] | None:
        if "candidate_deploy" not in self.completed:
            return None
        return self._run("rollback", self.spec["rollback"]["command"])


def _lock_images(batch_dir: Path) -> list[Path]:
    root = batch_dir / "sealed_lock" / "images" / "lock"
    images = sorted(path for path in root.glob("*") if path.is_file())
    if not images:
        raise ValueError("现场增量 lock 为空，无法执行候选 FPS 验收。")
    return images


def benchmark_shadow_engine(
    engine: Any,
    image_paths: Sequence[Path],
    *,
    probe_size: int,
    warmup_rounds: int,
    rounds: int,
    confidence: float,
) -> Dict[str, Any]:
    if probe_size <= 0 or rounds <= 0 or warmup_rounds < 0:
        raise ValueError("现场 FPS 验收参数非法。")
    sources: list[tuple[Image.Image, str]] = []
    try:
        for index in range(probe_size):
            path = image_paths[index % len(image_paths)]
            with Image.open(path) as source:
                source.load()
                sources.append((source.convert("RGB"), f"onsite-{index:04d}-{path.name}"))
        for _ in range(warmup_rounds):
            engine.predict_batch(sources, confidence, "auto")
        elapsed_rows = []
        for _ in range(rounds):
            started = time.perf_counter()
            results = engine.predict_batch(sources, confidence, "auto")
            elapsed = time.perf_counter() - started
            if len(results) != probe_size:
                raise RuntimeError("现场候选 batch 返回图像数不完整。")
            elapsed_rows.append(elapsed)
    finally:
        for image, _name in sources:
            image.close()
    fps_rows = [probe_size / value for value in elapsed_rows]
    return {
        "schema_version": 1,
        "kind": "onsite_incremental_shadow_fps",
        "probe_size": probe_size,
        "warmup_rounds": warmup_rounds,
        "rounds": rounds,
        "elapsed_seconds": elapsed_rows,
        "fps_by_round": fps_rows,
        "median_fps": float(statistics.median(fps_rows)),
        "timing_scope": "complete_image_inference",
    }


class OnsiteIncrementalWorkflow:
    """Candidate-first 4+2+n orchestration for a CUDA training host."""

    def __init__(
        self,
        config: Mapping[str, Any],
        store: Any | None,
        manager: Any | None,
        event_log: StructuredEventLog | None,
        *,
        run_root: str | Path | None = None,
        capability_probe: Callable[[str | Path | None, str], Mapping[str, Any]] = probe_cuda_training,
        shadow_loader: Callable[..., tuple[Any, Dict[str, Any]]] = shadow_load_generation,
        promoter: Callable[..., Dict[str, Any]] | None = None,
        rollback: Callable[..., Dict[str, Any]] | None = None,
        lineage_freezer: Callable[..., Path | None] = freeze_accepted_batch,
        deployment_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.config = dict(config)
        self.store = store
        self.manager = manager
        self.event_log = event_log
        self.run_root = _resolve_run_root(run_root)
        self.capability_probe = capability_probe
        self.shadow_loader = shadow_loader
        self.promoter = promoter or promote_generation
        self.rollback_generation = rollback or rollback_generation
        self.lineage_freezer = lineage_freezer
        self.deployment_runner = deployment_runner
        self.progress_callback = progress_callback

    def _production(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        generation = self.config["generation"]
        source = resolve_path(generation["registry"])
        default_source = resolve_path("models/generations.json")
        runtime = resolve_path(generation["runtime_registry"])
        # A plan is a genuinely read-only operation.  Prefer an already valid
        # runtime registry, otherwise inspect the verified source directly;
        # unlike active_generation_registry(), this never initializes state.
        registry_path = source
        if source == default_source and runtime.is_file():
            try:
                load_generation_registry(runtime)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                registry_path = source
            else:
                registry_path = runtime
        registry = load_generation_registry(registry_path)
        production_id = str(registry["channels"]["production"])
        return registry, dict(registry["generations_by_id"][production_id])

    def plan(
        self,
        bundle: str | Path,
        *,
        class_names: str | None = None,
        target: str = "auto",
        deployment_spec: str | Path | None = None,
        probe_training: bool = True,
        allow_provisional_names: bool = False,
    ) -> Dict[str, Any]:
        bundle_info = inspect_onsite_bundle(bundle)
        registry, production = self._production()
        platform = runtime_platform_info(self.config)
        resolved_target = (
            "ascend310b"
            if target == "auto" and platform["device_family"] == "ascend_310b"
            else "x86"
            if target == "auto"
            else str(target)
        )
        if resolved_target not in {"x86", "ascend310b"}:
            raise ValueError(f"不支持的现场部署目标：{resolved_target}")
        training = dict(self.config["incremental_workbench"]["training"])
        capability = (
            dict(
                self.capability_probe(
                    training.get("python"), str(training.get("device", "0"))
                )
            )
            if probe_training
            else {"ready": None, "not_probed": True}
        )
        spec = None
        if deployment_spec is not None:
            spec = load_ascend_deployment_spec(deployment_spec)
        deployment_ready = resolved_target == "x86" or spec is not None
        active_ids = sorted(int(value) for value in production["classes"])
        active_specialists = [
            model_id
            for model_id in production.get("model_members", [])
            if registry["models_by_id"][str(model_id)]["role"]
            in {"class_incremental_expert", "target_incremental_expert"}
        ]
        reasons = []
        source_class_ids = [int(value) for value in bundle_info["source_class_ids"]]
        override_names = [
            value.strip() for value in str(class_names or "").split(",") if value.strip()
        ]
        if override_names:
            if len(override_names) != len(source_class_ids):
                reasons.append(
                    "--class-names 数量必须与标签中实际出现的类别数一致。"
                )
            declared_names = {
                str(source_id): override_names[index]
                for index, source_id in enumerate(source_class_ids)
                if index < len(override_names)
            }
        else:
            declared_names = dict(bundle_info["declared_classes"])
        missing_names = [
            class_id
            for class_id in source_class_ids
            if str(class_id) not in declared_names
        ]
        if missing_names and not allow_provisional_names:
            reasons.append(
                "数据包未声明全部实际类别名称；缺少局部类别 ID："
                + ",".join(str(value) for value in missing_names)
            )
        normalized_names = [
            str(declared_names[str(class_id)]).strip().casefold()
            for class_id in source_class_ids
            if str(class_id) in declared_names
        ]
        if len(normalized_names) != len(set(normalized_names)):
            reasons.append("本轮多个局部类别不能使用相同名称。")
        active_names = {
            str(registry["class_map"][class_id]).strip().casefold()
            for class_id in active_ids
        }
        existing_names = sorted(set(normalized_names) & active_names)
        if existing_names:
            reasons.append(
                "现场 4+2+n 只接受 production 未学习的新类别；重复名称："
                + ",".join(existing_names)
            )
        used_global_ids = {int(value) for value in registry["class_map"]}
        bound_ids: set[int] = set()
        next_global_id = max(used_global_ids, default=-1) + 1
        predicted_ids = []
        for source_id in source_class_ids:
            if source_id not in used_global_ids and source_id not in bound_ids:
                global_id = source_id
            else:
                while next_global_id in used_global_ids or next_global_id in bound_ids:
                    next_global_id += 1
                global_id = next_global_id
                next_global_id += 1
            predicted_ids.append(global_id)
            bound_ids.add(global_id)
        declared_count = len(source_class_ids)
        if len(active_ids) < 6:
            reasons.append("现场 4+2+n 要求当前 production 至少已有六类。")
        if platform["architecture"] == "arm":
            reasons.append(
                "真正的新类别检测专家不能只在 310B production 环境训练；"
                "请在 CUDA 节点执行同一命令，再把候选交给板端部署编排。"
            )
        if probe_training and capability.get("ready") is not True:
            reasons.append(str(capability.get("error") or "CUDA/Ultralytics 训练环境不可用。"))
        if self.config["incremental_workbench"]["lifecycle"].get("auto_continue") is not True:
            reasons.append("现场一键流程要求 incremental lifecycle auto_continue=true。")
        if not deployment_ready:
            reasons.append("Ascend310B 自动部署必须提供完整 deployment spec。")
        specialist_budget = int(self.config["routing"]["max_specialists_per_image"])
        required_specialists = len(active_specialists) + (1 if source_class_ids else 0)
        if resolved_target == "x86" and required_specialists > specialist_budget:
            reasons.append(
                "现场新专家会超过每图专家预算："
                f"需要 {required_specialists}，当前允许 {specialist_budget}。"
            )
        return {
            "schema_version": 1,
            "kind": "onsite_incremental_plan",
            "ready": not reasons,
            "protocol": "4plus2_plus_n_class_incremental",
            "bundle": bundle_info,
            "platform": platform,
            "training_capability": capability,
            "deployment_target": resolved_target,
            "deployment_spec": spec["path"] if spec else None,
            "production_before": str(production["id"]),
            "active_class_ids_before": active_ids,
            "active_class_count_before": len(active_ids),
            "active_specialist_count_before": len(active_specialists),
            "specialist_budget": specialist_budget,
            "required_specialist_count": required_specialists,
            "declared_new_class_count": declared_count,
            "declared_new_classes": declared_names,
            "predicted_new_class_ids": predicted_ids,
            "predicted_final_class_count": len(active_ids) + len(predicted_ids),
            "stages": [
                "preflight",
                "ingest_and_audit",
                "register_new_classes",
                "seal_train_dev_lock",
                "train_new_expert",
                "dev_calibration",
                "cumulative_lock_recheck",
                "candidate_fps_gate",
                "candidate_deploy",
                "atomic_promote_or_rollback",
            ],
            "blocking_reasons": reasons,
        }

    def _new_state(
        self,
        plan: Mapping[str, Any],
        *,
        name: str | None,
        class_names: str | None,
        auto_deploy: bool,
    ) -> tuple[Path, Dict[str, Any]]:
        run_id = _safe_run_id(
            f"onsite-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )
        run_dir = (self.run_root / run_id).resolve()
        if self.run_root not in run_dir.parents or run_dir.exists():
            raise FileExistsError(f"现场增量运行目录不可用：{run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        state = {
            "schema_version": 1,
            "kind": "onsite_incremental_run",
            "run_id": run_id,
            "status": "PREFLIGHT_PASSED",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "name": name,
            "class_names_override": class_names,
            "auto_deploy": bool(auto_deploy),
            "production_before": plan["production_before"],
            "deployment_target": plan["deployment_target"],
            "plan": dict(plan),
            "history": [],
        }
        self._write_state(run_dir, state, "PREFLIGHT_PASSED")
        return run_dir, state

    def _write_state(
        self,
        run_dir: Path,
        state: Dict[str, Any],
        status: str,
        **details: Any,
    ) -> None:
        state["status"] = status
        state["updated_at"] = utc_now()
        state.update(details)
        state.setdefault("history", []).append(
            {"status": status, "at": state["updated_at"]}
        )
        _atomic_json(run_dir / "state.json", state)
        if self.event_log is not None:
            self.event_log.append(
                f"onsite.incremental.{status.lower()}",
                level="error" if status in FAILED_STATES else "info",
                component="incremental",
                run_id=str(state["run_id"]),
                batch_id=state.get("batch_id"),
                job_id=state.get("job_id"),
                generation_id=state.get("candidate_generation_id"),
                details=details,
            )
        self._emit_progress(
            {
                "run_id": state["run_id"],
                "status": status,
                "at": state["updated_at"],
                **details,
            }
        )

    def _emit_progress(self, payload: Mapping[str, Any]) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(dict(payload))
        except Exception:
            # Console rendering or an optional observer must never mutate the
            # training/deployment transaction.
            return

    def _wait_for_training_job(self, batch_id: str, job_id: str) -> Dict[str, Any]:
        if self.manager is None or not callable(getattr(self.manager, "get", None)):
            raise RuntimeError("现场训练管理器不支持可观察任务查询。")
        poll_seconds = max(
            0.05,
            min(
                5.0,
                float(self.config["incremental_workbench"]["poll_interval_ms"])
                / 1000.0,
            ),
        )
        started = time.monotonic()
        last_heartbeat = started
        while True:
            job = dict(self.manager.get(batch_id, job_id))
            status = str(job.get("status") or "UNKNOWN")
            if status in TRAINING_TERMINAL_STATES:
                return job
            now = time.monotonic()
            if now - last_heartbeat >= 30.0:
                self._emit_progress(
                    {
                        "run_id": None,
                        "status": "TRAINING_RUNNING",
                        "job_id": job_id,
                        "job_status": status,
                        "elapsed_seconds": round(now - started, 1),
                    }
                )
                last_heartbeat = now
            time.sleep(poll_seconds)

    def _round_contract(
        self,
        run_dir: Path,
        state: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> Dict[str, Any]:
        bindings = [dict(item) for item in manifest["audit"]["class_bindings"]]
        payload = {
            "schema_version": 1,
            "protocol": "4plus2_plus_n_class_incremental",
            "run_id": state["run_id"],
            "parent_generation_id": state["production_before"],
            "round_id": f"onsite_{state['run_id']}",
            "new_class_ids": sorted(int(item["global_class_id"]) for item in bindings),
            "new_classes": bindings,
            "learning_data_scope": "incremental_dataset_only",
            "validation_data_scope": "incremental_dataset_only",
            "base_detector_weights_frozen": True,
            "old_expert_weights_frozen": True,
            "old_raw_image_count": 0,
            "label_projection": "current_round_classes_only",
            "scene_sensor_counted_as_incremental_learning": False,
        }
        path = run_dir / "round_contract.yaml"
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return {**payload, "path": rel_path(path)}

    def _validate_new_classes(
        self,
        manifest: Mapping[str, Any],
        *,
        allow_provisional_names: bool,
    ) -> None:
        audit = manifest.get("audit") or {}
        if manifest.get("status") != "AUDITED":
            raise ValueError(str(manifest.get("error") or "现场数据审计未通过。"))
        if audit.get("incremental_mode") != "class_incremental":
            raise ValueError("现场 4+2+n 入口只接受真正的新增类别，不接受已有类目标更新。")
        bindings = audit.get("class_bindings") or []
        if not bindings or any(item.get("is_existing_class") is True for item in bindings):
            raise ValueError("现场批次必须全部是当前 production 尚未学习的新类别。")
        normalized_names = [
            str(item.get("display_name") or "").strip().casefold()
            for item in bindings
        ]
        if any(not value for value in normalized_names) or len(normalized_names) != len(
            set(normalized_names)
        ):
            raise ValueError("现场批次每个新类别必须有非空且互不重复的名称。")
        if audit.get("requires_class_confirmation") and not allow_provisional_names:
            raise ValueError(
                "现场批次类别名称尚未确认；请在数据包写入 classes.yaml/data.yaml，"
                "或使用 --class-names。"
            )

    def _close_engine(self, engine: Any) -> None:
        close = getattr(engine, "close", None)
        if callable(close):
            close()

    def run(
        self,
        bundle: str | Path,
        *,
        name: str | None = None,
        class_names: str | None = None,
        target: str = "auto",
        deployment_spec: str | Path | None = None,
        auto_deploy: bool = True,
        allow_provisional_names: bool = False,
        fps_probe_size: int | None = None,
        fps_warmup_rounds: int = 2,
        fps_rounds: int | None = None,
    ) -> Dict[str, Any]:
        plan = self.plan(
            bundle,
            class_names=class_names,
            target=target,
            deployment_spec=deployment_spec,
            probe_training=True,
            allow_provisional_names=allow_provisional_names,
        )
        blocking = list(plan["blocking_reasons"])
        if blocking:
            raise RuntimeError("现场增量预检失败：" + "；".join(blocking))
        if self.store is None or self.manager is None or self.event_log is None:
            raise RuntimeError("现场增量正式运行缺少工作台服务。")
        run_dir, state = self._new_state(
            plan,
            name=name,
            class_names=class_names,
            auto_deploy=auto_deploy,
        )
        parent_generation_id = str(plan["production_before"])
        external: AscendDeploymentExecutor | None = None
        local_promoted = False
        try:
            archive = Path(bundle).expanduser().resolve()
            manifest = self.store.create(
                archive.name,
                archive.read_bytes(),
                name,
                class_names,
            )
            state["batch_id"] = str(manifest["batch_id"])
            self._write_state(
                run_dir,
                state,
                "AUDITED",
                batch_id=state["batch_id"],
                audit=manifest.get("audit"),
            )
            self._validate_new_classes(
                manifest,
                allow_provisional_names=allow_provisional_names,
            )
            contract = self._round_contract(run_dir, state, manifest)
            self._write_state(
                run_dir,
                state,
                "CLASSES_REGISTERED",
                round_contract=contract,
            )
            manifest = self.store.inject(state["batch_id"])
            self._write_state(
                run_dir,
                state,
                "DATA_READY",
                split_counts=manifest["injection"]["counts"],
            )
            queued_job = self.manager.start(state["batch_id"], wait=False)
            state["job_id"] = str(queued_job["job_id"])
            self._write_state(
                run_dir,
                state,
                "TRAINING_STARTED",
                job_id=state["job_id"],
                training_job=queued_job,
            )
            job = self._wait_for_training_job(
                state["batch_id"], state["job_id"]
            )
            self._write_state(
                run_dir,
                state,
                str(job.get("status") or "FAILED"),
                job_id=state["job_id"],
                training_job=job,
            )
            if str(job.get("status")) in FAILED_STATES:
                return state
            lifecycle = job.get("lifecycle_result") or {}
            if str(lifecycle.get("status")) != "ACCEPTED":
                raise RuntimeError(
                    "现场训练没有停在已通过累计 lock 的候选状态："
                    f"{lifecycle.get('status') or job.get('status')}"
                )
            generation = lifecycle.get("generation") or {}
            candidate_id = str(generation.get("generation_id") or "")
            parent_generation_id = str(
                generation.get("parent_generation_id") or parent_generation_id
            )
            recheck = lifecycle.get("recheck") or {}
            if not candidate_id or recheck.get("accepted") is not True:
                raise RuntimeError("现场候选缺少有效代际或累计 lock 门禁。")
            state["candidate_generation_id"] = candidate_id
            state["parent_generation_id"] = parent_generation_id
            self._write_state(
                run_dir,
                state,
                "CANDIDATE_ACCEPTED",
                candidate_generation_id=candidate_id,
                parent_generation_id=parent_generation_id,
                recheck_manifest=recheck.get("manifest"),
                metrics=recheck.get("metrics"),
                gates=recheck.get("gates"),
            )
            if not auto_deploy:
                return state

            if plan["deployment_target"] == "ascend310b":
                candidate_manifest = json.loads(
                    (
                        self.store._batch_dir(state["batch_id"])
                        / "training"
                        / state["job_id"]
                        / "candidate_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                context = {
                    "repo_root": ROOT,
                    "run_root": run_dir,
                    "run_id": state["run_id"],
                    "bundle": Path(bundle).expanduser().resolve(),
                    "batch_id": state["batch_id"],
                    "job_id": state["job_id"],
                    "candidate_weight": resolve_path(candidate_manifest["best_weight"]),
                    "candidate_generation_id": candidate_id,
                    "parent_generation_id": parent_generation_id,
                    "generation_registry": active_generation_registry(self.config),
                    "lock_root": self.store._batch_dir(state["batch_id"])
                    / "sealed_lock",
                    "new_class_ids": ",".join(
                        str(value) for value in contract["new_class_ids"]
                    ),
                    "target_fps": float(self.config["performance"]["target_api_fps"]),
                }
                external = AscendDeploymentExecutor(
                    load_ascend_deployment_spec(str(deployment_spec)),
                    run_dir,
                    context,
                    runner=self.deployment_runner,
                )
                deployment = external.execute()
                self._write_state(
                    run_dir,
                    state,
                    "ASCEND_GATES_PASSED",
                    ascend_deployment=deployment,
                )
                shadow_engine = None
                shadow_smoke = None
                performance = deployment["reports"]["fps_gate"]["payload"]
            else:
                shadow_engine, shadow_smoke = self.shadow_loader(
                    self.config, candidate_id
                )
                try:
                    performance = benchmark_shadow_engine(
                        shadow_engine,
                        _lock_images(self.store._batch_dir(state["batch_id"])),
                        probe_size=int(
                            fps_probe_size
                            or self.config["performance"]["batch_probe_size"]
                        ),
                        warmup_rounds=int(fps_warmup_rounds),
                        rounds=int(
                            fps_rounds
                            or self.config["performance"]["benchmark_rounds"]
                        ),
                        confidence=float(self.config["inference"]["confidence_default"]),
                    )
                    target_fps = float(self.config["performance"]["target_api_fps"])
                    performance["target_fps"] = target_fps
                    performance["passed"] = performance["median_fps"] >= target_fps
                    _atomic_json(run_dir / "candidate-fps.json", performance)
                    if not performance["passed"]:
                        self._write_state(
                            run_dir,
                            state,
                            "REJECTED",
                            rejection_reason="candidate_fps_gate_failed",
                            performance=performance,
                        )
                        return state
                    self._write_state(
                        run_dir,
                        state,
                        "FPS_GATE_PASSED",
                        performance=performance,
                        shadow_smoke=shadow_smoke,
                    )
                finally:
                    self._close_engine(shadow_engine)

            promotion = self.promoter(
                self.config, candidate_id, str(recheck["manifest"])
            )
            local_promoted = True
            accepted_lineage = self.lineage_freezer(
                self.config["incremental_workbench"],
                state["batch_id"],
                candidate_id,
                str(manifest["injection"]["dataset_fingerprint"]),
                manifest["files"],
            )
            self.store.update_training(
                state["batch_id"],
                state["job_id"],
                "PROMOTED",
                onsite_run_id=state["run_id"],
                promotion=promotion,
            )
            self._write_state(
                run_dir,
                state,
                "PROMOTED",
                promotion=promotion,
                performance=performance,
                accepted_lineage=rel_path(accepted_lineage)
                if accepted_lineage
                else None,
                final_class_count=plan["active_class_count_before"]
                + len(contract["new_class_ids"]),
            )
            return state
        except BaseException as exc:
            rollback_records: Dict[str, Any] = {}
            rollback_error = None
            cancelled = isinstance(exc, KeyboardInterrupt)
            if cancelled and state.get("batch_id") and state.get("job_id"):
                try:
                    cancellation = self.manager.cancel(
                        str(state["batch_id"]), str(state["job_id"])
                    )
                    rollback_records["training"] = {
                        "status": cancellation.get("status"),
                        "job_id": state["job_id"],
                    }
                except Exception as cancellation_exc:
                    rollback_records["training"] = {
                        "status": "cancel_request_failed",
                        "error": str(cancellation_exc),
                    }
            if local_promoted:
                try:
                    rollback_records["local"] = self.rollback_generation(
                        self.config, parent_generation_id
                    )
                except Exception as rollback_exc:  # pragma: no cover - severe recovery path
                    rollback_error = str(rollback_exc)
            if external is not None:
                try:
                    board_rollback = external.rollback()
                    if board_rollback is not None:
                        rollback_records["ascend310b"] = board_rollback
                except Exception as rollback_exc:  # pragma: no cover - severe recovery path
                    rollback_error = (
                        f"{rollback_error}; {rollback_exc}"
                        if rollback_error
                        else str(rollback_exc)
                    )
            failure_status = (
                "ROLLBACK_FAILED"
                if rollback_error
                else "CANCELLED"
                if cancelled
                else "FAILED"
            )
            self._write_state(
                run_dir,
                state,
                failure_status,
                error=str(exc),
                error_type=type(exc).__name__,
                rollback=rollback_records or None,
                rollback_error=rollback_error,
            )
            return state

    def status(self, run_id: str) -> Dict[str, Any]:
        safe = _safe_run_id(run_id)
        path = (self.run_root / safe / "state.json").resolve()
        if self.run_root not in path.parents or not path.is_file():
            raise KeyError(run_id)
        return json.loads(path.read_text(encoding="utf-8"))
