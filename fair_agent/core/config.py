from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Union

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "agent_pipeline.yaml"
ASCEND_CONFIG = ROOT / "configs" / "agent_pipeline_ascend310b.yaml"
AUTO_CONFIG = "auto"
CONFIG_SCHEMA_VERSION = 3
ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
SECRET_PARTS = {"password", "passwd", "secret", "token", "api_key", "private_key"}
X86_MACHINE_NAMES = {"x86_64", "amd64", "x64", "i386", "i486", "i586", "i686"}
ARM_MACHINE_NAMES = {"aarch64", "arm64", "arm", "armv7l", "armv8l"}
PROTECTED_PREFIXES = (
    "model.expected_sha256",
    "assets.checksums",
    "assets.generation_registry",
    "web.generation_registry",
    "web.generation_channel",
    "generation.registry",
    "generation.runtime_registry",
    "tensorrt_backend.engines",
    "tensorrt_backend.context_engine",
    "tensorrt_backend.validated",
    "tensorrt_backend.validation_report",
    "tensorrt_backend.expected_version",
    "tensorrt_backend.expected_compute_capability",
    "ascend_backend.build_manifest",
    "ascend_backend.build_manifest_sha256",
    "ascend_backend.validation_report",
    "ascend_backend.validation_report_sha256",
    "ascend_backend.validated",
    "ascend_backend.validation_candidate",
)
KNOWN_TOP_LEVEL = {
    "schema_version", "runtime", "web", "inference", "routing", "decoding",
    "storage", "ui", "performance", "native_backend", "ascend_backend", "tensorrt_backend", "model", "assets", "automation",
    "generation", "submission", "blackboard", "detector", "functional_models", "inputs",
    "incremental", "decision",
    "logging", "incremental_workbench", "gates", "incremental_guardian",
}
KNOWN_SECTION_KEYS = {
    "runtime": {"mode", "local_python", "default_device", "server_host", "server_port"},
    "web": {"generation_registry", "generation_channel", "functional_registry"},
    "inference": {"backend", "imgsz", "specialist_imgsz", "iou", "max_det", "batch_size", "confidence_min", "confidence_max", "confidence_default", "warmup_iterations", "warmup_batch_size", "warmup_width", "warmup_height", "preload_specialists", "quantize", "cudnn_benchmark", "compile"},
    "routing": {
        "incremental_enabled", "require_acceptance_passed", "consensus_iou", "fusion_iou",
        "max_specialists_per_image", "conflict_iou", "conflict_incremental_coverage", "conflict_base_confidence",
        "specialist_margin", "preserve_base_class_owners",
        "detection_evidence_weight", "context_evidence_weight",
        "neutral_context_score", "default_routing_prior",
        "parallel_model_execution", "parallel_context_execution", "parallel_context_batch_execution", "max_model_workers",
    },
    "decoding": {"backend", "workers", "opencv_threads"},
    "storage": {"max_items", "ttl_seconds", "max_bytes"},
    "ui": {"history_limit", "result_cache_limit", "health_poll_ms", "toast_duration_ms", "default_view"},
    "performance": {
        "target_api_fps", "target_p95_ms", "benchmark_rounds", "warmup_requests",
        "benchmark_split", "report_root", "concurrent_requests", "batch_probe_size",
        "auto_start_server", "server_start_timeout_seconds", "request_timeout_seconds",
    },
    "native_backend": {"library", "base_engine", "engines", "context_engine", "precision", "require_exact_gpu", "validated"},
    "ascend_backend": {
        "device_id", "soc_version", "cann_version", "precision", "execution_mode",
        "model_layout",
        "encoded_preprocessing", "context_mode", "memory_mode", "schedule_mode", "detailed_event_timing",
        "submit_order", "collect_order", "stream_priorities",
        "validated", "validation_candidate", "validation_report",
        "validation_report_sha256", "build_manifest", "build_manifest_sha256",
        "models", "context_model", "dvpp_scene_resize_stages",
    },
    "tensorrt_backend": {
        "expected_version", "expected_compute_capability", "require_exact_gpu", "validated",
        "precision", "workspace_gib", "dynamic", "minimum_spatial_size", "engines", "context_engine", "export",
        "validation",
        "validation_report", "int8_calibration", "mixed_precision",
    },
    "generation": {
        "registry", "runtime_registry", "recheck_lock_split", "report_root", "candidate_id",
        "calibrated_threshold", "auto_promote", "shadow_smoke_images",
    },
    "gates": {"official_hard", "advisory"},
    "incremental_guardian": {"enabled", "dynamic_confusion", "recovery_actions"},
    "model": {"weights", "expected_sha256"},
    "logging": {"root", "max_file_bytes", "retained_files", "request_bodies"},
    "incremental_workbench": {
        "root", "max_archive_bytes", "max_extracted_bytes", "max_extracted_files",
        "max_image_pixels", "allowed_image_extensions", "validation_fraction",
        "minimum_images", "preview_limit", "job_log_tail_lines", "poll_interval_ms", "training",
        "lock_fraction", "split_seed", "lineage", "lifecycle",
    },
}


def resolve_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def detect_host_architecture(machine: str | None = None) -> str:
    """Normalize the current host to the two deployment architectures we ship."""

    detected = str(machine if machine is not None else platform.machine()).strip().lower()
    normalized = detected.replace("-", "_")
    if normalized in X86_MACHINE_NAMES:
        return "x86"
    if normalized in ARM_MACHINE_NAMES or normalized.startswith("armv"):
        return "arm"
    raise ValueError(
        f"不支持的设备架构：{detected or 'unknown'}；"
        "当前发布仅支持 x86/x86_64 与 ARM/aarch64。"
    )


def _backend_runtime_profile(backend: str) -> Dict[str, str]:
    if backend == "ascend_acl":
        return {"device_family": "ascend_310b", "model_format": "om"}
    if backend in {"tensorrt_engine", "tensorrt_native"}:
        return {"device_family": "x86_cuda", "model_format": "engine"}
    return {"device_family": "x86_cuda", "model_format": "pt"}


def select_runtime_config(
    path: Union[str, Path, None] = None,
    *,
    machine: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Select the x86/CUDA or ARM/Ascend config without probing accelerators.

    An explicit path wins. ``AGILE_AGENT_CONFIG`` remains the deployment-level
    override for default/auto callers. Architecture selection is deterministic
    and intentionally never falls back from one model family to the other.
    """

    environment = os.environ if environ is None else environ
    raw_machine = str(machine if machine is not None else platform.machine()).strip()
    architecture = detect_host_architecture(raw_machine)
    raw_path = "" if path is None else str(path).strip()
    auto_requested = not raw_path or raw_path.lower() == AUTO_CONFIG
    legacy_default = raw_path == str(DEFAULT_CONFIG)
    configured = str(environment.get("AGILE_AGENT_CONFIG") or "").strip()

    if configured and (auto_requested or legacy_default):
        selected = resolve_path(configured)
        source = "environment"
    elif auto_requested and architecture == "arm" and environment.get(
        "AGILE_AGENT_ASCEND_RELEASE"
    ):
        selected = (
            Path(str(environment["AGILE_AGENT_ASCEND_RELEASE"])).expanduser()
            / "configs"
            / "agent_pipeline_ascend310b.yaml"
        )
        source = "ascend_release"
    elif auto_requested:
        selected = ASCEND_CONFIG if architecture == "arm" else DEFAULT_CONFIG
        source = "architecture"
    else:
        selected = resolve_path(raw_path)
        source = "explicit"

    return {
        "config_path": selected,
        "machine": raw_machine or "unknown",
        "architecture": architecture,
        "selection": source,
        "automatic": source in {"architecture", "ascend_release"},
    }


def runtime_platform_info(
    config: Mapping[str, Any], machine: str | None = None
) -> Dict[str, Any]:
    """Describe the detected host and the model family selected by a config."""

    stored = config.get("_runtime_platform")
    metadata = dict(stored) if isinstance(stored, Mapping) else {}
    raw_machine = str(machine if machine is not None else metadata.get("machine") or platform.machine())
    architecture = detect_host_architecture(raw_machine)
    backend = str((config.get("inference") or {}).get("backend") or "unknown")
    profile = _backend_runtime_profile(backend)
    expected_family = "ascend_310b" if architecture == "arm" else "x86_cuda"
    return {
        **metadata,
        "machine": raw_machine,
        "architecture": architecture,
        "backend": backend,
        **profile,
        "architecture_match": profile["device_family"] == expected_family,
    }


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _expand_environment(value: Any, path: str = "") -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item, f"{path}.{key}" if path else str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, str):
        match = ENV_PATTERN.fullmatch(value)
        if match:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"配置引用的环境变量不存在：{path} -> {name}")
            return os.environ[name]
    return value


def parse_yaml_value(value: str) -> Any:
    parsed = yaml.safe_load(value)
    if isinstance(parsed, (dict, list)):
        return parsed
    return parsed


def split_key(key: str) -> List[str]:
    parts = [part for part in key.split(".") if part]
    if not parts or len(parts) != len(key.split(".")):
        raise ValueError(f"配置键路径无效：{key}")
    return parts


def get_key(data: Mapping[str, Any], key: str) -> Any:
    current: Any = data
    for part in split_key(key):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(key)
        current = current[part]
    return current


def set_key(data: MutableMapping[str, Any], key: str, value: Any, create: bool = False) -> None:
    parts = split_key(key)
    current: MutableMapping[str, Any] = data
    for part in parts[:-1]:
        child = current.get(part)
        if child is None and create:
            child = {}
            current[part] = child
        if not isinstance(child, MutableMapping):
            raise KeyError(key)
        current = child
    if not create and parts[-1] not in current:
        raise KeyError(key)
    current[parts[-1]] = value


def unset_key(data: MutableMapping[str, Any], key: str) -> None:
    parts = split_key(key)
    current: MutableMapping[str, Any] = data
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            raise KeyError(key)
        current = child
    if parts[-1] not in current:
        raise KeyError(key)
    del current[parts[-1]]


def is_protected_key(key: str) -> bool:
    return any(key == prefix or key.startswith(prefix + ".") for prefix in PROTECTED_PREFIXES)


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str] | None) -> Dict[str, Any]:
    result = copy.deepcopy(dict(config))
    for raw in overrides or []:
        if "=" not in raw:
            raise ValueError(f"CLI覆盖必须使用 key=value：{raw}")
        key, value = raw.split("=", 1)
        set_key(result, key, parse_yaml_value(value), create=False)
    return result


def config_sha256(config: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if not str(key).startswith("_")}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def redact_config(value: Any, path: str = "") -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            output[key] = "***" if str(key).lower() in SECRET_PARTS else redact_config(item, child_path)
        return output
    if isinstance(value, list):
        return [redact_config(item, path) for item in value]
    return value


def _require_mapping(config: Mapping[str, Any], key: str, errors: List[str]) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} 必须是映射")
        return {}
    return value


def _number(section: Mapping[str, Any], key: str, errors: List[str], minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        value = float(section[key])
    except (KeyError, TypeError, ValueError):
        errors.append(f"缺少或非法数值参数：{key}")
        return 0.0
    if minimum is not None and value < minimum:
        errors.append(f"{key} 不得小于 {minimum}")
    if maximum is not None and value > maximum:
        errors.append(f"{key} 不得大于 {maximum}")
    return value


def validate_config(
    config: Dict[str, Any],
    *,
    allow_unverified_tensorrt_hashes: bool = False,
) -> None:
    errors: List[str] = []
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 {CONFIG_SCHEMA_VERSION}")
    unknown = sorted(
        set(config)
        - KNOWN_TOP_LEVEL
        - {"_config_path", "_config_sha256", "_config_overrides", "_runtime_platform"}
    )
    if unknown:
        errors.append("未知顶层配置：" + ", ".join(unknown))
    for section_name, allowed_keys in KNOWN_SECTION_KEYS.items():
        section = config.get(section_name)
        if isinstance(section, Mapping):
            unknown_keys = sorted(set(section) - allowed_keys)
            if unknown_keys:
                errors.append(f"{section_name}包含未知字段：" + ", ".join(unknown_keys))

    runtime = _require_mapping(config, "runtime", errors)
    default_device = str(runtime.get("default_device", ""))
    if not default_device.isdigit() or int(default_device) < 0:
        errors.append("runtime.default_device 必须是非负 GPU 编号")
    if runtime.get("mode") != "local":
        errors.append("runtime.mode 必须为 local")
    server_host = str(runtime.get("server_host", ""))
    if server_host not in {"127.0.0.1", "localhost", "::1"}:
        errors.append("runtime.server_host 必须是本机回环地址")
    _number(runtime, "server_port", errors, 1, 65535)

    web = _require_mapping(config, "web", errors)
    for key in ("generation_registry", "generation_channel", "functional_registry"):
        if not web.get(key):
            errors.append(f"web.{key}不能为空")

    inference = _require_mapping(config, "inference", errors)
    if inference.get("backend") not in {"ultralytics_cuda", "tensorrt_engine", "tensorrt_native", "ascend_acl"}:
        errors.append("inference.backend 必须为 ultralytics_cuda、tensorrt_engine、tensorrt_native 或 ascend_acl")
    _number(inference, "imgsz", errors, 32)
    _number(inference, "specialist_imgsz", errors, 32)
    _number(inference, "iou", errors, 0.01, 1.0)
    _number(inference, "max_det", errors, 1)
    _number(inference, "batch_size", errors, 1)
    _number(inference, "warmup_iterations", errors, 1)
    _number(inference, "warmup_batch_size", errors, 1)
    _number(inference, "warmup_width", errors, 1)
    _number(inference, "warmup_height", errors, 1)
    confidence_min = _number(inference, "confidence_min", errors, 0.00001, 1.0)
    confidence_max = _number(inference, "confidence_max", errors, 0.01, 1.0)
    confidence_default = _number(
        inference, "confidence_default", errors, 0.00001, 1.0
    )
    if not confidence_min <= confidence_default <= confidence_max:
        errors.append("inference.confidence_default 必须位于最小值与最大值之间")
    if not all(isinstance(inference.get(key), bool) for key in ("preload_specialists", "cudnn_benchmark", "compile")):
        errors.append("inference.preload_specialists、cudnn_benchmark与compile必须为布尔值")
    if inference.get("quantize") not in {None, 16}:
        errors.append("inference.quantize必须为null或16")

    routing = _require_mapping(config, "routing", errors)
    for key in (
        "consensus_iou", "fusion_iou", "conflict_iou", "conflict_base_confidence",
        "specialist_margin", "detection_evidence_weight", "context_evidence_weight",
        "neutral_context_score", "default_routing_prior",
    ):
        _number(routing, key, errors, 0.0, 1.0)
    if routing.get("conflict_incremental_coverage") is not None:
        _number(routing, "conflict_incremental_coverage", errors, 0.0, 1.0)
    if abs(float(routing.get("detection_evidence_weight", 0)) + float(routing.get("context_evidence_weight", 0)) - 1.0) > 1e-9:
        errors.append("routing的检测证据权重与上下文证据权重之和必须为1")
    _number(routing, "max_specialists_per_image", errors, 1)
    _number(routing, "max_model_workers", errors, 1)
    if not all(isinstance(routing.get(key), bool) for key in (
        "incremental_enabled", "require_acceptance_passed", "preserve_base_class_owners",
    )):
        errors.append("routing中的开关必须为布尔值")
    if not all(isinstance(routing.get(key), bool) for key in ("parallel_model_execution", "parallel_context_execution", "parallel_context_batch_execution")):
        errors.append("routing中的并行执行开关必须为布尔值")
    decoding = _require_mapping(config, "decoding", errors)
    if decoding.get("backend") not in {"pillow", "opencv"}:
        errors.append("decoding.backend必须为pillow或opencv")
    _number(decoding, "workers", errors, 1, 32)
    _number(decoding, "opencv_threads", errors, 0, 32)
    storage = _require_mapping(config, "storage", errors)
    for key in ("max_items", "ttl_seconds", "max_bytes"):
        _number(storage, key, errors, 1)
    logging = _require_mapping(config, "logging", errors)
    if not logging.get("root"):
        errors.append("logging.root不能为空")
    _number(logging, "max_file_bytes", errors, 1024)
    _number(logging, "retained_files", errors, 1, 100)
    if logging.get("request_bodies") is not False:
        errors.append("logging.request_bodies必须为false，禁止记录上传内容")
    workbench = _require_mapping(config, "incremental_workbench", errors)
    if not workbench.get("root"):
        errors.append("incremental_workbench.root不能为空")
    for key in ("max_archive_bytes", "max_extracted_bytes", "max_extracted_files", "max_image_pixels"):
        _number(workbench, key, errors, 1)
    extensions = workbench.get("allowed_image_extensions")
    if not isinstance(extensions, list) or not extensions or not all(
        isinstance(value, str) and value.startswith(".") for value in extensions
    ):
        errors.append("incremental_workbench.allowed_image_extensions必须是带点号的非空扩展名列表")
    _number(workbench, "validation_fraction", errors, 0.01, 0.50)
    if "lock_fraction" in workbench:
        _number(workbench, "lock_fraction", errors, 0.01, 0.50)
    if "split_seed" in workbench:
        _number(workbench, "split_seed", errors, 0)
    _number(workbench, "minimum_images", errors, 2)
    _number(workbench, "preview_limit", errors, 1, 100)
    _number(workbench, "job_log_tail_lines", errors, 10, 2000)
    _number(workbench, "poll_interval_ms", errors, 500, 60000)
    workbench_training = workbench.get("training")
    if not isinstance(workbench_training, Mapping):
        errors.append("incremental_workbench.training必须是映射")
    else:
        required_training = {
            "python", "initial_weights", "device", "imgsz", "batch", "epochs", "patience",
            "workers", "optimizer", "lr0", "seed", "deterministic", "amp",
        }
        missing_training = sorted(required_training - set(workbench_training))
        if missing_training:
            errors.append("incremental_workbench.training缺少：" + ", ".join(missing_training))
        if workbench_training.get("device") is None or not str(workbench_training.get("device")).isdigit():
            errors.append("incremental_workbench.training.device必须是GPU编号")
        for key in ("imgsz", "batch", "epochs", "patience", "workers", "seed"):
            _number(workbench_training, key, errors, 0 if key in {"patience", "workers"} else 1)
        _number(workbench_training, "lr0", errors, 0.0000001, 1.0)
        if not all(isinstance(workbench_training.get(key), bool) for key in ("deterministic", "amp")):
            errors.append("incremental_workbench.training.deterministic与amp必须为布尔值")
    lineage = workbench.get("lineage")
    if not isinstance(lineage, Mapping):
        errors.append("incremental_workbench.lineage必须是映射")
    else:
        required_lineage = {"required", "root", "base_manifest", "base_split_manifest", "auto_initialize_base", "cache_roots"}
        missing_lineage = sorted(required_lineage - set(lineage))
        if missing_lineage:
            errors.append("incremental_workbench.lineage缺少：" + ", ".join(missing_lineage))
        if not isinstance(lineage.get("required"), bool) or not isinstance(lineage.get("auto_initialize_base"), bool):
            errors.append("incremental_workbench.lineage required/auto_initialize_base必须为布尔值")
        if not isinstance(lineage.get("cache_roots"), list):
            errors.append("incremental_workbench.lineage.cache_roots必须为列表")
    lifecycle = workbench.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        errors.append("incremental_workbench.lifecycle必须是映射")
    else:
        if not isinstance(lifecycle.get("auto_continue"), bool):
            errors.append("incremental_workbench.lifecycle.auto_continue必须为布尔值")
        _number(lifecycle, "calibration_target_precision", errors, 0.0, 1.0)
        threshold_min = _number(lifecycle, "threshold_min", errors, 0.0, 1.0)
        threshold_max = _number(lifecycle, "threshold_max", errors, 0.0, 1.0)
        _number(lifecycle, "threshold_step", errors, 0.0001, 1.0)
        if threshold_min >= threshold_max:
            errors.append("增量阈值扫描下界必须小于上界")
    ui = _require_mapping(config, "ui", errors)
    for key in ("history_limit", "result_cache_limit", "health_poll_ms", "toast_duration_ms"):
        _number(ui, key, errors, 1)
    performance = _require_mapping(config, "performance", errors)
    if not isinstance(performance.get("auto_start_server"), bool):
        errors.append("performance.auto_start_server必须为布尔值")
    _number(performance, "target_api_fps", errors, 1)
    _number(performance, "target_p95_ms", errors, 1)
    _number(performance, "server_start_timeout_seconds", errors, 1)
    _number(performance, "request_timeout_seconds", errors, 1)
    _number(performance, "benchmark_rounds", errors, 1)
    _number(performance, "warmup_requests", errors, 0)
    _number(performance, "concurrent_requests", errors, 1)
    _number(performance, "batch_probe_size", errors, 1)
    if not performance.get("benchmark_split") or not performance.get("report_root"):
        errors.append("performance必须声明benchmark_split和report_root")

    generation = _require_mapping(config, "generation", errors)
    if not generation.get("registry") or not generation.get("runtime_registry") or not generation.get("recheck_lock_split") or not generation.get("report_root"):
        errors.append("generation必须声明registry、runtime_registry、recheck_lock_split和report_root")
    if not isinstance(generation.get("auto_promote"), bool):
        errors.append("generation.auto_promote必须为布尔值")
    _number(generation, "shadow_smoke_images", errors, 1, 32)
    _number(generation, "calibrated_threshold", errors, 0.01, 1.0)
    gates = _require_mapping(config, "gates", errors)
    official_hard = gates.get("official_hard")
    advisory = gates.get("advisory")
    if not isinstance(official_hard, Mapping):
        errors.append("gates.official_hard必须是映射")
    else:
        unknown_hard = sorted(set(official_hard) - {
            "base_map50_min", "new_map50_min", "krr_min", "old_data_overlap_max",
        })
        if unknown_hard:
            errors.append("gates.official_hard包含未知字段：" + ", ".join(unknown_hard))
        for key in ("base_map50_min", "new_map50_min", "krr_min"):
            _number(official_hard, key, errors, 0.0, 1.0)
        _number(official_hard, "old_data_overlap_max", errors, 0.0)
    if not isinstance(advisory, Mapping):
        errors.append("gates.advisory必须是映射")
    else:
        unknown_advisory = sorted(set(advisory) - {
            "cumulative_map50_min", "lock_precision_min",
            "false_activation_rate_max", "latency_proxy_ms_max",
        })
        if unknown_advisory:
            errors.append("gates.advisory包含未知字段：" + ", ".join(unknown_advisory))
        for key in ("cumulative_map50_min", "lock_precision_min", "false_activation_rate_max"):
            _number(advisory, key, errors, 0.0, 1.0)
        _number(advisory, "latency_proxy_ms_max", errors, 0.0)

    guardian = _require_mapping(config, "incremental_guardian", errors)
    if not isinstance(guardian.get("enabled"), bool):
        errors.append("incremental_guardian.enabled必须为布尔值")
    confusion = guardian.get("dynamic_confusion")
    if not isinstance(confusion, Mapping):
        errors.append("incremental_guardian.dynamic_confusion必须是映射")
    else:
        unknown_confusion = sorted(set(confusion) - {
            "enabled", "match_iou", "min_support", "specialist_deficit_padding",
            "specialist_deficit_cap",
        })
        if unknown_confusion:
            errors.append("incremental_guardian.dynamic_confusion包含未知字段：" + ", ".join(unknown_confusion))
        if not isinstance(confusion.get("enabled"), bool):
            errors.append("incremental_guardian.dynamic_confusion.enabled必须为布尔值")
        _number(confusion, "match_iou", errors, 0.01, 1.0)
        _number(confusion, "min_support", errors, 1.0)
        _number(confusion, "specialist_deficit_padding", errors, 0.0, 1.0)
        _number(confusion, "specialist_deficit_cap", errors, 0.0, 1.0)
    recovery_actions = guardian.get("recovery_actions")
    if not isinstance(recovery_actions, Mapping) or not recovery_actions:
        errors.append("incremental_guardian.recovery_actions必须是非空映射")
    elif any(
        not isinstance(code, str)
        or not isinstance(actions, list)
        or not actions
        or any(not isinstance(action, str) or not action for action in actions)
        for code, actions in recovery_actions.items()
    ):
        errors.append("incremental_guardian.recovery_actions必须映射到非空动作字符串列表")

    native = _require_mapping(config, "native_backend", errors)
    if native.get("precision") not in {"fp16", "fp32", "int8"}:
        errors.append("native_backend.precision 非法")
    if not isinstance(native.get("require_exact_gpu"), bool) or not isinstance(native.get("validated"), bool):
        errors.append("native_backend.require_exact_gpu与validated必须为布尔值")
    native_engines = native.get("engines")
    if not isinstance(native_engines, Mapping) or not native_engines:
        errors.append("native_backend.engines必须是非空映射")
    else:
        for source, entry in native_engines.items():
            if not isinstance(source, str) or not isinstance(entry, Mapping) or not entry.get("path"):
                errors.append("native_backend.engines条目非法")
                continue
            unknown = sorted(set(entry) - {"path", "sha256"})
            if unknown:
                errors.append(f"native_backend.engines.{source}包含未知字段：" + ", ".join(unknown))
            digest = entry.get("sha256")
            if native.get("validated") is True and (
                not isinstance(digest, str) or len(digest) != 64
            ):
                errors.append(f"已验收native engine缺少SHA256：{source}")
    if inference.get("backend") == "tensorrt_native":
        for key in ("library", "base_engine", "context_engine"):
            if not native.get(key):
                errors.append(f"TensorRT后端缺少 native_backend.{key}")

    ascend = _require_mapping(config, "ascend_backend", errors)
    model_layout = ascend.get("model_layout")
    if model_layout != "independent_yolo26_e2e_v1":
        errors.append("ascend_backend.model_layout必须为independent_yolo26_e2e_v1")
    if not str(ascend.get("device_id", "")).isdigit():
        errors.append("ascend_backend.device_id必须是非负设备编号")
    if ascend.get("soc_version") != "Ascend310B1":
        errors.append("ascend_backend.soc_version必须为Ascend310B1")
    if not isinstance(ascend.get("validated"), bool):
        errors.append("ascend_backend.validated必须为布尔值")
    if not isinstance(ascend.get("validation_candidate", False), bool):
        errors.append("ascend_backend.validation_candidate必须为布尔值")
    if ascend.get("precision") not in {"mixed_float16", "origin"}:
        errors.append("ascend_backend.precision非法")
    if ascend.get("execution_mode") != "async_stream":
        errors.append("ascend_backend.execution_mode必须为async_stream")
    if ascend.get("encoded_preprocessing") != "dvpp":
        errors.append("ascend_backend.encoded_preprocessing必须为dvpp")
    if ascend.get("context_mode") != "model":
        errors.append("ascend_backend.context_mode必须为model")
    if ascend.get("memory_mode", "pageable") not in {"pageable", "pinned"}:
        errors.append("ascend_backend.memory_mode非法")
    if ascend.get("schedule_mode") != "unified_enqueue":
        errors.append("ascend_backend.schedule_mode必须为unified_enqueue")
    if not isinstance(ascend.get("detailed_event_timing", True), bool):
        errors.append("ascend_backend.detailed_event_timing必须为布尔值")
    model_roles = {"scene", "base", "specialist"}
    for key in ("submit_order", "collect_order"):
        order = ascend.get(key, ["scene", "base", "specialist"])
        if (
            not isinstance(order, list)
            or len(order) != len(model_roles)
            or any(not isinstance(value, str) for value in order)
            or set(order) != model_roles
        ):
            errors.append(
                f"ascend_backend.{key}必须是scene/base/specialist的无重复全排列"
            )
    stream_priorities = ascend.get("stream_priorities")
    if stream_priorities is not None and (
        not isinstance(stream_priorities, Mapping)
        or set(stream_priorities) != model_roles
        or any(
            value not in {"high", "normal", "low"}
            for value in stream_priorities.values()
        )
    ):
        errors.append(
            "ascend_backend.stream_priorities必须完整映射scene/base/specialist到high/normal/low"
        )
    scene_resize_stages = ascend.get("dvpp_scene_resize_stages", [])
    if not isinstance(scene_resize_stages, list) or len(scene_resize_stages) > 4:
        errors.append("ascend_backend.dvpp_scene_resize_stages必须是最多4级的尺寸列表")
    else:
        for index, stage in enumerate(scene_resize_stages):
            if (
                not isinstance(stage, list)
                or len(stage) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 16
                    or value > 4096
                    or value % 2
                    for value in stage
                )
            ):
                errors.append(
                    f"ascend_backend.dvpp_scene_resize_stages[{index}]必须是16-4096范围内的偶数宽高"
                )
    if (
        ascend.get("memory_mode", "pageable") == "pinned"
        and ascend.get("execution_mode", "synchronous") != "async_stream"
    ):
        errors.append("Ascend锁页内存要求async_stream执行模式")
    build_manifest = ascend.get("build_manifest")
    build_manifest_sha256 = ascend.get("build_manifest_sha256")
    validation_report = ascend.get("validation_report")
    validation_report_sha256 = ascend.get("validation_report_sha256")
    if bool(build_manifest) != bool(build_manifest_sha256):
        errors.append("ascend_backend.build_manifest与SHA256必须同时配置")
    if build_manifest_sha256 and (
        not isinstance(build_manifest_sha256, str) or len(build_manifest_sha256) != 64
    ):
        errors.append("ascend_backend.build_manifest_sha256非法")
    if validation_report_sha256 and not validation_report:
        errors.append("ascend_backend.validation_report_sha256缺少对应报告")
    if validation_report_sha256 and (
        not isinstance(validation_report_sha256, str)
        or len(validation_report_sha256) != 64
    ):
        errors.append("ascend_backend.validation_report_sha256非法")
    if ascend.get("validated") is True and ascend.get("encoded_preprocessing") == "dvpp":
        if not build_manifest or not build_manifest_sha256:
            errors.append("已验收Ascend DVPP配置缺少构建清单及SHA256")
        if not validation_report or not validation_report_sha256:
            errors.append("已验收Ascend DVPP配置缺少验证报告及SHA256")
    if ascend.get("validation_candidate") is True:
        if ascend.get("validated") is not False:
            errors.append("Ascend候选验证模式要求validated=false")
        if ascend.get("encoded_preprocessing") != "dvpp":
            errors.append("Ascend候选验证模式仅用于DVPP候选")
        if not build_manifest or not build_manifest_sha256:
            errors.append("Ascend候选验证模式缺少构建清单及SHA256")
    ascend_models = ascend.get("models")
    if not isinstance(ascend_models, Mapping) or not ascend_models:
        errors.append("ascend_backend.models必须是非空映射")
    else:
        for source, entry in ascend_models.items():
            if not isinstance(source, str) or not isinstance(entry, Mapping) or not entry.get("path"):
                errors.append("ascend_backend.models条目非法")
                continue
            unknown = sorted(
                set(entry) - {"path", "sha256", "output_contract", "max_det", "class_count"}
            )
            if unknown:
                errors.append(f"ascend_backend.models.{source}包含未知字段：" + ", ".join(unknown))
            output_contract = entry.get("output_contract")
            if output_contract != "yolo26_e2e_v1":
                errors.append(
                    f"ascend_backend.models.{source}.output_contract必须为yolo26_e2e_v1"
                )
            contract_fields = {"max_det", "class_count"}
            configured_contract_fields = contract_fields & set(entry)
            if output_contract == "yolo26_e2e_v1":
                required_fields = {"max_det", "class_count"}
                if configured_contract_fields != required_fields:
                    errors.append(
                        f"ascend_backend.models.{source}.yolo26_e2e_v1缺少固定输出参数"
                    )
                for name in required_fields:
                    value = entry.get(name)
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        errors.append(f"ascend_backend.models.{source}.{name}非法")
            digest = entry.get("sha256")
            if (ascend.get("validated") is True or ascend.get("validation_candidate") is True) and (
                not isinstance(digest, str) or len(digest) != 64
            ):
                errors.append(f"已验收Ascend OM缺少SHA256：{source}")
        e2e_entries = sum(
            isinstance(entry, Mapping)
            and entry.get("output_contract") == "yolo26_e2e_v1"
            for entry in ascend_models.values()
        )
        if e2e_entries != 2 or len(ascend_models) != 2:
            errors.append("independent_yolo26_e2e_v1要求恰好两个yolo26_e2e_v1 OM")
        if inference.get("imgsz") != inference.get("specialist_imgsz"):
            errors.append("YOLO26 E2E独立布局要求Base与Specialist推理分辨率一致")
    context_model = ascend.get("context_model")
    if not isinstance(context_model, Mapping) or not context_model.get("path"):
        errors.append("ascend_backend.context_model非法")
    elif (ascend.get("validated") is True or ascend.get("validation_candidate") is True) and (
        not isinstance(context_model.get("sha256"), str)
        or len(str(context_model.get("sha256"))) != 64
    ):
        errors.append("已验收Ascend context OM缺少SHA256")
    if (
        inference.get("backend") == "ascend_acl"
        and ascend.get("validated") is not True
        and ascend.get("validation_candidate") is not True
    ):
        errors.append("Ascend后端必须先完成golden验收")

    tensorrt_backend = _require_mapping(config, "tensorrt_backend", errors)
    if tensorrt_backend.get("precision") not in {"fp16", "fp32", "int8"}:
        errors.append("tensorrt_backend.precision 非法")
    if not isinstance(tensorrt_backend.get("require_exact_gpu"), bool) or not isinstance(
        tensorrt_backend.get("validated"), bool
    ):
        errors.append("tensorrt_backend.require_exact_gpu与validated必须为布尔值")
    if not isinstance(tensorrt_backend.get("dynamic"), bool):
        errors.append("tensorrt_backend.dynamic必须为布尔值")
    minimum_spatial_size = _number(tensorrt_backend, "minimum_spatial_size", errors, 32)
    if int(minimum_spatial_size) % 32:
        errors.append("tensorrt_backend.minimum_spatial_size必须是32的倍数")
    mixed_precision = tensorrt_backend.get("mixed_precision")
    if not isinstance(mixed_precision, Mapping):
        errors.append("tensorrt_backend.mixed_precision必须是映射")
    else:
        unknown_mixed = sorted(set(mixed_precision) - {
            "enabled", "constraint", "fp16_layer_patterns", "minimum_matched_layers",
        })
        if unknown_mixed:
            errors.append("tensorrt_backend.mixed_precision包含未知字段：" + ", ".join(unknown_mixed))
        if not isinstance(mixed_precision.get("enabled"), bool):
            errors.append("tensorrt_backend.mixed_precision.enabled必须为布尔值")
        if mixed_precision.get("constraint") not in {"obey", "prefer"}:
            errors.append("tensorrt_backend.mixed_precision.constraint必须为obey或prefer")
        patterns = mixed_precision.get("fp16_layer_patterns")
        if not isinstance(patterns, list) or not patterns or not all(
            isinstance(pattern, str) and pattern for pattern in patterns
        ):
            errors.append("tensorrt_backend.mixed_precision.fp16_layer_patterns必须是非空字符串列表")
        _number(mixed_precision, "minimum_matched_layers", errors, 1)
        if mixed_precision.get("enabled") is True and tensorrt_backend.get("precision") != "int8":
            errors.append("混合精度层约束仅在precision=int8时允许启用")
    validation = tensorrt_backend.get("validation")
    if not isinstance(validation, Mapping):
        errors.append("tensorrt_backend.validation必须是映射")
    else:
        unknown_validation = sorted(set(validation) - {
            "max_overall_map50_delta_warning", "max_per_class_map50_delta_warning",
            "evaluation_batch_size",
        })
        if unknown_validation:
            errors.append("tensorrt_backend.validation包含未知字段：" + ", ".join(unknown_validation))
        _number(validation, "max_overall_map50_delta_warning", errors, 0.0, 1.0)
        _number(validation, "max_per_class_map50_delta_warning", errors, 0.0, 1.0)
        evaluation_batch_size = _number(validation, "evaluation_batch_size", errors, 1)
        engine_batches = [
            int(entry.get("batch_size", 0))
            for entry in (tensorrt_backend.get("engines") or {}).values()
            if isinstance(entry, Mapping)
        ]
        if engine_batches and evaluation_batch_size > min(engine_batches):
            errors.append("TensorRT精度评测batch不得超过任一检测engine的最大batch")
    int8_calibration = tensorrt_backend.get("int8_calibration")
    if not isinstance(int8_calibration, Mapping):
        errors.append("tensorrt_backend.int8_calibration必须是映射")
    else:
        unknown_int8 = sorted(set(int8_calibration) - {
            "enabled", "auto_quantize_incremental", "representative_split", "cache_root",
            "threshold_split", "batch_size", "max_images", "minimum_images_per_class", "seed",
        })
        if unknown_int8:
            errors.append("tensorrt_backend.int8_calibration包含未知字段：" + ", ".join(unknown_int8))
        if not isinstance(int8_calibration.get("enabled"), bool) or not isinstance(
            int8_calibration.get("auto_quantize_incremental"), bool
        ):
            errors.append("INT8 enabled/auto_quantize_incremental必须为布尔值")
        if (
            not int8_calibration.get("representative_split")
            or not int8_calibration.get("threshold_split")
            or not int8_calibration.get("cache_root")
        ):
            errors.append("INT8校准必须声明representative_split、threshold_split和cache_root")
        calibration_batch = _number(int8_calibration, "batch_size", errors, 1)
        calibration_images = _number(int8_calibration, "max_images", errors, 1)
        _number(int8_calibration, "minimum_images_per_class", errors, 1)
        _number(int8_calibration, "seed", errors, 0)
        if calibration_batch > calibration_images:
            errors.append("INT8校准batch不得大于最大校准图像数")
        if tensorrt_backend.get("precision") == "int8" and int8_calibration.get("enabled") is not True:
            errors.append("precision=int8时必须启用INT8校准")
    if tensorrt_backend.get("validated") is True and not tensorrt_backend.get("validation_report"):
        errors.append("已启用TensorRT后端必须登记精度与性能验收报告")
    inactive_tensorrt_profile = (
        inference.get("backend") != "tensorrt_engine"
        and tensorrt_backend.get("validated") is False
    )
    allow_missing_engine_hashes = (
        allow_unverified_tensorrt_hashes or inactive_tensorrt_profile
    ) and tensorrt_backend.get("validated") is False
    _number(tensorrt_backend, "workspace_gib", errors, 0.1)
    require_tensorrt_identity = (
        allow_unverified_tensorrt_hashes
        or inference.get("backend") == "tensorrt_engine"
        or tensorrt_backend.get("validated") is True
    )
    if require_tensorrt_identity and (
        not tensorrt_backend.get("expected_version")
        or not tensorrt_backend.get("expected_compute_capability")
    ):
        errors.append("tensorrt_backend必须声明TensorRT版本与GPU计算能力")
    export = tensorrt_backend.get("export")
    if not isinstance(export, Mapping):
        errors.append("tensorrt_backend.export必须是映射")
    else:
        unknown_export = sorted(set(export) - {
            "device", "overwrite", "onnx_opset", "simplify", "context_checkpoint",
        })
        if unknown_export:
            errors.append("tensorrt_backend.export包含未知字段：" + ", ".join(unknown_export))
        if not str(export.get("device", "")).isdigit():
            errors.append("tensorrt_backend.export.device必须是非负GPU编号")
        if not isinstance(export.get("overwrite"), bool) or not isinstance(export.get("simplify"), bool):
            errors.append("tensorrt_backend.export.overwrite与simplify必须为布尔值")
        _number(export, "onnx_opset", errors, 13, 20)
        if not export.get("context_checkpoint"):
            errors.append("tensorrt_backend.export.context_checkpoint必填")
    engines = tensorrt_backend.get("engines")
    if not isinstance(engines, Mapping) or not engines:
        errors.append("tensorrt_backend.engines必须是非空映射")
    else:
        for source, entry in engines.items():
            if not isinstance(source, str) or not isinstance(entry, Mapping):
                errors.append("tensorrt_backend.engines条目非法")
                continue
            unknown_engine = sorted(set(entry) - {
                "path", "sha256", "imgsz", "batch_size", "min_batch_size", "opt_batch_size",
                "opt_height", "opt_width",
            })
            if unknown_engine:
                errors.append(f"TensorRT engine {source}包含未知字段：" + ", ".join(unknown_engine))
            raw_digest = entry.get("sha256")
            digest = str(raw_digest or "")
            missing_hash_allowed = allow_missing_engine_hashes and raw_digest is None
            if not entry.get("path") or (
                not missing_hash_allowed
                and (len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest))
            ):
                errors.append(f"TensorRT engine {source}缺少有效路径或SHA256")
            _number(entry, "imgsz", errors, 32)
            if int(entry.get("imgsz", 0) or 0) < minimum_spatial_size:
                errors.append(f"TensorRT engine {source} 的imgsz不得小于minimum_spatial_size")
            opt_height = _number(entry, "opt_height", errors, 32)
            opt_width = _number(entry, "opt_width", errors, 32)
            if int(opt_height) % 32 or int(opt_width) % 32:
                errors.append(f"TensorRT engine {source} 的opt_height/opt_width必须是32的倍数")
            if max(opt_height, opt_width) > int(entry.get("imgsz", 0) or 0):
                errors.append(f"TensorRT engine {source} 的最优空间尺寸不得超过imgsz")
            max_engine_batch = _number(entry, "batch_size", errors, 1)
            min_engine_batch = _number(entry, "min_batch_size", errors, 1)
            opt_engine_batch = _number(entry, "opt_batch_size", errors, 1)
            if not min_engine_batch <= opt_engine_batch <= max_engine_batch:
                errors.append(f"TensorRT engine {source} batch profile必须满足min<=opt<=max")
    context_engine = tensorrt_backend.get("context_engine")
    if not isinstance(context_engine, Mapping):
        errors.append("tensorrt_backend.context_engine必须是映射")
    else:
        unknown_context = sorted(set(context_engine) - {"path", "sha256", "imgsz", "batch_size", "min_batch_size", "opt_batch_size"})
        if unknown_context:
            errors.append("TensorRT context engine包含未知字段：" + ", ".join(unknown_context))
        raw_digest = context_engine.get("sha256")
        digest = str(raw_digest or "")
        missing_hash_allowed = allow_missing_engine_hashes and raw_digest is None
        if not context_engine.get("path") or (
            not missing_hash_allowed
            and (len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest))
        ):
            errors.append("TensorRT context engine缺少有效路径或SHA256")
        _number(context_engine, "imgsz", errors, 32)
        context_max_batch = _number(context_engine, "batch_size", errors, 1)
        context_min_profile = _number(context_engine, "min_batch_size", errors, 1)
        context_opt_profile = _number(context_engine, "opt_batch_size", errors, 1)
        if not context_min_profile <= context_opt_profile <= context_max_batch:
            errors.append("TensorRT context batch profile必须满足min<=opt<=max")

    model = _require_mapping(config, "model", errors)
    if not model.get("weights"):
        errors.append("model.weights is required")
    expected = str(model.get("expected_sha256") or "")
    if len(expected) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in expected):
        errors.append("model.expected_sha256 must be a 64-character hexadecimal digest")
    assets = _require_mapping(config, "assets", errors)
    if not assets.get("manifest") or not assets.get("checksums"):
        errors.append("assets.manifest 与 assets.checksums 必填")
    if not isinstance(assets.get("required"), list) or not assets.get("required"):
        errors.append("assets.required must be a non-empty list")
    functional = _require_mapping(config, "functional_models", errors)
    if not functional.get("registry") or int(functional.get("required_count") or 0) < 3:
        errors.append("functional_models必须声明至少3种功能模型")
    incremental = _require_mapping(config, "incremental", errors)
    supported_modes = incremental.get("supported_modes", [])
    if incremental.get("task_type") != "incremental_object_detection":
        errors.append("incremental.task_type must be incremental_object_detection")
    if set(supported_modes) != {"class_incremental", "target_incremental"}:
        errors.append("incremental.supported_modes必须包含两种增量模式")
    if incremental.get("primary_mode") not in supported_modes:
        errors.append("incremental.primary_mode未注册")
    if incremental.get("learning_data_scope") != "incremental_dataset_only":
        errors.append("incremental.learning_data_scope must be incremental_dataset_only")
    automation = _require_mapping(config, "automation", errors)
    if not isinstance(automation.get("allowed_output_roots"), list) or not automation.get("allowed_output_roots"):
        errors.append("automation.allowed_output_roots must be a non-empty list")
    _number(automation, "max_steps_per_run", errors, 1, 100)
    actions = config.get("decision", {}).get("actions", {}) if isinstance(config.get("decision"), Mapping) else {}
    if not isinstance(actions, dict) or not actions:
        errors.append("decision.actions must be a non-empty mapping")
    else:
        for name, action in actions.items():
            if not isinstance(action, dict) or action.get("risk_level") not in {"low", "medium", "high"}:
                errors.append(f"decision.actions.{name} 非法")
            elif not action.get("handler") and not action.get("argv"):
                errors.append(f"decision.actions.{name} needs handler or argv")
    if errors:
        raise ValueError("Invalid agent config: " + "; ".join(errors))


def runtime_config_path(
    path: Union[str, Path, None] = None,
    *,
    machine: str | None = None,
) -> Path:
    return Path(select_runtime_config(path, machine=machine)["config_path"])


def runtime_overrides(overrides: Iterable[str] | None = None) -> List[str]:
    if overrides is not None:
        return list(overrides)
    raw = os.environ.get("AGILE_AGENT_OVERRIDES", "")
    if not raw:
        return []
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("AGILE_AGENT_OVERRIDES 必须是JSON字符串数组")
    return value


def load_config(
    path: Union[str, Path, None] = None,
    overrides: Iterable[str] | None = None,
    *,
    allow_unverified_tensorrt_hashes: bool = False,
) -> Dict[str, Any]:
    selection = select_runtime_config(path)
    config_path = Path(selection["config_path"])
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Agent config must be a mapping: {config_path}")
    resolved = _expand_environment(raw)
    active_overrides = runtime_overrides(overrides)
    data = apply_overrides(resolved, active_overrides)
    data["_config_path"] = rel_path(config_path)
    data["_config_overrides"] = active_overrides
    validate_config(
        data,
        allow_unverified_tensorrt_hashes=allow_unverified_tensorrt_hashes,
    )
    selection_metadata = {
        key: value for key, value in selection.items() if key != "config_path"
    }
    selection_metadata["config_path"] = rel_path(config_path)
    data["_runtime_platform"] = runtime_platform_info(
        {
            **data,
            "_runtime_platform": selection_metadata,
        }
    )
    data["_config_sha256"] = config_sha256(data)
    return data


def _audit_config_write(config_path: Path, before: Mapping[str, Any], after: Mapping[str, Any], operation: str) -> Path:
    root = resolve_path("reports/config_audit")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = root / "backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    (backup_dir / config_path.name).write_text(yaml.safe_dump(dict(before), allow_unicode=True, sort_keys=False), encoding="utf-8")
    event = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "operation": operation,
        "config": rel_path(config_path),
        "before_sha256": config_sha256(before),
        "after_sha256": config_sha256(after),
        "backup": rel_path(backup_dir / config_path.name),
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return backup_dir


def write_config(path: Union[str, Path], data: Mapping[str, Any], operation: str) -> Path:
    config_path = resolve_path(path)
    before = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(before, dict):
        raise ValueError("原配置不是映射")
    clean = {key: value for key, value in data.items() if not str(key).startswith("_")}
    validate_config(dict(clean))
    backup_dir = _audit_config_write(config_path, before, clean, operation)
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(clean, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(temporary, config_path)
    from .runtime_log import event_log_from_config

    event_log_from_config(clean).append(
        "config.changed",
        component="configuration",
        details={
            "operation": operation,
            "config": rel_path(config_path),
            "before_sha256": config_sha256(before),
            "after_sha256": config_sha256(clean),
            "backup": rel_path(backup_dir / config_path.name),
        },
    )
    return config_path


def configured_python(config: Dict[str, Any]) -> Path:
    configured = os.environ.get("AGILE_AGENT_PYTHON") or config["runtime"].get(
        "local_python"
    )
    return Path(configured).expanduser() if configured else Path(sys.executable)


def inference_backend_options(config: Mapping[str, Any]) -> Dict[str, Any]:
    if str(config["inference"]["backend"]) == "tensorrt_engine":
        return dict(config["tensorrt_backend"])
    if str(config["inference"]["backend"]) == "ascend_acl":
        return dict(config["ascend_backend"])
    return dict(config["native_backend"])
