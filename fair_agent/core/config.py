from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Union

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "agent_pipeline.yaml"
CONFIG_SCHEMA_VERSION = 2
ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
SECRET_PARTS = {"password", "passwd", "secret", "token", "api_key", "private_key"}
PROTECTED_PREFIXES = (
    "model.expected_sha256",
    "assets.checksums",
    "assets.generation_registry",
    "web.generation_registry",
    "web.generation_channel",
    "generation.registry",
    "tensorrt_backend.engines",
    "tensorrt_backend.context_engine",
    "tensorrt_backend.validated",
    "tensorrt_backend.expected_version",
    "tensorrt_backend.expected_compute_capability",
)
KNOWN_TOP_LEVEL = {
    "schema_version", "seed", "runtime", "web", "inference", "routing", "limits",
    "storage", "ui", "performance", "native_backend", "tensorrt_backend", "model", "assets", "automation",
    "generation", "submission", "blackboard", "detector", "functional_models", "inputs", "modules",
    "policies", "thresholds", "incremental", "specialist_acceptance", "decision",
    "logging", "incremental_workbench",
}
KNOWN_SECTION_KEYS = {
    "runtime": {"mode", "local_python", "default_device", "server_host", "server_port"},
    "web": {"generation_registry", "generation_channel", "detector_weights", "functional_registry", "model_manifest"},
    "inference": {"backend", "imgsz", "specialist_imgsz", "iou", "max_det", "batch_size", "confidence_min", "confidence_max", "confidence_default", "warmup_iterations", "warmup_batch_size", "warmup_width", "warmup_height", "preload_specialists", "quantize", "cudnn_benchmark", "compile"},
    "routing": {
        "incremental_enabled", "require_acceptance_passed", "consensus_iou", "fusion_iou",
        "max_specialists_per_image", "conflict_iou", "conflict_base_confidence",
        "specialist_margin", "detection_evidence_weight", "context_evidence_weight",
        "neutral_context_score", "default_routing_prior",
        "parallel_model_execution", "parallel_context_execution", "parallel_context_batch_execution", "max_model_workers",
    },
    "limits": {
        "max_file_bytes", "max_batch_files", "max_batch_bytes", "max_image_pixels",
        "allowed_image_formats", "decode_backend", "decode_workers",
    },
    "storage": {"max_items", "ttl_seconds", "max_bytes"},
    "ui": {"history_limit", "result_cache_limit", "health_poll_ms", "toast_duration_ms", "default_view"},
    "performance": {
        "target_api_fps", "target_p95_ms", "benchmark_rounds", "warmup_requests",
        "benchmark_split", "report_root", "concurrent_requests", "batch_probe_size",
        "auto_start_server", "server_start_timeout_seconds", "request_timeout_seconds",
    },
    "native_backend": {"library", "base_engine", "context_engine", "precision", "require_exact_gpu", "validated"},
    "tensorrt_backend": {
        "expected_version", "expected_compute_capability", "require_exact_gpu", "validated",
        "precision", "workspace_gib", "dynamic", "engines", "context_engine", "export",
    },
    "generation": {"registry", "recheck_lock_split", "report_root", "candidate_id", "calibrated_threshold", "acceptance"},
    "model": {"weights", "expected_sha256"},
    "logging": {"root", "max_file_bytes", "retained_files", "request_bodies"},
    "incremental_workbench": {
        "root", "max_archive_bytes", "max_extracted_bytes", "max_extracted_files",
        "max_image_pixels", "allowed_image_extensions", "require_labels", "validation_fraction",
        "minimum_images", "preview_limit", "job_log_tail_lines", "poll_interval_ms", "training",
    },
}


def resolve_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def validate_config(config: Dict[str, Any]) -> None:
    errors: List[str] = []
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 {CONFIG_SCHEMA_VERSION}")
    unknown = sorted(set(config) - KNOWN_TOP_LEVEL - {"_config_path", "_config_sha256", "_config_overrides"})
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

    inference = _require_mapping(config, "inference", errors)
    if inference.get("backend") not in {"ultralytics_cuda", "tensorrt_engine", "tensorrt_native"}:
        errors.append("inference.backend 必须为 ultralytics_cuda、tensorrt_engine 或 tensorrt_native")
    _number(inference, "imgsz", errors, 32)
    _number(inference, "specialist_imgsz", errors, 32)
    _number(inference, "iou", errors, 0.01, 1.0)
    _number(inference, "max_det", errors, 1)
    _number(inference, "batch_size", errors, 1)
    _number(inference, "warmup_iterations", errors, 1)
    _number(inference, "warmup_batch_size", errors, 1)
    _number(inference, "warmup_width", errors, 1)
    _number(inference, "warmup_height", errors, 1)
    confidence_min = _number(inference, "confidence_min", errors, 0.01, 1.0)
    confidence_max = _number(inference, "confidence_max", errors, 0.01, 1.0)
    confidence_default = _number(inference, "confidence_default", errors, 0.01, 1.0)
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
    if abs(float(routing.get("detection_evidence_weight", 0)) + float(routing.get("context_evidence_weight", 0)) - 1.0) > 1e-9:
        errors.append("routing的检测证据权重与上下文证据权重之和必须为1")
    _number(routing, "max_specialists_per_image", errors, 1)
    _number(routing, "max_model_workers", errors, 1)
    if not isinstance(routing.get("incremental_enabled"), bool) or not isinstance(routing.get("require_acceptance_passed"), bool):
        errors.append("routing中的开关必须为布尔值")
    if not all(isinstance(routing.get(key), bool) for key in ("parallel_model_execution", "parallel_context_execution", "parallel_context_batch_execution")):
        errors.append("routing中的并行执行开关必须为布尔值")

    limits = _require_mapping(config, "limits", errors)
    for key in ("max_file_bytes", "max_batch_files", "max_batch_bytes", "max_image_pixels"):
        _number(limits, key, errors, 1)
    formats = limits.get("allowed_image_formats")
    if not isinstance(formats, list) or not formats or not all(isinstance(item, str) for item in formats):
        errors.append("limits.allowed_image_formats 必须是非空字符串列表")
    if limits.get("decode_backend") not in {"pillow", "opencv"}:
        errors.append("limits.decode_backend必须为pillow或opencv")
    _number(limits, "decode_workers", errors, 1, 32)
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
    if not isinstance(workbench.get("require_labels"), bool):
        errors.append("incremental_workbench.require_labels必须为布尔值")
    _number(workbench, "validation_fraction", errors, 0.01, 0.50)
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
    if not generation.get("registry") or not generation.get("recheck_lock_split") or not generation.get("report_root"):
        errors.append("generation必须声明registry、recheck_lock_split和report_root")
    _number(generation, "calibrated_threshold", errors, 0.01, 1.0)
    acceptance = generation.get("acceptance")
    if not isinstance(acceptance, Mapping):
        errors.append("generation.acceptance必须是映射")
    else:
        for key in ("min_base_map50", "min_new_map50", "min_krr", "min_combined_map50", "min_lock_precision", "max_false_activation_rate"):
            _number(acceptance, key, errors, 0.0, 1.0)

    native = _require_mapping(config, "native_backend", errors)
    if native.get("precision") not in {"fp16", "fp32", "int8"}:
        errors.append("native_backend.precision 非法")
    if not isinstance(native.get("require_exact_gpu"), bool) or not isinstance(native.get("validated"), bool):
        errors.append("native_backend.require_exact_gpu与validated必须为布尔值")
    if inference.get("backend") == "tensorrt_native":
        for key in ("library", "base_engine", "context_engine"):
            if not native.get(key):
                errors.append(f"TensorRT后端缺少 native_backend.{key}")

    tensorrt_backend = _require_mapping(config, "tensorrt_backend", errors)
    if tensorrt_backend.get("precision") not in {"fp16", "fp32", "int8"}:
        errors.append("tensorrt_backend.precision 非法")
    if not isinstance(tensorrt_backend.get("require_exact_gpu"), bool) or not isinstance(
        tensorrt_backend.get("validated"), bool
    ):
        errors.append("tensorrt_backend.require_exact_gpu与validated必须为布尔值")
    if not isinstance(tensorrt_backend.get("dynamic"), bool):
        errors.append("tensorrt_backend.dynamic必须为布尔值")
    _number(tensorrt_backend, "workspace_gib", errors, 0.1)
    if not tensorrt_backend.get("expected_version") or not tensorrt_backend.get("expected_compute_capability"):
        errors.append("tensorrt_backend必须声明TensorRT版本与GPU计算能力")
    export = tensorrt_backend.get("export")
    if not isinstance(export, Mapping):
        errors.append("tensorrt_backend.export必须是映射")
    else:
        unknown_export = sorted(set(export) - {
            "device", "overwrite", "onnx_opset", "simplify", "context_checkpoint",
            "context_min_batch", "context_opt_batch",
        })
        if unknown_export:
            errors.append("tensorrt_backend.export包含未知字段：" + ", ".join(unknown_export))
        if not str(export.get("device", "")).isdigit():
            errors.append("tensorrt_backend.export.device必须是非负GPU编号")
        if not isinstance(export.get("overwrite"), bool) or not isinstance(export.get("simplify"), bool):
            errors.append("tensorrt_backend.export.overwrite与simplify必须为布尔值")
        _number(export, "onnx_opset", errors, 13, 20)
        min_batch = _number(export, "context_min_batch", errors, 1)
        opt_batch = _number(export, "context_opt_batch", errors, 1)
        if min_batch > opt_batch:
            errors.append("TensorRT context最小batch不得大于最优batch")
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
            unknown_engine = sorted(set(entry) - {"path", "sha256", "imgsz", "batch_size"})
            if unknown_engine:
                errors.append(f"TensorRT engine {source}包含未知字段：" + ", ".join(unknown_engine))
            digest = str(entry.get("sha256") or "")
            if not entry.get("path") or len(digest) != 64 or any(
                ch not in "0123456789abcdefABCDEF" for ch in digest
            ):
                errors.append(f"TensorRT engine {source}缺少有效路径或SHA256")
            _number(entry, "imgsz", errors, 32)
            _number(entry, "batch_size", errors, 1)
    context_engine = tensorrt_backend.get("context_engine")
    if not isinstance(context_engine, Mapping):
        errors.append("tensorrt_backend.context_engine必须是映射")
    else:
        unknown_context = sorted(set(context_engine) - {"path", "sha256", "imgsz", "batch_size"})
        if unknown_context:
            errors.append("TensorRT context engine包含未知字段：" + ", ".join(unknown_context))
        digest = str(context_engine.get("sha256") or "")
        if not context_engine.get("path") or len(digest) != 64 or any(
            ch not in "0123456789abcdefABCDEF" for ch in digest
        ):
            errors.append("TensorRT context engine缺少有效路径或SHA256")
        _number(context_engine, "imgsz", errors, 32)
        context_max_batch = _number(context_engine, "batch_size", errors, 1)
        if isinstance(export, Mapping) and float(export.get("context_opt_batch", 0) or 0) > context_max_batch:
            errors.append("TensorRT context最优batch不得大于engine最大batch")

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


def runtime_config_path(path: Union[str, Path] = DEFAULT_CONFIG) -> Path:
    if str(path) == str(DEFAULT_CONFIG) and os.environ.get("AGILE_AGENT_CONFIG"):
        return resolve_path(os.environ["AGILE_AGENT_CONFIG"])
    return resolve_path(path)


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


def load_config(path: Union[str, Path] = DEFAULT_CONFIG, overrides: Iterable[str] | None = None) -> Dict[str, Any]:
    config_path = runtime_config_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Agent config must be a mapping: {config_path}")
    resolved = _expand_environment(raw)
    active_overrides = runtime_overrides(overrides)
    data = apply_overrides(resolved, active_overrides)
    data["_config_path"] = rel_path(config_path)
    data["_config_overrides"] = active_overrides
    validate_config(data)
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
    configured = config["runtime"].get("local_python")
    return Path(configured).expanduser() if configured else Path(sys.executable)


def inference_backend_options(config: Mapping[str, Any]) -> Dict[str, Any]:
    if str(config["inference"]["backend"]) == "tensorrt_engine":
        return dict(config["tensorrt_backend"])
    return dict(config["native_backend"])
