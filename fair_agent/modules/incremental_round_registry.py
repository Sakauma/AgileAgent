from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

from fair_agent.core.config import resolve_path


DEFAULT_ROUND_REGISTRY = "configs/incremental_round_registry_4plus2.yaml"


def _int_mapping(raw: Any, field: str) -> Dict[int, int]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{field} 必须是非空映射")
    try:
        mapping = {int(key): int(value) for key, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须使用整数类别 ID") from exc
    if len(mapping) != len(set(mapping.values())):
        raise ValueError(f"{field} 的全局类别 ID 不能重复")
    return mapping


def _class_ids(raw: Any, field: str) -> list[int]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field} 必须是非空列表")
    try:
        values = [int(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须使用整数类别 ID") from exc
    if len(values) != len(set(values)):
        raise ValueError(f"{field} 不能包含重复类别")
    return values


def _split_contract(
    raw: Any, round_id: str, *, allow_base: bool = False
) -> Dict[str, str]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{round_id} 缺少 Increment split 配置")
    splits = {key: str(raw.get(key) or "") for key in ("train", "dev", "lock")}
    if any(not value for value in splits.values()):
        raise ValueError(f"{round_id} 必须登记 train/dev/lock")
    if not allow_base and any(
        "base_" in Path(value).name.lower() for value in splits.values()
    ):
        raise ValueError(f"{round_id} 的增量学习 split 不能指向 Base 清单")
    return splits


def load_incremental_round_registry(
    path: str | Path = DEFAULT_ROUND_REGISTRY,
) -> Dict[str, Any]:
    """Load and validate the ordered class-incremental round contract.

    The registry describes detector learning only. Scene-SensorNet and scene-aware
    calibration are deliberately represented as an external system-calibration
    contract and never become incremental-learning inputs.
    """

    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"增量轮次注册表不存在：{resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or int(raw.get("schema_version") or 0) != 1:
        raise ValueError("增量轮次注册表版本不受支持")

    raw_classes = raw.get("classes")
    if not isinstance(raw_classes, Mapping) or not raw_classes:
        raise ValueError("增量轮次注册表缺少类别注册表")
    classes: Dict[int, Dict[str, Any]] = {}
    for raw_id, value in raw_classes.items():
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"类别 ID 非法：{raw_id}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"类别登记项必须是映射：{class_id}")
        name = str(value.get("name") or "").strip()
        introduced_in = str(value.get("introduced_in") or "").strip()
        if not name or not introduced_in:
            raise ValueError(f"类别 {class_id} 缺少 name 或 introduced_in")
        classes[class_id] = {
            **dict(value),
            "id": class_id,
            "name": name,
            "introduced_in": introduced_in,
        }
    if len({item["name"] for item in classes.values()}) != len(classes):
        raise ValueError("类别名称不能重复")

    raw_base = raw.get("base")
    if not isinstance(raw_base, Mapping):
        raise ValueError("增量轮次注册表缺少 Base 代际")
    base_ids = _class_ids(raw_base.get("class_ids"), "base.class_ids")
    base_mapping = _int_mapping(raw_base.get("local_to_global"), "base.local_to_global")
    if set(base_ids) != set(base_mapping.values()) or set(base_ids) - set(classes):
        raise ValueError("Base 类别集合与局部映射/类别注册表不一致")
    if any(classes[class_id]["introduced_in"] != "base" for class_id in base_ids):
        raise ValueError("Base 类别必须登记 introduced_in: base")
    if raw_base.get("learning_data_scope") != "base_dataset_only":
        raise ValueError("Base 代际必须声明 base_dataset_only")
    base_generation_id = str(raw_base.get("generation_id") or "").strip()
    base_model_id = str(raw_base.get("model_id") or "").strip()
    if not base_generation_id or not base_model_id:
        raise ValueError("Base 代际必须登记 generation_id 和 model_id")
    base_splits = _split_contract(raw_base.get("splits"), "base", allow_base=True)
    if any("base_" not in Path(value).name.lower() for value in base_splits.values()):
        raise ValueError("Base 代际必须使用 Base train/dev/lock 清单")
    base = {
        **dict(raw_base),
        "class_ids": base_ids,
        "local_to_global": base_mapping,
        "generation_id": base_generation_id,
        "model_id": base_model_id,
        "splits": base_splits,
    }

    raw_rounds = raw.get("rounds")
    if not isinstance(raw_rounds, list) or len(raw_rounds) < 2:
        raise ValueError("正式类别增量协议至少需要两轮新增类别")
    rounds: list[Dict[str, Any]] = []
    rounds_by_id: Dict[str, Dict[str, Any]] = {}
    known = set(base_ids)
    previous_generation = base_generation_id
    generation_ids = {base_generation_id}
    model_ids = {base_model_id}
    for expected_index, value in enumerate(raw_rounds, start=1):
        if not isinstance(value, Mapping):
            raise ValueError(f"第 {expected_index} 轮配置必须是映射")
        round_id = str(value.get("round_id") or "").strip()
        round_index = int(value.get("round_index") or 0)
        generation_id = str(value.get("generation_id") or "").strip()
        parent_generation_id = str(value.get("parent_generation_id") or "").strip()
        specialist = value.get("specialist")
        if not isinstance(specialist, Mapping):
            raise ValueError(f"{round_id or expected_index} 缺少 specialist")
        model_id = str(specialist.get("model_id") or "").strip()
        new_ids = _class_ids(value.get("new_class_ids"), f"{round_id}.new_class_ids")
        local_to_global = _int_mapping(
            specialist.get("local_to_global"), f"{round_id}.specialist.local_to_global"
        )
        if (
            not round_id
            or round_index != expected_index
            or not generation_id
            or not model_id
        ):
            raise ValueError(f"第 {expected_index} 轮标识或顺序非法")
        if round_id in rounds_by_id or generation_id in generation_ids or model_id in model_ids:
            raise ValueError(f"轮次、代际或专家 ID 重复：{round_id}")
        if parent_generation_id != previous_generation:
            raise ValueError(f"{round_id} 未指向上一轮冻结代际 {previous_generation}")
        if set(new_ids) != set(local_to_global.values()):
            raise ValueError(f"{round_id} 的新增类别与专家局部映射不一致")
        if set(new_ids) & known or set(new_ids) - set(classes):
            raise ValueError(f"{round_id} 包含已学习或未注册类别")
        if any(classes[class_id]["introduced_in"] != round_id for class_id in new_ids):
            raise ValueError(f"{round_id} 与类别 introduced_in 登记不一致")
        if value.get("learning_data_scope") != "incremental_dataset_only":
            raise ValueError(f"{round_id} 必须声明 incremental_dataset_only")
        if (
            value.get("validation_data_scope") != "incremental_dataset_only"
            or value.get("base_detector_weights_frozen") is not True
            or value.get("old_expert_weights_frozen") is not True
            or int(value.get("old_raw_image_count", -1)) != 0
            or value.get("image_selector") != "contains_current_round_class"
            or value.get("label_projection") != "current_round_classes_only"
        ):
            raise ValueError(f"{round_id} 的冻结、验证或标签投影契约不完整")
        splits = _split_contract(value.get("splits"), round_id)
        source_splits = _split_contract(
            value.get("source_splits") or value.get("splits"),
            f"{round_id}.source_splits",
        )
        known_before = sorted(known)
        known.update(new_ids)
        learned_ids = _class_ids(value.get("learned_class_ids"), f"{round_id}.learned_class_ids")
        if set(learned_ids) != known:
            raise ValueError(f"{round_id} 的累计类别集合不正确")
        round_spec = {
            **dict(value),
            "round_id": round_id,
            "round_index": round_index,
            "generation_id": generation_id,
            "parent_generation_id": parent_generation_id,
            "new_class_ids": new_ids,
            "old_class_ids": known_before,
            "learned_class_ids": learned_ids,
            "splits": splits,
            "source_splits": source_splits,
            "specialist": {
                **dict(specialist),
                "model_id": model_id,
                "local_to_global": local_to_global,
            },
        }
        rounds.append(round_spec)
        rounds_by_id[round_id] = round_spec
        generation_ids.add(generation_id)
        model_ids.add(model_id)
        previous_generation = generation_id

    if set(classes) != known:
        raise ValueError("存在未被 Base 或任一增量轮次引入的类别")
    system_calibration = raw.get("system_calibration")
    if not isinstance(system_calibration, Mapping) or (
        system_calibration.get("phase") != "system_calibration"
        or system_calibration.get("counted_as_incremental_learning") is not False
        or system_calibration.get("detector_weights_updated") is not False
        or system_calibration.get("scene_sensor_is_incremental_learner") is not False
        or system_calibration.get("frozen_before_joint_evaluation") is not True
    ):
        raise ValueError("场景系统必须明确排除在增量学习之外")
    joint_evaluation = raw.get("joint_evaluation")
    if not isinstance(joint_evaluation, Mapping) or (
        joint_evaluation.get("phase") != "joint_evaluation"
        or joint_evaluation.get("counted_as_incremental_learning") is not False
        or joint_evaluation.get("detector_weights_updated") is not False
        or joint_evaluation.get("model_selection_allowed") is not False
        or joint_evaluation.get("lineage_required") is not True
        or set(joint_evaluation.get("metrics_per_round") or [])
        != {"new_map50", "krr", "full_map50"}
    ):
        raise ValueError("逐轮联合评估契约不完整")
    calibration_defaults = raw.get("system_calibration_defaults")
    if not isinstance(calibration_defaults, Mapping):
        raise ValueError("缺少 system_calibration_defaults")
    try:
        default_thresholds = {
            int(key): float(value)
            for key, value in dict(
                calibration_defaults.get("threshold_by_class") or {}
            ).items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("系统校准初始阈值格式非法") from exc
    if set(default_thresholds) != set(classes) or any(
        not 0.01 <= value <= 1.0 for value in default_thresholds.values()
    ):
        raise ValueError("系统校准初始阈值必须完整覆盖类别注册表")

    return {
        **dict(raw),
        "path": resolved,
        "classes": classes,
        "class_names": {key: value["name"] for key, value in classes.items()},
        "base": base,
        "rounds": rounds,
        "rounds_by_id": rounds_by_id,
        "system_calibration": dict(system_calibration),
        "joint_evaluation": dict(joint_evaluation),
        "system_calibration_defaults": {
            **dict(calibration_defaults),
            "threshold_by_class": default_thresholds,
        },
    }


def select_round(registry: Mapping[str, Any], round_id: str) -> Dict[str, Any]:
    try:
        return dict(registry["rounds_by_id"][round_id])
    except KeyError as exc:
        valid = ", ".join(registry.get("rounds_by_id", {}))
        raise ValueError(f"未登记的增量轮次 {round_id}；可选：{valid}") from exc


def rounds_through(
    registry: Mapping[str, Any], round_id: str
) -> list[Dict[str, Any]]:
    selected = select_round(registry, round_id)
    return [
        dict(value)
        for value in registry["rounds"]
        if int(value["round_index"]) <= int(selected["round_index"])
    ]


def introduced_class_names(
    registry: Mapping[str, Any], class_ids: Iterable[int]
) -> Dict[int, str]:
    names = registry["class_names"]
    return {int(class_id): str(names[int(class_id)]) for class_id in class_ids}
