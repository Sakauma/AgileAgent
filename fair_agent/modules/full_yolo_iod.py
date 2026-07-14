from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import yaml
from PIL import Image

from fair_agent.modules.strict_incremental import (
    GLOBAL_CLASS_NAMES,
    image_class_ids,
    read_split,
    read_yolo_labels,
    source_label,
)


OFFICIAL_COMMIT = "3d9d05a6e88561c88916657367412f9adb7341a7"
OFFICIAL_OLD_CLASSES = ("soldier", "small aircraft", "tank")
OFFICIAL_NEW_CLASS = "warship"
OFFICIAL_ALL_CLASSES = (*OFFICIAL_OLD_CLASSES, OFFICIAL_NEW_CLASS)
GLOBAL_TO_OFFICIAL = {0: 0, 1: 1, 3: 2, 2: 3}
OFFICIAL_TO_GLOBAL = {value: key for key, value in GLOBAL_TO_OFFICIAL.items()}
OFFICIAL_BASE_CONFIG = "third_party/mmyolo/configs/yolov8/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py"
_RELATIVE_BASE_FRAGMENT = (
    "'../../third_party/mmyolo/configs/yolov8/'\n"
    "    'yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py'"
)


def load_full_yolo_iod_config(path: str | Path) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("完整 YOLO-IOD 配置顶层必须是映射")
    validate_full_yolo_iod_config(data)
    return data


def validate_full_yolo_iod_config(config: Mapping[str, Any]) -> None:
    experiment = config.get("experiment", {})
    if experiment.get("protocol") != "strict-p02":
        raise ValueError("完整 YOLO-IOD 首版只允许 strict-p02")
    if experiment.get("official_commit") != OFFICIAL_COMMIT:
        raise ValueError("官方 YOLO-IOD 提交哈希与已审计版本不一致")
    classes = config.get("classes", {})
    if tuple(classes.get("old", ())) != OFFICIAL_OLD_CLASSES:
        raise ValueError("官方模型旧类顺序必须为 soldier、small aircraft、tank")
    if classes.get("new") != OFFICIAL_NEW_CLASS:
        raise ValueError("strict-p02 新增类必须为 warship")
    mapping = {int(key): int(value) for key, value in classes.get("global_to_official", {}).items()}
    if mapping != GLOBAL_TO_OFFICIAL:
        raise ValueError("赛题类别到官方 YOLO-IOD 类别映射错误")
    devices = [str(item) for item in config.get("runtime", {}).get("devices", [])]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("runtime.devices 必须声明不重复的 GPU")
    if int(config.get("schema_version", 1)) >= 2:
        if devices != ["1"]:
            raise ValueError("完整 YOLO-IOD r04 及后续实验只允许使用 GPU 1")
        dataset = config.get("dataset", {})
        if dataset.get("incremental_validation_only") is not True:
            raise ValueError("增量阶段验证必须只读取增量数据")
        cpr = config.get("cpr", {})
        if cpr.get("enabled") is not False:
            raise ValueError("strict-p02 无旧类共现时必须禁用 CPR")
        if cpr.get("mode") != "disabled_no_old_class_cooccurrence":
            raise ValueError("CPR 禁用原因必须写入配置")
        if int(cpr.get("required_incremental_old_class_gt_count", -1)) != 0:
            raise ValueError("CPR 禁用门禁必须要求增量旧类 GT 数量为 0")
        target_batch = int(config.get("training", {}).get("target_effective_batch_size", 0))
        if target_batch != 16:
            raise ValueError("完整 YOLO-IOD 单卡实验的目标有效 batch 必须为 16")
        for stage, row in training_batch_plan(config).items():
            if row["effective_batch_size"] != target_batch:
                raise ValueError(
                    f"{stage} 有效 batch 错误：expected={target_batch} "
                    f"actual={row['effective_batch_size']}"
                )


def cpr_is_enabled(config: Mapping[str, Any]) -> bool:
    cpr = config.get("cpr", {})
    if "enabled" in cpr:
        return bool(cpr["enabled"])
    return str(cpr.get("incremental", "enabled")) == "enabled"


def training_batch_plan(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    runtime_devices = [str(item) for item in config.get("runtime", {}).get("devices", [])]
    training = config.get("training", {})
    result: Dict[str, Dict[str, Any]] = {}
    for name, key in (("base", "base"), ("current", "current_teacher"), ("final", "final")):
        stage = training.get(key, {})
        devices = [str(item) for item in stage.get("devices", runtime_devices)]
        micro_batch = int(stage.get("batch_per_gpu", 0))
        accumulation = int(stage.get("gradient_accumulation_steps", 1))
        if not devices or micro_batch <= 0 or accumulation <= 0:
            raise ValueError(f"{name} 的设备、micro-batch 和梯度累积必须为正数")
        result[name] = {
            "devices": devices,
            "micro_batch_per_gpu": micro_batch,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": micro_batch * len(devices) * accumulation,
        }
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def _ensure_runtime_directories(data_root: Path) -> None:
    (data_root / "importance").mkdir(parents=True, exist_ok=True)


def _categories() -> list[Dict[str, Any]]:
    return [
        {"id": index + 1, "name": name, "supercategory": "target"}
        for index, name in enumerate(OFFICIAL_ALL_CLASSES)
    ]


def _coco_document(
    images: Sequence[Path],
    allowed_global_ids: Iterable[int],
) -> Dict[str, Any]:
    allowed = {int(value) for value in allowed_global_ids}
    coco_images = []
    annotations = []
    annotation_id = 1
    for image_id, image_path in enumerate(images, 1):
        with Image.open(image_path) as image:
            width, height = image.size
        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        for class_id, x, y, box_width, box_height in read_yolo_labels(source_label(image_path)):
            if class_id not in allowed:
                continue
            x1 = max(0.0, (x - box_width / 2.0) * width)
            y1 = max(0.0, (y - box_height / 2.0) * height)
            x2 = min(float(width), (x + box_width / 2.0) * width)
            y2 = min(float(height), (y + box_height / 2.0) * height)
            pixel_width = max(0.0, x2 - x1)
            pixel_height = max(0.0, y2 - y1)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": GLOBAL_TO_OFFICIAL[class_id] + 1,
                    "bbox": [x1, y1, pixel_width, pixel_height],
                    "area": pixel_width * pixel_height,
                    "iscrowd": 0,
                    # YOLO only supplies boxes. A rectangular polygon preserves
                    # those boxes through YOLO-World's official mask-refine pipeline.
                    "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
                    "score": 1.0,
                }
            )
            annotation_id += 1
    return {
        "info": {"description": "AgileAgent strict-p02 YOLO-IOD reproduction"},
        "licenses": [],
        "images": coco_images,
        "annotations": annotations,
        "categories": _categories(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_coco(
    path: Path,
    images: Sequence[Path],
    allowed_global_ids: Iterable[int],
) -> Dict[str, int]:
    document = _coco_document(images, allowed_global_ids)
    _write_json(path, document)
    return {
        "images": len(document["images"]),
        "annotations": len(document["annotations"]),
    }


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"官方配置模板替换项数量错误：{old!r} count={count}")
    return text.replace(old, new)


def _replace_all(text: str, old: str, new: str, minimum: int = 1) -> str:
    count = text.count(old)
    if count < minimum:
        raise ValueError(f"官方配置模板缺少替换项：{old!r}")
    return text.replace(old, new)


def _tuple_literal(values: Sequence[str]) -> str:
    return repr(tuple(values))


def _rewrite_base_dependency(template: str, dependency: str | Path) -> str:
    return _replace_once(template, _RELATIVE_BASE_FRAGMENT, repr(Path(dependency).as_posix()))


def _inject_gradient_accumulation(text: str, steps: int) -> str:
    constructor = "    constructor='YOLOWv5OptimizerConstructor')"
    replacement = f"    accumulative_counts={int(steps)},\n{constructor}"
    return _replace_once(text, constructor, replacement)


def _adapt_stage_config(
    template: str,
    *,
    base_dependency: str,
    class_names: Sequence[str],
    train_ann: str,
    val_ann: str,
    text_path: str,
    data_root: str,
    image_prefix_train: str,
    image_prefix_val: str,
    importance_path: str,
    batch: int,
    accumulation_steps: int,
    epochs: int,
) -> str:
    output = _rewrite_base_dependency(template, base_dependency)
    output = _replace_once(output, "num_classes = 40", f"num_classes = {len(class_names)}")
    output = _replace_once(
        output, "num_training_classes = 40", f"num_training_classes = {len(class_names)}"
    )
    output = _replace_once(output, "max_epochs = 20", f"max_epochs = {epochs}")
    output = _replace_once(
        output, "train_batch_size_per_gpu = 16", f"train_batch_size_per_gpu = {batch}"
    )
    output = _inject_gradient_accumulation(output, accumulation_steps)
    start = output.index("classes = (")
    end = output.index("\n\n# model settings", start)
    output = output[:start] + f"classes = {_tuple_literal(class_names)}" + output[end:]
    output = _replace_all(output, "data_root='data/coco'", f"data_root='{data_root}'")
    output = _replace_once(
        output,
        "ann_file='loco_annotations/40+40(order)/instances_train2017_part1.json'",
        f"ann_file='{train_ann}'",
    )
    output = _replace_once(
        output,
        "ann_file='loco_annotations/40+40(order)/instances_val2017_part1.json'",
        f"ann_file='{val_ann}'",
    )
    output = _replace_all(output, "data_prefix=dict(img='train2017/')", f"data_prefix=dict(img='{image_prefix_train}')")
    output = _replace_all(output, "data_prefix=dict(img='val2017/')", f"data_prefix=dict(img='{image_prefix_val}')")
    output = _replace_all(
        output,
        "data/coco/loco_annotations/40+40(order)/loco_class_texts_curs1.json",
        text_path,
    )
    output = _replace_once(
        output,
        "data/coco/loco_annotations/40+40(order)/t1_importance_iks.pt",
        importance_path,
    )
    output = _replace_once(
        output,
        "ann_file='data/coco/loco_annotations/40+40(order)/instances_val2017_part1.json'",
        f"ann_file='{data_root}/{val_ann}'",
    )
    return output


def _adapt_final_config(
    template: str,
    *,
    base_dependency: str,
    data_root: str,
    train_ann: str,
    val_ann: str,
    text_path: str,
    image_prefix_train: str,
    image_prefix_val: str,
    importance_path: str,
    base_config: str,
    base_checkpoint: str,
    current_config: str,
    current_checkpoint: str,
    init_checkpoint: str,
    batch: int,
    accumulation_steps: int,
    epochs: int,
) -> str:
    output = _rewrite_base_dependency(template, base_dependency)
    output = _replace_once(output, "num_classes = 80", "num_classes = 4")
    output = _replace_once(output, "num_training_classes = 80", "num_training_classes = 4")
    output = _replace_once(output, "ori_num_classes = 40", "ori_num_classes = 3")
    output = _replace_once(output, "max_epochs = 20", f"max_epochs = {epochs}")
    output = _replace_once(output, "train_batch_size_per_gpu = 12", f"train_batch_size_per_gpu = {batch}")
    output = _inject_gradient_accumulation(output, accumulation_steps)
    output = _replace_once(output, "load_from = './weights/loco_40_40_t1.pth'", f"load_from = '{init_checkpoint}'")
    start = output.index("classes = (")
    end = output.index("\n\n# model settings", start)
    output = output[:start] + f"classes = {_tuple_literal(OFFICIAL_ALL_CLASSES)}" + output[end:]
    output = _replace_once(
        output,
        "config='work_dirs/yolo_iod_loco_coco_40_40_task0/yolo_iod_loco_coco_40_40_task0.py'",
        f"config='{base_config}'",
    )
    output = _replace_once(
        output,
        "ckpt='work_dirs/yolo_iod_loco_coco_40_40_task0/epoch_20.pth'",
        f"ckpt='{base_checkpoint}'",
    )
    output = _replace_once(
        output,
        "config='work_dirs/yolo_iod_loco_coco_40_40_stage1/yolo_iod_loco_coco_40_40_stage1.py'",
        f"config='{current_config}'",
    )
    output = _replace_once(
        output,
        "ckpt='work_dirs/yolo_iod_loco_coco_40_40_stage1/epoch_20.pth'",
        f"ckpt='{current_checkpoint}'",
    )
    output = _replace_all(output, "data_root='data/coco'", f"data_root='{data_root}'")
    output = _replace_once(
        output,
        "ann_file='loco_annotations/40+40(order)/instances_train2017_part1_ps.json'",
        f"ann_file='{train_ann}'",
    )
    output = _replace_once(
        output,
        "ann_file='loco_annotations/40+40(order)/instances_val2017_part1.json'",
        f"ann_file='{val_ann}'",
    )
    output = _replace_all(output, "data_prefix=dict(img='train2017/')", f"data_prefix=dict(img='{image_prefix_train}')")
    output = _replace_all(output, "data_prefix=dict(img='val2017/')", f"data_prefix=dict(img='{image_prefix_val}')")
    output = _replace_all(
        output,
        "data/coco/loco_annotations/40+40(order)/loco_class_texts_stage1.json",
        text_path,
        minimum=2,
    )
    output = _replace_once(
        output,
        "data/coco/loco_annotations/40+40(order)/t1_importance_iod.pt",
        importance_path,
    )
    output = _replace_once(
        output,
        "ann_file='data/coco/loco_annotations/40+40(order)/instances_val2017_part1.json'",
        f"ann_file='{data_root}/{val_ann}'",
    )
    return output


def generate_official_configs(
    config: Mapping[str, Any],
    official_repo: Path,
    data_root: Path,
) -> Dict[str, str]:
    run_id = str(config["experiment"]["run_id"])
    runtime = config["runtime"]
    training = config["training"]
    batch_plan = training_batch_plan(config)
    incremental_validation_only = bool(config.get("dataset", {}).get("incremental_validation_only", False))
    final_val_name = "current_dev.json" if incremental_validation_only else "final_dev.json"
    final_train_name = "current_train_cpr.json" if cpr_is_enabled(config) else "current_train.json"
    relative_data_root = data_root.relative_to(official_repo).as_posix()
    annotation_root = "annotations"
    text_root = f"{relative_data_root}/texts"
    config_root = official_repo / "configs" / "agileagent_3_1" / run_id
    config_root.mkdir(parents=True, exist_ok=False)
    work_root = official_repo / "work_dirs" / "agileagent_3_1" / run_id
    base_work = work_root / "base"
    current_work = work_root / "current"
    final_work = work_root / "final"
    base_config = config_root / "base.py"
    current_config = config_root / "current.py"
    final_config = config_root / "final.py"
    base_lock_config = config_root / "base_lock.py"
    final_lock_config = config_root / "final_lock.py"
    stage_template = (
        official_repo / "configs/loco_40_40/yolo_iod_loco_coco_40_40_stage1.py"
    ).read_text(encoding="utf-8")
    final_template = (
        official_repo / "configs/loco_40_40/yolo_iod_loco_coco_40_40_task1.py"
    ).read_text(encoding="utf-8")
    base_dependency = (official_repo / OFFICIAL_BASE_CONFIG).as_posix()

    base_text = _adapt_stage_config(
        stage_template,
        base_dependency=base_dependency,
        class_names=OFFICIAL_OLD_CLASSES,
        train_ann=f"{annotation_root}/base_train.json",
        val_ann=f"{annotation_root}/base_dev.json",
        text_path=f"{text_root}/old.json",
        data_root=relative_data_root,
        image_prefix_train="images/train/",
        image_prefix_val="images/val/",
        importance_path=f"{relative_data_root}/importance/base_iks.pt",
        batch=int(training["base"]["batch_per_gpu"]),
        accumulation_steps=batch_plan["base"]["gradient_accumulation_steps"],
        epochs=int(training["base"]["epochs"]),
    )
    current_text = _adapt_stage_config(
        stage_template,
        base_dependency=base_dependency,
        class_names=(OFFICIAL_NEW_CLASS,),
        train_ann=f"{annotation_root}/current_train.json",
        val_ann=f"{annotation_root}/current_dev.json",
        text_path=f"{text_root}/current.json",
        data_root=relative_data_root,
        image_prefix_train="images/train/",
        image_prefix_val="images/val/",
        importance_path=f"{relative_data_root}/importance/current_iks.pt",
        batch=int(training["current_teacher"]["batch_per_gpu"]),
        accumulation_steps=batch_plan["current"]["gradient_accumulation_steps"],
        epochs=int(training["current_teacher"]["epochs"]),
    )
    final_text = _adapt_final_config(
        final_template,
        base_dependency=base_dependency,
        data_root=relative_data_root,
        train_ann=f"{annotation_root}/{final_train_name}",
        val_ann=f"{annotation_root}/{final_val_name}",
        text_path=f"{text_root}/all.json",
        image_prefix_train="images/train/",
        image_prefix_val="images/val/",
        importance_path=f"{relative_data_root}/importance/final_iks.pt",
        base_config=base_config.relative_to(official_repo).as_posix(),
        base_checkpoint=(base_work / f"epoch_{training['base']['epochs']}.pth").relative_to(official_repo).as_posix(),
        current_config=current_config.relative_to(official_repo).as_posix(),
        current_checkpoint=(current_work / f"epoch_{training['current_teacher']['epochs']}.pth").relative_to(official_repo).as_posix(),
        init_checkpoint=(work_root / "final_init.pth").relative_to(official_repo).as_posix(),
        batch=int(training["final"]["batch_per_gpu"]),
        accumulation_steps=batch_plan["final"]["gradient_accumulation_steps"],
        epochs=int(training["final"]["epochs"]),
    )
    base_lock_text = base_text.replace(
        f"ann_file='{annotation_root}/base_dev.json'", "ann_file='annotations/lock_old.json'"
    ).replace("data_prefix=dict(img='images/val/')", "data_prefix=dict(img='images/lock/')")
    base_lock_text = base_lock_text.replace(
        f"ann_file='{relative_data_root}/{annotation_root}/base_dev.json'",
        f"ann_file='{relative_data_root}/annotations/lock_old.json'",
    )
    final_lock_text = final_text.replace(
        f"ann_file='{annotation_root}/{final_val_name}'", "ann_file='annotations/lock_full.json'"
    ).replace("data_prefix=dict(img='images/val/')", "data_prefix=dict(img='images/lock/')")
    final_lock_text = final_lock_text.replace(
        f"ann_file='{relative_data_root}/{annotation_root}/{final_val_name}'",
        f"ann_file='{relative_data_root}/annotations/lock_full.json'",
    )
    for path, content in (
        (base_config, base_text),
        (current_config, current_text),
        (final_config, final_text),
        (base_lock_config, base_lock_text),
        (final_lock_config, final_lock_text),
    ):
        path.write_text(content, encoding="utf-8")
    return {
        "config_root": str(config_root),
        "base_config": str(base_config),
        "current_config": str(current_config),
        "final_config": str(final_config),
        "base_lock_config": str(base_lock_config),
        "final_lock_config": str(final_lock_config),
        "base_work_dir": str(base_work),
        "current_work_dir": str(current_work),
        "final_work_dir": str(final_work),
        "final_init": str(work_root / "final_init.pth"),
    }


def prepare_full_yolo_iod(config: Mapping[str, Any]) -> Dict[str, Any]:
    official_repo = Path(config["paths"]["official_repo"]).resolve()
    if not official_repo.is_dir():
        raise FileNotFoundError(f"官方 YOLO-IOD 仓库不存在：{official_repo}")
    run_id = str(config["experiment"]["run_id"])
    data_root = official_repo / "data" / "agileagent" / run_id
    if data_root.exists():
        raise FileExistsError(f"拒绝覆盖完整 YOLO-IOD 数据视图：{data_root}")
    split_paths = config["paths"]["source_splits"]
    train = read_split(split_paths["train"])
    dev = read_split(split_paths["val"])
    lock_path = Path(split_paths["lock"]).resolve()
    new_global_id = 2

    def partition(rows: Sequence[Path]) -> tuple[list[Path], list[Path]]:
        base, current = [], []
        for image in rows:
            classes = image_class_ids(image)
            if new_global_id in classes:
                if classes != {new_global_id}:
                    raise ValueError(f"strict-p02 新增类图像存在旧类共现：{image.name}")
                current.append(image)
            else:
                base.append(image)
        return base, current

    base_train, current_train = partition(train)
    base_dev, current_dev = partition(dev)
    expected = config["dataset"]["expected_counts"]
    actual = {
        "source_train": len(train),
        "source_dev": len(dev),
        "base_train": len(base_train),
        "base_dev": len(base_dev),
        "current_train": len(current_train),
        "current_dev": len(current_dev),
    }
    for key, value in expected.items():
        if key != "lock" and actual.get(key) != int(value):
            raise ValueError(f"数据数量不符：{key} expected={value} actual={actual.get(key)}")

    for split_name, rows in (("train", train), ("val", dev)):
        for source in rows:
            _link_or_copy(source, data_root / "images" / split_name / source.name)
    _ensure_runtime_directories(data_root)
    annotations = data_root / "annotations"
    counts: Dict[str, Dict[str, int]] = {
        "base_train": _write_coco(annotations / "base_train.json", base_train, {0, 1, 3}),
        "base_dev": _write_coco(annotations / "base_dev.json", base_dev, {0, 1, 3}),
        "current_train": _write_coco(annotations / "current_train.json", current_train, {2}),
        "current_dev": _write_coco(annotations / "current_dev.json", current_dev, {2}),
    }
    if not bool(config.get("dataset", {}).get("incremental_validation_only", False)):
        counts["final_dev"] = _write_coco(
            annotations / "final_dev.json", dev, set(GLOBAL_CLASS_NAMES)
        )
    _write_json(data_root / "texts" / "old.json", [[name] for name in OFFICIAL_OLD_CLASSES])
    _write_json(data_root / "texts" / "current.json", [[OFFICIAL_NEW_CLASS]])
    _write_json(data_root / "texts" / "all.json", [[name] for name in OFFICIAL_ALL_CLASSES])
    generated_configs = generate_official_configs(config, official_repo, data_root)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "protocol": "strict-p02",
        "method": "official_yolo_iod_full",
        "official_commit": OFFICIAL_COMMIT,
        "official_class_order": list(OFFICIAL_ALL_CLASSES),
        "global_to_official": GLOBAL_TO_OFFICIAL,
        "official_to_global": OFFICIAL_TO_GLOBAL,
        "learning_data_scope": "incremental_dataset_only",
        "old_raw_image_count": 0,
        "base_contains_new_class_images": False,
        "incremental_old_class_gt_count": 0,
        "cpr_stage0": "not_applicable_no_future_class_images_in_base_stage",
        "cpr_incremental": (
            "enabled" if cpr_is_enabled(config) else "disabled_no_old_class_cooccurrence"
        ),
        "cpr_gate": {
            "incremental_old_class_gt_count": 0,
            "decision": "run" if cpr_is_enabled(config) else "skip",
        },
        "batch_plan": training_batch_plan(config),
        "counts": actual,
        "annotation_counts": counts,
        "source_split_sha256": {
            name: sha256_file(Path(path).resolve()) for name, path in split_paths.items()
        },
        "lock_split_stems": [image.stem for image in read_split(lock_path)],
        "lock_materialized_after_freeze": False,
        "data_root": str(data_root),
        "generated_configs": generated_configs,
        "runtime_devices": list(config["runtime"]["devices"]),
    }
    _write_json(data_root / "manifest.json", manifest)
    return manifest


def materialize_full_yolo_iod_lock(config: Mapping[str, Any], frozen_checkpoint: Path) -> Dict[str, Any]:
    official_repo = Path(config["paths"]["official_repo"]).resolve()
    run_id = str(config["experiment"]["run_id"])
    data_root = official_repo / "data" / "agileagent" / run_id
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("lock_materialized_after_freeze"):
        raise FileExistsError("完整 YOLO-IOD lock-val 已经物化")
    if not frozen_checkpoint.is_file():
        raise FileNotFoundError(f"最终冻结权重不存在：{frozen_checkpoint}")
    split_path = Path(config["paths"]["source_splits"]["lock"]).resolve()
    if sha256_file(split_path) != manifest["source_split_sha256"]["lock"]:
        raise ValueError("lock-val 划分在训练后发生变化")
    lock = read_split(split_path)
    if [image.stem for image in lock] != manifest["lock_split_stems"]:
        raise ValueError("lock-val stem 清单在训练后发生变化")
    for source in lock:
        _link_or_copy(source, data_root / "images" / "lock" / source.name)
    manifest["annotation_counts"]["lock_old"] = _write_coco(
        data_root / "annotations" / "lock_old.json", lock, {0, 1, 3}
    )
    manifest["annotation_counts"]["lock_full"] = _write_coco(
        data_root / "annotations" / "lock_full.json", lock, set(GLOBAL_CLASS_NAMES)
    )
    manifest["lock_materialized_after_freeze"] = True
    manifest["frozen_checkpoint"] = str(frozen_checkpoint)
    manifest["frozen_checkpoint_sha256"] = sha256_file(frozen_checkpoint)
    _write_json(manifest_path, manifest)
    return manifest


def count_cpr_pseudo_labels(source_json: Path, cpr_json: Path) -> Dict[str, int]:
    source = json.loads(source_json.read_text(encoding="utf-8"))
    refined = json.loads(cpr_json.read_text(encoding="utf-8"))
    source_ids = {int(row["id"]) for row in source.get("annotations", [])}
    pseudo = [row for row in refined.get("annotations", []) if int(row["id"]) not in source_ids]
    old_category_ids = {1, 2, 3}
    return {
        "source_annotations": len(source.get("annotations", [])),
        "refined_annotations": len(refined.get("annotations", [])),
        "pseudo_annotations": len(pseudo),
        "old_class_pseudo_annotations": sum(
            int(row.get("category_id", -1)) in old_category_ids for row in pseudo
        ),
    }


def summarize_disabled_cpr(source_json: Path, reason: str) -> Dict[str, Any]:
    source = json.loads(source_json.read_text(encoding="utf-8"))
    source_count = len(source.get("annotations", []))
    return {
        "enabled": False,
        "status": "skipped_by_data_gate",
        "reason": reason,
        "source_annotations": source_count,
        "refined_annotations": source_count,
        "pseudo_annotations": 0,
        "old_class_pseudo_annotations": 0,
    }


def write_command_manifest(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> Dict[str, Any]:
    official_repo = Path(config["paths"]["official_repo"]).resolve()
    python = Path(config["runtime"]["python"]).resolve()
    generated = manifest["generated_configs"]
    runtime_devices = [str(item) for item in config["runtime"]["devices"]]
    devices = ",".join(runtime_devices)
    gpu_count = len(runtime_devices)
    work = Path(generated["base_work_dir"]).parent
    training = config["training"]
    final_devices = [str(item) for item in training["final"].get("devices", runtime_devices)]
    if not final_devices or len(final_devices) != len(set(final_devices)):
        raise ValueError("training.final.devices 必须声明不重复的 GPU")
    checkpoints = {
        "base": str(Path(generated["base_work_dir"]) / f"epoch_{training['base']['epochs']}.pth"),
        "current": str(
            Path(generated["current_work_dir"]) / f"epoch_{training['current_teacher']['epochs']}.pth"
        ),
        "final": str(Path(generated["final_work_dir"]) / f"epoch_{training['final']['epochs']}.pth"),
    }
    common_env = {
        "CUDA_VISIBLE_DEVICES": devices,
        "PATH": f"{python.parent}:{os.environ.get('PATH', '')}",
        "PYTHONPATH": str(official_repo),
    }
    commands = {
        "base": [
            "bash",
            str(official_repo / "tools/dist_train_gps.sh"),
            generated["base_config"],
            str(gpu_count),
            "--amp",
            "--work-dir",
            generated["base_work_dir"],
        ],
        "current": [
            "bash",
            str(official_repo / "tools/dist_train_gps.sh"),
            generated["current_config"],
            str(gpu_count),
            "--amp",
            "--work-dir",
            generated["current_work_dir"],
        ],
        "final": [
            "bash",
            str(official_repo / "tools/dist_train_gps.sh"),
            generated["final_config"],
            str(len(final_devices)),
            "--amp",
            "--work-dir",
            generated["final_work_dir"],
        ],
    }
    result = {
        "schema_version": 1,
        "run_id": config["experiment"]["run_id"],
        "cwd": str(official_repo),
        "env": common_env,
        "stage_devices": {
            "base": runtime_devices,
            "current": runtime_devices,
            "final": final_devices,
        },
        "batch_plan": training_batch_plan(config),
        "commands": commands,
        "checkpoints": checkpoints,
        "final_init": generated["final_init"],
        "work_root": str(work),
    }
    _write_json(Path(manifest["data_root"]) / "command_manifest.json", result)
    return result
