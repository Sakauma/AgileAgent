from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def is_class_output_key(key: str) -> bool:
    lowered = key.lower()
    return ".cv3." in lowered and (
        lowered.endswith(".2.weight") or lowered.endswith(".2.bias")
    )


def is_task_shared_key(key: str, exclude_patterns: Iterable[str]) -> bool:
    lowered = key.lower()
    return not any(str(pattern).lower() in lowered for pattern in exclude_patterns)


def merge_task_vectors(
    reference: Mapping[str, Any],
    old_task: Mapping[str, Any],
    new_task: Mapping[str, Any],
    template: Mapping[str, Any],
    *,
    alpha_old: float,
    alpha_new: float,
    shared_key_exclude: Sequence[str],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Merge compatible shared tensors and preserve old task-specific tensors."""
    import torch

    merged: Dict[str, Any] = {}
    merged_keys: list[str] = []
    preserved_keys: list[str] = []
    for key, target in template.items():
        ref = reference.get(key)
        old = old_task.get(key)
        new = new_task.get(key)
        compatible = (
            ref is not None
            and old is not None
            and new is not None
            and ref.shape == old.shape == new.shape == target.shape
            and torch.is_floating_point(ref)
            and is_task_shared_key(key, shared_key_exclude)
        )
        if compatible:
            value = ref.float() + alpha_old * (old.float() - ref.float())
            value = value + alpha_new * (new.float() - ref.float())
            merged[key] = value.to(dtype=target.dtype)
            merged_keys.append(key)
        elif old is not None and old.shape == target.shape:
            merged[key] = old.detach().clone().to(dtype=target.dtype)
            preserved_keys.append(key)
        elif new is not None and new.shape == target.shape:
            merged[key] = new.detach().clone().to(dtype=target.dtype)
        elif ref is not None and ref.shape == target.shape:
            merged[key] = ref.detach().clone().to(dtype=target.dtype)
        else:
            merged[key] = target.detach().clone()
    return merged, {
        "merged_shared_tensor_count": len(merged_keys),
        "preserved_old_tensor_count": len(preserved_keys),
        "shared_key_exclude": list(shared_key_exclude),
    }


def copy_incremental_head_rows(
    state: Dict[str, Any],
    old_state: Mapping[str, Any],
    new_state: Mapping[str, Any],
    base_local_to_global: Mapping[int, int],
    new_global_id: int,
) -> Dict[str, int]:
    """Copy old class rows from the base detector and the new row from its task model."""
    old_rows = 0
    new_rows = 0
    mapping = {int(local): int(global_id) for local, global_id in base_local_to_global.items()}
    for key, value in state.items():
        if not is_class_output_key(key):
            continue
        old_value = old_state.get(key)
        new_value = new_state.get(key)
        if old_value is None or new_value is None:
            raise ValueError(f"增量检测头缺少分类张量：{key}")
        if value.shape[1:] != old_value.shape[1:] or value.shape[1:] != new_value.shape[1:]:
            raise ValueError(f"增量检测头分类张量形状不兼容：{key}")
        for local_id, global_id in mapping.items():
            value[global_id].copy_(old_value[local_id].to(value.device, value.dtype))
            old_rows += 1
        value[int(new_global_id)].copy_(new_value[0].to(value.device, value.dtype))
        new_rows += 1
    if not old_rows or not new_rows:
        raise ValueError("没有找到 YOLO11 分类输出行")
    return {"copied_old_rows": old_rows, "copied_new_rows": new_rows}


def build_duet_checkpoint(
    reference_weight: Path,
    base_weight: Path,
    current_weight: Path,
    output_weight: Path,
    *,
    class_names: Mapping[int, str],
    base_local_to_global: Mapping[int, int],
    new_global_id: int,
    alpha_old: float,
    alpha_new: float,
    shared_key_exclude: Sequence[str],
) -> Dict[str, Any]:
    """Build one four-class YOLO checkpoint from two task vectors and disjoint head rows."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    reference_yolo = YOLO(str(reference_weight))
    base_yolo = YOLO(str(base_weight))
    current_yolo = YOLO(str(current_weight))
    model_yaml = deepcopy(current_yolo.model.yaml)
    model_yaml["nc"] = len(class_names)
    expanded = DetectionModel(cfg=model_yaml, ch=3, nc=len(class_names), verbose=False)
    expanded.names = {int(key): str(value) for key, value in class_names.items()}

    merged, report = merge_task_vectors(
        reference_yolo.model.state_dict(),
        base_yolo.model.state_dict(),
        current_yolo.model.state_dict(),
        expanded.state_dict(),
        alpha_old=float(alpha_old),
        alpha_new=float(alpha_new),
        shared_key_exclude=shared_key_exclude,
    )
    report.update(
        copy_incremental_head_rows(
            merged,
            base_yolo.model.state_dict(),
            current_yolo.model.state_dict(),
            base_local_to_global,
            new_global_id,
        )
    )
    expanded.load_state_dict(merged, strict=True)
    reference_yolo.model = expanded
    reference_yolo.ckpt = {}
    output_weight.parent.mkdir(parents=True, exist_ok=False)
    reference_yolo.save(output_weight)
    report.update(
        {
            "method": "duet_yolo11s",
            "alpha_old": float(alpha_old),
            "alpha_new": float(alpha_new),
            "output_weight": str(output_weight),
        }
    )
    return report


def initialize_unified_student(
    student: Any,
    base: Any,
    current_teacher: Any,
    base_local_to_global: Mapping[int, int],
    new_global_id: int,
) -> Dict[str, int]:
    """Initialize a four-class student with old rows and a trained new-class row."""
    import torch

    student_state = student.state_dict()
    base_state = base.state_dict()
    current_state = current_teacher.state_dict()
    with torch.no_grad():
        for key, target in student_state.items():
            old = base_state.get(key)
            if old is not None and old.shape == target.shape:
                target.copy_(old)
        copied = copy_incremental_head_rows(
            student_state,
            base_state,
            current_state,
            base_local_to_global,
            new_global_id,
        )
    student.load_state_dict(student_state, strict=True)
    return copied


def _importance_masks(
    model: Any,
    reference_state: Mapping[str, Any],
    current_state: Mapping[str, Any],
    ratio: float,
    excluded_patterns: Sequence[str],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Select output kernels whose current-task vector has the largest magnitude."""
    import torch

    masks: Dict[str, Any] = {}
    selected = 0
    available = 0
    by_parameter: Dict[str, Dict[str, int]] = {}
    for name, parameter in model.named_parameters():
        if any(pattern.lower() in name.lower() for pattern in excluded_patterns):
            continue
        reference = reference_state.get(name)
        current = current_state.get(name)
        if reference is None or current is None or reference.shape != current.shape:
            continue
        if parameter.ndim == 0:
            continue
        score = (current.float() - reference.float()).abs()
        score = score.reshape(score.shape[0], -1).mean(dim=1)
        count = score.numel()
        keep = min(count, max(1, int(math.ceil(count * float(ratio)))))
        indices = torch.topk(score, keep, largest=True, sorted=False).indices
        channel_mask = torch.zeros(count, dtype=parameter.dtype, device=parameter.device)
        channel_mask[indices.to(parameter.device)] = 1
        shape = [count] + [1] * (parameter.ndim - 1)
        masks[name] = channel_mask.reshape(shape)
        selected += keep
        available += count
        by_parameter[name] = {"selected": keep, "available": count}
    return masks, {
        "selection_ratio": float(ratio),
        "selected_output_channels": selected,
        "available_output_channels": available,
        "parameters": by_parameter,
    }


def _set_teacher_dtype(teacher: Any, image: Any) -> None:
    first = next(teacher.parameters(), None)
    if first is not None and (first.device != image.device or first.dtype != image.dtype):
        teacher.to(device=image.device, dtype=image.dtype)
    teacher.eval()


def _confidence_mask(scores: Any, quantile: float, minimum: float) -> Any:
    import torch

    confidence = scores.detach().sigmoid().amax(dim=1)
    threshold = torch.quantile(confidence.float().flatten(), float(quantile))
    threshold = max(float(threshold.item()), float(minimum))
    return confidence >= threshold


def _classification_distillation(student: Any, teacher: Any, mask: Any) -> Any:
    import torch.nn.functional as F

    if not mask.any():
        return student.new_tensor(0.0)
    student_rows = student.permute(0, 2, 1)[mask]
    teacher_rows = teacher.detach().permute(0, 2, 1)[mask]
    return F.mse_loss(student_rows.sigmoid(), teacher_rows.sigmoid())


def _bbox_distillation(student: Any, teacher: Any, mask: Any) -> Any:
    import torch.nn.functional as F

    if student.shape != teacher.shape or not mask.any():
        return student.new_tensor(0.0)
    batch, channels, anchors = student.shape
    reg_max = channels // 4
    student_dfl = student.view(batch, 4, reg_max, anchors).permute(0, 3, 1, 2)
    teacher_dfl = teacher.detach().view(batch, 4, reg_max, anchors).permute(0, 3, 1, 2)
    student_rows = student_dfl[mask]
    teacher_rows = teacher_dfl[mask]
    return F.kl_div(
        F.log_softmax(student_rows.float(), dim=-1),
        F.softmax(teacher_rows.float(), dim=-1),
        reduction="batchmean",
    )


def _feature_distillation(student_features: Sequence[Any], teacher_features: Sequence[Any]) -> Any:
    import torch.nn.functional as F

    losses = []
    for student, teacher in zip(student_features, teacher_features):
        if student.shape != teacher.shape:
            continue
        scale = teacher.detach().float().square().mean().sqrt().clamp_min(1e-4)
        losses.append(F.mse_loss(student.float() / scale, teacher.detach().float() / scale))
    return sum(losses) / len(losses) if losses else student_features[0].new_tensor(0.0)


def _criterion_class() -> type:
    import torch
    import torch.nn.functional as F
    from ultralytics.utils.loss import v8DetectionLoss

    class IncrementalDetectionLoss(v8DetectionLoss):
        def __init__(
            self,
            model: Any,
            *,
            method: str,
            old_teacher: Any | None,
            current_teacher: Any | None,
            old_class_indices: Sequence[int],
            new_class_index: int,
            weights: Mapping[str, float],
            quantile: float,
            minimum_confidence: float,
            direction_terms: Sequence[tuple[Any, Any, Any]],
        ) -> None:
            super().__init__(model)
            self.method = method
            self.model = model
            self.old_teacher = old_teacher
            self.current_teacher = current_teacher
            self.old_class_indices = tuple(int(value) for value in old_class_indices)
            self.new_class_index = int(new_class_index)
            self.weights = {str(key): float(value) for key, value in weights.items()}
            self.quantile = float(quantile)
            self.minimum_confidence = float(minimum_confidence)
            self.direction_terms = list(direction_terms)
            self.last_auxiliary: Dict[str, float] = {}

        def __call__(self, preds: Any, batch: Mapping[str, Any]) -> tuple[Any, Any]:
            parsed = self.parse_output(preds)
            detector_loss, loss_items = super().loss(parsed, batch)
            image = batch["img"]
            batch_size = int(image.shape[0])
            auxiliary: Dict[str, Any] = {}

            old_output = None
            if self.old_teacher is not None:
                _set_teacher_dtype(self.old_teacher, image)
                with torch.no_grad():
                    old_output = self.parse_output(self.old_teacher(image))
            current_output = None
            if self.current_teacher is not None:
                _set_teacher_dtype(self.current_teacher, image)
                with torch.no_grad():
                    current_output = self.parse_output(self.current_teacher(image))

            if old_output is not None:
                old_mask = _confidence_mask(
                    old_output["scores"], self.quantile, self.minimum_confidence
                )
                if self.method == "yolo_iod_lite":
                    old_index = torch.tensor(
                        self.old_class_indices, device=parsed["scores"].device, dtype=torch.long
                    )
                    old_scores = parsed["scores"].index_select(1, old_index)
                    auxiliary["old_class"] = _classification_distillation(
                        old_scores, old_output["scores"], old_mask
                    )
                auxiliary["old_bbox"] = _bbox_distillation(
                    parsed["boxes"], old_output["boxes"], old_mask
                )
                auxiliary["old_feature"] = _feature_distillation(
                    parsed["feats"], old_output["feats"]
                )

            if current_output is not None:
                current_mask = _confidence_mask(
                    current_output["scores"], self.quantile, self.minimum_confidence
                )
                new_scores = parsed["scores"][:, self.new_class_index : self.new_class_index + 1]
                auxiliary["current_class"] = _classification_distillation(
                    new_scores, current_output["scores"], current_mask
                )
                auxiliary["current_bbox"] = _bbox_distillation(
                    parsed["boxes"], current_output["boxes"], current_mask
                )
                auxiliary["current_feature"] = _feature_distillation(
                    parsed["feats"], current_output["feats"]
                )

            if self.direction_terms:
                direction_losses = []
                for parameter, reference, old_delta in self.direction_terms:
                    current_delta = parameter.float() - reference
                    denominator = (current_delta.norm() * old_delta.norm()).clamp_min(1e-12)
                    cosine = torch.dot(current_delta.flatten(), old_delta.flatten()) / denominator
                    direction_losses.append(F.relu(-cosine))
                auxiliary["direction"] = (
                    torch.stack(direction_losses).mean()
                    if direction_losses
                    else parsed["boxes"].new_tensor(0.0)
                )

            total = detector_loss.sum()
            for name, value in auxiliary.items():
                total = total + batch_size * self.weights.get(name, 0.0) * value
            self.last_auxiliary = {
                name: float(value.detach().float().item()) for name, value in auxiliary.items()
            }
            return total, loss_items

    return IncrementalDetectionLoss


def _trainer_model(trainer: Any) -> Any:
    return trainer.model.module if hasattr(trainer.model, "module") else trainer.model


def configure_duet_specialist(
    trainer: Any,
    *,
    base_weight: Path,
    reference_weight: Path,
    settings: Mapping[str, Any],
) -> None:
    """Attach DuET-style distillation and directional consistency to a new-task detector."""
    from ultralytics import YOLO

    model = _trainer_model(trainer)
    base = YOLO(str(base_weight)).model
    reference = YOLO(str(reference_weight)).model
    reference_state = reference.state_dict()
    old_state = base.state_dict()
    excluded = tuple(settings.get("shared_key_exclude", ["model.23"]))
    direction_terms = []
    for name, parameter in model.named_parameters():
        ref = reference_state.get(name)
        old = old_state.get(name)
        if (
            ref is None
            or old is None
            or ref.shape != old.shape
            or ref.shape != parameter.shape
            or not is_task_shared_key(name, excluded)
        ):
            continue
        ref_device = ref.detach().float().to(parameter.device)
        old_delta = old.detach().float().to(parameter.device) - ref_device
        if float(old_delta.norm().item()) > 1e-12:
            direction_terms.append((parameter, ref_device, old_delta))
    criterion = _criterion_class()(
        model,
        method="duet_yolo11s",
        old_teacher=base,
        current_teacher=None,
        old_class_indices=(),
        new_class_index=0,
        weights=settings.get("loss_weights", {}),
        quantile=float(settings.get("distill_quantile", 0.75)),
        minimum_confidence=float(settings.get("minimum_teacher_confidence", 0.01)),
        direction_terms=direction_terms,
    )
    model.criterion = criterion
    trainer._incremental_method_audit = {
        "method": "duet_yolo11s",
        "direction_tensor_count": len(direction_terms),
        "shared_key_exclude": list(excluded),
        "loss_weights": dict(settings.get("loss_weights", {})),
        "old_raw_image_count": 0,
    }


def configure_yolo_iod_lite_student(
    trainer: Any,
    *,
    base_weight: Path,
    current_teacher_weight: Path,
    reference_weight: Path,
    base_local_to_global: Mapping[int, int],
    new_global_id: int,
    settings: Mapping[str, Any],
) -> None:
    """Initialize and constrain the unified student using IKS and asymmetric KD."""
    import torch
    from ultralytics import YOLO

    model = _trainer_model(trainer)
    base = YOLO(str(base_weight)).model
    current = YOLO(str(current_teacher_weight)).model
    reference = YOLO(str(reference_weight)).model
    copied = initialize_unified_student(
        model, base, current, base_local_to_global, new_global_id
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    masks, selection = _importance_masks(
        model,
        reference.state_dict(),
        current.state_dict(),
        float(settings.get("kernel_selection_ratio", 0.25)),
        tuple(settings.get("kernel_selection_exclude", [".cv3."])),
    )
    parameters = dict(model.named_parameters())
    for name, mask in masks.items():
        parameter = parameters[name]
        parameter.requires_grad_(True)
        parameter.register_hook(lambda gradient, value=mask: gradient * value)

    old_ids = sorted(int(value) for value in base_local_to_global.values())
    protected = []
    branches = getattr(model.model[-1], "cv3", None)
    if branches is None:
        raise RuntimeError("YOLO11 检测头缺少 cv3 分类分支")
    for branch in branches:
        classifier = branch[-1]
        classifier.weight.requires_grad_(True)
        classifier.bias.requires_grad_(True)
        weight = classifier.weight[old_ids].detach().clone()
        bias = classifier.bias[old_ids].detach().clone()
        protected.append((old_ids, weight, bias))

        def row_mask(gradient: Any, class_id: int = int(new_global_id)) -> Any:
            mask = torch.zeros_like(gradient)
            mask[class_id] = 1
            return gradient * mask

        classifier.weight.register_hook(row_mask)
        classifier.bias.register_hook(row_mask)
    trainer._protected_old_rows = protected

    criterion = _criterion_class()(
        model,
        method="yolo_iod_lite",
        old_teacher=base,
        current_teacher=current,
        old_class_indices=old_ids,
        new_class_index=int(new_global_id),
        weights=settings.get("loss_weights", {}),
        quantile=float(settings.get("distill_quantile", 0.75)),
        minimum_confidence=float(settings.get("minimum_teacher_confidence", 0.01)),
        direction_terms=(),
    )
    model.criterion = criterion
    ema = getattr(getattr(trainer, "ema", None), "ema", None)
    if ema is not None:
        ema.load_state_dict(model.state_dict())
        trainer.ema.updates = 0
    trainer._incremental_method_audit = {
        "method": "yolo_iod_lite",
        "head_rows": copied,
        "kernel_selection": selection,
        "loss_weights": dict(settings.get("loss_weights", {})),
        "cpr": {
            "status": "not_applicable",
            "reason": "warship_increment_has_no_old_class_cooccurrence",
            "pseudo_label_count": 0,
        },
        "old_raw_image_count": 0,
    }
    restore_protected_old_rows(trainer)


def restore_protected_old_rows(trainer: Any) -> None:
    """Undo optimizer weight decay on frozen old class rows in model and EMA."""
    import torch

    protected = getattr(trainer, "_protected_old_rows", [])
    if not protected:
        return
    candidates = [_trainer_model(trainer)]
    ema = getattr(getattr(trainer, "ema", None), "ema", None)
    if ema is not None:
        candidates.append(ema)
    with torch.no_grad():
        for candidate in candidates:
            for module in candidate.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            branches = getattr(candidate.model[-1], "cv3", None)
            if branches is None:
                continue
            for branch, (old_ids, weight, bias) in zip(branches, protected):
                classifier = branch[-1]
                classifier.weight[old_ids].copy_(weight.to(classifier.weight.device, classifier.weight.dtype))
                classifier.bias[old_ids].copy_(bias.to(classifier.bias.device, classifier.bias.dtype))


def old_classification_row_drift(
    base_weight: Path,
    student_weight: Path,
    base_local_to_global: Mapping[int, int],
) -> float:
    from ultralytics import YOLO

    base_state = YOLO(str(base_weight)).model.state_dict()
    student_state = YOLO(str(student_weight)).model.state_dict()
    maximum = 0.0
    for key, before in base_state.items():
        if not is_class_output_key(key) or key not in student_state:
            continue
        after = student_state[key]
        for local_id, global_id in base_local_to_global.items():
            delta = (after[int(global_id)].float() - before[int(local_id)].float()).abs().max()
            maximum = max(maximum, float(delta.item()))
    return maximum


def shared_parameter_relative_drift(
    base_weight: Path,
    student_weight: Path,
    shared_key_exclude: Sequence[str] = ("model.23",),
) -> float:
    import torch
    from ultralytics import YOLO

    base_state = YOLO(str(base_weight)).model.state_dict()
    student_state = YOLO(str(student_weight)).model.state_dict()
    numerator = torch.tensor(0.0)
    denominator = torch.tensor(0.0)
    for key, before in base_state.items():
        after = student_state.get(key)
        if (
            after is None
            or before.shape != after.shape
            or not torch.is_floating_point(before)
            or not is_task_shared_key(key, shared_key_exclude)
        ):
            continue
        numerator += (after.float() - before.float()).square().sum()
        denominator += before.float().square().sum()
    return float((numerator.sqrt() / denominator.sqrt().clamp_min(1e-12)).item())
