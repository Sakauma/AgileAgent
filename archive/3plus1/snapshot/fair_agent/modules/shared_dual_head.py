from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from ultralytics.nn.modules.head import Detect

from fair_agent.core.hashes import sha256_file


YOLO_DETECT_HEAD_PREFIX = "model.23."
DUAL_HEAD_OUTPUT_NAMES = ("old_raw_output", "new_raw_output")


class ResidualAdaptedDetect(Detect):
    """Apply zero-initialized residual 1x1 adapters before a YOLO Detect head."""

    def __init__(self, detect: torch.nn.Module, channels: list[int]) -> None:
        torch.nn.Module.__init__(self)
        if len(channels) != int(detect.nl):
            raise ValueError("residual adapter数量与Detect特征层数量不一致")
        self.detect = detect
        self.adapters = torch.nn.ModuleList(
            torch.nn.Conv2d(channel, channel, 1, bias=True)
            for channel in channels
        )
        for adapter in self.adapters:
            torch.nn.init.zeros_(adapter.weight)
            torch.nn.init.zeros_(adapter.bias)
        for name in ("i", "f", "type", "np", "nc", "nl", "reg_max", "no"):
            setattr(self, name, getattr(detect, name))
        self.type = f"{type(self).__module__}.{type(self).__name__}"
        self.np = sum(parameter.numel() for parameter in self.parameters())
        self._synchronize_detect_tensors()

    def _synchronize_detect_tensors(self) -> None:
        for name in ("stride", "anchors", "strides"):
            if hasattr(self.detect, name):
                setattr(self, name, getattr(self.detect, name))

    def _apply(self, fn: Any) -> "ResidualAdaptedDetect":
        super()._apply(fn)
        for name in ("stride", "anchors", "strides"):
            value = getattr(self.detect, name, None)
            if isinstance(value, torch.Tensor):
                setattr(self.detect, name, fn(value))
        self._synchronize_detect_tensors()
        return self

    def forward(self, features: list[torch.Tensor]) -> Any:
        if len(features) != len(self.adapters):
            raise RuntimeError("residual adapter收到的特征层数量不匹配")
        if getattr(self, "export_collapsed", False):
            adapted = [
                adapter(feature)
                for feature, adapter in zip(features, self.adapters)
            ]
        else:
            adapted = [
                feature + adapter(feature)
                for feature, adapter in zip(features, self.adapters)
            ]
        return self.detect(adapted)


def collapse_residual_adapter_for_export(
    wrapper: ResidualAdaptedDetect,
) -> dict[str, Any]:
    """Fold each identity skip into its 1x1 adapter before FP16 export."""

    if not isinstance(wrapper, ResidualAdaptedDetect):
        raise TypeError("只能折叠ResidualAdaptedDetect")
    if getattr(wrapper, "export_collapsed", False):
        raise ValueError("residual adapter已经折叠")
    with torch.no_grad():
        for adapter in wrapper.adapters:
            channels = int(adapter.weight.shape[0])
            identity = torch.eye(
                channels,
                device=adapter.weight.device,
                dtype=adapter.weight.dtype,
            ).reshape(channels, channels, 1, 1)
            adapter.weight.add_(identity)
    wrapper.export_collapsed = True
    return {
        "kind": "identity_folded_1x1",
        "adapter_count": len(wrapper.adapters),
        "explicit_add_removed": True,
    }


def attach_residual_feature_adapters(model: Any) -> dict[str, Any]:
    """Wrap the final Detect head with trainable, zero-initialized adapters."""

    detect = model.model[-1]
    if isinstance(detect, ResidualAdaptedDetect):
        raise ValueError("模型已经包含residual adapter")
    channels = [int(branch[0].conv.in_channels) for branch in detect.cv2]
    wrapper = ResidualAdaptedDetect(detect, channels)
    model.model[-1] = wrapper
    return {
        "channels": channels,
        "parameter_count": sum(
            parameter.numel() for parameter in wrapper.adapters.parameters()
        ),
        "zero_initialized": all(
            int(torch.count_nonzero(parameter).item()) == 0
            for parameter in wrapper.adapters.parameters()
        ),
    }


def residual_adapter_detection_trainer() -> type[Any]:
    """Return a trainer that injects adapters after Ultralytics rebuilds the model."""

    from ultralytics.models.yolo.detect import DetectionTrainer

    class P10ResidualAdapterTrainer(DetectionTrainer):
        def get_model(
            self,
            cfg: str | None = None,
            weights: str | None = None,
            verbose: bool = True,
        ) -> Any:
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            self.residual_adapter_report = attach_residual_feature_adapters(model)
            if not isinstance(model.model[-1], ResidualAdaptedDetect):
                raise RuntimeError("P10 residual adapter注入失败")
            return model

    return P10ResidualAdapterTrainer


def copy_shared_backbone_state(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    head_prefix: str = YOLO_DETECT_HEAD_PREFIX,
) -> int:
    """Copy every compatible backbone/neck tensor while leaving the Detect head intact."""

    copied = 0
    for key, source_value in source.items():
        if key.startswith(head_prefix):
            continue
        target_value = target.get(key)
        if target_value is None or target_value.shape != source_value.shape:
            raise ValueError(f"共享骨干张量不兼容：{key}")
        target_value.copy_(
            source_value.to(device=target_value.device, dtype=target_value.dtype)
        )
        copied += 1
    if copied <= 0:
        raise ValueError("没有找到可复制的共享骨干张量")
    return copied


def maximum_shared_backbone_drift(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    head_prefix: str = YOLO_DETECT_HEAD_PREFIX,
) -> float:
    maximum = 0.0
    compared = 0
    for key, reference_value in reference.items():
        if key.startswith(head_prefix):
            continue
        candidate_value = candidate.get(key)
        if candidate_value is None or candidate_value.shape != reference_value.shape:
            raise ValueError(f"共享骨干张量不兼容：{key}")
        difference = (
            candidate_value.detach().float().cpu()
            - reference_value.detach().float().cpu()
        ).abs()
        if difference.numel():
            maximum = max(maximum, float(difference.max().item()))
        compared += 1
    if compared <= 0:
        raise ValueError("没有找到可比较的共享骨干张量")
    return maximum


def build_shared_head_training_checkpoint(
    base_weight: str | Path,
    head_init_weight: str | Path,
    target: str | Path,
    *,
    residual_adapter: bool = False,
) -> dict[str, Any]:
    """Create a one-class training checkpoint with Base features and Specialist head."""

    import torch
    from ultralytics import YOLO

    base_path = Path(base_weight).resolve()
    head_init_path = Path(head_init_weight).resolve()
    target_path = Path(target).resolve()
    if target_path.exists():
        raise FileExistsError(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    base = YOLO(str(base_path))
    hybrid = YOLO(str(head_init_path))
    state = hybrid.model.state_dict()
    with torch.no_grad():
        copied = copy_shared_backbone_state(base.model.state_dict(), state)
    hybrid.model.load_state_dict(state, strict=True)
    adapter = (
        attach_residual_feature_adapters(hybrid.model)
        if residual_adapter
        else None
    )
    hybrid.ckpt = {}
    hybrid.save(target_path)
    saved = YOLO(str(target_path)).model.state_dict()
    drift = maximum_shared_backbone_drift(base.model.state_dict(), saved)
    if drift != 0.0:
        raise RuntimeError(f"共享骨干初始化发生漂移：{drift}")
    return {
        "path": str(target_path),
        "sha256": sha256_file(target_path),
        "base_weight": str(base_path),
        "base_weight_sha256": sha256_file(base_path),
        "head_init_weight": str(head_init_path),
        "head_init_weight_sha256": sha256_file(head_init_path),
        "residual_adapter": adapter,
        "shared_tensor_count": copied,
        "shared_max_drift": drift,
    }


class SharedBackboneFreezeGuard:
    """Keep Base parameters, buffers and EMA bitwise fixed during head-only training."""

    def __init__(self, base_weight: str | Path) -> None:
        from ultralytics import YOLO

        self.base_path = Path(base_weight).resolve()
        self.reference = {
            key: value.detach().cpu().clone()
            for key, value in YOLO(str(self.base_path)).model.state_dict().items()
            if not key.startswith(YOLO_DETECT_HEAD_PREFIX)
        }

    @staticmethod
    def _model(trainer: Any) -> Any:
        model = trainer.model
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def _set_backbone_eval(model: Any) -> None:
        import torch

        for module in model.model[:23].modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    def batch_start(self, trainer: Any) -> None:
        self._set_backbone_eval(self._model(trainer))

    def restore(self, trainer: Any) -> None:
        import torch

        candidates = [self._model(trainer)]
        ema = getattr(getattr(trainer, "ema", None), "ema", None)
        if ema is not None:
            candidates.append(ema.module if hasattr(ema, "module") else ema)
        with torch.no_grad():
            for model in candidates:
                state = model.state_dict()
                for key, reference in self.reference.items():
                    target = state.get(key)
                    if target is None or target.shape != reference.shape:
                        raise RuntimeError(f"训练中的共享骨干张量不兼容：{key}")
                    target.copy_(
                        reference.to(device=target.device, dtype=target.dtype)
                    )
            self._set_backbone_eval(self._model(trainer))

    def weight_drift(self, candidate_weight: str | Path) -> float:
        from ultralytics import YOLO

        candidate = YOLO(str(Path(candidate_weight).resolve())).model.state_dict()
        return maximum_shared_backbone_drift(self.reference, candidate)


def configure_map50_checkpointing(trainer: Any) -> None:
    """Make best.pt follow the competition's mAP50 score instead of mAP50-95."""

    from types import MethodType

    metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
    box = getattr(metrics, "box", None)
    if box is None:
        raise RuntimeError("Ultralytics validator 尚未初始化，无法按 mAP50 选择权重")

    def map50_fitness(metric: Any) -> float:
        return float(metric.map50)

    box.fitness = MethodType(map50_fitness, box)
    trainer._checkpoint_metric = "metrics/mAP50(B)"


def compose_shared_dual_head(base_model: Any, new_head_model: Any) -> Any:
    """Compose one export module that evaluates the Base graph once and two heads."""

    import torch

    if len(base_model.model) != len(new_head_model.model):
        raise ValueError("Base与新类模型层数不一致")
    old_head = base_model.model[-1]
    new_head = new_head_model.model[-1]
    if list(old_head.f) != list(new_head.f):
        raise ValueError("Base与新类Detect head的特征层索引不一致")

    class SharedBackboneDualHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                copy.deepcopy(list(base_model.model[:-1]))
            )
            self.old_head = copy.deepcopy(old_head)
            self.new_head = copy.deepcopy(new_head)
            self.save = set(int(value) for value in base_model.save)

        @staticmethod
        def _prediction(value: Any) -> Any:
            if isinstance(value, tuple):
                return value[0]
            return value

        def forward(self, images: Any) -> tuple[Any, Any]:
            outputs: list[Any] = []
            value = images
            for layer in self.layers:
                if layer.f != -1:
                    value = (
                        outputs[layer.f]
                        if isinstance(layer.f, int)
                        else [
                            value if index == -1 else outputs[index]
                            for index in layer.f
                        ]
                    )
                value = layer(value)
                outputs.append(value if int(layer.i) in self.save else None)
            features = [outputs[index] for index in self.old_head.f]
            old_output = self.old_head(list(features))
            new_output = self.new_head(list(features))
            return self._prediction(old_output), self._prediction(new_output)

    return SharedBackboneDualHead()


def build_shared_dual_head_export_module(
    base_weight: str | Path,
    new_head_weight: str | Path,
    device: str = "cuda:0",
) -> tuple[Any, dict[str, Any]]:
    from ultralytics import YOLO

    base_path = Path(base_weight).resolve()
    new_path = Path(new_head_weight).resolve()
    base = copy.deepcopy(YOLO(str(base_path)).model).to(device).eval()
    new = copy.deepcopy(YOLO(str(new_path)).model).to(device).eval()
    drift = maximum_shared_backbone_drift(
        base.state_dict(), new.state_dict()
    )
    if drift != 0.0:
        raise RuntimeError(f"新类head checkpoint共享骨干发生漂移：{drift}")
    if hasattr(base, "fuse"):
        base = base.fuse().eval()
    if hasattr(new, "fuse"):
        new = new.fuse().eval()
    adapter_export = (
        collapse_residual_adapter_for_export(new.model[-1])
        if isinstance(new.model[-1], ResidualAdaptedDetect)
        else None
    )
    module = compose_shared_dual_head(base, new).to(device).eval()
    for layer in module.modules():
        if hasattr(layer, "export"):
            layer.export = True
        if hasattr(layer, "format"):
            layer.format = "onnx"
        if hasattr(layer, "dynamic"):
            layer.dynamic = False
    return module, {
        "base_weight": str(base_path),
        "base_weight_sha256": sha256_file(base_path),
        "new_head_weight": str(new_path),
        "new_head_weight_sha256": sha256_file(new_path),
        "shared_max_drift": drift,
        "residual_adapter_export": adapter_export,
    }


def export_shared_dual_head_onnx(
    module: Any,
    sample: Any,
    target: str | Path,
    *,
    opset: int = 17,
) -> dict[str, Any]:
    import torch

    from fair_agent.modules.ascend_detection_export import (
        _without_optional_onnx_postprocessing,
    )

    path = Path(target).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".onnx", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with _without_optional_onnx_postprocessing(), torch.inference_mode():
            torch.onnx.export(
                module,
                sample,
                temporary,
                input_names=["images"],
                output_names=list(DUAL_HEAD_OUTPUT_NAMES),
                opset_version=int(opset),
                dynamic_axes=None,
                do_constant_folding=True,
                dynamo=False,
            )
        if temporary.stat().st_size <= 0:
            raise RuntimeError("共享双head ONNX为空")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "opset": int(opset),
        "input_name": "images",
        "output_names": list(DUAL_HEAD_OUTPUT_NAMES),
    }
