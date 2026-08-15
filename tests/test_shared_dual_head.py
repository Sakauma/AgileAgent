from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from fair_agent.modules.shared_dual_head import (  # noqa: E402
    ResidualAdaptedDetect,
    collapse_residual_adapter_for_export,
    compose_shared_dual_head,
    copy_shared_backbone_state,
    maximum_shared_backbone_drift,
)


def test_shared_state_copy_preserves_new_detect_head() -> None:
    source = {
        "model.0.weight": torch.tensor([1.0, 2.0]),
        "model.23.cv3.weight": torch.tensor([3.0]),
    }
    target = {
        "model.0.weight": torch.tensor([9.0, 9.0]),
        "model.23.cv3.weight": torch.tensor([8.0]),
    }

    copied = copy_shared_backbone_state(source, target)

    assert copied == 1
    assert target["model.0.weight"].tolist() == [1.0, 2.0]
    assert target["model.23.cv3.weight"].tolist() == [8.0]
    assert maximum_shared_backbone_drift(source, target) == 0.0


def test_shared_state_drift_rejects_missing_or_changed_tensor() -> None:
    reference = {"model.0.weight": torch.tensor([1.0])}

    assert maximum_shared_backbone_drift(
        reference, {"model.0.weight": torch.tensor([1.5])}
    ) == 0.5
    with pytest.raises(ValueError, match="不兼容"):
        maximum_shared_backbone_drift(reference, {})


class GraphLayer(torch.nn.Module):
    def __init__(self, index, function, source=-1):
        super().__init__()
        self.i = index
        self.f = source
        self.function = function

    def forward(self, value):
        return self.function(value)


class GraphHead(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.f = [0, 1]
        self.scale = scale

    def forward(self, features):
        return (features[0] + features[1]) * self.scale


class GraphModel:
    def __init__(self, head):
        self.model = torch.nn.ModuleList(
            [
                GraphLayer(0, lambda value: value * 2),
                GraphLayer(1, lambda value: value + 1),
                head,
            ]
        )
        self.save = [0, 1]


def test_shared_graph_executes_backbone_once_and_returns_two_heads() -> None:
    module = compose_shared_dual_head(
        GraphModel(GraphHead(1)),
        GraphModel(GraphHead(10)),
    )

    old, new = module(torch.tensor([2.0]))

    assert old.tolist() == [9.0]
    assert new.tolist() == [90.0]


class AdapterBranch(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, 4, 1)


class AdapterDetect(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nl = 3
        self.cv2 = torch.nn.ModuleList(
            torch.nn.Sequential(AdapterBranch(channels))
            for channels in (2, 3, 4)
        )
        self.i = 23
        self.f = [16, 19, 22]
        self.type = "Detect"
        self.np = 1
        self.nc = 1
        self.reg_max = 16
        self.no = 65
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

    def forward(self, features):
        return [feature.mean() for feature in features]


def test_residual_adapter_is_zero_initialized_and_identity() -> None:
    detect = AdapterDetect()
    wrapper = ResidualAdaptedDetect(detect, [2, 3, 4])
    features = [
        torch.randn(1, channels, 5, 5) for channels in (2, 3, 4)
    ]

    expected = detect([feature.clone() for feature in features])
    actual = wrapper([feature.clone() for feature in features])

    assert all(
        torch.count_nonzero(parameter).item() == 0
        for parameter in wrapper.adapters.parameters()
    )
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))
    assert wrapper.f == [16, 19, 22]
    assert wrapper.stride.tolist() == [8.0, 16.0, 32.0]


def test_residual_adapter_export_collapse_removes_add_without_drift() -> None:
    wrapper = ResidualAdaptedDetect(AdapterDetect(), [2, 3, 4])
    features = [
        torch.randn(1, channels, 5, 5) for channels in (2, 3, 4)
    ]
    with torch.no_grad():
        for adapter in wrapper.adapters:
            adapter.weight.uniform_(-0.2, 0.2)
            adapter.bias.uniform_(-0.1, 0.1)
    expected = wrapper([feature.clone() for feature in features])

    report = collapse_residual_adapter_for_export(wrapper)
    actual = wrapper([feature.clone() for feature in features])

    assert report == {
        "kind": "identity_folded_1x1",
        "adapter_count": 3,
        "explicit_add_removed": True,
    }
    assert all(
        torch.allclose(left, right, atol=1e-6, rtol=1e-6)
        for left, right in zip(actual, expected)
    )
    with pytest.raises(ValueError, match="已经折叠"):
        collapse_residual_adapter_for_export(wrapper)
