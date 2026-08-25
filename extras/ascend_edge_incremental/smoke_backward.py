#!/usr/bin/env python3
"""Prove that a real optimizer step executes on the Ascend NPU."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    import torch
    import torch_npu  # noqa: F401, PLC0415

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable; refusing CPU fallback")
    torch.manual_seed(20260825)
    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    features = torch.randn(256, 12, device=device)
    target = (features[:, :3].sum(dim=1, keepdim=True) > 0).float()
    model = torch.nn.Sequential(
        torch.nn.Linear(12, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)
    criterion = torch.nn.BCEWithLogitsLoss()
    initial = [value.detach().cpu().clone() for value in model.parameters()]
    losses: list[float] = []
    iteration_ms: list[float] = []
    for _ in range(60):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), target)
        loss.backward()
        optimizer.step()
        torch.npu.synchronize()
        losses.append(float(loss.detach().cpu()))
        iteration_ms.append((time.perf_counter() - started) * 1000.0)
    changed_l2 = sum(
        float(torch.sum((after.detach().cpu() - before) ** 2))
        for before, after in zip(initial, model.parameters())
    ) ** 0.5
    steady = sorted(iteration_ms[10:])
    report = {
        "status": "passed" if losses[-1] < losses[0] and changed_l2 > 0 else "failed",
        "platform": "Ascend310B1",
        "device": torch.npu.get_device_name(args.device_id),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "cpu_fallback_used": False,
        "iterations": len(losses),
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "loss_reduction_ratio": 1.0 - losses[-1] / losses[0],
        "parameter_delta_l2": changed_l2,
        "first_iteration_ms": iteration_ms[0],
        "steady_iteration_median_ms": steady[len(steady) // 2],
        "steady_iteration_p95_ms": steady[
            min(len(steady) - 1, int(len(steady) * 0.95))
        ],
        "max_memory_allocated_mb": torch.npu.max_memory_allocated() / 1024**2,
        "max_memory_reserved_mb": torch.npu.max_memory_reserved() / 1024**2,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
