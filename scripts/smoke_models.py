#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.core.hashes import verify_sha256s


def main() -> int:
    parser = argparse.ArgumentParser(description="在 x86 NVIDIA GPU 上校验并加载发布的 YOLO 权重。")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "local_infer_gpu.yaml")
    parser.add_argument("--load-only", action="store_true", help="只加载模型，不执行合成图推理。")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    predict = dict(config["predict"])
    device = int(predict["device"])
    imgsz = int(predict["imgsz"])
    batch_size = int(predict["batch"])

    checksum_path = ROOT / "models" / "SHA256SUMS.txt"
    checksums = verify_sha256s(checksum_path)
    if not checksums["valid"]:
        raise SystemExit(f"Model checksum verification failed: {checksums['errors']}")

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as exc:
        raise SystemExit("缺少 Ultralytics 或 PyTorch，请安装推理依赖。") from exc

    if not torch.cuda.is_available():
        raise SystemExit("默认推理设备为 GPU，但当前 PyTorch 无法使用 CUDA。请检查显卡驱动和 CUDA 版 PyTorch。")

    model_paths = sorted((ROOT / "models").rglob("*.pt"))
    image = Image.new("RGB", (imgsz, imgsz), color=(0, 0, 0))
    results = []
    base_model = None
    for path in model_paths:
        started = time.perf_counter()
        model = YOLO(str(path))
        if path.parent.name == "base":
            base_model = model
        prediction_count = None
        if not args.load_only:
            prediction = model.predict(source=image, imgsz=imgsz, device=device, verbose=False)
            prediction_count = len(prediction)
            if prediction_count != 1:
                raise RuntimeError(f"模型 {path} 返回了异常的结果数量：{prediction_count}")
        results.append({
            "model": path.relative_to(ROOT).as_posix(),
            "device": device,
            "imgsz": imgsz,
            "loaded": True,
            "synthetic_results": prediction_count,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        })

    batch_result_count = None
    if not args.load_only:
        if base_model is None:
            raise RuntimeError("未找到基础模型，无法执行批量推理验收")
        batch_results = base_model.predict(
            source=[image] * batch_size,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            verbose=False,
        )
        batch_result_count = len(batch_results)
        if batch_result_count != batch_size:
            raise RuntimeError(f"批量推理结果数量异常：expected={batch_size} actual={batch_result_count}")

    print(json.dumps({
        "checksums": checksums,
        "config": args.config.relative_to(ROOT).as_posix(),
        "device": device,
        "imgsz": imgsz,
        "batch": batch_size,
        "batch_result_count": batch_result_count,
        "models": results,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
