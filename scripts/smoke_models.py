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
from fair_agent.modules.functional_models import validate_functional_models


def main() -> int:
    parser = argparse.ArgumentParser(description="在 x86 NVIDIA GPU 上校验并加载发布的 YOLO 权重。")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "local_infer_gpu.yaml")
    parser.add_argument("--registry", type=Path, default=ROOT / "configs" / "functional_models.yaml")
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
    functional = validate_functional_models(args.registry)
    if not functional["valid"] or not functional["all_x86_gpu_ready"]:
        raise SystemExit(f"功能模型注册表验收失败：{functional['errors']}")

    try:
        from ultralytics import YOLO
        import torch
        from fair_agent.models.context import evaluate_context_paths, load_context_model, predict_context
        from fair_agent.modules.web_inference import WebInferenceEngine
        from fair_agent.web.app import build_web_settings
    except ImportError as exc:
        raise SystemExit("缺少 Ultralytics 或 PyTorch，请安装推理依赖。") from exc

    if not torch.cuda.is_available():
        raise SystemExit("默认推理设备为 GPU，但当前 PyTorch 无法使用 CUDA。请检查显卡驱动和 CUDA 版 PyTorch。")

    detection_entry = next(item for item in functional["models"] if item["function"] == "multimodal_target_detection")
    incremental_entry = next(item for item in functional["models"] if item["function"] == "incremental_object_detection")
    context_entry = next(item for item in functional["models"] if item["function"] == "context_perception")
    model_paths = [ROOT / detection_entry["artifacts"][0]["path"]]
    model_paths.extend(ROOT / artifact["path"] for artifact in incremental_entry["artifacts"])
    image = Image.new("RGB", (imgsz, imgsz), color=(0, 0, 0))
    results = []
    base_model = None
    for path in model_paths:
        started = time.perf_counter()
        model = YOLO(str(path))
        if path == model_paths[0]:
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

    context_weights = ROOT / context_entry["artifacts"][0]["path"]
    context_model, context_checkpoint = load_context_model(context_weights, f"cuda:{device}")
    context_prediction = None
    context_lock_evaluation = None
    context_reference_comparable = None
    lock_paths = []
    if not args.load_only:
        context_prediction = predict_context(context_model, context_checkpoint, image, f"cuda:{device}")
        lock_split = ROOT / "splits" / "strict_3plus1" / "scene_test.txt"
        if lock_split.exists():
            lock_paths = [ROOT / line.strip() for line in lock_split.read_text(encoding="utf-8").splitlines() if line.strip()]
            context_lock_evaluation = evaluate_context_paths(context_model, context_checkpoint, lock_paths, f"cuda:{device}", batch_size=batch_size)
            expected_metrics = json.loads((ROOT / "models" / "context" / "scene_sensor_metrics.json").read_text(encoding="utf-8"))["lock"]
            context_reference_comparable = int(expected_metrics.get("image_count", -1)) == len(lock_paths)
            if context_reference_comparable:
                for name in ["sensor_accuracy", "scene_accuracy", "joint_accuracy"]:
                    if abs(float(context_lock_evaluation[name]) - float(expected_metrics[name])) > 1e-9:
                        raise RuntimeError(f"Scene-SensorNet lock 指标不一致：{name}")

    orchestration_results = []
    if not args.load_only:
        del base_model
        del model
        del prediction
        del batch_results
        del context_model
        torch.cuda.empty_cache()
        settings = build_web_settings()
        engine = WebInferenceEngine(
            settings["detector_path"],
            settings["context_path"],
            device_index=settings["device_index"],
            predict_options=settings["predict"],
            incremental_protocols=settings["protocols"],
            class_names=settings["class_names"],
            base_class_ids=settings["base_class_ids"],
            base_local_to_global=settings.get("base_local_to_global"),
            routing_options=settings["routing"],
            generation_id=settings["generation_id"],
            base_model_id=settings["base_model_id"],
            class_owners=settings["class_owners"],
        )
        samples = []
        for sensor in ("ir", "sar"):
            sample = next((path for path in lock_paths if path.name.startswith(f"{sensor}_")), None)
            if sample is not None:
                with Image.open(sample) as source:
                    samples.append((sample.name, source.convert("RGB")))
        if not samples:
            samples = [("synthetic.png", image)]
        for filename, sample_image in samples:
            prediction = engine.predict(sample_image, filename, confidence=0.50, incremental_protocol="auto")
            decision = prediction.get("agent", {}).get("decision", {})
            if decision.get("final_detection_count") != prediction.get("detection_count"):
                raise RuntimeError(f"Agent 编排结果数量不一致：{filename}")
            if not isinstance(decision.get("executed_protocols"), list) or not isinstance(
                decision.get("fusion_summary"), dict
            ):
                raise RuntimeError(f"Agent 编排缺少决策轨迹：{filename}")
            if decision.get("generation_id") != settings["generation_id"]:
                raise RuntimeError(f"Agent 编排使用了错误模型代际：{filename}")
            orchestration_results.append({
                "sample": filename,
                "detection_count": prediction["detection_count"],
                "models_used": prediction["agent"]["models_used"],
                "executed_protocols": decision["executed_protocols"],
                "skipped_protocols": decision["skipped_protocols"],
                "fusion_summary": decision["fusion_summary"],
            })

    print(json.dumps({
        "checksums": checksums,
        "config": args.config.relative_to(ROOT).as_posix(),
        "device": device,
        "imgsz": imgsz,
        "batch": batch_size,
        "batch_result_count": batch_result_count,
        "functional_model_count": functional["model_count"],
        "functional_roles": [item["function"] for item in functional["models"]],
        "context_model": {
            "model": context_entry["id"],
            "weights": context_entry["artifacts"][0]["path"],
            "loaded": True,
            "synthetic_prediction": context_prediction,
            "lock_evaluation": context_lock_evaluation,
            "historical_reference_comparable": context_reference_comparable,
        },
        "agent_orchestration": orchestration_results,
        "yolo_models": results,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
