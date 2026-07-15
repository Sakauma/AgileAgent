from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from PIL import Image

from fair_agent.core.config import rel_path, resolve_path
from fair_agent.core.hashes import sha256_file
from fair_agent.core.runtime_log import StructuredEventLog, utc_now
from fair_agent.modules.generation_management import (
    build_candidate_confusion_graph,
    promote_generation,
    recheck_generation,
    register_generation_deployment,
    register_trained_candidate,
    rollback_generation,
    shadow_load_generation,
)
from fair_agent.modules.incremental_lineage import freeze_accepted_batch
from fair_agent.modules.strict_incremental import evaluate_ap50, precision_recall
from fair_agent.modules.tensorrt_export import export_incremental_int8_engine
from fair_agent.modules.web_inference import remap_specialist_records_dynamic, result_records


PromotionCallback = Callable[[str, str, Any, Mapping[str, Any]], Dict[str, Any]]
RollbackCallback = Callable[[str], Dict[str, Any]]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _ground_truth(images: list[Path], local_to_global: Mapping[int, int]) -> list[Dict[str, Any]]:
    rows = []
    for image in images:
        with Image.open(image) as source:
            width, height = source.size
        label = image.with_suffix(".txt")
        if not label.is_file() and "images" in image.parts:
            parts = list(image.parts)
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            label = Path(*parts).with_suffix(".txt")
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            class_id = local_to_global[int(fields[0])]
            x, y, w, h = [float(value) for value in fields[1:]]
            rows.append({
                "image_id": image.stem,
                "class_id": class_id,
                "xyxy": [(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height],
            })
    return rows


def calibrate_candidate(
    config: Mapping[str, Any],
    batch_dir: Path,
    candidate_manifest: Mapping[str, Any],
    local_to_global: Mapping[int, int],
) -> Dict[str, Any]:
    from ultralytics import YOLO

    images = sorted((batch_dir / "prepared" / "images" / "val").glob("*"))
    images = [path for path in images if path.is_file()]
    if not images:
        raise ValueError("增量dev为空，无法逐类校准阈值。")
    lifecycle = config["incremental_workbench"]["lifecycle"]
    threshold_min = float(lifecycle["threshold_min"])
    threshold_max = float(lifecycle["threshold_max"])
    threshold_step = float(lifecycle["threshold_step"])
    target_precision = float(lifecycle["calibration_target_precision"])
    model = YOLO(str(resolve_path(candidate_manifest["best_weight"])))
    predictions = []
    batch_size = int(config["inference"]["batch_size"])
    started = time.perf_counter()
    for offset in range(0, len(images), batch_size):
        chunk = images[offset: offset + batch_size]
        sources = []
        for path in chunk:
            with Image.open(path) as source:
                source.load()
                sources.append(source.convert("RGB"))
        results = model.predict(
            source=sources,
            device=str(config["runtime"]["default_device"]),
            imgsz=int(config["inference"]["specialist_imgsz"]),
            conf=min(0.001, threshold_min),
            iou=float(config["inference"]["iou"]),
            max_det=int(config["inference"]["max_det"]),
            batch=len(sources),
            verbose=False,
        )
        for path, result in zip(chunk, results):
            for row in remap_specialist_records_dynamic(
                result_records(result), local_to_global, {}, "calibration"
            ):
                predictions.append({**row, "image_id": path.stem})
    ground_truth = _ground_truth(images, local_to_global)
    class_ids = sorted(set(local_to_global.values()))
    steps = int(math.floor((threshold_max - threshold_min) / threshold_step)) + 1
    thresholds = [round(threshold_min + index * threshold_step, 10) for index in range(steps)]
    per_class = {}
    all_target_precisions_reached = True
    for class_id in class_ids:
        curve = []
        for threshold in thresholds:
            metrics = precision_recall(predictions, ground_truth, class_id, threshold)
            precision = float(metrics["precision"])
            recall = float(metrics["recall"])
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            curve.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1})
        eligible = [row for row in curve if row["precision"] >= target_precision]
        target_precision_reached = bool(eligible)
        selected = max(
            eligible or curve,
            key=(
                (lambda row: (row["recall"], row["f1"], row["threshold"]))
                if eligible else (lambda row: (row["f1"], row["precision"], row["threshold"]))
            ),
        )
        class_map50 = float(evaluate_ap50(predictions, ground_truth, [class_id])["map50"])
        per_class[str(class_id)] = {
            **selected,
            "map50": class_map50,
            "target_precision": target_precision,
            "target_precision_reached": target_precision_reached,
            "curve": curve,
        }
        all_target_precisions_reached = (
            all_target_precisions_reached and target_precision_reached
        )
    output_path = batch_dir / "calibration" / f"{candidate_manifest['job_id']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output_path.with_name(output_path.stem + "-predictions.jsonl")
    ground_truth_path = output_path.with_name(output_path.stem + "-ground-truth.jsonl")
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    ground_truth_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ground_truth),
        encoding="utf-8",
    )
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "job_id": candidate_manifest["job_id"],
        "dev_image_count": len(images),
        "dev_only": True,
        "target_precision": target_precision,
        "per_class_thresholds": {key: value["threshold"] for key, value in per_class.items()},
        "per_class_metrics": per_class,
        "new_map50": float(evaluate_ap50(predictions, ground_truth, class_ids)["map50"]),
        "target_precision_reached": all_target_precisions_reached,
        "calibrated": len(per_class) == len(class_ids),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "predictions": rel_path(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "ground_truth": rel_path(ground_truth_path),
        "ground_truth_sha256": sha256_file(ground_truth_path),
    }
    _atomic_json(output_path, payload)
    payload["calibration_sources"] = {str(class_id): rel_path(output_path) for class_id in class_ids}
    payload["path"] = rel_path(output_path)
    payload["sha256"] = sha256_file(output_path)
    return payload


def quantize_incremental_candidate(
    config: Mapping[str, Any],
    batch_dir: Path,
    generation_id: str,
    candidate_manifest: Mapping[str, Any],
    local_to_global: Mapping[int, int],
) -> Dict[str, Any] | None:
    inference = config["inference"]
    backend = config["tensorrt_backend"]
    calibration = backend["int8_calibration"]
    if not (
        str(inference["backend"]) == "tensorrt_engine"
        and str(backend["precision"]) == "int8"
        and calibration["enabled"] is True
        and calibration["auto_quantize_incremental"] is True
    ):
        return None
    image_root = batch_dir / "prepared" / "images"
    images = sorted(
        path
        for split in ("train", "val")
        for path in (image_root / split).glob("*")
        if path.is_file()
    )
    if not images:
        raise ValueError("增量INT8校准没有可访问的train/dev图像。")
    lock_manifest_path = batch_dir / "sealed_lock" / "lock_manifest.json"
    lock_manifest = json.loads(lock_manifest_path.read_text(encoding="utf-8"))
    forbidden_stems = [str(row["stem"]) for row in lock_manifest.get("files", [])]
    deployment = export_incremental_int8_engine(
        config,
        candidate_manifest["best_weight"],
        generation_id,
        images,
        sorted(local_to_global),
        forbidden_stems,
    )
    registration = register_generation_deployment(
        config, generation_id, "tensorrt_int8", deployment
    )
    return {"deployment": deployment, "registration": registration}


class IncrementalLifecycle:
    def __init__(
        self,
        store: Any,
        config: Mapping[str, Any],
        event_log: StructuredEventLog,
        promotion_callback: PromotionCallback | None = None,
        rollback_callback: RollbackCallback | None = None,
    ) -> None:
        self.store = store
        self.config = dict(config)
        self.event_log = event_log
        self.promotion_callback = promotion_callback
        self.rollback_callback = rollback_callback

    def _rollback(
        self,
        batch_id: str,
        job_id: str,
        candidate_id: str,
        target_id: str,
        failed_stage: str,
        error: Exception,
    ) -> Dict[str, Any]:
        try:
            rollback = (
                self.rollback_callback(target_id)
                if self.rollback_callback
                else rollback_generation(self.config, target_id)
            )
        except Exception as rollback_error:
            self._state(
                batch_id,
                job_id,
                "ROLLBACK_FAILED",
                generation_id=candidate_id,
                rollback_target=target_id,
                failed_stage=failed_stage,
                lifecycle_error=str(error),
                lifecycle_error_type=type(error).__name__,
                rollback_error=str(rollback_error),
                rollback_error_type=type(rollback_error).__name__,
            )
            raise RuntimeError(
                f"增量生命周期在{failed_stage}失败，且无法恢复父代际：{rollback_error}"
            ) from rollback_error
        self._state(
            batch_id,
            job_id,
            "ROLLED_BACK",
            generation_id=candidate_id,
            rollback_target=target_id,
            failed_stage=failed_stage,
            lifecycle_error=str(error),
            lifecycle_error_type=type(error).__name__,
            rollback=rollback,
        )
        return rollback

    def _state(self, batch_id: str, job_id: str, state: str, **details: Any) -> None:
        self.store.update_training(batch_id, job_id, state, **details)
        self.event_log.append(
            f"incremental.lifecycle.{state.lower()}", component="incremental",
            batch_id=batch_id, job_id=job_id, details=details,
        )
        canonical_events = {
            "CALIBRATED": "incremental.dev_calibration.completed",
            "DIAGNOSING": "incremental.dev_diagnosis.started",
            "DIAGNOSED": "incremental.dev_diagnosis.completed",
            "RECOVERY_REQUIRED": "incremental.recovery.selected",
            "REGISTERED_CANDIDATE": "generation.registered",
            "QUANTIZING": "incremental.quantization.started",
            "QUANTIZED": "incremental.quantization.completed",
            "LOCK_RECHECKING": "incremental.lock_recheck.started",
            "ACCEPTED": "incremental.lock_recheck.completed",
            "REJECTED": "incremental.candidate.rejected",
            "SHADOW_LOADING": "generation.shadow_load.started",
            "PROMOTED": "generation.shadow_load.completed",
            "ROLLED_BACK": "generation.rollback.completed",
            "ROLLBACK_FAILED": "generation.rollback.failed",
        }
        canonical = canonical_events.get(state)
        if canonical:
            self.event_log.append(
                canonical,
                level="error" if state == "ROLLBACK_FAILED" else (
                    "warning" if state in {"REJECTED", "ROLLED_BACK", "RECOVERY_REQUIRED"} else "info"
                ),
                component="generation" if canonical.startswith("generation.") else "incremental",
                batch_id=batch_id,
                job_id=job_id,
                generation_id=details.get("generation_id"),
                details=details,
            )

    def run(self, batch_id: str, job_id: str) -> Dict[str, Any]:
        batch_dir = self.store._batch_dir(batch_id)
        batch = self.store.get(batch_id)
        candidate_path = batch_dir / "training" / job_id / "candidate_manifest.json"
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        local_to_global = {
            int(key): int(value) for key, value in batch["audit"]["local_to_global"].items()
        }
        self._state(batch_id, job_id, "CALIBRATING")
        calibration = calibrate_candidate(self.config, batch_dir, candidate, local_to_global)
        if not calibration["calibrated"]:
            self._state(batch_id, job_id, "REJECTED", calibration=calibration,
                        rejection_reason="dev_precision_calibration_failed")
            return {"status": "REJECTED", "calibration": calibration}
        self._state(batch_id, job_id, "CALIBRATED", calibration=calibration)
        guardian = self.config.get("incremental_guardian")
        if isinstance(guardian, Mapping) and bool(guardian.get("enabled")):
            new_threshold = float(self.config["gates"]["official_hard"]["new_map50_min"])
            if float(calibration["new_map50"]) < new_threshold:
                recovery_actions = list(
                    guardian.get("recovery_actions", {}).get("NEW_KNOWLEDGE_UNDERFIT", [])
                )
                self._state(
                    batch_id,
                    job_id,
                    "RECOVERY_REQUIRED",
                    diagnosis="NEW_KNOWLEDGE_UNDERFIT",
                    dev_new_map50=float(calibration["new_map50"]),
                    required_new_map50=new_threshold,
                    recovery_actions=recovery_actions,
                    lock_unsealed=False,
                )
                self._state(
                    batch_id,
                    job_id,
                    "REJECTED",
                    rejection_reason="dev_full_score_candidate_not_reached",
                    recovery_actions=recovery_actions,
                    lock_unsealed=False,
                )
                return {
                    "status": "REJECTED",
                    "calibration": calibration,
                    "diagnosis": "NEW_KNOWLEDGE_UNDERFIT",
                    "recovery_actions": recovery_actions,
                    "lock_unsealed": False,
                }
        batch = self.store.get(batch_id)
        registered = register_trained_candidate(self.config, batch, candidate, calibration)
        generation_id = registered["generation_id"]
        parent_generation_id = registered["parent_generation_id"]
        self._state(batch_id, job_id, "REGISTERED_CANDIDATE", generation=registered)
        confusion_graph = None
        if isinstance(guardian, Mapping) and bool(guardian.get("enabled")):
            self._state(batch_id, job_id, "DIAGNOSING", generation_id=generation_id, source="incremental_dev_only")
            try:
                confusion_graph = build_candidate_confusion_graph(
                    self.config, batch_dir, generation_id, calibration
                )
            except Exception as exc:
                self._state(
                    batch_id,
                    job_id,
                    "REJECTED",
                    generation_id=generation_id,
                    rejection_reason="dev_confusion_diagnosis_failed",
                    diagnosis_error=str(exc),
                    diagnosis_error_type=type(exc).__name__,
                    lock_unsealed=False,
                )
                return {
                    "status": "REJECTED",
                    "generation": registered,
                    "calibration": calibration,
                    "rejection_reason": "dev_confusion_diagnosis_failed",
                    "error": str(exc),
                    "lock_unsealed": False,
                }
            self._state(
                batch_id,
                job_id,
                "DIAGNOSED",
                generation_id=generation_id,
                confusion_graph=confusion_graph,
            )
        quantization = None
        inference_config = self.config.get("inference", {})
        tensorrt_config = self.config.get("tensorrt_backend", {})
        int8_config = tensorrt_config.get("int8_calibration", {})
        should_quantize = (
            str(inference_config.get("backend")) == "tensorrt_engine"
            and str(tensorrt_config.get("precision")) == "int8"
            and bool(int8_config.get("auto_quantize_incremental"))
        )
        if should_quantize:
            self._state(batch_id, job_id, "QUANTIZING", generation_id=generation_id)
            try:
                quantization = quantize_incremental_candidate(
                    self.config, batch_dir, generation_id, candidate, local_to_global
                )
            except Exception as exc:
                self._state(
                    batch_id,
                    job_id,
                    "REJECTED",
                    generation_id=generation_id,
                    rejection_reason="int8_quantization_failed",
                    quantization_error=str(exc),
                    quantization_error_type=type(exc).__name__,
                )
                return {
                    "status": "REJECTED",
                    "generation": registered,
                    "rejection_reason": "int8_quantization_failed",
                    "error": str(exc),
                }
            self._state(
                batch_id, job_id, "QUANTIZED", generation_id=generation_id,
                quantization=quantization,
            )
        self._state(batch_id, job_id, "LOCK_RECHECKING", generation_id=generation_id)
        recheck = recheck_generation(self.config, generation_id)
        if not recheck["accepted"]:
            self._state(batch_id, job_id, "REJECTED", generation_id=generation_id,
                        recheck_manifest=recheck["manifest"], recheck=recheck,
                        rejection_reason="deployment_gates_failed")
            return {
                "status": "REJECTED", "generation": registered,
                "quantization": quantization, "confusion_graph": confusion_graph, "recheck": recheck,
            }
        self._state(batch_id, job_id, "ACCEPTED", generation_id=generation_id,
                    recheck_manifest=recheck["manifest"], recheck=recheck)
        if not bool(self.config["generation"]["auto_promote"]):
            return {
                "status": "ACCEPTED", "generation": registered,
                "quantization": quantization, "confusion_graph": confusion_graph, "recheck": recheck,
            }
        self._state(batch_id, job_id, "SHADOW_LOADING", generation_id=generation_id)
        failed_stage = "shadow_load"
        try:
            shadow_engine, smoke = shadow_load_generation(self.config, generation_id)
            failed_stage = "promotion"
            if self.promotion_callback:
                promotion = self.promotion_callback(generation_id, recheck["manifest"], shadow_engine, smoke)
            else:
                promotion = promote_generation(self.config, generation_id, recheck["manifest"])
                promotion["shadow_smoke"] = smoke
            failed_stage = "lineage_freeze"
            accepted_lineage = freeze_accepted_batch(
                self.config["incremental_workbench"], batch_id, generation_id,
                str(batch["injection"]["dataset_fingerprint"]), batch["files"],
            )
        except Exception as exc:
            rollback = self._rollback(
                batch_id, job_id, generation_id, parent_generation_id, failed_stage, exc
            )
            return {
                "status": "ROLLED_BACK",
                "generation": registered,
                "quantization": quantization,
                "confusion_graph": confusion_graph,
                "recheck": recheck,
                "failed_stage": failed_stage,
                "error": str(exc),
                "rollback": rollback,
            }
        self._state(batch_id, job_id, "PROMOTED", generation_id=generation_id,
                    promotion=promotion, accepted_lineage=rel_path(accepted_lineage) if accepted_lineage else None)
        return {
            "status": "PROMOTED", "generation": registered,
            "quantization": quantization, "confusion_graph": confusion_graph,
            "recheck": recheck, "promotion": promotion,
        }
