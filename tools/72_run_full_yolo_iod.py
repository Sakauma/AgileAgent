from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fair_agent.modules.full_yolo_iod import (
    OFFICIAL_COMMIT,
    cpr_is_enabled,
    count_cpr_pseudo_labels,
    load_full_yolo_iod_config,
    materialize_full_yolo_iod_lock,
    prepare_full_yolo_iod,
    sha256_file,
    summarize_disabled_cpr,
    write_command_manifest,
)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _official_commit(repo: Path) -> str | None:
    if not (repo / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _official_patch_status(repo: Path, patch: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(patch),
        "exists": patch.is_file(),
        "sha256": sha256_file(patch) if patch.is_file() else None,
        "status": "missing",
    }
    if not patch.is_file() or not (repo / ".git").is_dir():
        return result
    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if reverse.returncode == 0:
        result["status"] = "applied"
        return result
    forward = subprocess.run(
        ["git", "apply", "--check", str(patch)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    result["status"] = "applicable" if forward.returncode == 0 else "invalid"
    if forward.returncode != 0:
        result["error"] = forward.stderr.strip()
    return result


def _ensure_official_patch(config: Mapping[str, Any], official: Path) -> Dict[str, Any]:
    patch = _resolve_repo_path(config["official_patch"]["path"])
    status = _official_patch_status(official, patch)
    if status["status"] == "applicable":
        subprocess.run(["git", "apply", str(patch)], cwd=official, check=True)
        status = _official_patch_status(official, patch)
    if status["status"] != "applied":
        raise RuntimeError(f"YOLO-IOD 兼容补丁不可用：{status}")
    return status


def preflight(config: Mapping[str, Any]) -> Dict[str, Any]:
    official = Path(config["paths"]["official_repo"]).resolve()
    python = Path(config["runtime"]["python"]).resolve()
    split_checks = {}
    for name, value in config["paths"]["source_splits"].items():
        path = _resolve_repo_path(value)
        split_checks[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    import_check = None
    if python.is_file():
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import torch,mmcv,mmdet,mmengine,mmyolo,transformers,albumentations,numpy; "
                "print(torch.__version__, torch.version.cuda, mmcv.__version__, "
                "mmdet.__version__, mmengine.__version__, mmyolo.__version__, transformers.__version__, "
                "'albumentations=' + albumentations.__version__, 'numpy=' + numpy.__version__)",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        import_check = {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    run_id = str(config["experiment"]["run_id"])
    data_root = official / "data" / "agileagent" / run_id
    config_root = official / "configs" / "agileagent_3_1" / run_id
    patch = _resolve_repo_path(config["official_patch"]["path"])
    patch_status = _official_patch_status(official, patch)
    checks = {
        "official_repo": str(official),
        "official_repo_exists": official.is_dir(),
        "official_commit": _official_commit(official),
        "expected_commit": OFFICIAL_COMMIT,
        "python": str(python),
        "python_exists": python.is_file(),
        "imports": import_check,
        "official_patch": patch_status,
        "splits": split_checks,
        "data_output_conflict": data_root.exists(),
        "config_output_conflict": config_root.exists(),
    }
    errors = []
    if not checks["official_repo_exists"]:
        errors.append("official_repo_missing")
    if checks["official_commit"] != OFFICIAL_COMMIT:
        errors.append("official_commit_mismatch")
    if patch_status["status"] not in {"applicable", "applied"}:
        errors.append("official_patch_invalid")
    if not checks["python_exists"]:
        errors.append("runtime_python_missing")
    elif not import_check or import_check["returncode"] != 0:
        errors.append("runtime_import_failed")
    compatibility = config.get("compatibility", {})
    for package in ("albumentations", "numpy"):
        expected = str(compatibility.get(package, "")).strip()
        if expected and (not import_check or f"{package}={expected}" not in import_check["stdout"]):
            errors.append(f"runtime_{package}_version_mismatch")
    if not all(item["exists"] for item in split_checks.values()):
        errors.append("source_split_missing")
    if checks["data_output_conflict"] or checks["config_output_conflict"]:
        errors.append("run_id_output_conflict")
    return {
        "schema_version": 1,
        "mode": "full_yolo_iod_preflight",
        "run_id": run_id,
        "ready": not errors,
        "checks": checks,
        "errors": errors,
    }


def _runtime_env(command_manifest: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, str]:
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in command_manifest["env"].items()})
    environment.update(
        {str(key): str(value) for key, value in config.get("runtime", {}).get("env", {}).items()}
    )
    return environment


def _run_logged(
    name: str,
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_root: Path,
    action_log: Path,
) -> None:
    log_path = log_root / f"{name}.log"
    started = time.time()
    log_mode = "append" if log_path.is_file() and log_path.stat().st_size > 0 else "create"
    with action_log.open("a", encoding="utf-8") as audit:
        audit.write(
            json.dumps(
                {
                    "stage": name,
                    "event": "start",
                    "time": started,
                    "argv": list(command),
                    "log_mode": log_mode,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    with log_path.open("a", encoding="utf-8") as output:
        if log_mode == "append":
            output.write(f"\n===== resume {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            output.flush()
        result = subprocess.run(
            list(command), cwd=cwd, env=dict(env), stdout=output, stderr=subprocess.STDOUT, check=False
        )
    finished = time.time()
    with action_log.open("a", encoding="utf-8") as audit:
        audit.write(
            json.dumps(
                {
                    "stage": name,
                    "event": "finish" if result.returncode == 0 else "error",
                    "time": finished,
                    "elapsed_seconds": finished - started,
                    "returncode": result.returncode,
                    "log": str(log_path),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    if result.returncode != 0:
        raise RuntimeError(f"{name} 失败，详见 {log_path}")


def _record_skip(action_log: Path, stage: str, artifact: Path) -> None:
    with action_log.open("a", encoding="utf-8") as audit:
        audit.write(
            json.dumps(
                {
                    "stage": stage,
                    "event": "skip_existing",
                    "time": time.time(),
                    "artifact": str(artifact),
                    "sha256": sha256_file(artifact) if artifact.is_file() else None,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _record_policy_skip(action_log: Path, stage: str, artifact: Path, reason: str) -> None:
    with action_log.open("a", encoding="utf-8") as audit:
        audit.write(
            json.dumps(
                {
                    "stage": stage,
                    "event": "skip_by_policy",
                    "time": time.time(),
                    "artifact": str(artifact),
                    "sha256": sha256_file(artifact),
                    "reason": reason,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _resume_command(command: Sequence[str], work_dir: Path) -> list[str]:
    marker = work_dir / "last_checkpoint"
    if not marker.is_file():
        return list(command)
    checkpoint_value = marker.read_text(encoding="utf-8").strip()
    if not checkpoint_value:
        raise ValueError(f"续训标记为空：{marker}")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = (work_dir / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"续训检查点不存在：{checkpoint}")
    return [*command, "--resume", str(checkpoint)]


def _ensure_pretrained(config: Mapping[str, Any], official: Path) -> Path:
    pretrained = config["pretrained"]
    target = official / pretrained["path"]
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        urls = pretrained.get("urls") or [pretrained.get("url")]
        urls = [str(url) for url in urls if url]
        if not urls:
            raise ValueError("pretrained.urls 至少需要配置一个下载地址")
        errors = []
        for url in urls:
            result = subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "5",
                    "--retry-delay",
                    "2",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "10",
                    "--speed-time",
                    "60",
                    "--speed-limit",
                    "1024",
                    "--continue-at",
                    "-",
                    "--output",
                    str(temporary),
                    url,
                ],
                check=False,
            )
            if result.returncode == 0 and temporary.is_file() and temporary.stat().st_size > 0:
                break
            errors.append(f"{url}: curl={result.returncode}")
        else:
            raise RuntimeError("YOLO-World 预训练权重下载失败：" + "; ".join(errors))
        temporary.replace(target)
    expected = str(pretrained.get("sha256", "")).strip()
    if expected and sha256_file(target) != expected:
        raise ValueError("YOLO-World 预训练权重 SHA256 不匹配")
    return target


def _cpr_command(
    python: Path,
    official: Path,
    manifest: Mapping[str, Any],
    command_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[str]:
    data_root = Path(manifest["data_root"])
    source = data_root / "annotations/current_train.json"
    output = data_root / "annotations/current_train_cpr.json"
    base_config = command_manifest["commands"]["base"][2]
    base_checkpoint = command_manifest["checkpoints"]["base"]
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(official / 'script')!r}); "
        "from pseudo_label_sc import main; "
        f"main({base_config!r}, {base_checkpoint!r}, "
        f"{str(data_root / 'texts/old.json')!r}, {str(source)!r}, {str(output)!r}, "
        f"{str(data_root / 'images/train')!r}, "
        f"{float(config['cpr']['score_threshold'])!r}, {float(config['cpr']['iou_threshold'])!r})"
    )
    return [str(python), "-c", code]


def run_all(config: Mapping[str, Any], *, prepare_only: bool = False) -> Dict[str, Any]:
    official = Path(config["paths"]["official_repo"]).resolve()
    python = Path(config["runtime"]["python"]).resolve()
    official_patch = _ensure_official_patch(config, official)
    _ensure_pretrained(config, official)
    run_id = str(config["experiment"]["run_id"])
    data_root = official / "data" / "agileagent" / run_id
    manifest_path = data_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = prepare_full_yolo_iod(config)
    command_manifest = write_command_manifest(config, manifest)
    if prepare_only:
        return {"status": "prepared", "manifest": manifest, "commands": command_manifest}
    report_root = _resolve_repo_path(config["paths"]["report_root"]) / run_id
    report_root.mkdir(parents=True, exist_ok=True)
    action_log = report_root / "action_log.jsonl"
    environment = _runtime_env(command_manifest, config)
    cwd = Path(command_manifest["cwd"])

    base_checkpoint = Path(command_manifest["checkpoints"]["base"])
    current_checkpoint = Path(command_manifest["checkpoints"]["current"])
    final_checkpoint = Path(command_manifest["checkpoints"]["final"])
    if base_checkpoint.is_file():
        _record_skip(action_log, "01_base_train", base_checkpoint)
    else:
        base_work_dir = Path(manifest["generated_configs"]["base_work_dir"])
        base_command = _resume_command(command_manifest["commands"]["base"], base_work_dir)
        _run_logged(
            "01_base_train",
            base_command,
            cwd=cwd,
            env=environment,
            log_root=report_root,
            action_log=action_log,
        )
    if current_checkpoint.is_file():
        _record_skip(action_log, "02_current_teacher_train", current_checkpoint)
    else:
        current_work_dir = Path(manifest["generated_configs"]["current_work_dir"])
        current_command = _resume_command(
            command_manifest["commands"]["current"], current_work_dir
        )
        _run_logged(
            "02_current_teacher_train",
            current_command,
            cwd=cwd,
            env=environment,
            log_root=report_root,
            action_log=action_log,
        )
    cpr_source = data_root / "annotations/current_train.json"
    cpr_output = data_root / "annotations/current_train_cpr.json"
    if not cpr_is_enabled(config):
        reason = str(config["cpr"]["mode"])
        if int(manifest.get("incremental_old_class_gt_count", -1)) != 0:
            raise ValueError("CPR 禁用门禁失败：增量数据存在旧类 GT")
        _record_policy_skip(action_log, "03_cpr", cpr_source, reason)
        cpr_counts = summarize_disabled_cpr(cpr_source, reason)
    elif cpr_output.is_file():
        _record_skip(action_log, "03_cpr", cpr_output)
        cpr_counts = {
            "enabled": True,
            "status": "completed",
            **count_cpr_pseudo_labels(cpr_source, cpr_output),
        }
    else:
        cpr_command = _cpr_command(python, official, manifest, command_manifest, config)
        _run_logged(
            "03_cpr",
            cpr_command,
            cwd=cwd,
            env=environment,
            log_root=report_root,
            action_log=action_log,
        )
        cpr_counts = {
            "enabled": True,
            "status": "completed",
            **count_cpr_pseudo_labels(cpr_source, cpr_output),
        }
    _json_write(report_root / "cpr_summary.json", cpr_counts)
    if final_checkpoint.is_file():
        _record_skip(action_log, "04_final_iod_train", final_checkpoint)
    else:
        final_work_dir = Path(manifest["generated_configs"]["final_work_dir"])
        final_command = _resume_command(command_manifest["commands"]["final"], final_work_dir)
        if "--resume" not in final_command:
            shutil.copy2(base_checkpoint, command_manifest["final_init"])
        final_environment = dict(environment)
        final_environment["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(item) for item in command_manifest["stage_devices"]["final"]
        )
        _run_logged(
            "04_final_iod_train",
            final_command,
            cwd=cwd,
            env=final_environment,
            log_root=report_root,
            action_log=action_log,
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("lock_materialized_after_freeze"):
        materialize_full_yolo_iod_lock(config, final_checkpoint)
    evaluations = config["evaluation"]
    base_test = [
        str(python),
        str(official / "tools/test.py"),
        manifest["generated_configs"]["base_lock_config"],
        command_manifest["checkpoints"]["base"],
        "--work-dir",
        str(report_root / "base_lock"),
        "--out",
        str(report_root / "base_lock_predictions.pkl"),
    ]
    final_test = [
        str(python),
        str(official / "tools/test.py"),
        manifest["generated_configs"]["final_lock_config"],
        str(final_checkpoint),
        "--work-dir",
        str(report_root / "final_lock"),
        "--out",
        str(report_root / "final_lock_predictions.pkl"),
    ]
    evaluation_env = dict(environment)
    evaluation_env["CUDA_VISIBLE_DEVICES"] = str(evaluations.get("device", config["runtime"]["devices"][0]))
    base_predictions = report_root / "base_lock_predictions.pkl"
    final_predictions = report_root / "final_lock_predictions.pkl"
    if base_predictions.is_file():
        _record_skip(action_log, "05_base_lock_eval", base_predictions)
    else:
        _run_logged(
            "05_base_lock_eval",
            base_test,
            cwd=cwd,
            env=evaluation_env,
            log_root=report_root,
            action_log=action_log,
        )
    if final_predictions.is_file():
        _record_skip(action_log, "06_final_lock_eval", final_predictions)
    else:
        _run_logged(
            "06_final_lock_eval",
            final_test,
            cwd=cwd,
            env=evaluation_env,
            log_root=report_root,
            action_log=action_log,
        )
    result = {
        "schema_version": 1,
        "status": "evaluation_complete",
        "run_id": run_id,
        "method": "official_yolo_iod_full",
        "official_commit": OFFICIAL_COMMIT,
        "official_patch": official_patch,
        "cpr": cpr_counts,
        "base_checkpoint": command_manifest["checkpoints"]["base"],
        "base_sha256": sha256_file(Path(command_manifest["checkpoints"]["base"])),
        "current_checkpoint": command_manifest["checkpoints"]["current"],
        "current_sha256": sha256_file(Path(command_manifest["checkpoints"]["current"])),
        "final_checkpoint": str(final_checkpoint),
        "final_sha256": sha256_file(final_checkpoint),
        "report_root": str(report_root),
    }
    _json_write(report_root / "run_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="运行官方完整 YOLO-IOD strict-p02 复现实验。")
    parser.add_argument("--config", default="configs/full_yolo_iod_p02_r04.yaml")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config_path = _resolve_repo_path(args.config)
    config = load_full_yolo_iod_config(config_path)
    if args.check_only:
        result = preflight(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ready"] else 2
    result = run_all(config, prepare_only=args.prepare_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
