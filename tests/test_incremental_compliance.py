from fair_agent.modules.incremental_compliance import (
    evaluate_incremental_metrics,
    verify_class_incremental_learning_scope,
    verify_new_images_only,
)
from fair_agent.modules.status import parse_incremental


def test_new_images_only_accepts_exact_incremental_set() -> None:
    result = verify_new_images_only(["generated/images/a.png", "generated/images/b.png"], ["new/images/b.png", "new/images/a.png"])
    assert result["compliant"] is True
    assert result["old_raw_image_count"] == 0


def test_new_images_only_rejects_old_raw_image() -> None:
    result = verify_new_images_only(["generated/images/a.png", "generated/images/old.png"], ["new/images/a.png"])
    assert result["compliant"] is False
    assert result["unexpected_stems"] == ["old"]


def test_incremental_decision_requires_compliance_and_both_metrics() -> None:
    assert evaluate_incremental_metrics(0.60, 0.95, True)["passed"] is True
    assert evaluate_incremental_metrics(0.90, 0.99, False)["passed"] is False
    assert evaluate_incremental_metrics(0.59, 0.99, True)["passed"] is False
    assert evaluate_incremental_metrics(0.90, 0.94, True)["passed"] is False


def test_class_incremental_scope_checks_training_validation_and_classes() -> None:
    result = verify_class_incremental_learning_scope(
        ["generated/train/a.png"],
        ["generated/val/b.png"],
        ["increment/train/a.png"],
        ["increment/val/b.png"],
        ["soldier", "tank", "warship"],
        ["small_aircraft"],
    )
    assert result["task_type"] == "class_incremental_object_detection"
    assert result["learning_data_scope"] == "incremental_dataset_only"
    assert result["learning_scope_verified"] is True
    assert result["old_raw_image_count"] == 0


def test_class_incremental_scope_rejects_old_validation_data() -> None:
    result = verify_class_incremental_learning_scope(
        ["generated/train/a.png"],
        ["generated/val/old.png"],
        ["increment/train/a.png"],
        ["increment/val/b.png"],
        ["soldier"],
        ["small_aircraft"],
    )
    assert result["learning_scope_verified"] is False
    assert result["old_raw_image_count"] == 1


def test_class_incremental_scope_rejects_overlap_and_invalid_class_partition() -> None:
    result = verify_class_incremental_learning_scope(
        ["generated/train/a.png"],
        ["generated/val/a.png"],
        ["increment/train/a.png"],
        ["increment/val/a.png"],
        ["soldier", "tank"],
        ["tank"],
    )
    assert result["train_validation_overlap"] == ["a"]
    assert result["class_partition_valid"] is False
    assert result["learning_scope_verified"] is False


def test_new_images_only_content_check_rejects_renamed_old_image(tmp_path) -> None:
    allowed = tmp_path / "allowed" / "a.png"
    generated = tmp_path / "generated" / "a.png"
    allowed.parent.mkdir()
    generated.parent.mkdir()
    allowed.write_bytes(b"incremental-image")
    generated.write_bytes(b"old-image-renamed")
    result = verify_new_images_only([generated], [allowed], verify_content=True)
    assert result["compliant"] is False
    assert result["content_mismatches"] == ["a"]
    assert result["old_raw_image_count"] == 1


def test_required_compliance_never_falls_back_to_legacy_replay_metrics(tmp_path) -> None:
    legacy = tmp_path / "legacy.csv"
    legacy.write_text(
        "protocol,new_map50_after,krr\n"
        "p01_new_small_aircraft,0.99,1.0\n",
        encoding="utf-8",
    )
    config = {
        "inputs": {
            "incremental_compliant_metrics": str(tmp_path / "missing.csv"),
            "incremental_metrics": [str(legacy)],
        },
        "incremental": {
            "require_compliant_no_old_data": True,
            "task_type": "class_incremental_object_detection",
            "learning_data_scope": "incremental_dataset_only",
            "expected_protocols": ["p01_new_small_aircraft"],
        },
        "thresholds": {"min_new_class_map50": 0.60, "min_krr": 0.95},
    }
    result = parse_incremental(config)
    assert result["protocols"] == []
    assert result["complete"] is False
    assert result["source"] == "missing_compliant_metrics"
