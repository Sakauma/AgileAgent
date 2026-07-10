from fair_agent.modules.incremental_compliance import evaluate_incremental_metrics, verify_new_images_only


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
