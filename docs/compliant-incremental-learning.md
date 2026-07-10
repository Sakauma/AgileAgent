# Compliant Incremental Learning

## Constraint

During an incremental round, training may use only the newly supplied images and annotations. Old raw training images must never enter the student train or validation split.

## Method

The primary implementation for each 3+1 protocol is:

1. Freeze the detector trained on the three base classes.
2. Remap the newly introduced global class ID to class `0` inside an isolated specialist dataset.
3. Train a one-class specialist only on the incremental images and labels.
4. Keep the base detector immutable for old-class inference.
5. Compose predictions by taking old classes from the frozen base and the new class from the specialist.
6. Map specialist class `0` back to the global class ID.
7. Evaluate the composed class-wise output on the full test view.

The repository also includes a teacher pseudo-label distillation ablation. It generates old-class pseudo labels only on incremental images, but it never admits old raw images into training.

## Audit Contract

Every generated protocol contains `manifest.json` with:

- `training_source_policy: new_incremental_images_only`
- exact training image count
- new-class ground-truth object count
- old-class pseudo-label count
- `old_raw_image_count`, which must equal zero
- a strict stem comparison between generated training images and the authorized new-image split

The training runner refuses to start when the compliance check fails.

## Acceptance

A protocol passes only when all conditions hold:

```text
old_raw_image_count = 0
frozen_parameter_max_abs_drift = 0
New-mAP50 >= 0.60
KRR >= 0.95
```

KRR is computed as `old_mAP50_after / old_mAP50_before` on the same full test view. Per-protocol outputs are isolated under `reports/incremental_no_old_distill/<protocol>/`; aggregation never overwrites individual evidence.
