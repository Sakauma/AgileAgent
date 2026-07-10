# AgileAgent

AgileAgent is an auditable IR/SAR rapid-learning workbench built around a frozen YOLO11s detector, policy-driven diagnostics, compliant incremental learning, and submission packaging. It provides both a CLI and a Streamlit interface while keeping training, inference, and packaging behind explicit safety gates.

## Highlights

- Dataset audit, metadata generation, and sensor-aware splitting.
- Blackboard state with fingerprints, freshness, blockers, and artifact hashes.
- Policy decisions with risk levels and an allowlist for automatic execution.
- No-old-data incremental learning using a frozen base detector and class-routed new-class specialists.
- Submission inference with weight verification, unique output directories, manifests, and zip packages.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[workbench,dev]"
```

Install inference dependencies only in an environment with a compatible PyTorch/CUDA stack:

```bash
pip install -e ".[inference]"
```

## Agent CLI

```bash
python -m fair_agent.cli doctor
python -m fair_agent.cli refresh
python -m fair_agent.cli decide --sensor sar --scene urban --class-focus soldier
python -m fair_agent.cli pipeline --mode dryrun
python -m fair_agent.cli serve
```

## Compliant Incremental Learning

The incremental stage never replays old raw images. The base detector stays immutable for old classes, while a one-class specialist learns only from the newly supplied images. Predictions are composed by class, so old knowledge retention is structural rather than dependent on rehearsal. Teacher pseudo-label distillation remains available as an auditable ablation.

```bash
python tools/27_build_new_class_specialist_dataset.py \
  --config configs/incremental_no_old_distill_yolo11s.yaml \
  --protocol p01_new_small_aircraft

python tools/24_run_no_old_distill.py \
  --config configs/incremental_no_old_distill_yolo11s.yaml \
  --protocol p01_new_small_aircraft --device 0

python tools/25_aggregate_no_old_distill.py \
  --config configs/incremental_no_old_distill_yolo11s.yaml
```

To build the teacher pseudo-label ablation instead:

```bash
python tools/23_build_no_old_distill_dataset.py \
  --config configs/incremental_no_old_distill_yolo11s.yaml \
  --protocol p01_new_small_aircraft
```

See [docs/compliant-incremental-learning.md](docs/compliant-incremental-learning.md) for the protocol and audit contract.

## Data Policy

This repository intentionally excludes competition datasets, labels, model weights, predictions, reports, PDFs, SSH credentials, and generated run directories. Supply authorized data and frozen weights locally using the paths in `configs/`.
