# Implementation Plan: MNIST Digit Classifier Pipeline

**Branch**: `001-build-mnist-classifier` | **Date**: 2026-05-12 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `/specs/001-build-mnist-classifier/spec.md`

## Summary

Build a fully containerised MNIST digit classifier. The system trains a feed-forward neural network, logs per-epoch and per-run timing metrics to CSV, generates learning-curve and classification-quality visualisations, and packages everything in a minimal Docker image (`python:3.11-slim`). On the host, both CPU and XPU (Intel integrated GPU) are valid execution targets; inside the container only CPU is supported. The Dockerfile uses a host-mounted volume for MNIST data and downloads it automatically when absent.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: PyTorch (CPU wheel via PyPI inside Docker; nightly XPU wheel for host), torchvision, matplotlib (Agg), stdlib csv  
**Storage**: Local filesystem — `data/` (MNIST), `results/` (CSV metrics, PNGs, `model.pt`)  
**Testing**: pytest — unit, contract, integration suites  
**Target Platform**: Linux (Docker container, `python:3.11-slim`); host Linux with CPU or XPU  
**Project Type**: CLI tool + Docker image  
**Performance Goals**: ≥98% test-set accuracy on standard MNIST run; per-epoch timing in CSV  
**Constraints**: CPU-only inside Docker; XPU passthrough excluded; no XPU/GPU driver deps in image; no pandas; stdlib csv only; no auto-device fallback  
**Scale/Scope**: Single-machine, single-dataset (MNIST 0–9); local results storage

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Testing Standards** ✅ — Test-first workflow already established; unit + integration + contract tests exist for all source modules; 80% coverage target enforced.
- **Code Quality Standards** ✅ — All public interfaces typed; linting/static analysis required to pass before commit.
- **UX Consistency** ✅ — Fail-fast errors with actionable messages for invalid device, missing data, missing prerequisite files; stable CLI flag conventions.
- **Performance Requirements** ✅ — ≥98% accuracy defined in SC-002; per-epoch `elapsed_seconds` and per-run `training_time_seconds` logged; CPU vs XPU timing comparison chart produced by `analyze.py`.
- **Commit Control** ✅ — Explicit user approval required before every commit (Constitution V; FR-012).

No violations to justify.

## Project Structure

### Documentation (this feature)

```text
specs/001-build-mnist-classifier/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── __init__.py
├── device.py            # Explicit cpu/xpu validation; fail-fast
├── data.py              # MNIST DataLoader; 90/10 train/val split
├── metrics.py           # stdlib csv writers; timing columns
├── model.py             # MnistClassifier 784→256(ReLU)→128(ReLU)→10
├── train.py             # CLI entry point; training loop; predictions
└── analyze.py           # Learning curves, device/time comparison, confusion matrix

tests/
├── unit/
│   ├── test_model.py
│   ├── test_device.py
│   ├── test_metrics.py
│   ├── test_data.py
│   ├── test_analyze_curves.py
│   └── test_analyze_classification.py
├── contract/
│   ├── test_train_cli.py
│   └── test_analyze_cli.py
└── integration/
    ├── test_train_pipeline.py
    ├── test_analyze_curves.py
    └── test_classification_analysis.py

requirements.txt         # XPU index scoped to torch + torchvision only
requirements-docker.txt  # NEW — matplotlib + pytest; no torch (installed inline in Dockerfile)
Dockerfile               # NEW — python:3.11-slim; CPU-only PyTorch wheel
.dockerignore            # NEW — exclude data/, results/, .venv/, __pycache__/, .git/
.gitignore
```

**Structure Decision**: Single-project layout. Existing `src/` and `tests/` tree retained. New files are `Dockerfile`, `.dockerignore`, and `requirements-docker.txt` at the project root.

## Complexity Tracking

No Constitution violations requiring justification.

---

## Phase 0 — Research

*See [research.md](research.md) for full findings.*

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Docker base image | `python:3.11-slim` | Minimal footprint; glibc present for PyTorch wheels; no build tools needed at runtime |
| PyTorch wheel (container) | `torch` from `https://download.pytorch.org/whl/cpu` | XPU nightly wheel is host-only; container uses standard CPU build |
| PyTorch wheel (host) | `torch --index-url https://download.pytorch.org/whl/nightly/xpu` | Required for XPU device; scoped per-package in requirements.txt |
| Data provisioning | Volume mount (`-v ./data:/app/data`); download fallback | Avoids baking dataset into image; self-contained when no volume provided |
| XPU in container | Excluded — CPU only | Device passthrough requires driver deps that inflate image and break portability |
| `requirements.txt` split | `--index-url` per package (torch, torchvision only) | `matplotlib` and `pytest` come from PyPI; XPU index must not be applied globally |

---

## Phase 1 — Design & Contracts

*See [data-model.md](data-model.md) and [contracts/](contracts/) for artifacts.*

### New Files

#### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch in a dedicated layer (large, rarely changes)
RUN pip install --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu

# Install remaining runtime deps
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy source code
COPY src/ src/
COPY tests/ tests/

# Optional volumes: mount host data/results or let container download data
VOLUME ["/app/data"]
VOLUME ["/app/results"]

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--device", "cpu", "--data", "/app/data", "--results", "/app/results"]
```

#### `requirements-docker.txt`

```
matplotlib
pytest
```

#### `.dockerignore`

```
__pycache__/
*.pyc
.venv/
venv/
data/
results/
*.pt
.DS_Store
Thumbs.db
*.tmp
*.swp
.git/
```

### Contract: Docker CLI Usage

```bash
# Build image
docker build -t mnist-classifier .

# Train (CPU, default settings, host data mounted):
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  mnist-classifier

# Train with custom epochs:
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  mnist-classifier --epochs 20

# Run analysis (override entrypoint):
docker run --rm \
  -v "$(pwd)/results:/app/results" \
  --entrypoint python mnist-classifier \
  -m src.analyze --results /app/results

# Run full test suite inside container:
docker run --rm mnist-classifier pytest tests/ -v
```

---

## Constitution Check (Post-Design)

- **Testing Standards** ✅ — New Docker tasks will include contract tests: image builds successfully, default entrypoint exits 0 on a smoke run.
- **Code Quality Standards** ✅ — No new Python source; Dockerfile follows layer-caching best practices (large deps first, source last).
- **UX Consistency** ✅ — Default `CMD` provides a working out-of-box invocation with explicit `--device cpu`; error messages from `src.train` propagate via exit code.
- **Performance Requirements** ✅ — Accuracy and timing budgets unchanged; CPU-only container performance is acceptable for CI/portability use cases.
- **Commit Control** ✅ — All new files committed only with explicit user approval.
