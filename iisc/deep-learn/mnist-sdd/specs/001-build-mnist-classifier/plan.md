# Implementation Plan: MNIST Digit Classifier Pipeline

**Branch**: `001-build-mnist-classifier` | **Date**: 2026-05-11 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-build-mnist-classifier/spec.md`

## Summary

Build a compact PyTorch MNIST classifier with fixed architecture
(`784 -> 256 -> 128 -> 10`), explicit device execution (`cpu` or `xpu`),
deterministic CSV logging, and offline analysis utilities. The implementation
must log epoch timing and total training time, then compare both model quality
and elapsed time between CPU and XPU runs.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: PyTorch, torchvision, matplotlib  
**Storage**: Local filesystem CSV files, plots, and model checkpoints  
**Testing**: pytest (unit, contract, and integration checks)  
**Target Platform**: Linux laptop/workstation with CPU and integrated XPU  
**Project Type**: CLI-oriented ML application  
**Performance Goals**: 
- Log `elapsed_seconds` per epoch and `training_time_seconds` per run in CSV outputs
- Produce CPU vs XPU time comparison output for completed paired runs
- Preserve model quality target: MNIST test accuracy >= 98% for standard run  
**Constraints**:
- Device must be explicitly selected by user (`cpu` or `xpu`)
- Torch CUDA/GPU detection is not used in this environment
- If `xpu` is selected but unavailable, fail fast with actionable error and no fallback
- Minimal dependency footprint (no pandas)  
**Scale/Scope**: MNIST digits 0-9, local experiment workflow, single feature scope

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Testing Standards: PASS. Test-first tasks planned per user story (unit + integration + contract).
- Code Quality Standards: PASS. Type hints and small focused modules are required in implementation tasks.
- UX Consistency: PASS. CLI arguments and failure messages are explicit and stable.
- Performance Requirements: PASS. Epoch/run time logging and CPU/XPU timing comparison are included.
- Commit Control: PASS. Workflow keeps explicit user approval before any commit.

## Project Structure

### Documentation (this feature)

```text
specs/001-build-mnist-classifier/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── __init__.py
├── model.py
├── train.py
├── analyze.py
├── device.py
├── data.py
└── metrics.py

tests/
├── unit/
├── integration/
└── contract/
```

**Structure Decision**: Single Python CLI project. Keep training pipeline and
analysis in separate modules so time and quality comparisons can run offline
from persisted CSV artifacts.

## Phase 0: Research Outcomes

- Device strategy: use `cpu` and `xpu` as supported runtime targets.
- Runtime detection strategy: check XPU availability via PyTorch XPU API; do not
  rely on CUDA checks in this project.
- Timing strategy: capture epoch duration and total run duration in CSV, then
  compute and visualize CPU vs XPU time deltas in analysis.

## Phase 1: Design Outputs

- `research.md` updated with explicit CPU/XPU decisions and alternatives.
- `data-model.md` updated to include timing fields and comparison entity.
- `contracts/cli.md` updated with `cpu|xpu` contract and timing outputs.
- `quickstart.md` updated with paired-run comparison flow.
- `.github/copilot-instructions.md` already references this feature plan.

## Post-Design Constitution Re-Check

- Testing Standards: PASS. Independent testable slices retained for US1, US2, US3.
- Code Quality Standards: PASS. Public interfaces remain typed and modular by design.
- UX Consistency: PASS. CLI contract now explicit for `cpu|xpu` and clear errors.
- Performance Requirements: PASS. Training-time logging and CPU/XPU time comparison integrated.
- Commit Control: PASS. No auto-commit behavior introduced.

## Complexity Tracking

No constitution violations requiring justification.
