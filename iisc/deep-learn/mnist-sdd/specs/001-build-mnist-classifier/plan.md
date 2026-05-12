# Implementation Plan: MNIST Digit Classifier Pipeline

**Branch**: `001-build-mnist-classifier` | **Date**: 2026-05-12 | **Spec**: `specs/001-build-mnist-classifier/spec.md`
**Input**: Feature specification from `specs/001-build-mnist-classifier/spec.md`

## Summary

Build and maintain a reproducible MNIST training and analysis pipeline that uses explicit device selection (`cpu` or `xpu`), fails fast when the selected device is unavailable, records timing and quality metrics, emits concise progress logs to console and per-run files, supports CPU-only container execution, and enforces code-quality gates. The standard success target is test accuracy >= 97% for standard runs.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: `torch`, `torchvision`, `matplotlib`, stdlib `csv`, stdlib `logging`, `pytest`  
**Storage**: Local filesystem artifacts (CSV, PNG, model checkpoint, plain-text run logs)  
**Testing**: `pytest` (unit, integration, contract) — test-first for all new and changed behavior  
**Target Platform**: Linux host; Docker runtime (`python:3.11-slim`)  
**Project Type**: CLI ML workflow (training + analysis)  
**Performance Goals**:
- MNIST test accuracy >= 97% for standard training runs (SC-002)
- Primary workflow (train/evaluate/visualize) succeeds in >= 95% of valid runs (SC-001)
- Track `elapsed_seconds` (epoch) and `training_time_seconds` (run) for regression checks  
**Constraints**:
- Device is explicit (`cpu` or `xpu`) with no automatic fallback
- If `xpu` is selected and unavailable, fail with actionable error
- Logs are plain text and must include epoch fields: `epoch`, `elapsed_seconds`, `loss`, `accuracy`
- Docker image is CPU-only, uses `python:3.11-slim`, volume mount + download fallback
- All code must pass linting and static type checks before commit  
**Scale/Scope**:
- MNIST digits (0-9) only
- Single-model architecture: 784 -> 256 -> 128 -> 10
- Local/lab execution with local output directories

## Constitution Check (Pre-Design)

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Testing Standards: PASS — test-first enforced per user story; tests precede implementation tasks.
- Code Quality Standards: PASS — lint + static analysis required by constitution; explicit code-quality validation gate added.
- UX Consistency: PASS — fail-fast device validation, actionable errors, stable CLI flags, stable outputs.
- Performance Requirements: PASS — SC-002 (≥97% test accuracy) and SC-001 (≥95% workflow success) are measurable and have explicit validation tasks.
- Commit Control: PASS — explicit user approval gate task is required before any commit; no auto-commit.

## Project Structure

### Documentation (this feature)

```text
specs/001-build-mnist-classifier/
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   |-- cli.md
|   `-- docker-cli.md
`-- tasks.md
```

### Source Code (repository root)

```text
src/
|-- __init__.py
|-- analyze.py
|-- data.py
|-- device.py
|-- metrics.py
|-- model.py
`-- train.py

tests/
|-- contract/
|   |-- test_analyze_cli.py
|   |-- test_docker_files.py
|   `-- test_train_cli.py
|-- integration/
|   |-- test_analyze_curves.py
|   |-- test_classification_analysis.py
|   `-- test_train_pipeline.py
`-- unit/
    |-- test_analyze_classification.py
    |-- test_analyze_curves.py
    |-- test_data.py
    |-- test_device.py
    |-- test_metrics.py
    `-- test_model.py
```

**Structure Decision**: Single Python CLI project with clear split between runtime modules (`src/`) and test layers.

## Phase 0: Research Summary

Research decisions are documented in `specs/001-build-mnist-classifier/research.md` and resolve all prior clarifications:
- Explicit `cpu|xpu` device contract and fail-fast behavior (Decision 1, 2)
- Epoch and run timing persistence (Decision 3)
- CPU vs XPU analysis outputs (Decision 4)
- Minimal dependencies: `torch`, `torchvision`, `matplotlib`, stdlib only (Decision 5)
- Docker CPU-only portability using `python:3.11-slim` (Decision 6)
- CPU-only PyTorch wheel in Docker; XPU wheel on host (Decision 7)
- Data volume mount with download fallback (Decision 8)
- Plain-text progress logging (console + `run_<run_id>.log`) (Decision 9–12)
- Standard run quality target >= 97% test accuracy (Decision 13)
- FR-014 runtime validation: volume mount + download fallback each need explicit test coverage (Decision 14)

No unresolved technical clarifications remain.

## Phase 1: Design & Contracts

Design outputs are captured in:
- `specs/001-build-mnist-classifier/data-model.md`
- `specs/001-build-mnist-classifier/contracts/cli.md`
- `specs/001-build-mnist-classifier/contracts/docker-cli.md`
- `specs/001-build-mnist-classifier/quickstart.md`

Design coverage includes:
- Run, epoch metrics, evaluation snapshots, timing comparisons, progress-log event modeling
- CLI argument/behavior/output contracts for training and analysis
- Docker build/run contract for CPU-only containerized workflow
- Data mount and download-fallback behavior explicitly described in docker-cli.md
- Local and Docker quickstart flows aligned with module invocation (`python -m src.*`)
- Lint/type validation instruction added to quickstart notes

## Constitution Check (Post-Design)

- Testing Standards: PASS — all user stories have test tasks before implementation; foundational test tasks precede implementation tasks.
- Code Quality Standards: PASS — explicit lint/type validation task added to Polish phase.
- UX Consistency: PASS — user-visible behavior consistent across local and Docker usage; error messages are actionable.
- Performance Requirements: PASS — SC-001 and SC-002 both have measurable validation tasks.
- Commit Control: PASS — T045 is a dedicated explicit user approval gate task; no task may commit without it.

## Complexity Tracking

No constitutional violations or justified complexity exceptions were required for this plan.
