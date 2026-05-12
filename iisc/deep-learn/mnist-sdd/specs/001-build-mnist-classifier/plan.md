# Implementation Plan: MNIST Digit Classifier Pipeline

**Branch**: `001-build-mnist-classifier` | **Date**: 2026-05-12 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-build-mnist-classifier/spec.md`

## Summary

Deliver a reproducible MNIST training and analysis pipeline that supports explicit host runtime device selection (`cpu` or `xpu`) and a CPU-only Docker workflow. The system persists metrics and artifacts, generates analysis plots, and now includes explicit progress logging for both training and analysis: concise per-epoch plain-text logs to console and per-run files named `run_<run_id>.log`.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: PyTorch, torchvision, matplotlib, stdlib `csv`, stdlib `logging`
**Storage**: Local filesystem (`data/`, `results/`, CSV files, plots, model checkpoint, per-run log files)
**Testing**: pytest (unit, contract, integration)
**Target Platform**: Linux host (CPU/XPU) and Linux container (`python:3.11-slim`, CPU-only)
**Project Type**: CLI-oriented ML application with Docker packaging
**Performance Goals**:
- MNIST test accuracy >= 98% for standard runs
- Persist `elapsed_seconds` per epoch and `training_time_seconds` per run
- Emit concise per-epoch progress logs without per-batch noise
**Constraints**:
- Explicit device selection only; no automatic fallback
- Container execution is CPU-only
- Progress logs must be plain text in console and `run_<run_id>.log` per run
- Each per-epoch line must include: `epoch`, `elapsed_seconds`, `loss`, `accuracy`
- No pandas dependency
**Scale/Scope**: Local single-user workflow, MNIST digits 0-9, single-results-directory artifacts

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Testing Standards: PASS. Existing test-first workflow; logging adds new contract and unit checks.
- Code Quality Standards: PASS. Public interfaces remain typed; logging via stdlib keeps complexity low.
- UX Consistency: PASS. Clear, concise progress logs and actionable error messages are retained.
- Performance Requirements: PASS. Timing metrics are preserved; logging granularity constrained to per-epoch for low overhead.
- Commit Control: PASS. Any commit still requires explicit user approval in-session.

No constitutional violations identified.

## Project Structure

### Documentation (this feature)

```text
specs/001-build-mnist-classifier/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── cli.md
│   └── docker-cli.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── __init__.py
├── device.py
├── data.py
├── metrics.py
├── model.py
├── train.py
└── analyze.py

tests/
├── unit/
├── integration/
└── contract/

Dockerfile
.dockerignore
requirements.txt
requirements-docker.txt
```

**Structure Decision**: Keep a single-project layout with modular CLI entry points. Logging integration is implemented within `src/train.py` and `src/analyze.py` using shared formatting conventions.

## Complexity Tracking

No complexity exceptions requiring justification.

---

## Phase 0: Research Outcomes

All technical unknowns are resolved in [research.md](research.md). Key outcomes:

1. Container base image remains `python:3.11-slim`.
2. Host uses XPU wheels only for `torch` and `torchvision`; container uses CPU wheels.
3. Data strategy remains volume mount with download fallback.
4. Logging strategy is fixed:
   - Destination: console + per-run file in results directory
   - Format: plain text
   - Granularity: concise per-epoch + lifecycle events
   - Required per-epoch fields: `epoch`, `elapsed_seconds`, `loss`, `accuracy`
   - File naming: `run_<run_id>.log`

## Phase 1: Design & Contracts

### Data Model Updates

- Add logging entity semantics to capture run-level and per-epoch log events.
- Preserve compatibility with existing CSV entities; logs are supplemental observability artifacts.

### Contract Updates

- Update CLI contract to define module invocation (`python -m src.train`, `python -m src.analyze`).
- Add logging behavior contract:
  - Train and analyze both emit lifecycle log messages to console and file.
  - Per-epoch train logs contain `epoch`, `elapsed_seconds`, `loss`, `accuracy`.
  - Per-run log filename is `run_<run_id>.log` in `results_dir`.

### Quickstart Updates

- Local invocations use module mode.
- Expected outputs now include per-run plain-text log files.

## Post-Design Constitution Re-Check

- Testing Standards: PASS. Logging behavior can be asserted via contract tests and artifact checks.
- Code Quality Standards: PASS. Logging approach uses stdlib and clear formatting requirements.
- UX Consistency: PASS. Progress feedback is concise and predictable across runs.
- Performance Requirements: PASS. Logging volume bounded to per-epoch granularity.
- Commit Control: PASS. No commit automation introduced.
