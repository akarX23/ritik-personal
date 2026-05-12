# Implementation Plan: MNIST Digit Classifier Pipeline

**Branch**: `001-build-mnist-classifier` | **Date**: 2026-05-12 | **Spec**: `specs/001-build-mnist-classifier/spec.md`
**Input**: Feature specification from `specs/001-build-mnist-classifier/spec.md`

## Summary

Update the existing MNIST pipeline to support multi-batch training in one command (`--batches`), persist combined CSV rows with explicit `batch_size`, and produce `results.md` from historical rows in the results directory. The report must include configuration metadata plus two comparison tables: final split metrics per batch size and epoch-sampled comparisons every 10 epochs plus final epoch.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: `torch`, `torchvision`, `matplotlib`, stdlib `csv`, stdlib `logging`, `pytest`  
**Storage**: Local filesystem artifacts (CSV, PNG, model checkpoint, plain-text run logs, markdown report)  
**Testing**: `pytest` (unit, integration, contract) with test-first updates for new CLI and reporting behavior  
**Target Platform**: Linux host; Docker runtime (`python:3.11-slim`)  
**Project Type**: CLI ML workflow (training + analysis)  
**Performance Goals**:
- Standard CPU run keeps SC-002 at test accuracy >= 97%
- Multi-batch runs maintain measurable per-batch timings and reproducible comparisons
- Primary workflow reliability remains >= 95% successful valid runs (SC-001)
**Constraints**:
- Device remains explicit (`cpu` or `xpu`) with no fallback
- Multi-batch execution is sequential per batch size
- Shared CSV outputs must include `batch_size` and distinct `run_id` per batch execution
- `results.md` compares all matching historical rows in selected results directory
- Code-quality gates (lint and type checks) remain mandatory before commit approval
**Scale/Scope**:
- MNIST digits only (0-9)
- Fixed architecture: 784 -> 256 -> 128 -> 10
- Local results directories may accumulate multiple historical runs for cross-batch comparison

## Constitution Check (Pre-Design)

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Testing Standards: PASS - New multi-batch and report requirements have explicit test-first expectations.
- Code Quality Standards: PASS - Changes remain in typed Python modules and contract docs; lint/type gates retained.
- UX Consistency: PASS - New CLI behavior and report structure are explicit and deterministic.
- Performance Requirements: PASS - Per-batch timing/accuracy comparisons are measurable and preserved in shared outputs.
- Commit Control: PASS - No commit is planned without explicit user approval.

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

**Structure Decision**: Keep single-project CLI layout and evolve existing modules/contracts for multi-batch reporting features.

## Phase 0: Research Summary

Existing decisions (1-14) remain valid. Additional decisions for new clarifications are captured in updated `research.md`:
- `--batches` comma-separated CLI contract
- Sequential batch execution strategy with distinct run IDs
- Shared CSV schema extension with `batch_size`
- `results.md` generation format and historical-row comparison scope
- Epoch sampling rule (every 10 epochs plus final epoch)

No unresolved clarifications remain.

## Phase 1: Design & Contracts

Updated artifacts for new requirements:
- `specs/001-build-mnist-classifier/data-model.md` (adds batch-aware run/report entities)
- `specs/001-build-mnist-classifier/contracts/cli.md` (extends train/analyze contracts for `--batches` and `results.md`)
- `specs/001-build-mnist-classifier/quickstart.md` (adds multi-batch usage and report validation steps)

Agent context reference remains correct in `.github/copilot-instructions.md` and points to this plan path.

## Constitution Check (Post-Design)

- Testing Standards: PASS - New report scope and batch semantics are contract-testable.
- Code Quality Standards: PASS - Added requirements stay within existing quality gates.
- UX Consistency: PASS - User-facing CLI/report contracts are explicit.
- Performance Requirements: PASS - Batch-wise and epoch-wise comparison outputs preserve measurable outcomes.
- Commit Control: PASS - Commit approval remains explicit and external to planning output.

## Complexity Tracking

No constitutional violations or complexity exceptions were required.
