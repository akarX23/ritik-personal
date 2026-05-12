---
description: "Task list for MNIST Digit Classifier Pipeline implementation"
---

# Tasks: MNIST Digit Classifier Pipeline

**Input**: Design documents from `/specs/001-build-mnist-classifier/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md, contracts/docker-cli.md

**Tests**: Test tasks are REQUIRED by this feature's test-first requirements.  
**Organization**: Tasks are grouped by user story so each story remains independently testable.

## Format: `[ID] [P?] [Story] Description`

- [P] indicates parallelizable work (different files, no blocking dependency)
- [Story] label is required for user story phases only (`[US1]`, `[US2]`, `[US3]`)
- Every task includes an exact file path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Align dependency and project scaffolding with current plan/contracts.

- [ ] T001 Update host dependencies in `requirements.txt` to keep XPU wheel index scoped to `torch` and `torchvision` only
- [ ] T002 [P] Ensure docker runtime dependencies in `requirements-docker.txt` include non-torch packages only (`matplotlib`, `pytest`)
- [ ] T003 [P] Ensure root ignore rules in `.gitignore` and `.dockerignore` are aligned with data/results/model/log artifacts
- [ ] T004 Verify package entrypoint markers in `src/__init__.py` and `tests/__init__.py` for module execution support

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared behavior required by all stories before feature work.

**CRITICAL**: No user story implementation begins until this phase is complete.

- [ ] T005 Implement/confirm explicit device resolution contract (`cpu` or `xpu`) with fail-fast behavior in `src/device.py`
- [ ] T006 [P] Implement/confirm MNIST loader split behavior and defaults in `src/data.py`
- [ ] T007 [P] Implement/confirm CSV writer schemas in `src/metrics.py` including timing fields and run summary fields
- [ ] T008 Add shared logging setup helper for plain-text console + file logging in `src/train.py`
- [ ] T009 Add deterministic per-run log naming (`run_<run_id>.log`) integration with run identifiers in `src/train.py`

**Checkpoint**: Foundational runtime, metrics, and logging primitives are ready.

---

## Phase 3: User Story 1 - Train a Reliable MNIST Classifier (Priority: P1) 🎯 MVP

**Goal**: Train MNIST reliably with explicit device selection, persisted artifacts, and concise progress logging.

**Independent Test**: Run one CPU training cycle with module invocation and verify model/checkpoint/CSV/log artifacts plus fail-fast invalid device behavior.

### Tests for User Story 1 (REQUIRED)

- [ ] T010 [P] [US1] Add unit tests for model forward shape and deterministic behavior in `tests/unit/test_model.py`
- [ ] T011 [P] [US1] Add unit tests for explicit device validation and no-fallback behavior in `tests/unit/test_device.py`
- [ ] T012 [P] [US1] Add unit tests for metrics CSV schema writing in `tests/unit/test_metrics.py`
- [ ] T013 [P] [US1] Add contract tests for `python -m src.train` CLI arguments and required `--device` in `tests/contract/test_train_cli.py`
- [ ] T014 [P] [US1] Add integration test for one-epoch train run artifact generation in `tests/integration/test_train_pipeline.py`
- [ ] T015 [P] [US1] Add integration/contract test for training log behavior (console+file, required epoch fields, file naming) in `tests/contract/test_train_cli.py`

### Implementation for User Story 1

- [ ] T016 [US1] Implement/confirm MNIST classifier architecture `784->256->128->10` in `src/model.py`
- [ ] T017 [US1] Implement/confirm training CLI parser and module entrypoint behavior in `src/train.py`
- [ ] T018 [US1] Implement/confirm training epoch loop, optimizer/loss flow, and per-epoch metrics persistence in `src/train.py`
- [ ] T019 [US1] Implement/confirm validation/test checkpoint evaluation during training and CSV persistence in `src/train.py`
- [ ] T020 [US1] Implement concise per-epoch progress log lines with `epoch`, `elapsed_seconds`, `loss`, `accuracy` in `src/train.py`
- [ ] T021 [US1] Implement lifecycle logging (run start/end, evaluation, checkpoint, errors) to console and `run_<run_id>.log` in `src/train.py`
- [ ] T022 [US1] Implement/confirm `run_summary.csv` final write with `training_time_seconds` and final metrics plus model checkpoint save in `src/train.py`

**Checkpoint**: User Story 1 is independently runnable and testable (MVP).

---

## Phase 4: User Story 2 - Visualize Learning Curves (Priority: P2)

**Goal**: Produce learning and device/time comparison plots from persisted run outputs with clear analysis lifecycle logs.

**Independent Test**: Execute analysis on fixture and smoke-run results; verify required plot files and log behavior.

### Tests for User Story 2 (REQUIRED)

- [ ] T023 [P] [US2] Add contract tests for `python -m src.analyze` required args and missing-results errors in `tests/contract/test_analyze_cli.py`
- [ ] T024 [P] [US2] Add unit tests for learning curve generation from synthetic CSV data in `tests/unit/test_analyze_curves.py`
- [ ] T025 [P] [US2] Add integration test for end-to-end curve generation from fixture results in `tests/integration/test_analyze_curves.py`
- [ ] T026 [P] [US2] Add tests for analysis lifecycle logging to console + `run_<run_id>.log` in `tests/contract/test_analyze_cli.py`

### Implementation for User Story 2

- [ ] T027 [US2] Implement/confirm analyze CLI parser and entrypoint for module invocation in `src/analyze.py`
- [ ] T028 [P] [US2] Implement/confirm learning curve plot generation in `src/analyze.py`
- [ ] T029 [P] [US2] Implement/confirm CPU-vs-XPU quality comparison plot generation in `src/analyze.py`
- [ ] T030 [US2] Implement/confirm CPU-vs-XPU training time comparison plot generation in `src/analyze.py`
- [ ] T031 [US2] Implement lifecycle logging to console and per-run log file for analysis workflow in `src/analyze.py`

**Checkpoint**: User Story 2 works independently with fixture data and real run outputs.

---

## Phase 5: User Story 3 - Analyze Classification Quality (Priority: P3)

**Goal**: Generate confusion matrix and class-wise precision/recall/F1 visualizations from test predictions.

**Independent Test**: Run classification analysis against fixture and real predictions; verify plot outputs and dependency errors.

### Tests for User Story 3 (REQUIRED)

- [ ] T032 [P] [US3] Add unit tests for confusion matrix plot generation in `tests/unit/test_analyze_classification.py`
- [ ] T033 [P] [US3] Add unit tests for classification report plot generation in `tests/unit/test_analyze_classification.py`
- [ ] T034 [P] [US3] Add integration test for classification analysis from predictions fixture in `tests/integration/test_classification_analysis.py`

### Implementation for User Story 3

- [ ] T035 [US3] Implement/confirm test prediction export (`predictions.csv`) in `src/train.py`
- [ ] T036 [P] [US3] Implement/confirm confusion matrix plot generation in `src/analyze.py`
- [ ] T037 [P] [US3] Implement/confirm classification report plot generation in `src/analyze.py`
- [ ] T038 [US3] Implement/confirm missing-predictions dependency error handling in `src/analyze.py`

**Checkpoint**: User Story 3 is independently testable and complete.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Containerization, docs alignment, validation, and release gates.

- [ ] T039 [P] Align Docker image build/runtime contract in `Dockerfile` with CPU-only wheel source and module entrypoint
- [ ] T040 [P] Add/confirm Docker contract tests for support files and directives in `tests/contract/test_docker_files.py`
- [ ] T041 Update quickstart examples and outputs (module invocation + logging artifact expectations) in `specs/001-build-mnist-classifier/quickstart.md`
- [ ] T042 Validate quickstart scenarios end-to-end (host train/analyze and docker train/analyze commands)
- [ ] T043 [P] Verify standard run target `>= 0.98` test accuracy and timing/log fields in `results/*/run_summary.csv` and `results/*/run_<run_id>.log`
- [ ] T044 Request explicit user approval before any commit operation for these changes

---

## Dependencies & Execution Order

### Phase Dependencies

- Phase 1 (Setup): no dependencies
- Phase 2 (Foundational): depends on Phase 1; blocks all user stories
- Phase 3 (US1): depends on Phase 2
- Phase 4 (US2): depends on Phase 2 and US1 artifact contracts
- Phase 5 (US3): depends on Phase 2 and US1 predictions export contract
- Phase 6 (Polish): depends on completion of user stories

### User Story Dependencies

- US1 (P1): first deliverable (MVP), independent after foundational work
- US2 (P2): independent with fixture data, integrates with US1 outputs for full workflow
- US3 (P3): independent with fixture predictions, integrates with US1 outputs for full workflow

### Within Each User Story

- Tests are written before implementation
- CLI/contract behavior is established before integration flow checks
- Core artifact generation precedes analysis/visualization features

### Parallel Opportunities

- Setup: T002, T003, T004 can run in parallel
- Foundational: T006 and T007 can run in parallel after T005; T008 and T009 follow foundational primitives
- US1 tests: T010-T015 parallelizable by file focus
- US1 implementation: T016/T017 parallelizable; T018-T022 follow loop
- US2 tests: T023-T026 parallelizable
- US2 implementation: T028/T029 parallelizable after T027
- US3 tests: T032-T034 parallelizable
- US3 implementation: T036/T037 parallelizable after T035
- Polish: T039 and T040 parallelizable; T043 in parallel with T041 after runnable workflows exist

---

## Parallel Example: User Story 1

```text
Run in parallel after Phase 2:
- T010 tests/unit/test_model.py
- T011 tests/unit/test_device.py
- T012 tests/unit/test_metrics.py
- T013 tests/contract/test_train_cli.py
- T014 tests/integration/test_train_pipeline.py
- T015 tests/contract/test_train_cli.py (logging assertions)

Then implement:
- T016 src/model.py
- T017 src/train.py (CLI)
- T018-T022 src/train.py (loop, eval, logging, summary)
```

## Parallel Example: User Story 2

```text
Run in parallel:
- T024 tests/unit/test_analyze_curves.py
- T025 tests/integration/test_analyze_curves.py
- T026 tests/contract/test_analyze_cli.py

Then implement:
- T027 src/analyze.py (CLI)
- T028 src/analyze.py (learning curves)
- T029 src/analyze.py (device quality comparison)
- T030 src/analyze.py (time comparison)
- T031 src/analyze.py (analysis logging)
```

## Implementation Strategy

### MVP First (US1)

1. Complete Setup + Foundational phases
2. Complete US1 tests and implementation
3. Validate one smoke training run with required artifacts and logs
4. Demo/deploy MVP before moving to later stories

### Incremental Delivery

1. Deliver US1 (training + artifacts + logging)
2. Deliver US2 (learning/time comparison analysis)
3. Deliver US3 (classification quality analysis)
4. Finish cross-cutting docker/doc/performance/approval tasks

### Team Parallelization

1. One engineer focuses on train/runtime path (US1)
2. One engineer focuses on analyze path (US2/US3)
3. One engineer focuses on contracts, docker, and docs (Phase 6)
