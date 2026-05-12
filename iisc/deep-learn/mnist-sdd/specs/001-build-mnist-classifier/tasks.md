---
description: "Task list for MNIST Digit Classifier Pipeline implementation"
---

# Tasks: MNIST Digit Classifier Pipeline

**Input**: Design documents from /specs/001-build-mnist-classifier/  
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Test tasks are mandatory. Constitution requires test-first: every test task precedes its implementation task.
**Organization**: Tasks grouped by user story so each story is independently implementable and testable.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Align project scaffolding, dependency definitions, and baseline test layout.

- [X] T001 Align host dependencies for XPU host runtime in requirements.txt
- [X] T002 [P] Align container dependency list in requirements-docker.txt
- [X] T003 [P] Align docker build context exclusions in .dockerignore
- [X] T004 [P] Align repository ignore patterns for artifacts in .gitignore
- [X] T005 Verify package entrypoint marker for module execution in src/__init__.py
- [X] T006 Verify test package marker for module-based test discovery in tests/__init__.py

---

## Phase 2: Foundational Tests (Test-First Gate)

**Purpose**: Write tests for all shared runtime contracts BEFORE any implementation starts.

**CRITICAL (constitution §I)**: These tests must be written and confirmed to FAIL before Phase 3 implementation tasks begin.

- [X] T007 [P] Write unit tests for explicit cpu/xpu validation and fail-fast no-fallback behavior in tests/unit/test_device.py
- [X] T008 [P] Write unit tests for MNIST loader split sizes, DataLoader return types, and deterministic seeding in tests/unit/test_data.py
- [X] T009 [P] Write unit tests for CSV schema columns, append behavior, and timing fields in tests/unit/test_metrics.py
- [X] T010 Write contract test asserting run-id is stable string and log filename matches run_<run_id>.log pattern in tests/contract/test_train_cli.py
- [X] T011 Write contract test for console + file log output with required epoch fields (epoch, elapsed_seconds, loss, accuracy) in tests/contract/test_train_cli.py

**Checkpoint**: All foundational tests written and confirmed failing — implementation may begin.

---

## Phase 3: Foundational Implementation (Blocking Prerequisites)

**Purpose**: Implement the shared runtime contracts the tests above target.

- [X] T012 Implement explicit cpu/xpu validation and fail-fast device checks in src/device.py
- [X] T013 [P] Implement MNIST loading and train/validation/test split behavior in src/data.py
- [X] T014 [P] Implement stable CSV schemas including timing fields in src/metrics.py
- [X] T015 Add run-id generation and reusable log filename wiring in src/metrics.py
- [X] T016 Implement shared plain-text logger setup (console plus file) in src/train.py
- [X] T017 Implement reusable run log naming pattern run_<run_id>.log in src/train.py

**Checkpoint**: Foundation complete — foundational tests now pass — user stories can proceed.

---

## Phase 4: User Story 1 — Train a Reliable MNIST Classifier (Priority: P1) 🎯 MVP

**Goal**: Train and evaluate MNIST with explicit device control, persisted artifacts, and concise per-epoch progress logs.

**Independent Test**: Run one training cycle via module invocation and confirm model, CSV, and log outputs plus fail-fast device behavior.

### Tests for User Story 1 (MANDATORY — write before implementation)

- [X] T018 [P] [US1] Add unit tests for model tensor shape and determinism in tests/unit/test_model.py
- [X] T019 [P] [US1] Add contract tests for train CLI argument contract and required --device in tests/contract/test_train_cli.py
- [X] T020 [P] [US1] Add integration smoke test for train artifact generation (CSV, model.pt, log file) in tests/integration/test_train_pipeline.py
- [X] T021 [P] [US1] Add contract assertions for per-epoch log fields and run_<run_id>.log file naming in tests/contract/test_train_cli.py

### Implementation for User Story 1

- [X] T022 [US1] Implement fixed MLP architecture 784->256->128->10 in src/model.py
- [X] T023 [US1] Implement train CLI parser and module entrypoint behavior in src/train.py
- [X] T024 [US1] Implement training loop with optimizer/loss and per-epoch metrics writes in src/train.py
- [X] T025 [US1] Implement periodic validation and test snapshot persistence in src/train.py
- [X] T026 [US1] Implement concise per-epoch log line with epoch/elapsed_seconds/loss/accuracy in src/train.py
- [X] T027 [US1] Implement run summary persistence with training_time_seconds and checkpoint save in src/train.py

**Checkpoint**: User Story 1 is independently complete and demoable (MVP).

---

## Phase 5: User Story 2 — Visualize Learning Curves (Priority: P2)

**Goal**: Generate train/validation/test curves and CPU-vs-XPU quality/time comparison outputs.

**Independent Test**: Run analysis on prepared metrics and verify required plot files are generated.

### Tests for User Story 2 (MANDATORY — write before implementation)

- [X] T028 [P] [US2] Add contract tests for analyze CLI required arguments and missing-results error in tests/contract/test_analyze_cli.py
- [X] T029 [P] [US2] Add unit tests for learning-curve plot generation from synthetic CSV in tests/unit/test_analyze_curves.py
- [X] T030 [P] [US2] Add integration tests for end-to-end curve rendering from fixture results in tests/integration/test_analyze_curves.py
- [X] T031 [P] [US2] Add contract assertions for analysis lifecycle logging behavior in tests/contract/test_analyze_cli.py

### Implementation for User Story 2

- [X] T032 [US2] Implement analyze CLI parser and execution entrypoint in src/analyze.py
- [X] T033 [P] [US2] Implement train/validation/test learning curve plots in src/analyze.py
- [X] T034 [P] [US2] Implement CPU-vs-XPU quality comparison plot generation in src/analyze.py
- [X] T035 [US2] Implement CPU-vs-XPU training time comparison plot generation in src/analyze.py
- [X] T036 [US2] Implement analysis lifecycle logging to console and run log file in src/analyze.py

**Checkpoint**: User Story 2 is independently complete using fixture or real metrics.

---

## Phase 6: User Story 3 — Analyze Classification Quality (Priority: P3)

**Goal**: Produce confusion matrix and class-wise precision/recall/F1 visualizations from prediction artifacts.

**Independent Test**: Run analysis on predictions fixture and verify both classification outputs are created.

### Tests for User Story 3 (MANDATORY — write before implementation)

- [X] T037 [P] [US3] Add unit tests for confusion-matrix output generation in tests/unit/test_analyze_classification.py
- [X] T038 [P] [US3] Add unit tests for classification-report visualization output in tests/unit/test_analyze_classification.py
- [X] T039 [P] [US3] Add integration test for classification analysis workflow in tests/integration/test_classification_analysis.py

### Implementation for User Story 3

- [X] T040 [US3] Implement prediction export for final test pass (predictions.csv) in src/train.py
- [X] T041 [P] [US3] Implement confusion matrix plotting from predictions.csv in src/analyze.py
- [X] T042 [P] [US3] Implement class-wise precision/recall/F1 report plotting in src/analyze.py
- [X] T043 [US3] Implement missing-input dependency error handling for classification analysis in src/analyze.py

**Checkpoint**: User Story 3 is independently complete and testable.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Finalize container contract, code quality gates, docs, acceptance validation, and commit-gate controls.

### Container Coverage (FR-013, FR-014)

- [X] T044 [P] Align CPU-only container build/runtime contract in Dockerfile
- [X] T045 [P] Add or update docker contract coverage for support files in tests/contract/test_docker_files.py
- [X] T046 [P] Add integration test for FR-014 mounted-data path: verify torchvision reads existing files when /app/data is mounted in tests/integration/test_train_pipeline.py
- [X] T047 [P] Add integration test for FR-014 download-fallback path: verify torchvision auto-downloads when data dir is absent in tests/integration/test_train_pipeline.py
- [X] T048 Add integration test for full container workflow execution (train + analyze + verify artifacts) in tests/contract/test_docker_files.py

### Code Quality Gate (CAR-002, constitution §II)

- [X] T049 Run lint and static analysis on all changed files in src/ and tests/ and confirm zero violations

### Acceptance Validation (SC-001, SC-002)

- [X] T050 Verify SC-002: standard CPU run achieves test accuracy >= 0.97 using results/standard-cpu/run_summary.csv
- [X] T051 Verify SC-001: repeat primary workflow N=3 times and confirm all succeed (>=95% pass rate baseline)
- [ ] T052 Validate quickstart host and container flow examples and expected output list in specs/001-build-mnist-classifier/quickstart.md

### Commit Gate (FR-012, CAR-005, constitution §V — MANDATORY)

- [ ] T053 Request explicit user approval before any commit; present change summary and validation results

---

## Dependencies & Execution Order

### Phase Dependencies

- Phase 1 (Setup): no dependencies.
- Phase 2 (Foundational Tests): depends on Phase 1; BLOCKS Phase 3 implementation.
- Phase 3 (Foundational Implementation): depends on Phase 2 tests failing; BLOCKS all stories.
- Phase 4 (US1): depends on Phase 3.
- Phase 5 (US2): depends on Phase 3 and run metrics contracts.
- Phase 6 (US3): depends on Phase 3 and predictions contract.
- Phase 7 (Polish): depends on completion of selected stories.
- T053 (Commit Gate): must be the last task before any commit is made.

### User Story Dependencies

- US1 (P1): independent after foundational work; defines MVP.
- US2 (P2): independent with fixture data; full value when US1 outputs are present.
- US3 (P3): independent with fixture predictions; full value when US1 outputs are present.

### Within Each User Story

- Write tests first and ensure they fail before implementation.
- Implement core generation paths before analysis/visualization layers.
- Complete story-scoped validations before moving to lower-priority stories.

### Parallel Opportunities

- Setup: T002, T003, T004 can run in parallel.
- Phase 2 foundational tests: T007, T008, T009 can run in parallel.
- US1 tests: T018-T021 can run in parallel.
- US2 tests: T028-T031 can run in parallel.
- US3 tests: T037-T039 can run in parallel.
- US2 implementation: T033 and T034 can run in parallel after T032.
- US3 implementation: T041 and T042 can run in parallel after T040.
- Polish: T044, T045, T046, T047 can run in parallel; T049 after source changes settle.

---

## Parallel Example: User Story 1

- T018 tests/unit/test_model.py
- T019 tests/contract/test_train_cli.py (CLI args)
- T020 tests/integration/test_train_pipeline.py
- T021 tests/contract/test_train_cli.py (log assertions)

---

## Parallel Example: User Story 2

- T028 tests/contract/test_analyze_cli.py
- T029 tests/unit/test_analyze_curves.py
- T030 tests/integration/test_analyze_curves.py
- T031 tests/contract/test_analyze_cli.py (logging)

---

## Implementation Strategy

### MVP First (User Story 1)

1. Complete Phase 1 (Setup) and Phase 2 (Foundational Tests).
2. Complete Phase 3 (Foundational Implementation).
3. Complete US1 tests and implementation (Phase 4).
4. Validate artifacts, device/logging behavior, and fail-fast contract.
5. Demo MVP before adding additional stories.

### Incremental Delivery

1. Deliver US1: training/evaluation/logging baseline.
2. Deliver US2: learning curves + device quality/time comparisons.
3. Deliver US3: classification quality visualizations.
4. Run Polish (Phase 7): code quality, container, acceptance, commit-gate.

### Parallel Team Strategy

1. All members complete Setup together.
2. All members complete Foundational Tests (test-first gate).
3. One stream: runtime/training path (US1, Foundational Implementation).
4. Another stream: analysis/visualization path (US2/US3).
5. Cross-cutting stream: container, docs, acceptance, code quality.
6. Everyone syncs on T053 before any commit.
