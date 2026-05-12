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

- [X] T053 Request explicit user approval before any commit; present change summary and validation results (completed after Phase 6 validation in prior session)

---

## Phase 8: Extended User Story 1 — Multi-Batch Training (Priority: P1) 🔄 ✅ COMPLETE

**Goal**: Extend training pipeline to support multiple batch sizes in one command via `--batches` flag, maintain distinct run_ids per batch, and persist batch_size in shared CSV files.

**Independent Test**: Run multi-batch training command and confirm each batch size gets distinct run_id, batch_size appears in all metric rows, and sequential execution completes successfully.

### Tests for Multi-Batch Training (MANDATORY — write before implementation)

- [X] T054 [P] [US1] Contract test for `--batches` CLI flag parsing (comma-separated list format) in tests/contract/test_train_cli.py
- [X] T055 [P] [US1] Unit test for batch-size loop sequencing logic in tests/unit/test_train.py
- [X] T056 [P] [US1] Integration test for distinct run_id per batch in shared CSV files in tests/integration/test_train_pipeline.py
- [X] T057 [P] [US1] Contract test for batch_size column presence in metrics_train.csv, metrics_validation.csv, metrics_test.csv, run_summary.csv in tests/contract/test_train_cli.py
- [X] T058 [US1] Integration test for backward compatibility: single --batch flag still works when --batches is not used in tests/integration/test_train_pipeline.py

### Implementation for Multi-Batch Training

- [X] T059 [US1] Extend src/train.py CLI argument parser to accept --batches comma-separated list (e.g., `--batches 32,64,128`) in src/train.py
- [X] T060 [US1] Implement batch-size loop in src/train.py that parses --batches, iterates over each size sequentially (depends on T059)
- [X] T061 [P] [US1] Update CSV column schemas in src/metrics.py to include optional batch_size field in TRAIN_COLUMNS, EVAL_COLUMNS, SUMMARY_COLUMNS, PREDICTIONS_COLUMNS in src/metrics.py
- [X] T062 [US1] Modify training loop in src/train.py to write batch_size to all metric CSV rows during multi-batch execution (depends on T060, T061)
- [X] T063 [US1] Extend run_id generation in src/train.py to create distinct run_id for each batch-size execution in multi-batch mode (depends on T060)
- [X] T064 [US1] Update per-epoch logging in src/train.py to include batch context in log output for multi-batch runs (depends on T063)
- [X] T065 [US1] Validate and test CSV append behavior in multi-batch mode: all rows append to same file with distinct run_ids and batch_size values (depends on T062, T063)

**Checkpoint**: ✅ Multi-batch training fully functional with distinct run_ids, batch_size tracking, and backward-compatible single-batch mode. All 28 tests passing.

---

## Phase 9: Extended User Story 2 — Results Report Generation (Priority: P2) 🔄 ✅ COMPLETE

**Goal**: Extend analyze pipeline to generate results.md with final-metrics table (per batch size), epoch-comparison table (sampled every 10 epochs plus final epoch), and configuration metadata. Compare all historical matching rows in results directory.

**Independent Test**: Run analysis on multi-batch training outputs and verify results.md contains both required tables with correct structure, configuration section, and row counts matching sampled epochs.

### Tests for Results Report Generation (MANDATORY — write before implementation)

- [X] T066 [P] [US2] Contract test for results.md file generation and presence in results directory in tests/contract/test_analyze_cli.py
- [X] T067 [P] [US2] Unit test for final-metrics table construction (batch_size, train/validation/test loss+accuracy) in tests/unit/test_analyze.py
- [X] T068 [P] [US2] Unit test for epoch-sampling logic (every 10 epochs + always include final epoch) in tests/unit/test_analyze.py
- [X] T069 [P] [US2] Integration test for results.md generation from multi-batch CSV files in tests/integration/test_analyze_curves.py
- [X] T070 [US2] Integration test for historical-row comparison scope: verify results.md includes all matching rows from results-directory CSVs (depends on T069)

### Implementation for Results Report Generation

- [X] T071 [US2] Implement epoch-sampling function in src/analyze.py to extract rows at multiples of 10 and always include final epoch in src/analyze.py
- [X] T072 [P] [US2] Implement final-metrics table generation in src/analyze.py: query all unique batch_size values and construct table with train_loss, train_accuracy, validation_loss, validation_accuracy, test_loss, test_accuracy in src/analyze.py
- [X] T073 [P] [US2] Implement epoch-comparison table generation in src/analyze.py: sample epochs using T071 logic, group by batch_size, show per-split metrics in src/analyze.py
- [X] T074 [US2] Implement configuration metadata section in src/analyze.py: extract device, epochs, learning_rate, batch_size list from compared runs (depends on T072)
- [X] T075 [US2] Implement historical-row comparison scope in src/analyze.py: query all matching rows across selected results-directory CSVs (not just current-command rows) (depends on T072, T073)
- [X] T076 [US2] Integrate epoch-sampling, final-metrics, epoch-comparison, and config sections into complete results.md generation function in src/analyze.py (depends on T071, T072, T073, T074, T075)
- [X] T077 [US2] Add lifecycle logging for results report generation in src/analyze.py: log start, section completion, and final generation events to console and run log (depends on T076)
- [X] T078 [US2] Ensure results.md is written to correct results directory using CLI arguments in src/analyze.py (depends on T076)

**Checkpoint**: ✅ Results report generation fully functional with both tables, configuration metadata, historical-row comparison, and proper file placement.

---

## Phase 10: Extended Polish & Cross-Cutting Concerns for Multi-Batch Features

**Purpose**: Code quality gates, documentation validation, acceptance testing, and final commit approval for multi-batch and results.md features.

### Test Coverage & Validation

- [X] T079 [P] Run flake8 lint on modified files: src/train.py, src/analyze.py, src/metrics.py with max-line-length 120
- [X] T080 [P] Run mypy type checks on modified files: src/train.py, src/analyze.py, src/metrics.py with --ignore-missing-imports
- [X] T081 Run full pytest test suite including new multi-batch and results.md tests: `pytest tests/ -v` and confirm all pass
- [X] T082 Validate SC-001 (≥95% workflow success): Run multi-batch training 3 times with --batches 32,64,128 and confirm all succeed with proper artifacts
- [X] T083 Validate SC-002 (≥97% test accuracy): Confirm final test_accuracy from multi-batch runs meets >= 0.97 threshold
- [X] T084 Validate quickstart multi-batch examples: Execute commands from quickstart.md section "Run multi-batch experiments" and verify all outputs
- [X] T085 Validate results.md structure: Run analysis on multi-batch results, confirm results.md contains "Final Metrics by Batch Size" and "Epoch Comparison" sections with correct row counts

### Documentation & Acceptance

- [X] T086 Update results.md expected outputs in quickstart.md to document new files and table structure (if needed)
- [X] T087 Run full integration example from quickstart: multi-batch train → analyze → verify results.md structure and content

### Commit Gate for Extended Features (FR-012, CAR-005)

- [ ] T088 Request explicit user approval for multi-batch and results.md implementation; present validation results from T079–T087

---

## Phase 11: Filter-Batch Analysis Extension (FR-026/027/028)

**Goal**: Add optional `--filter-batch INT` / `-b INT` flag to `src/analyze.py` that scopes learning-curve plots and the epoch-comparison table to a single batch size, while keeping the Final Metrics table unfiltered. Fail fast with an actionable error when the specified batch size is absent.

**Independent Test**: Run `python -m src.analyze -r results/phase10-check/run-smoke --filter-batch 64` — curves and epoch-comparison must contain only batch-64 rows; final-metrics table must still show all three batch sizes (32, 64, 128).

### Tests for Filter-Batch Extension (MANDATORY — write first, must FAIL before implementation) ⚠️

- [X] T089 [P] [US4] Contract test: `--filter-batch` / `-b` flag parses as optional int with no default in tests/contract/test_analyze_cli.py
- [X] T090 [P] [US4] Contract test: `run_analysis` exits non-zero with error listing available batches when `--filter-batch` value has no matching rows, using a synthetic CSV fixture in tests/contract/test_analyze_cli.py
- [X] T091 [P] [US4] Unit test: `filter_rows_by_batch(rows, batch_size)` returns only rows whose `batch_size` field matches the given int in tests/unit/test_analyze.py
- [X] T092 [P] [US4] Unit test: `_build_epoch_comparison_table` applies `filter_batch` when set and returns unfiltered rows when `filter_batch` is `None`, verified with synthetic multi-batch CSV data in tests/unit/test_analyze.py
- [X] T093 [US4] Integration test: run full analysis on `results/phase10-check/run-smoke` with `filter_batch=64`; assert generated curves reference only batch-64 rows and epoch-comparison table omits rows for batches 32 and 128 in tests/integration/test_analyze_curves.py

### Implementation for Filter-Batch Extension (after all T089–T093 tests fail)

- [X] T094 [US4] Add `--filter-batch INT` / `-b INT` optional argument (default `None`) to `build_parser()` in src/analyze.py
- [X] T095 [US4] Implement `filter_rows_by_batch(rows: list[dict], batch_size: int) -> list[dict]` helper in src/analyze.py that returns rows where `int(row["batch_size"]) == batch_size`
- [X] T096 [US4] Add fail-fast validation in `run_analysis()` in src/analyze.py: if `filter_batch` is set and no train rows match, print error listing all unique `batch_size` values found in CSV files and call `sys.exit(1)`
- [X] T097 [US4] Apply `filter_rows_by_batch` to train/val/test row lists in `plot_learning_curves()` when `filter_batch` is not `None` in src/analyze.py
- [X] T098 [US4] Apply `filter_rows_by_batch` to sampled rows inside `_build_epoch_comparison_table()` when `filter_batch` is not `None` in src/analyze.py (Final Metrics table must NOT be filtered — FR-027)
- [X] T099 [US4] Thread `filter_batch` parameter from parsed args through `run_analysis()` into `plot_learning_curves()` and `generate_results_report()` calls in src/analyze.py

**Checkpoint**: `--filter-batch 64` on a 3-batch results directory produces curves and epoch-comparison for batch 64 only; final-metrics table still shows all batches; missing batch exits with error listing available sizes.

---

## Phase 12: Polish for Filter-Batch Feature

**Purpose**: Code quality gates, acceptance validation, and commit approval for FR-026/027/028.

### Code Quality Gates (FR-011, CAR-002)

- [X] T100 [P] Run flake8 on src/analyze.py with `--max-line-length 120` and confirm zero violations
- [X] T101 [P] Run mypy on src/analyze.py with `--ignore-missing-imports` and confirm zero type errors

### Test Suite & Acceptance Validation

- [X] T102 Run full pytest suite: `pytest tests/ -v` and confirm all T089–T093 tests pass alongside all prior tests
- [X] T103 Validate FR-026 backward compatibility: run `python -m src.analyze -r results/phase10-check/run-smoke` without `--filter-batch` and confirm output is identical to pre-filter behavior (all batch sizes in curves and tables)
- [X] T104 Validate FR-027: run `python -m src.analyze -r results/phase10-check/run-smoke --filter-batch 64` and confirm `results.md` Final Metrics table shows rows for all three batch sizes (32, 64, 128) while the Epoch Comparison table contains only batch-64 rows
- [X] T105 Validate FR-028: run `python -m src.analyze -r results/phase10-check/run-smoke --filter-batch 999` and confirm non-zero exit code and error message that lists available batch sizes (32, 64, 128)
- [X] T106 Validate quickstart example: execute `python -m src.analyze -r ./results/phase10-check/run-smoke --filter-batch 64` as documented in quickstart.md and verify plots and results.md are produced without errors

### Commit Gate (FR-012, CAR-005)

- [X] T107 Request explicit user approval for filter-batch implementation; present validation results from T100–T106 before any commit

---

## Extended Dependencies & Execution Order

### Phase Dependencies (Extended)

- Phases 1–7: ✅ Already complete (commit 1d771c6)
- Phase 8 (US1 Multi-Batch): depends on Phases 1–7 completion
  - Tests T054–T058: must pass before implementation T059–T065
  - T060 depends on T059; T062 depends on T060, T061; T063 depends on T060; T064 depends on T063
- Phase 9 (US2 Results Report): depends on Phase 8 completion (requires multi-batch training outputs)
  - Tests T066–T070: must pass before implementation T071–T078
  - T074 depends on T072; T075 depends on T072, T073; T076 depends on T071–T075; T077, T078 depend on T076
- Phase 10 (Polish Extended): depends on Phase 9 completion
  - T079, T080 can run in parallel [P]
  - T081–T087 sequential, each validates different aspect
  - T088 is final gate, requires explicit approval
- Phase 11 (Filter-Batch Extension): depends on Phase 10 (committed baseline)
  - Tests T089–T093: must be written and FAIL before implementation T094–T099
  - T089–T092 can run in parallel [P]; T093 depends on T089–T092
  - T094 must be complete before T095–T099
  - T095 required by T097, T098; T096 required by T099; T097–T099 can be parallelized after T094, T095
- Phase 12 (Filter-Batch Polish): depends on Phase 11 completion
  - T100, T101 can run in parallel [P]
  - T102–T106 sequential, each validates different acceptance criterion
  - T107 is final gate, requires explicit approval

### Multi-Batch Training Flow (Phase 8)

```
T054–T057: Parallel tests (different test files)
   ↓
T058: Integration test (depends on all above)
   ↓
T059: Parse --batches CLI
   ↓
T060: Implement batch loop (depends on T059)
   ↓
T061 (parallel): CSV column schema
T062: Write batch_size to rows (depends on T060, T061)
T063: Distinct run_id per batch (depends on T060)
   ↓
T064: Update logging (depends on T063)
T065: Validate append behavior (depends on T062, T063)
```

### Results Report Flow (Phase 9)

```
T066–T069: Parallel tests (different test files)
   ↓
T070: Integration test (depends on T069)
   ↓
T071: Epoch-sampling function
   ↓
T072, T073: Parallel table generation (both needed)
   ↓
T074: Config section (depends on T072)
T075: Historical-row query (depends on T072, T073)
   ↓
T076: Full integration (depends on T071–T075)
   ↓
T077, T078: Logging and file placement (depend on T076)
```

### Filter-Batch Extension Flow (Phase 11)

```
T089–T092: Parallel contract + unit tests
   ↓
T093: Integration test (depends on T089–T092 passing conceptually; must FAIL before impl)
   ↓
T094: Add --filter-batch arg to build_parser()
   ↓
T095: filter_rows_by_batch() helper (depends on T094)
   ↓
T096, T097, T098: Parallel — fail-fast check, plot filter, epoch-comparison filter (each depends on T095)
   ↓
T099: Thread filter_batch through run_analysis() (depends on T096–T098)
```

### Parallel Opportunities (Extended)

**Phase 8 Tests:**
- T054, T055, T056, T057 can run in parallel

**Phase 8 Implementation (after tests pass):**
- T061 [P] can start immediately (independent CSV schema)
- T060 blocking T062, T063 once complete
- T064, T065 after T063, T062 respectively

**Phase 9 Tests:**
- T066, T067, T068, T069 can run in parallel

**Phase 9 Implementation (after tests pass):**
- T072, T073 can run in parallel (both construct different tables)
- T074 can start once T072 completes
- T075 can start once T072, T073 complete
- T076 requires T071–T075
- T077, T078 can run after T076

**Phase 10 Polish:**
- T079, T080 [P] lint/type checks can run in parallel
- T081–T087 sequential, each validates previous layers

**Phase 11 Tests:**
- T089, T090, T091, T092 [P] can run in parallel

**Phase 11 Implementation (after tests pass):**
- T095, T096 can start once T094 completes
- T097, T098, T099 can start once T095 completes; T096 also blocks T099

**Phase 12 Polish:**
- T100, T101 [P] lint/type checks can run in parallel
- T102–T106 sequential, each validates a distinct acceptance criterion

---

## Extended Implementation Strategy

### Multi-Batch MVP (Phases 1–8)

1. ✅ Phases 1–7: Complete from prior session (commit 1d771c6)
2. **Phase 8 (US1 Multi-Batch)**:
   - T054–T058: Write and validate all tests FAIL
   - T059–T065: Implement multi-batch training
   - **VALIDATE**: Run `python -m src.train -d cpu --batches 32,64,128` and verify distinct run_ids and batch_size in CSVs
3. **Full Multi-Batch MVP Ready**

### With Results Report (Phases 1–9)

1. ✅ Phases 1–7: Complete
2. ✅ Phase 8: Multi-batch training complete
3. **Phase 9 (US2 Results Report)**:
   - T066–T070: Write and validate all tests FAIL
   - T071–T078: Implement results.md generation
   - **VALIDATE**: Run `python -m src.analyze -r ./results` and verify both tables in results.md
4. **Full Feature Complete (MVP + Multi-Batch + Reporting)**

### Final Validation (Phase 10)

1. Code quality gates (T079–T080)
2. Test suite and acceptance validation (T081–T087)
3. User approval and commit (T088)

### Filter-Batch Validation (Phase 12)

1. Code quality gates (T100–T101)
2. Test suite and acceptance validation (T102–T106)
3. User approval and commit (T107)

---

## Task Summary

| Phase | Tasks | Focus | Status |
|-------|-------|-------|--------|
| 1: Setup | T001–T006 | Project structure | ✅ Complete |
| 2: Foundational Tests | T007–T011 | Test-first gate | ✅ Complete |
| 3: Foundational Impl | T012–T017 | Shared runtime | ✅ Complete |
| 4: US1 (Training) | T018–T027 | MNIST training | ✅ Complete |
| 5: US2 (Curves) | T028–T036 | Learning visualization | ✅ Complete |
| 6: US3 (Classification) | T037–T043 | Quality metrics | ✅ Complete |
| 7: Polish | T044–T053 | Code quality, container, acceptance | ✅ Complete |
| 8: US1 Extended | T054–T065 | **Multi-batch training** | ✅ Complete |
| 9: US2 Extended | T066–T078 | **Results report generation** | ✅ Complete |
| 10: Final Polish | T079–T088 | Code quality for new features, approval | ✅ Complete |
| 11: Filter-Batch Extension | T089–T099 | **--filter-batch flag (FR-026/027/028)** | ⏳ Pending |
| 12: Filter-Batch Polish | T100–T107 | Code quality, acceptance, approval | ⏳ Pending |
| **Total** | **107 Tasks** | MNIST with multi-batch + reporting + filter-batch | |

---

## Notes

- **[X] tasks**: Already complete from prior session (commit 1d771c6)
- **[ ] tasks**: Pending implementation for multi-batch (T054–T065) and results.md (T066–T078) features
- **[P] marker**: Tasks can run in parallel (different files, no dependencies)
- **[Story] label**: Maps task to user story (US1, US2, US3) for traceability
- **Test-First**: All T054–T058, T066–T070 tests must be written and FAIL before implementation begins
- **Backward Compatibility**: Single --batch flag mode remains functional when --batches is not specified
- **Filter-Batch (FR-026/027/028)**: T089–T093 tests must FAIL before T094–T099 implementation; tests use existing `results/phase10-check/run-smoke` fixture data — no training re-runs
- **MVP Increments**:
  - Commit 1d771c6: US1–US3 complete (T001–T053) ✅
  - Commit b69fdb5: Phase 8 (T054–T065) adds multi-batch capability ✅
  - Commit f16df24: Phases 9–10 (T066–T088) adds results.md + quality gates ✅
  - Next: Phase 11 (T089–T099) adds filter-batch to analyze.py
  - Then: Phase 12 (T100–T107) validates and gates commit

---

## Dependencies & Execution Order

## Dependencies & Execution Order

### Phase Dependencies (Original)

- Phase 1 (Setup): no dependencies.
- Phase 2 (Foundational Tests): depends on Phase 1; BLOCKS Phase 3 implementation.
- Phase 3 (Foundational Implementation): depends on Phase 2 tests failing; BLOCKS all stories.
- Phase 4 (US1): depends on Phase 3.
- Phase 5 (US2): depends on Phase 3 and run metrics contracts.
- Phase 6 (US3): depends on Phase 3 and predictions contract.
- Phase 7 (Polish): depends on completion of selected stories.

### Phase Dependencies (Extended — Multi-Batch & Results)

- Phases 1–7: ✅ Already complete (commit 1d771c6)
- Phase 8 (US1 Multi-Batch): depends on Phases 1–7 completion
  - Tests T054–T058: must pass before implementation T059–T065
  - T060 depends on T059; T062 depends on T060, T061; T063 depends on T060; T064 depends on T063
- Phase 9 (US2 Results Report): depends on Phase 8 completion (requires multi-batch training outputs)
  - Tests T066–T070: must pass before implementation T071–T078
  - T074 depends on T072; T075 depends on T072, T073; T076 depends on T071–T075; T077, T078 depend on T076
- Phase 10 (Polish Extended): depends on Phase 9 completion
  - T079, T080 can run in parallel [P]
  - T081–T087 sequential
  - T088 is final gate requiring explicit approval
- T053 (Prior Commit Gate) → T088 (Extended Commit Gate): Serial; T088 follows completion of Phase 9 testing

### User Story Dependencies

- US1 (P1): independent after foundational work; defines MVP. Extended in Phase 8 with multi-batch.
- US2 (P2): independent with fixture data; full value when US1 outputs are present. Extended in Phase 9 with results.md.
- US3 (P3): independent with fixture predictions; full value when US1 outputs are present. No Phase 9 extension needed.

### Within Each User Story

- Write tests first and ensure they fail before implementation.
- Implement core generation paths before analysis/visualization layers.
- Complete story-scoped validations before moving to lower-priority stories.

### Parallel Opportunities

- Setup (Phase 1): T002, T003, T004 in parallel.
- Phase 2: T007, T008, T009 in parallel.
- Phase 4 (US1 tests): T018–T021 in parallel.
- Phase 5 (US2 tests): T028–T031 in parallel.
- Phase 6 (US3 tests): T037–T039 in parallel.
- **Phase 8 (US1 Multi-Batch tests)**: T054–T057 in parallel; T058 after all tests.
- **Phase 8 (US1 Multi-Batch impl)**: T061 [P] independent; T060 → T062, T063; T064, T065 after their dependencies.
- **Phase 9 (US2 Results tests)**: T066–T069 in parallel; T070 after.
- **Phase 9 (US2 Results impl)**: T072, T073 in parallel; T074, T075 after; T076 last; T077, T078 after T076.
- Phase 10 Polish: T079, T080 [P] lint/type in parallel; T081–T087 sequential; T088 final gate.

---

## Parallel Example: User Story 1 – Multi-Batch Training (Phase 8)

```
Test Phase (parallel start):
  - T054: tests/contract/test_train_cli.py (--batches flag parsing)
  - T055: tests/unit/test_train.py (batch loop logic)
  - T056: tests/integration/test_train_pipeline.py (distinct run_ids in CSV)
  - T057: tests/contract/test_train_cli.py (batch_size column presence)

Integration Test (after T054–T057):
  - T058: tests/integration/test_train_pipeline.py (backward compatibility)

Implementation Phase (sequential):
  - T059: Parse --batches CLI
  - T060: Implement batch loop (after T059)
  - T061 (parallel): CSV column schema
  - T062: Write batch_size (after T060, T061)
  - T063: Distinct run_id per batch (after T060)
  - T064: Update logging (after T063)
  - T065: Validate append (after T062, T063)
```

---

## Parallel Example: User Story 2 – Results Report (Phase 9)

```
Test Phase (parallel start):
  - T066: tests/contract/test_analyze_cli.py (results.md generation)
  - T067: tests/unit/test_analyze.py (final-metrics table)
  - T068: tests/unit/test_analyze.py (epoch-sampling logic)
  - T069: tests/integration/test_analyze_curves.py (full generation from CSVs)

Integration Test (after T066–T069):
  - T070: tests/integration/test_analyze_curves.py (historical-row comparison)

Implementation Phase (sequential):
  - T071: Epoch-sampling function
  - T072, T073 (parallel): Final-metrics and epoch-comparison tables
  - T074: Config section (after T072)
  - T075: Historical-row query (after T072, T073)
  - T076: Full integration (after T071–T075)
  - T077, T078: Logging and file placement (after T076)
```

---

## Implementation Strategy (Extended)

### MVP First (User Stories 1–3)

1. ✅ Phase 1–7: Complete (commit 1d771c6)
2. Demo primary training, visualization, and classification workflows
3. Commit if needed

### Incremental Delivery with Multi-Batch (Phases 1–8)

1. ✅ Phases 1–7: Complete
2. **Phase 8 (US1 Extended - Multi-Batch Training)**:
   - T054–T058: Write tests, confirm FAIL
   - T059–T065: Implement and validate multi-batch
   - **DEMO**: `python -m src.train -d cpu --batches 32,64,128`; verify distinct run_ids and batch_size in CSVs
3. **Commit Phase 8** if desired (multi-batch training ready)

### Full Feature with Results Report (Phases 1–9)

1. ✅ Phases 1–7: Complete
2. ✅ Phase 8: Multi-batch training complete
3. **Phase 9 (US2 Extended - Results Report)**:
   - T066–T070: Write tests, confirm FAIL
   - T071–T078: Implement and validate results.md generation
   - **DEMO**: `python -m src.analyze -r ./results`; verify results.md with both tables and config
4. **Commit Phase 9** if desired (full feature complete)

### Final Validation & Commit (Phase 10)

1. Code quality gates: T079–T080
2. Test suite and acceptance: T081–T087
3. **Explicit user approval (T088)** before final commit
4. Commit when approved

### Team Execution (Multiple Developers)

If team available:
1. **Developer 1**: Phase 8 (US1 Multi-Batch Training)
   - Parallel tests T054–T057
   - Sequence implementation T059–T065
   - Validate outputs
2. **Developer 2**: Phase 9 (US2 Results Report)
   - Wait for Phase 8 completion
   - Parallel tests T066–T069
   - Sequence implementation T071–T078
   - Validate outputs
3. **Team Lead**: Phase 10 (Polish & Commit Gate)
   - Run quality gates T079–T080
   - Validate acceptance T081–T087
   - Manage user approval T088
   - Prepare and execute final commit


