---

description: "Task list for MNIST Digit Classifier Pipeline implementation"
---

# Tasks: MNIST Digit Classifier Pipeline

**Input**: Design documents from `/specs/001-build-mnist-classifier/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/cli.md ✅, contracts/docker-cli.md ✅

**Tests**: Test tasks are MANDATORY. Every user story includes tests written before implementation tasks.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no shared dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Exact file paths are included in each description

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create project skeleton, install dependencies, and configure tooling

- [X] T001 Create project directory structure: `src/`, `tests/unit/`, `tests/integration/`, `tests/contract/`, `data/`, `results/`
- [X] T002 Create `requirements.txt` with `torch` and `torchvision` from PyTorch nightly xpu index (`--index-url https://download.pytorch.org/whl/nightly/xpu`) plus `matplotlib` and `pytest`
- [X] T003 [P] Create `src/__init__.py` as empty package marker
- [X] T004 [P] Create `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`, `tests/contract/__init__.py` as empty package markers

**Checkpoint**: Project structure in place and dependencies defined — ready to build foundational modules

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core shared infrastructure that every user story depends on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Implement device validation in `src/device.py`: accept `"cpu"` or `"xpu"`, map to `torch.device`, raise `RuntimeError` with actionable message if selected device is unavailable; no fallback logic
- [X] T006 [P] Implement MNIST data loading in `src/data.py`: load train/val/test splits via `torchvision.datasets.MNIST`, return `DataLoader` objects for each split; accept `data_dir`, `batch_size`, `device` as arguments
- [X] T007 [P] Implement CSV metric writer in `src/metrics.py`: functions to append a row to `metrics_train.csv`, `metrics_validation.csv`, `metrics_test.csv`, and `run_summary.csv` using only the `csv` stdlib module; column schema matches `EpochMetrics` and `EvaluationSnapshot` entities in `data-model.md`

**Checkpoint**: Foundational modules ready — user story implementation can now begin

---

## Phase 3: User Story 1 — Train a Reliable MNIST Classifier (Priority: P1) 🎯 MVP

**Goal**: Train a `784 → 256 → 128 → 10` MLP on MNIST, log per-epoch metrics to CSV, run validation and test checks every 5 epochs, save model checkpoint, and respect explicit device selection.

**Independent Test**: Run one training cycle (`python src/train.py -e 1 -d cpu -r results/smoke`); confirm model checkpoint, `metrics_train.csv`, `metrics_validation.csv`, `metrics_test.csv`, and `run_summary.csv` are produced with correct column headers including timing fields; confirm training on an unavailable XPU fails with a clear error.

### Tests for User Story 1 (MANDATORY) ⚠️

> **Write these tests FIRST — they must FAIL before implementation begins**

- [X] T008 [P] [US1] Write unit tests for `MnistClassifier` forward pass shape, dtype, and no-loop batch guarantee in `tests/unit/test_model.py`
- [X] T009 [P] [US1] Write unit tests for `device.py` — valid device accepted, unavailable device raises `RuntimeError`, no fallback occurs — in `tests/unit/test_device.py`
- [X] T010 [P] [US1] Write unit tests for `metrics.py` CSV writer — correct columns written, rows appended, file created on first call — in `tests/unit/test_metrics.py`
- [X] T011 [P] [US1] Write contract tests for `train.py` CLI arguments (`-e`, `-r`, `-d`, `-b`, `-lr`, `-m`) and required `--device` enforcement in `tests/contract/test_train_cli.py`
- [X] T012 [P] [US1] Write integration test for one full training run: verify all four CSV output files exist with expected columns and at least one data row in `tests/integration/test_train.py`

### Implementation for User Story 1

- [X] T013 [US1] Implement `MnistClassifier` in `src/model.py`: `nn.Sequential` with layers `784 → 256 (ReLU) → 128 (ReLU) → 10`; typed `forward(x: torch.Tensor) -> torch.Tensor`; no raw Python loops over batch elements
- [X] T014 [US1] Implement `train.py` argument parser with `-e/--epochs` (default 10), `-r/--results` (default `./results`), `-d/--device` (required, `cpu` or `xpu`), `-b/--batch` (default 64), `-lr/--lr` (default 0.001), `-m/--data` (default `./data`) in `src/train.py`
- [X] T015 [US1] Implement training loop in `src/train.py`: load MNIST via `data.py`, validate device via `device.py`, train with CrossEntropyLoss and Adam, log per-epoch `loss`, `accuracy`, `elapsed_seconds`, `device` to `metrics_train.csv` via `metrics.py`
- [X] T016 [US1] Add validation and test evaluation every 5 epochs in `src/train.py`: write rows to `metrics_validation.csv` and `metrics_test.csv` with matching schema
- [X] T017 [US1] Add `run_summary.csv` write with `training_time_seconds` and model checkpoint save (`model.pt`) at end of training in `src/train.py`
- [X] T018 [US1] Add fail-fast error handling for missing data directory and interrupted-run logging in `src/train.py`

**Checkpoint**: User Story 1 fully functional and independently testable — MVP deliverable

---

## Phase 4: User Story 2 — Visualize Learning Curves (Priority: P2)

**Goal**: Read saved CSV files from a results directory and produce train/validation/test learning curve plots, CPU vs XPU quality comparison, and CPU vs XPU timing comparison when both device runs are present.

**Independent Test**: Point `analyze.py` at a completed training run results directory; confirm learning curves plus CPU/XPU quality and timing comparison outputs are produced with clear labels; confirm the script exits with a clear error if the results directory is missing or empty.

### Tests for User Story 2 (MANDATORY) ⚠️

> **Write these tests FIRST — they must FAIL before implementation begins**

- [X] T019 [P] [US2] Write contract tests for `analyze.py` CLI: `-r/--results` required, missing directory raises clear error — in `tests/contract/test_analyze_cli.py`
- [X] T020 [P] [US2] Write unit tests for curve-plotting functions: given synthetic CSV data, assert plot image file is written to the expected path — in `tests/unit/test_analyze_curves.py`
- [X] T021 [P] [US2] Write integration test for learning curves: use a fixture results directory with pre-made CSV files, assert all three curve image files are produced — in `tests/integration/test_curves.py`

### Implementation for User Story 2

- [X] T022 [US2] Implement `analyze.py` CLI entry point with `-r/--results` (required) argument and results-dir validation in `src/analyze.py`
- [X] T023 [P] [US2] Implement `plot_learning_curves()` in `src/analyze.py`: read `metrics_train.csv`, `metrics_validation.csv`, `metrics_test.csv`; plot loss and accuracy vs epoch for each split; save to `<results_dir>/learning_curves_loss.png` and `learning_curves_accuracy.png`
- [X] T024 [P] [US2] Implement `plot_device_comparison()` in `src/analyze.py`: when two subdirectories for `cpu` and `xpu` runs exist under `--results`, overlay their training loss/accuracy curves and save to `<results_dir>/device_comparison.png`
- [X] T025 [US2] Wire `plot_learning_curves()`, `plot_device_comparison()`, and `plot_time_comparison()` into `analyze.py` main execution flow, with clear dependency error if required CSV files are absent

**Checkpoint**: User Stories 1 and 2 independently functional

---

## Phase 5: User Story 3 — Analyze Classification Quality (Priority: P3)

**Goal**: Generate a confusion matrix image and a per-class precision/recall/F1 bar chart from test predictions stored after a completed training run.

**Independent Test**: Point `analyze.py` at a completed results directory that contains a `predictions.csv` (or equivalent); confirm `confusion_matrix.png` and `classification_report.png` are produced with all 10 MNIST digit classes labeled; confirm the script exits with a clear error if predictions data is missing.

### Tests for User Story 3 (MANDATORY) ⚠️

> **Write these tests FIRST — they must FAIL before implementation begins**

- [X] T026 [P] [US3] Write unit tests for confusion matrix plot function: given a synthetic 10×10 matrix, assert output file written with correct class labels — in `tests/unit/test_analyze_classification.py`
- [X] T027 [P] [US3] Write unit tests for classification report bar chart function: given synthetic per-class precision/recall/F1 values, assert plot file written — in `tests/unit/test_analyze_classification.py`
- [X] T028 [P] [US3] Write integration test for classification analysis: use a fixture results directory with a `predictions.csv`, assert `confusion_matrix.png` and `classification_report.png` are produced — in `tests/integration/test_classification.py`

### Implementation for User Story 3

- [X] T029 [US3] Add per-sample prediction logging to `src/train.py` test evaluation pass: write true label, predicted label columns to `predictions.csv` in `<results_dir>` for the final test epoch
- [X] T030 [P] [US3] Implement `plot_confusion_matrix()` in `src/analyze.py`: read `predictions.csv`, build 10×10 matrix, render as annotated heatmap, save to `<results_dir>/confusion_matrix.png`
- [X] T031 [P] [US3] Implement `plot_classification_report()` in `src/analyze.py`: compute per-class precision, recall, F1 from `predictions.csv`, render grouped bar chart for digits 0–9, save to `<results_dir>/classification_report.png`
- [X] T032 [US3] Add missing-predictions dependency error to `analyze.py` with actionable remediation message in `src/analyze.py`

**Checkpoint**: All three user stories independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Hardening, type safety, and final validation

- [X] T033 [P] Add type hints to all public functions in `src/model.py`, `src/train.py`, `src/data.py`, `src/device.py`, `src/metrics.py`, `src/analyze.py`
- [X] T034 [P] Add unit tests for `data.py` split sizes and `DataLoader` return types in `tests/unit/test_data.py`
- [ ] T035 Validate all quickstart.md scenarios end-to-end: smoke train run, learning curve generation, classification report generation, and CPU/XPU timing comparison output
- [ ] T036 [P] Confirm `run_summary.csv` includes final test accuracy ≥ 0.98 and `training_time_seconds` for a standard training run
- [ ] T037 Request explicit user approval before any commit of implementation changes

---

## Phase 7: User Story 4 — Containerize the Project (Priority: P4)

**Goal**: Package the entire project into a minimal Docker image using `python:3.11-slim`, CPU-only PyTorch wheel, and host-mounted volumes for data and results. The container must run training, analysis, and tests without any XPU/GPU driver dependencies.

**Independent Test**: `docker build -t mnist-classifier .` succeeds; `docker run --rm mnist-classifier python -m src.train --help` exits 0; `docker run --rm mnist-classifier pytest tests/unit/ -v` exits 0; `docker run --rm mnist-classifier python -m src.train --device xpu` exits non-zero with a clear device error.

### Tests for User Story 4 (MANDATORY) ⚠️

> **Write these tests FIRST — they must FAIL before the files are created**

  - [X] T038 [P] [US4] Write contract tests for Docker support files in `tests/contract/test_docker_files.py`: assert `Dockerfile` exists and contains `FROM python:3.11-slim`, `ENTRYPOINT ["python", "-m", "src.train"]`, `VOLUME ["/app/data"]`, `VOLUME ["/app/results"]`; assert `requirements-docker.txt` exists and does NOT contain `torch` with XPU index URL; assert `.dockerignore` exists and includes `data/` and `.git/`

### Implementation for User Story 4

  - [X] T039 [P] [US4] Create `requirements-docker.txt` at project root containing only `matplotlib` and `pytest` — no `torch` or `torchvision` entries (those are installed inline in Dockerfile from the CPU wheel index)
  - [X] T040 [P] [US4] Create `.dockerignore` at project root excluding: `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `data/`, `results/`, `*.pt`, `.DS_Store`, `Thumbs.db`, `*.tmp`, `*.swp`, `.git/`
  - [X] T041 [US4] Create `Dockerfile` at project root: base `python:3.11-slim`; install `torch torchvision` from `https://download.pytorch.org/whl/cpu` in a dedicated `RUN` layer; copy and install `requirements-docker.txt`; `COPY src/ src/` and `COPY tests/ tests/`; declare `VOLUME ["/app/data"]` and `VOLUME ["/app/results"]`; set `ENV PYTHONUNBUFFERED=1`; set `ENTRYPOINT ["python", "-m", "src.train"]`; set `CMD ["--device", "cpu", "--data", "/app/data", "--results", "/app/results"]`
  - [X] T042 [US4] Update `specs/001-build-mnist-classifier/quickstart.md` to add a Docker section: `docker build -t mnist-classifier .`, volume-mounted train invocation, analysis override invocation, and `pytest tests/` inside container

**Checkpoint**: Full project runnable inside Docker container — US4 independently deliverable

---

## Phase 8: Final Commit Approval

- [ ] T043 Request explicit user approval before committing Docker containerization changes (Dockerfile, .dockerignore, requirements-docker.txt, quickstart.md update, contract tests)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all user stories
- **Phase 3 (US1 — Train)**: Depends on Phase 2; no other story dependencies
- **Phase 4 (US2 — Curves)**: Depends on Phase 2; reads CSV artifacts produced by US1 `train.py`, but visualization logic is independently testable using fixture CSV data
- **Phase 5 (US3 — Classification)**: Depends on Phase 2 and T029 (predictions CSV added to `train.py`); independently testable with fixture data
- **Phase 6 (Polish)**: Depends on all user story phases complete
- **Phase 7 (US4 — Docker)**: Depends on Phase 1 (project structure); independent of Phases 3–5 (wraps existing source); can begin after Phase 2 is complete
- **Phase 8 (Final Commit)**: Depends on Phase 7 complete and T038 contract test passing

### User Story Dependencies

- **US1 (P1)**: Can start after Phase 2 — no dependencies on other stories
- **US2 (P2)**: Can start after Phase 2 — integrates with US1 CSV output but independently testable with fixtures
- **US3 (P3)**: Requires T029 (predictions CSV export in `train.py`) before integration testing; analysis logic independently testable with fixtures
- **US4 (P4)**: Depends only on `src/` and `tests/` existing (Phase 1 complete); no dependency on US1–US3 logic; fully parallelisable with Phases 3–6

### Within Each User Story

1. Tests written first (must FAIL before implementation)
2. Foundational utilities (`device.py`, `data.py`, `metrics.py`) before consuming modules
3. Model definition before training loop
4. Training loop before analysis scripts
5. Core implementation before integration testing

### Parallel Opportunities

- T003, T004 (Phase 1): fully parallel
- T006, T007 (Phase 2): parallel after T005 (device.py has no dependency on them)
- T008–T012 (US1 tests): all parallel — different test files
- T013–T018 (US1 implementation): T013 and T014 parallel; T015 depends on T013+T014; T016–T018 depend on T015
- T019–T021 (US2 tests): fully parallel
- T023, T024 (US2 plots): parallel — different plot functions in same file
- T026–T028 (US3 tests): fully parallel
- T030, T031 (US3 plots): parallel — different functions
- **T038 (US4 contract test)**: parallel with T039, T040 — different files
- **T039, T040 (US4 support files)**: parallel — independent files; T041 (Dockerfile) depends on both

---

## Parallel Example: User Story 1

```
Phase 2 (must complete first)
  ├── T005 (device.py)
  ├── T006 (data.py)        ← parallel
  └── T007 (metrics.py)     ← parallel

Phase 3 — US1 Tests (all parallel once Phase 2 done)
  ├── T008 (test_model.py)
  ├── T009 (test_device.py)
  ├── T010 (test_metrics.py)
  ├── T011 (test_train_cli.py)
  └── T012 (test_train.py)

Phase 3 — US1 Implementation (sequential after tests)
  T013 (model.py) ──┐
  T014 (train.py arg parser) ──┼──→ T015 (training loop) → T016 → T017 → T018
```

## Parallel Example: User Story 4 (Docker)

```
Phase 7 — US4 Tests (write first)
  └── T038 (test_docker_files.py)   ← must FAIL before T039–T041

Phase 7 — US4 Implementation
  T039 (requirements-docker.txt) ──┐
  T040 (.dockerignore)            ──┼──→ T041 (Dockerfile) → T042 (quickstart.md)

Phase 8 — Commit Approval
  └── T043 (user approval gate)
```

---

## Implementation Strategy

**MVP Scope**: Phase 1 + Phase 2 + Phase 3 (User Story 1) — delivers a working, accurate, device-explicit training pipeline with CSV metric output.

**Incremental Delivery**:
1. MVP: Working training run with CSV metrics and model checkpoint (US1)
2. Add learning curve plots (US2) — no model changes required
3. Add classification quality analysis (US3) — add predictions CSV export to train.py, then analysis plots

**Key Invariants** (must hold throughout):
- Device selection is always explicit; fail-fast on unavailability with no fallback
- No raw Python loops over batch elements in the model forward path
- All CSV column names consistent with `data-model.md` entity field names
- Timing fields (`elapsed_seconds`, `training_time_seconds`) are persisted and used in analysis
- All public functions carry type hints
- User approval required before any commit
