# Feature Specification: MNIST Digit Classifier Pipeline

**Feature Branch**: `001-build-mnist-classifier`  
**Created**: 2026-05-11  
**Status**: Draft  
**Input**: User description: "Create a neural network that is capable of classifying the handwritten digits from the MNIST dataset accurately. The project should develop the model, train on CPUs or GPUs, test the model, generate the train, validation, testing curves, and visualize the classification metrics for the dataset."

## Clarifications

### Session 2026-05-11

- Q: Should the system automatically switch to CPU if GPU is unavailable? → A: No. Device must be explicitly instructed by the user for training and testing; no automatic fallback is allowed.

### Session 2026-05-12

- Q: Should the container support XPU device passthrough, or run CPU-only inside Docker? → A: CPU-only inside Docker; no XPU/GPU driver dependencies in the image.
- Q: How should MNIST data be provisioned inside the container? → A: Host volume mount (`-v ./data:/app/data`); fall back to downloading via torchvision only if the data directory is absent.
- Q: What is the acceptable compressed image size ceiling? → A: No upper limit; use a small base image (e.g., `python:3.11-slim`).
- Q: What should be the primary logging destination for progress tracking? → A: Log to both console output and a per-run log file in the results directory.
- Q: What progress granularity should logs use during training? → A: Use concise per-epoch progress logs plus key lifecycle events.
- Q: Should logs use plain text lines or structured JSON lines? → A: Use plain text log lines.
- Q: What minimum fields should each per-epoch log line include? → A: epoch, elapsed_seconds, loss, accuracy.
- Q: What log file naming rule should be used for per-run logs? → A: Use `run_<run_id>.log` in the results directory.
- Q: What minimum accuracy threshold should the trained model achieve on the MNIST test set? → A: 0.97 (97%).
- Q: How should the training CLI accept multiple batch sizes? → A: Add a list flag using comma-separated values (for example `--batches 32,64,128`).
- Q: What comparison format should `results.md` require for batch-size analysis? → A: Include one final-metrics table per batch size (train/validation/test loss+accuracy) and one epoch-comparison table sampled every 10 epochs.
- Q: How should epoch sampling work when epochs are fewer than 10 or not divisible by 10? → A: Include multiples of 10 and always include the final epoch.
- Q: How should run identity be recorded for sequential multi-batch training in shared CSV files? → A: Use a distinct `run_id` for each batch-size execution and include `batch_size` in every row.
- Q: What run scope should `results.md` compare in the results directory? → A: Compare all matching historical rows in the results-directory CSV files.

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Train a Reliable MNIST Classifier (Priority: P1)

As an ML practitioner, I want to train a neural network on MNIST so that it learns to classify handwritten digits accurately.

**Why this priority**: Without a working and accurate model training workflow, no downstream testing or analysis delivers value.

**Independent Test**: Can be fully tested by running one training cycle and confirming model artifacts plus final validation and test performance are produced.

**Acceptance Scenarios**:

1. **Given** MNIST data is available, **When** the user starts training, **Then** the system trains a neural network and stores model outputs for later evaluation.
2. **Given** both CPU and XPU are possible execution targets, **When** the user explicitly selects a target device for training or testing, **Then** the system runs only on that selected device.
3. **Given** the user explicitly selects XPU and XPU is unavailable, **When** execution starts, **Then** the system fails with a clear device-availability error and does not switch to CPU automatically.
4. **Given** a completed training run, **When** evaluation is executed, **Then** the system reports final accuracy and loss for validation and test splits.

---

### User Story 2 - Visualize Learning Curves (Priority: P2)

As a practitioner, I want training, validation, and testing curves so that I can inspect learning behavior and detect issues like overfitting.

**Why this priority**: Curves make model behavior interpretable and directly support model tuning.

**Independent Test**: Can be tested by generating run visualizations from a completed training run and verifying all required curves are present and readable.

**Acceptance Scenarios**:

1. **Given** a completed run with recorded metrics, **When** visualization is generated, **Then** train, validation, and test performance curves are created and viewable.
2. **Given** multiple epochs of logged metrics, **When** the curves are rendered, **Then** each epoch is represented in chronological order with clear labels.

---

### User Story 3 - Analyze Classification Quality (Priority: P3)

As a practitioner, I want classification metrics visualizations so that I can understand class-level strengths and weaknesses.

**Why this priority**: Detailed quality analysis improves confidence and enables targeted model refinement.

**Independent Test**: Can be tested by generating a classification report visualization from test predictions and confirming class-level metrics are shown.

**Acceptance Scenarios**:

1. **Given** test predictions and true labels, **When** metric analysis is run, **Then** class-level precision, recall, and F1-score are produced.
2. **Given** class-level results are available, **When** visualization is generated, **Then** confusion matrix and summary metric views are presented.

---

### Edge Cases

- What happens when XPU is unavailable or inaccessible at runtime? If XPU was explicitly selected, execution must fail with actionable guidance and must not fall back to CPU.
- What happens when training is interrupted before completion? Partial run logs should remain available and the interruption should be clearly reported.
- How does the system handle corrupted or missing MNIST data files? The run must fail fast with actionable remediation guidance.
- What happens when one or more classes are underperforming significantly? Class-level metrics should make this visible without manual post-processing.
- What happens when visualization generation is requested before evaluation data exists? The system should return a clear dependency error.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST train a neural network model for MNIST handwritten digit classification.
- **FR-002**: System MUST require explicit user instruction of device target (CPU or XPU) for both training and testing execution.
- **FR-003**: System MUST split dataset usage into training, validation, and testing phases and preserve their distinct results.
- **FR-004**: System MUST evaluate trained model performance on validation and testing data after training completes.
- **FR-005**: System MUST record per-epoch training and validation metrics needed to generate learning curves.
- **FR-006**: System MUST produce train, validation, and test curve visualizations for the run.
- **FR-007**: System MUST generate classification quality outputs including class-level precision, recall, F1-score, and confusion matrix view.
- **FR-008**: System MUST persist run metadata and final outcomes so results are reproducible and reviewable.
- **FR-009**: System MUST provide clear failure feedback for missing data, interrupted runs, and invalid evaluation prerequisites.
- **FR-010**: System MUST NOT automatically switch devices; when the selected device is unavailable, it MUST fail fast with a clear error and remediation guidance.
- **FR-011**: System MUST define acceptance tests for core training workflow, evaluation workflow, and visualization workflow before implementation.
- **FR-012**: System MUST enforce explicit user approval before any commit operation related to this feature work.
- **FR-013**: System MUST provide a Dockerfile that packages all runtime dependencies and enables the full training, evaluation, and analysis workflow to run inside a container on CPU only; no XPU/GPU driver dependencies are included in the image. The base image MUST be a slim variant (e.g., `python:3.11-slim`); there is no compressed-size ceiling.
- **FR-014**: The container MUST accept a host-mounted volume at `/app/data` for MNIST data; if no volume is mounted and the data directory is absent, the training script MUST download MNIST via torchvision automatically.
- **FR-015**: System MUST emit progress logs to both console and a per-run plain-text log file stored in the selected results directory for training and analysis workflows, using concise per-epoch messages plus key lifecycle events.
- **FR-016**: Each per-epoch training progress log line MUST include at least `epoch`, `elapsed_seconds`, `loss`, and `accuracy` fields.
- **FR-017**: Per-run plain-text log files MUST use the naming format `run_<run_id>.log` inside the selected results directory.
- **FR-018**: Training CLI MUST accept multiple batch sizes through a comma-separated list flag `--batches` (for example `--batches 32,64,128`).
- **FR-019**: Training workflow MUST execute all configured epochs for each batch size from `--batches` sequentially and persist outputs in the same CSV files.
- **FR-020**: Persisted metrics CSV rows for train, validation, and test MUST include a `batch_size` column to support cross-batch comparisons in one shared file set.
- **FR-021**: Analysis workflow MUST generate `results.md` in the selected results directory with two comparison tables: (a) final train/validation/test loss+accuracy per batch size and (b) epoch comparison sampled every 10 epochs.
- **FR-022**: `results.md` MUST include the run configuration used to produce compared results, including at least device, epochs, learning rate, and the batch-size list.
- **FR-023**: The epoch-comparison table in `results.md` MUST include epoch rows at multiples of 10 and MUST always include the final epoch even when it is not a multiple of 10.
- **FR-024**: For multi-batch training commands, each batch-size execution MUST use a distinct `run_id` and every persisted metric row MUST include `batch_size` for unambiguous comparison in shared CSV files.
- **FR-025**: `results.md` MUST compare all matching historical rows present in the selected results-directory CSV files (not only rows from the current command).

### Constitution Alignment Requirements *(mandatory)*

- **CAR-001 Testing**: Specification MUST define test-first expectations for each user story.
- **CAR-002 Code Quality**: Specification MUST define lint/type/static-analysis expectations for changed scope.
- **CAR-003 UX Consistency**: Specification MUST define user-facing conventions for errors, validation, and outputs.
- **CAR-004 Performance**: Specification MUST define measurable performance budget(s) and regression criteria.
- **CAR-005 Commit Control**: Workflow MUST include explicit user approval before any commit action.

### Key Entities *(include if feature involves data)*

- **TrainingRun**: Represents one complete model training/evaluation attempt for a single batch-size execution, including unique run identifier, batch size, execution target (CPU/XPU), start/end timestamps, and status.
- **DatasetSplitMetrics**: Represents metrics for one split (train, validation, or test), including loss and accuracy values over epochs or final evaluation points.
- **ModelArtifact**: Represents persisted model outputs from a run, including model version reference and associated run identifier.
- **ClassificationReport**: Represents class-level evaluation output including precision, recall, F1-score per class, and aggregate summary values.
- **ConfusionMatrixView**: Represents class-to-class prediction distribution data used for visualization and error analysis.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The primary workflow (train, evaluate, and visualize) completes successfully in at least 95% of valid runs.
- **SC-002**: The trained model achieves at least 97% accuracy on the MNIST test set for a standard training run.
- **SC-003**: Train, validation, and test curves are generated for 100% of successful runs and are interpretable without additional manual processing.
- **SC-004**: Classification report and confusion matrix are produced for 100% of successful evaluations.
- **SC-005**: 100% of runs honor explicit device selection; zero runs auto-fallback from XPU to CPU when XPU is selected but unavailable.

## Assumptions


- Target users are practitioners running experiments in a local or lab environment with access to MNIST data.
- Initial scope is MNIST-only classification for digits 0-9; non-MNIST datasets are out of scope for this feature.
- Users explicitly select the compute device for training and testing before execution starts.
- Run outputs and visualizations are stored locally and do not require external reporting services.
- Standard model training and evaluation practices are acceptable defaults unless later refined in planning.
- The containerized workflow runs on CPU only; XPU/GPU device passthrough is not supported inside Docker and is used only on the host directly.
