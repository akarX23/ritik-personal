# Data Model: MNIST Digit Classifier Pipeline

## Entities

### TrainingRun

Represents one end-to-end training invocation.

Fields:
- `run_id`: unique run identifier
- `device`: selected execution device, `cpu` or `xpu`
- `epochs`: total epoch count
- `start_time`: run start timestamp
- `end_time`: run end timestamp
- `training_time_seconds`: total elapsed training time in seconds
- `status`: running, completed, or failed
- `results_dir`: output directory for artifacts

### EpochMetrics

Represents per-epoch measurements for a specific run and split.

Fields:
- `run_id`: parent run identifier
- `epoch`: epoch number
- `split`: train, validation, or test
- `loss`: scalar loss value
- `accuracy`: scalar accuracy value
- `elapsed_seconds`: epoch duration
- `device`: device used for the epoch

### EvaluationSnapshot

Represents a validation or test checkpoint captured during training.

Fields:
- `run_id`: parent run identifier
- `epoch`: checkpoint epoch number
- `split`: validation or test
- `loss`: scalar loss value
- `accuracy`: scalar accuracy value
- `elapsed_seconds`: evaluation duration for the snapshot
- `device`: device used during evaluation

### DeviceTimeComparison

Represents timing comparison between one CPU run and one XPU run.

Fields:
- `comparison_id`: unique comparison identifier
- `cpu_run_id`: linked `TrainingRun` id for CPU
- `xpu_run_id`: linked `TrainingRun` id for XPU
- `cpu_training_time_seconds`: total CPU run training time
- `xpu_training_time_seconds`: total XPU run training time
- `time_delta_seconds`: `cpu_training_time_seconds - xpu_training_time_seconds`
- `speedup_ratio`: `cpu_training_time_seconds / xpu_training_time_seconds`
- `generated_at`: timestamp when comparison was computed

### ClassificationReport

Represents class-wise evaluation results.

Fields:
- `run_id`: parent run identifier
- `labels`: class labels 0-9
- `precision`: per-class precision values
- `recall`: per-class recall values
- `f1_score`: per-class F1 values
- `support`: per-class sample counts

### ConfusionMatrix

Represents prediction-vs-target counts.

Fields:
- `run_id`: parent run identifier
- `labels`: class labels 0-9
- `matrix`: 10x10 count matrix

### ProgressLogEvent

Represents one emitted progress log event for a run.

Fields:
- `run_id`: parent run identifier
- `timestamp`: event timestamp
- `level`: log level (`INFO`, `WARNING`, `ERROR`)
- `event_type`: lifecycle or epoch
- `message`: plain-text log line content
- `epoch`: optional epoch index (required for epoch events)
- `elapsed_seconds`: optional elapsed time for epoch events
- `loss`: optional loss value for epoch events
- `accuracy`: optional accuracy value for epoch events
- `log_file`: path to `run_<run_id>.log`

## Relationships

- One `TrainingRun` has many `EpochMetrics` rows.
- One `TrainingRun` has zero or more `EvaluationSnapshot` rows.
- One `TrainingRun` has one `ClassificationReport` and one `ConfusionMatrix` after testing.
- One `DeviceTimeComparison` references two `TrainingRun` records (`cpu` and `xpu`).
- One `TrainingRun` has many `ProgressLogEvent` records.
- `analysis.py` consumes the persisted CSV outputs derived from these entities.

## Validation Rules

- `device` must be explicitly set to `cpu` or `xpu`.
- `epoch` values are positive integers.
- `loss` and `accuracy` are numeric scalars saved with consistent column names across CSV files.
- `elapsed_seconds` and `training_time_seconds` are non-negative numeric values.
- `speedup_ratio` is computed only when `xpu_training_time_seconds > 0`.
- `labels` must cover the 10 MNIST classes.
- Visualization inputs must exist before analysis runs.
- Per-epoch log events must include `epoch`, `elapsed_seconds`, `loss`, and `accuracy`.
- Log file name must match `run_<run_id>.log`.

### ContainerConfig

Represents the runtime configuration for a containerised invocation.

Fields:
- `base_image`: Docker base image tag (e.g., `python:3.11-slim`)
- `device`: always `cpu` inside container; no XPU/GPU driver deps included
- `data_mount`: optional host path bound to `/app/data` inside the container
- `results_mount`: optional host path bound to `/app/results` inside the container
- `download_fallback`: boolean; if `true`, torchvision downloads MNIST when `/app/data` is empty

Validation Rules:
- `device` inside container is always `cpu`; any other value is rejected at runtime by `src.device`
- `data_mount` is optional; when absent, `download_fallback` must be `true`
- `results_mount` should be provided to persist output beyond container lifetime