# Data Model: MNIST Digit Classifier Pipeline

## Entities

### TrainingRun

Represents one end-to-end training invocation.

Fields:
- `run_id`: unique run identifier
- `device`: selected execution device, `cpu` or `gpu`
- `epochs`: total epoch count
- `start_time`: run start timestamp
- `end_time`: run end timestamp
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
- `device`: device used during evaluation

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

## Relationships

- One `TrainingRun` has many `EpochMetrics` rows.
- One `TrainingRun` has zero or more `EvaluationSnapshot` rows.
- One `TrainingRun` has one `ClassificationReport` and one `ConfusionMatrix` after testing.
- `analysis.py` consumes the persisted CSV outputs derived from these entities.

## Validation Rules

- `device` must be explicitly set to `cpu` or `gpu`.
- `epoch` values are positive integers.
- `loss` and `accuracy` are numeric scalars saved with consistent column names across CSV files.
- `labels` must cover the 10 MNIST classes.
- Visualization inputs must exist before analysis runs.