# CLI Contract: MNIST Digit Classifier Pipeline

## `python -m src.train`

Purpose: Train the MNIST model, run scheduled validation/testing, and write metrics CSV files.

Arguments:
- `-e` or `--epochs`: number of training epochs, default `10`
- `-r` or `--results`: results directory, default `./results`
- `-d` or `--device`: required device selection, `cpu` or `xpu`
- `-b` or `--batch`: batch size, default `64`
- `--batches`: comma-separated batch sizes (for example `32,64,128`); when provided, runs sequential training for each batch size
- `-lr` or `--lr`: learning rate, default `0.001`
- `-m` or `--data`: MNIST data directory, default `./data`

Behavior:
- Trains the fixed `784 -> 256 -> 128 -> 10` PyTorch model.
- If `--batches` is provided, executes all epochs sequentially for each batch size in listed order.
- Uses a distinct `run_id` per batch-size execution.
- Records per-epoch loss, accuracy, elapsed time, and device in CSV files.
- Records `batch_size` in train/validation/test metric rows to support shared-file comparisons.
- Runs validation and test checks every 5 epochs.
- Logs total run time as `training_time_seconds` in `run_summary.csv`.
- Emits plain-text progress logs to console and to `<results_dir>/run_<run_id>.log`.
- Emits concise per-epoch log lines with `epoch`, `elapsed_seconds`, `loss`, and `accuracy`.
- Fails if the requested device is unavailable (no automatic fallback).
- Uses SC-002 baseline where a standard run is expected to reach test accuracy `>= 0.97`.

Outputs:
- `metrics_train.csv`
- `metrics_validation.csv`
- `metrics_test.csv`
- `run_summary.csv`
- `predictions.csv` (final test predictions and labels)
- model checkpoint file (`model.pt`)
- per-run plain-text log file (`run_<run_id>.log`)

## `python -m src.analyze`

Purpose: Generate curves and comparisons from saved CSV outputs.

Arguments:
- `-r` or `--results`: results directory, required
- `-b` or `--filter-batch`: optional integer; when provided, scopes learning-curve plots and the epoch-comparison table in `results.md` to rows matching this batch size only

Behavior:
- Reads CSV files from the results directory.
- Generates training, validation, and testing curves.
- Produces classification metrics visualizations.
- Produces CPU-vs-XPU comparison plots when both device runs are available.
- Produces CPU-vs-XPU training time comparison from `run_summary.csv` entries.
- Generates `results.md` with:
	- final metrics comparison table by batch size (train/validation/test loss and accuracy) — always shows ALL batch sizes, never filtered
	- epoch comparison table sampled every 10 epochs plus final epoch — filtered to `--filter-batch` when specified
	- configuration section including device, epochs, learning rate, and compared batch-size list
- When `--filter-batch` is provided, learning-curve plots include only rows where `batch_size` matches the specified value; the epoch-comparison table is likewise restricted to those rows.
- When `--filter-batch` is omitted, all-batches behavior is preserved (identical to pre-FR-026 behavior).
- When `--filter-batch` specifies a batch size with no matching rows, exits with a non-zero status and an error message listing all batch sizes found in the CSV files.
- Comparison scope for `results.md` includes all matching historical rows in selected results-directory CSV files.

Outputs:
- learning curve images
- classification report plots
- confusion matrix image
- device quality comparison image when both device datasets exist
- device time comparison image when both device summaries exist
- markdown report file (`results.md`)
- per-run plain-text log file (`run_<run_id>.log`)
